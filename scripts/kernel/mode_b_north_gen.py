#!/usr/bin/env python3
"""Mode-B G1: is the residual result-write TASK-AGNOSTIC? Train ONE value-only injector (conditioned on the result
value, NOT on which primitive produced it) across several tasks, then test it on a HELD-OUT task type it never saw.

If a value-only write faithfully injects on an UNSEEN task, the mechanism generalizes — Mode B is a general
"the answer is v" write, not a letter-count-specific hack (the B2 scope caveat). Recipe = B2 (North FROZEN, only
the injector trainable, autodiff through 24 MoE layers); the only change is a multi-task prompt distribution and a
value-only (task-blind) injector.

Tasks (each -> a single-digit result 0..9, distinct reasoning):
  letter_count  count a letter in a word          [TRAIN]
  sum_mod       (a+b) mod 10                       [TRAIN]
  len_mod       length of a word mod 10            [TRAIN]
  max_digit     largest digit in a list           [HELD-OUT — never trained]

Controls ("fail->pass is not proof"): deployment per task (inject true -> output==true); causal (inject WRONG ->
output follows it); placebo (decoupled target -> chance); ==0 when disabled. The killer metric is HELD-OUT-TASK
deployment: high => the write is genuinely value-only / task-agnostic.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_b_north_gen.py [--steps N]
"""
from __future__ import annotations
import os
import sys, os, json, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.utils import load
from kernel.mode_b import ModeBInjector, run_stack_modeb

MODEL = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")
K = 10
ALPHA = "abcdefghijklmnopqrstuvwxyz"
_ONLY = " Reply with only the digit."


def t_letter_count(rng):
    ch = rng.choice(ALPHA); k = rng.randint(0, 9)
    others = [c for c in ALPHA if c != ch]
    n = rng.randint(max(k, 4), max(k + 7, 11))
    s = [ch] * k + [rng.choice(others) for _ in range(n - k)]; rng.shuffle(s)
    return f"How many times does the letter '{ch}' appear in \"{''.join(s)}\"?" + _ONLY, k

def t_sum_mod(rng):
    a, b = rng.randint(0, 20), rng.randint(0, 20)
    return f"What is ({a} + {b}) mod 10?" + _ONLY, (a + b) % 10

def t_len_mod(rng):
    n = rng.randint(1, 19); w = "".join(rng.choice(ALPHA) for _ in range(n))
    return f"What is the length of the string \"{w}\" mod 10?" + _ONLY, n % 10

def t_max_digit(rng):
    ds = [rng.randint(0, 9) for _ in range(rng.randint(2, 6))]
    return f"What is the largest digit in the list {ds}?" + _ONLY, max(ds)

TASKS = {"letter_count": t_letter_count, "sum_mod": t_sum_mod, "len_mod": t_len_mod, "max_digit": t_max_digit}
TRAIN_TASKS = ["letter_count", "sum_mod", "len_mod"]
HELDOUT_TASK = "max_digit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-per-task", type=int, default=110)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size); site = int(model.args.num_hidden_layers) // 2
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(K)]
    onehot = mx.eye(K)

    def enc(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, reasoning=False)

    def gen_set(task, n, seed):
        rng = random.Random(seed); out = []
        for _ in range(n):
            p, r = TASKS[task](rng); out.append((enc(p), r))
        return out

    # train pool: 3 tasks; eval: all 4 (incl held-out task), fresh seeds
    train_pool = []
    for ti, t in enumerate(TRAIN_TASKS):
        train_pool += gen_set(t, args.n_per_task, seed=100 + ti)
    eval_sets = {t: gen_set(t, args.n_eval, seed=200 + i) for i, t in enumerate(TASKS)}

    def first_token(inj, ids, v):
        rv = onehot[v][None, :] if inj is not None else None
        logits = run_stack_modeb(model, mx.array([ids], dtype=mx.int32), inj, site, rv)
        return int(mx.argmax(logits[0, -1, :]).item())

    def emits(inj, ids, want):
        return tok.decode([first_token(inj, ids, want if inj is not None else 0)]).strip() == str(want)

    def train(placebo):
        rng = random.Random(7)
        inj = ModeBInjector(d, bottleneck=64, result_dim=K)
        inj.up.weight = mx.random.normal(inj.up.weight.shape) * 1e-3
        mx.eval(inj.parameters())
        opt = optim.Adam(learning_rate=args.lr)

        def loss_fn(injector, ids, rv, target):
            return nn.losses.cross_entropy(run_stack_modeb(model, ids, injector, site, rv)[:, -1, :], target, reduction="mean")

        lvg = nn.value_and_grad(inj, loss_fn)
        for ep in range(args.steps):
            order = list(range(len(train_pool))); rng.shuffle(order); tot = 0.0
            for i in order:
                ids, true_r = train_pool[i]
                v = rng.randint(0, K - 1) if (placebo or rng.random() < 0.5) else true_r
                target_v = rng.randint(0, K - 1) if placebo else v
                loss, grads = lvg(inj, mx.array([ids], dtype=mx.int32), onehot[v][None, :], mx.array([digit_ids[target_v]]))
                opt.update(inj, grads); mx.eval(inj.parameters(), opt.state)
                tot += float(loss)
            print(f"  [{'placebo' if placebo else 'real'}] epoch {ep+1}/{args.steps} loss={tot/len(order):.3f}", flush=True)
        return inj

    print(f"[train] site={site} train_tasks={TRAIN_TASKS} heldout={HELDOUT_TASK} n_train={len(train_pool)}", flush=True)
    real = train(placebo=False)
    placebo = train(placebo=True)

    deploy, unaided = {}, {}
    for t, ev in eval_sets.items():
        deploy[t] = round(sum(emits(real, ids, r) for ids, r in ev) / len(ev), 3)
        unaided[t] = round(sum(emits(None, ids, r) for ids, r in ev) / len(ev), 3)
    # causal + placebo measured on the held-out task (the hardest, most honest slice)
    ho = eval_sets[HELDOUT_TASK]; rng = random.Random(9); causal = 0
    for ids, r in ho:
        v = rng.choice([x for x in range(K) if x != r])
        causal += int(tok.decode([first_token(real, ids, v)]).strip() == str(v))
    causal = round(causal / len(ho), 3)
    placebo_dep = round(sum(emits(placebo, ids, r) for ids, r in ho) / len(ho), 3)
    real.armed = False
    disabled_eq = all(first_token(real, ids, r) == first_token(None, ids, r) for ids, r in ho[:10])
    real.armed = True

    ho_dep = deploy[HELDOUT_TASK]
    success = (ho_dep > 0.7 and all(deploy[t] > 0.7 for t in TASKS) and causal > 0.5
               and placebo_dep < 0.3 and disabled_eq)
    verdict = {
        "site": site, "train_tasks": TRAIN_TASKS, "heldout_task": HELDOUT_TASK,
        "deployment_per_task": deploy, "unaided_per_task": unaided,
        "heldout_deployment": ho_dep, "heldout_causal": causal, "heldout_placebo": placebo_dep,
        "disabled_eq": bool(disabled_eq), "chance": 1.0 / K,
        "VERDICT": ("TASK-AGNOSTIC: a value-only residual write injects faithfully across tasks INCLUDING the "
                    "held-out task type -> Mode B generalizes" if success else
                    "PARTIAL/NEGATIVE (honest): see per-task deployment / held-out metrics"),
    }
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(verdict, open(os.path.join(logdir, "mode_b_north_gen.json"), "w"), indent=2, default=str)
    print("\n===== Mode-B G1 (task-agnostic injection) =====", flush=True)
    for t in TASKS:
        tag = "  [HELD-OUT]" if t == HELDOUT_TASK else ""
        print(f"  {t:14s} deployment={deploy[t]:.3f}  unaided={unaided[t]:.3f}{tag}", flush=True)
    print(f"  held-out causal={causal}  held-out placebo={placebo_dep}  disabled=={disabled_eq}  chance={1.0/K:.2f}", flush=True)
    print(f"VERDICT: {verdict['VERDICT']}", flush=True)
    print("(-> logs/mode_b_north_gen.json)", flush=True)


if __name__ == "__main__":
    main()
