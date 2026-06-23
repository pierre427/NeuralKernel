#!/usr/bin/env python3
"""Mode-B B2 (the research CRUX): train the result-conditioned injector on REAL North to WRITE a verified result
into the residual stream so the model EMITS it — through 24 nonlinear MoE layers — instead of as tokens (Mode A).

Task: letter-count questions ("how many 'r' in <word>? reply with only the digit"). The kernel's handler knows
the true count; Mode B injects it as a one-hot vector at site 24. We train ONLY the injector (North frozen) to
make the answer token equal the INJECTED value v (a mix of true-count and random override values), so it learns
faithful result-WRITING, not the task. Then we test, against the independent oracle (word.count) and with the
project's controls ("fail->pass is not proof"):

  unaided        no injector                          -> baseline P(first token == true count)
  deployment     inject one-hot(true count)           -> P(output == true count)   [the use case]
  causal/faithful inject one-hot(v != true count)      -> P(output == v)            [output TRACKS injection => it
                                                          is genuinely WRITING the result, not recomputing it]
  placebo        injector trained with target DECOUPLED from the injected value (shuffled) -> should collapse
  disabled       armed=False                          -> == base (the B0 ==0 guarantee for a trained injector)

A real Mode-B success = deployment >> unaided AND causal/faithful HIGH AND placebo collapses. A partial/negative
result (e.g. it learns to recompute, or injection through the MoE stack is weak) is a legitimate finding for an
open research gap — reported honestly.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_b_north.py [--steps N]
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
K = 10                      # counts / injectable values 0..9 (single-digit answers)
ALPHA = "abcdefghijklmnopqrstuvwxyz"


def make_examples(n_per_count, ch="r", seed=0):
    rng = random.Random(seed)
    others = [c for c in ALPHA if c != ch]
    ex = []
    for k in range(K):
        for _ in range(n_per_count):
            length = rng.randint(max(k, 4), max(k + 7, 11))
            s = [ch] * k + [rng.choice(others) for _ in range(length - k)]
            rng.shuffle(s)
            w = "".join(s)
            assert w.count(ch) == k
            ex.append((w, ch, k))
    rng.shuffle(ex)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3, help="epochs over the train set (B=1)")
    ap.add_argument("--n-per-count", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size); nL = int(model.args.num_hidden_layers); site = nL // 2
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(K)]

    def enc(word, ch):
        msg = f"How many times does the letter '{ch}' appear in \"{word}\"? Reply with only the digit."
        return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, reasoning=False)

    train_ex = make_examples(args.n_per_count, seed=1)
    eval_ex = make_examples(max(1, args.n_eval // K), seed=2)[:args.n_eval]
    train_ids = [(enc(w, c), k) for (w, c, k) in train_ex]
    eval_ids = [(enc(w, c), k) for (w, c, k) in eval_ex]
    onehot = mx.eye(K)

    def first_token(injector, ids, v):
        rv = onehot[v][None, :] if injector is not None else None
        logits = run_stack_modeb(model, mx.array([ids], dtype=mx.int32), injector, site, rv)
        return int(mx.argmax(logits[0, -1, :]).item())

    def emits(injector, ids, want):
        t = first_token(injector, ids, want if injector is not None else 0)
        return tok.decode([t]).strip() == str(want)

    def train(placebo: bool):
        rng = random.Random(7)
        inj = ModeBInjector(d, bottleneck=64, result_dim=K)
        inj.up.weight = mx.random.normal(inj.up.weight.shape) * 1e-3      # small-random: gradient flows from step 1
        mx.eval(inj.parameters())                                        # (==0-when-disabled is via armed=False, B0)
        opt = optim.Adam(learning_rate=args.lr)

        def loss_fn(injector, ids, rv, target):
            logits = run_stack_modeb(model, ids, injector, site, rv)
            return nn.losses.cross_entropy(logits[:, -1, :], target, reduction="mean")

        lvg = nn.value_and_grad(inj, loss_fn)
        for ep in range(args.steps):
            order = list(range(len(train_ids))); rng.shuffle(order)
            tot = 0.0
            for i in order:
                ids, true_k = train_ids[i]
                v = rng.randint(0, K - 1) if (placebo or rng.random() < 0.5) else true_k   # inject value
                target_v = rng.randint(0, K - 1) if placebo else v   # placebo: target DECOUPLED from injected v
                rv = onehot[v][None, :]
                loss, grads = lvg(inj, mx.array([ids], dtype=mx.int32), rv, mx.array([digit_ids[target_v]]))
                opt.update(inj, grads); mx.eval(inj.parameters(), opt.state)
                tot += float(loss)
            print(f"  [{'placebo' if placebo else 'real'}] epoch {ep+1}/{args.steps} mean_loss={tot/len(order):.3f}", flush=True)
        return inj

    print(f"[train] site={site} steps={args.steps} n_train={len(train_ids)} lr={args.lr}", flush=True)
    real = train(placebo=False)
    placebo = train(placebo=True)

    # ---- eval (held-out) ----
    unaided = sum(emits(None, ids, k) for ids, k in eval_ids) / len(eval_ids)
    deploy = sum(emits(real, ids, k) for ids, k in eval_ids) / len(eval_ids)
    # causal/faithfulness: inject a WRONG value v != true; does the output follow v?
    rng = random.Random(9); causal_hits = 0
    for ids, k in eval_ids:
        v = rng.choice([x for x in range(K) if x != k])
        causal_hits += int(tok.decode([first_token(real, ids, v)]).strip() == str(v))
    causal = causal_hits / len(eval_ids)
    placebo_deploy = sum(emits(placebo, ids, k) for ids, k in eval_ids) / len(eval_ids)
    # disabled (armed=False) == unaided exactly (B0 guarantee for a trained injector)
    real.armed = False
    disabled_eq = all(first_token(real, ids, k) == first_token(None, ids, k) for ids, k in eval_ids[:12])
    real.armed = True

    chance = 1.0 / K
    verdict = {
        "site": site, "n_train": len(train_ids), "n_eval": len(eval_ids), "steps": args.steps, "chance": chance,
        "unaided_acc": round(unaided, 3), "deployment_acc": round(deploy, 3),
        "causal_faithfulness": round(causal, 3), "placebo_deploy_acc": round(placebo_deploy, 3),
        "disabled_eq_unaided": bool(disabled_eq),
    }
    success = (deploy > unaided + 0.2 and causal > 0.5 and placebo_deploy < deploy - 0.2 and disabled_eq)
    verdict["VERDICT"] = (
        "MODE-B WORKS: a trained residual write sets North's output (deployment>>unaided, output tracks the "
        "injected value, placebo collapses, ==0 when disabled)" if success else
        "PARTIAL/NEGATIVE (honest, open gap): see metrics — "
        + ("; ".join(s for s, c in [
            ("deployment not > unaided+0.2", not deploy > unaided + 0.2),
            ("causal/faithfulness <=0.5 (output does not track injection)", not causal > 0.5),
            ("placebo did not collapse", not placebo_deploy < deploy - 0.2),
            ("disabled != base", not disabled_eq)] if c) or "thresholds borderline"))

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(verdict, open(os.path.join(logdir, "mode_b_north.json"), "w"), indent=2, default=str)
    print("\n===== Mode-B B2 (North) =====", flush=True)
    for kk in ("unaided_acc", "deployment_acc", "causal_faithfulness", "placebo_deploy_acc", "disabled_eq_unaided"):
        print(f"  {kk:22s} {verdict[kk]}", flush=True)
    print(f"  chance={chance:.3f}", flush=True)
    print(f"VERDICT: {verdict['VERDICT']}", flush=True)
    print("(-> logs/mode_b_north.json)", flush=True)


if __name__ == "__main__":
    main()
