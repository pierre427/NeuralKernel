#!/usr/bin/env python3
"""Mode-B G2: MULTI-TOKEN value injection by PER-TOKEN re-injection. B2/G1 proved single-token value delivery
(inject one-hot(v) -> model emits v, ~100%, task-agnostic). A multi-token value (e.g. a 2-digit number) needs the
kernel to re-inject the position-appropriate digit at each answer step — exactly how a real kernel would drive it.

G2 tests whether per-token re-injection COMPOSES into a correct multi-digit value (or whether position-2 degrades
because its residual context now contains the already-emitted first digit). The injector is the SAME value-only
10-way digit injector; the kernel (which knows the verified value V) drives the right digit per step.

Task: a SECRET 2-digit number 00-99, only via injection; "state it as two digits". Train the injector on BOTH
answer positions: position-0 context = prompt; position-1 context = prompt + the (true) first digit token. Eval is
a per-token-DRIVEN decode (step0 inject d0 -> tok0; step1 with input prompt+tok0 inject d1 -> tok1) — using the
model's OWN tok0, not teacher-forced.

Controls ("fail->pass is not proof"): per-digit + FULL-value accuracy; causal (inject a different value V' ->
output follows V', both digits); placebo (decoupled target) collapses; ==0 disabled; unaided ~ chance (the value
is only available via injection). Run:
  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_b_north_g2.py [--steps N]
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
PROMPT = ("A secret 2-digit number (00-99) has been provided to you out of band. "
          "State it as exactly two digits, most significant first. Reply with only the two digits.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=140, help="values/epoch (each -> 2 position examples)")
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size); site = int(model.args.num_hidden_layers) // 2
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(K)]
    onehot = mx.eye(K)
    base_ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, reasoning=False)

    def fwd_last(inj, seq, digit):
        rv = onehot[digit][None, :] if inj is not None else None
        logits = run_stack_modeb(model, mx.array([seq], dtype=mx.int32), inj, site, rv)
        return int(mx.argmax(logits[0, -1, :]).item())

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
            tot = 0.0
            for _ in range(args.n_train):
                V = rng.randint(0, 99); d0, d1 = V // 10, V % 10
                pos = rng.randint(0, 1)
                seq = list(base_ids) if pos == 0 else list(base_ids) + [digit_ids[d0]]   # pos-1 ctx = prompt + d0
                dig = d0 if pos == 0 else d1
                tgt = rng.randint(0, 9) if placebo else dig
                loss, grads = lvg(inj, mx.array([seq], dtype=mx.int32), onehot[dig][None, :], mx.array([digit_ids[tgt]]))
                opt.update(inj, grads); mx.eval(inj.parameters(), opt.state)
                tot += float(loss)
            print(f"  [{'placebo' if placebo else 'real'}] epoch {ep+1}/{args.steps} loss={tot/args.n_train:.3f}", flush=True)
        return inj

    print(f"[train] site={site} task=2-digit value, per-token re-injection (pos0=prompt, pos1=prompt+d0)", flush=True)
    real = train(placebo=False)
    placebo = train(placebo=True)

    rng = random.Random(2)
    Vs = [rng.randint(0, 99) for _ in range(args.n_eval)]

    def drive(inj, V):
        """per-token-DRIVEN decode: inject d0 -> tok0; then input prompt+tok0, inject d1 -> tok1. Uses model's own tok0."""
        d0, d1 = V // 10, V % 10
        t0 = fwd_last(inj, list(base_ids), d0)
        t1 = fwd_last(inj, list(base_ids) + [t0], d1)
        return tok.decode([t0]).strip(), tok.decode([t1]).strip()

    def metrics(inj, Vs):
        pos0 = pos1 = full = 0
        for V in Vs:
            a, b = drive(inj, V); d0, d1 = V // 10, V % 10
            pos0 += int(a == str(d0)); pos1 += int(b == str(d1)); full += int(a == str(d0) and b == str(d1))
        n = len(Vs)
        return round(pos0 / n, 3), round(pos1 / n, 3), round(full / n, 3)

    p0, p1, full = metrics(real, Vs)
    # unaided: no injection -> the model cannot know the secret value
    un0 = un1 = unf = 0
    for V in Vs:
        d0, d1 = V // 10, V % 10
        t0 = fwd_last(None, list(base_ids), 0); t1 = fwd_last(None, list(base_ids) + [t0], 0)
        unf += int(tok.decode([t0]).strip() == str(d0) and tok.decode([t1]).strip() == str(d1))
    unaided_full = round(unf / len(Vs), 3)
    # causal: inject a DIFFERENT value's digits -> output follows it (both digits)
    causal = 0
    for V in Vs:
        Vp = (V + 37) % 100
        a, b = drive(real, Vp)
        causal += int(a == str(Vp // 10) and b == str(Vp % 10))
    causal = round(causal / len(Vs), 3)
    _, _, placebo_full = metrics(placebo, Vs)
    real.armed = False
    disabled_eq = all(fwd_last(real, list(base_ids), V // 10) == fwd_last(None, list(base_ids), 0) for V in Vs[:10])
    real.armed = True

    chance_full = 1.0 / 100
    success = (full > 0.7 and causal > 0.6 and placebo_full < 0.1 and disabled_eq)
    verdict = {
        "site": site, "task": "2-digit (00-99) per-token re-injection",
        "pos0_acc": p0, "pos1_acc": p1, "full_value_acc": full, "unaided_full": unaided_full,
        "causal_full": causal, "placebo_full": placebo_full, "disabled_eq": bool(disabled_eq),
        "chance_full": chance_full,
        "VERDICT": ("MULTI-TOKEN WORKS: per-token re-injection composes to a correct multi-digit value (full-value "
                    "acc high, output tracks the injected value, placebo collapses, ==0 disabled)" if success else
                    "PARTIAL/NEGATIVE (honest): see per-position vs full-value accuracy"),
    }
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(verdict, open(os.path.join(logdir, "mode_b_north_g2.json"), "w"), indent=2, default=str)
    print("\n===== Mode-B G2 (multi-token / per-token re-injection) =====", flush=True)
    print(f"  pos0(tens)={p0}  pos1(ones)={p1}  FULL-VALUE={full}  (unaided_full={unaided_full}, chance={chance_full})", flush=True)
    print(f"  causal_full={causal}  placebo_full={placebo_full}  disabled=={disabled_eq}", flush=True)
    print(f"VERDICT: {verdict['VERDICT']}", flush=True)
    print("(-> logs/mode_b_north_g2.json)", flush=True)


if __name__ == "__main__":
    main()
