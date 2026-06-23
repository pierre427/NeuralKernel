#!/usr/bin/env python3
"""Mode-B G3b: attack the G3 fact-use NEGATIVE with EARLY + TOKEN-LOCALIZED injection.

G3 found that injecting a fact and asking the model to TRANSFORM it failed (held-out op 0.0): a LATE (site 24),
BROADCAST (all positions) write acts as output-steering, not a readable fact. Hypothesis for the fix:
  * EARLY site (default 8) -> the downstream stack has 40 layers to read + use the fact (vs 24's late bias).
  * TOKEN-LOCALIZED -> write the value into ONE position's residual (like an attendable token), not a global bias.

Same protocol as G3 (secret digit ONLY via injection; train ops {state,+1,+2,double}; HELD-OUT op +3; single-digit
answer) and the same controls (held-out op, same-injection/different-prompt discriminator, echo, causal, placebo,
==0). Decisive metric: HELD-OUT op accuracy — if it rises well above G3's 0.0, fact-use is crackable.

Args let you ablate: --site, --localize/--broadcast, --pos {last,first-after-prompt}. Run:
  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_b_north_g3b.py [--site 8]
"""
from __future__ import annotations
import os
import sys, os, json, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load
from kernel.mode_b import ModeBInjector

MODEL = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")
K = 10
OPS = {
    "state":  ("State the secret digit.", lambda x: x % 10),
    "add1":   ("Add 1 to the secret digit (mod 10).", lambda x: (x + 1) % 10),
    "add2":   ("Add 2 to the secret digit (mod 10).", lambda x: (x + 2) % 10),
    "double": ("Double the secret digit (mod 10).", lambda x: (2 * x) % 10),
    "add3":   ("Add 3 to the secret digit (mod 10).", lambda x: (x + 3) % 10),   # HELD-OUT
}
TRAIN_OPS = ["state", "add1", "add2", "double"]
HELDOUT_OP = "add3"


def run_local(model, ids, injector, site, result_vec, pos, localize):
    """Prefill forward with the injector applied AFTER block `site`, optionally MASKED to a single position `pos`."""
    inner = model.model
    h = inner.embed_tokens(ids)
    for li, layer in enumerate(inner.layers):
        win = inner.window_size if layer.self_attn.use_sliding_window else None
        mask = create_attention_mask(h, None, window_size=win)
        h = layer(h, mask, None)
        if injector is not None and injector.armed and li == site:
            delta = injector.delta(h, result_vec).astype(h.dtype)
            if localize and pos is not None:
                T = h.shape[-2]
                delta = delta * (mx.arange(T) == pos).astype(h.dtype).reshape(1, T, 1)
            h = h + delta
    return inner.embed_tokens.as_linear(inner.norm(h)) * model.args.logit_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, default=8)
    ap.add_argument("--localize", dest="localize", action="store_true", default=True)
    ap.add_argument("--broadcast", dest="localize", action="store_false")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=260)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    print(f"[load] {MODEL}  | site={args.site} localize={args.localize}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size)
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(K)]
    onehot = mx.eye(K)

    def enc(op):
        msg = f"A single secret digit (0-9) has been provided to you out of band. {OPS[op][0]} Reply with only the resulting digit."
        return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, reasoning=False)
    ids_by_op = {op: enc(op) for op in OPS}
    pos_by_op = {op: len(ids_by_op[op]) - 1 for op in OPS}      # inject at the LAST prompt token (attended by the answer)

    def first_token(inj, op, X):
        rv = onehot[X][None, :] if inj is not None else None
        logits = run_local(model, mx.array([ids_by_op[op]], dtype=mx.int32), inj, args.site, rv, pos_by_op[op], args.localize)
        return int(mx.argmax(logits[0, -1, :]).item())

    def out_digit(inj, op, X):
        return tok.decode([first_token(inj, op, X)]).strip()

    def train(placebo):
        rng = random.Random(7)
        inj = ModeBInjector(d, bottleneck=64, result_dim=K)
        inj.up.weight = mx.random.normal(inj.up.weight.shape) * 1e-3
        mx.eval(inj.parameters())
        opt = optim.Adam(learning_rate=args.lr)

        def loss_fn(injector, ids, rv, pos, target):
            logits = run_local(model, ids, injector, args.site, rv, pos, args.localize)
            return nn.losses.cross_entropy(logits[:, -1, :], target, reduction="mean")

        lvg = nn.value_and_grad(inj, loss_fn)
        for ep in range(args.steps):
            tot = 0.0
            for _ in range(args.n_train):
                X = rng.randint(0, 9); op = rng.choice(TRAIN_OPS)
                tgt = rng.randint(0, 9) if placebo else OPS[op][1](X)
                loss, grads = lvg(inj, mx.array([ids_by_op[op]], dtype=mx.int32), onehot[X][None, :],
                                  pos_by_op[op], mx.array([digit_ids[tgt]]))
                opt.update(inj, grads); mx.eval(inj.parameters(), opt.state)
                tot += float(loss)
            print(f"  [{'placebo' if placebo else 'real'}] epoch {ep+1}/{args.steps} loss={tot/args.n_train:.3f}", flush=True)
        return inj

    print(f"[train] EARLY+LOCALIZED fact injection vs G3's late+broadcast (held-out op={HELDOUT_OP})", flush=True)
    real = train(placebo=False)
    placebo = train(placebo=True)

    rng = random.Random(2); Xs = [rng.randint(0, 9) for _ in range(args.n_eval)]
    def acc(inj, op): return round(sum(out_digit(inj, op, x) == str(OPS[op][1](x)) for x in Xs) / len(Xs), 3)

    train_acc = {op: acc(real, op) for op in TRAIN_OPS}
    heldout = acc(real, HELDOUT_OP)
    unaided = acc(None, HELDOUT_OP)
    placebo_h = acc(placebo, HELDOUT_OP)
    echo = round(sum(out_digit(real, HELDOUT_OP, x) == str(x) and OPS[HELDOUT_OP][1](x) != x for x in Xs)
                 / sum(1 for x in Xs if OPS[HELDOUT_OP][1](x) != x), 3)
    same_inj = round(sum(out_digit(real, "state", x) == str(x) and out_digit(real, HELDOUT_OP, x) == str((x + 3) % 10)
                         for x in Xs) / len(Xs), 3)
    real.armed = False
    disabled_eq = all(first_token(real, HELDOUT_OP, x) == first_token(None, HELDOUT_OP, x) for x in Xs[:10])
    real.armed = True

    cracked = heldout > 0.5 and heldout > placebo_h + 0.3 and same_inj > 0.4 and echo < 0.3 and disabled_eq
    improved = heldout > 0.2  # any meaningful lift over G3's 0.0
    verdict = {
        "config": {"site": args.site, "localize": args.localize, "pos": "last_prompt_token"},
        "train_op_acc": train_acc, "heldout_op_acc": heldout, "unaided": unaided, "placebo_heldout": placebo_h,
        "echo_rate": echo, "same_injection_diff_prompt": same_inj, "disabled_eq": bool(disabled_eq),
        "G3_baseline_heldout": 0.0,
        "VERDICT": ("FACT-USE CRACKED (early+localized): the model reads the injected fact and applies a HELD-OUT op"
                    if cracked else
                    f"IMPROVED over G3 but not solved (held-out {heldout} vs G3 0.0) — fact-use partial" if improved else
                    "STILL NEGATIVE: early+localized did not enable held-out fact-use either (deepens the G3 boundary)"),
    }
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(verdict, open(os.path.join(logdir, "mode_b_north_g3b.json"), "w"), indent=2, default=str)
    print("\n===== Mode-B G3b (early+localized fact injection) =====", flush=True)
    print(f"  train ops: {train_acc}", flush=True)
    print(f"  HELD-OUT op (+3) = {heldout}   (G3 was 0.0 | unaided={unaided} placebo={placebo_h})", flush=True)
    print(f"  same-injection/diff-prompt = {same_inj}   echo={echo}   disabled=={disabled_eq}", flush=True)
    print(f"VERDICT: {verdict['VERDICT']}", flush=True)
    print("(-> logs/mode_b_north_g3b.json)", flush=True)


if __name__ == "__main__":
    main()
