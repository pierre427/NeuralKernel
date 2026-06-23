#!/usr/bin/env python3
"""Mode-B G3 (the real prize): can a residual write inject a fact the MODEL USES in further reasoning — not just
emits? B2/G1 proved inject-as-final-answer (output == injected value). G3 tests inject-as-INTERMEDIATE-FACT:
the fact is a SECRET digit NOT in the prompt (so the model can ONLY know it from the injection), and the prompt
asks for a TRANSFORM op(X) mod 10. A correct transformed output means the model READ the injected fact and
operated on it.

Anti-confound controls (the whole point — "fail->pass is not proof"):
  * HELD-OUT OP: train ops {state, +1, +2, double}; TEST op (+3) is never trained. If +3 works, the injector
    can't have hardcoded it -> the prompt's op is being applied to the read fact.
  * SAME-INJECTION / DIFFERENT-PROMPT: identical injected X, prompt "state it" -> X AND prompt "+3" -> X+3.
    Output tracks the PROMPT while injection is fixed -> it's a readable fact, not a fixed output-steer.
  * ECHO confound: on a transform prompt, P(output == X) must be LOW (it's transforming, not echoing).
  * CAUSAL: inject X vs X' on the held-out op -> output op(X) vs op(X').
  * PLACEBO (decoupled target) collapses; ==0 when disabled.
Unaided (no injection) ~ chance, since the secret digit is ONLY available via the injection.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_b_north_g3.py [--steps N]
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
OPS = {  # name -> (request text, op(X)->result mod 10)
    "state":  ("State the secret digit.", lambda x: x % 10),
    "add1":   ("Add 1 to the secret digit (mod 10).", lambda x: (x + 1) % 10),
    "add2":   ("Add 2 to the secret digit (mod 10).", lambda x: (x + 2) % 10),
    "double": ("Double the secret digit (mod 10).", lambda x: (2 * x) % 10),
    "add3":   ("Add 3 to the secret digit (mod 10).", lambda x: (x + 3) % 10),   # HELD-OUT
}
TRAIN_OPS = ["state", "add1", "add2", "double"]
HELDOUT_OP = "add3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=280)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size); site = int(model.args.num_hidden_layers) // 2
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(K)]
    onehot = mx.eye(K)

    def enc(op):
        req = OPS[op][0]
        msg = f"A single secret digit (0-9) has been provided to you out of band. {req} Reply with only the resulting digit."
        return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, reasoning=False)

    ids_by_op = {op: enc(op) for op in OPS}                       # prompt is op-only (X is NOT in the text)

    def first_token(inj, op, X):
        rv = onehot[X][None, :] if inj is not None else None
        logits = run_stack_modeb(model, mx.array([ids_by_op[op]], dtype=mx.int32), inj, site, rv)
        return int(mx.argmax(logits[0, -1, :]).item())

    def out_digit(inj, op, X):
        return tok.decode([first_token(inj, op, X)]).strip()

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
                X = rng.randint(0, 9); op = rng.choice(TRAIN_OPS)
                tgt = rng.randint(0, 9) if placebo else OPS[op][1](X)     # placebo: target decoupled from fact+op
                loss, grads = lvg(inj, mx.array([ids_by_op[op]], dtype=mx.int32), onehot[X][None, :], mx.array([digit_ids[tgt]]))
                opt.update(inj, grads); mx.eval(inj.parameters(), opt.state)
                tot += float(loss)
            print(f"  [{'placebo' if placebo else 'real'}] epoch {ep+1}/{args.steps} loss={tot/args.n_train:.3f}", flush=True)
        return inj

    print(f"[train] site={site} train_ops={TRAIN_OPS} HELD-OUT op={HELDOUT_OP} (secret digit is NOT in the prompt)", flush=True)
    real = train(placebo=False)
    placebo = train(placebo=True)

    rng = random.Random(2)
    Xs = [rng.randint(0, 9) for _ in range(args.n_eval)]

    def acc(inj, op, xs):
        return round(sum(out_digit(inj, op, x) == str(OPS[op][1](x)) for x in xs) / len(xs), 3)

    train_op_acc = {op: acc(real, op, Xs) for op in TRAIN_OPS}
    heldout_acc = acc(real, HELDOUT_OP, Xs)
    unaided_heldout = acc(None, HELDOUT_OP, Xs)
    echo_on_heldout = round(sum(out_digit(real, HELDOUT_OP, x) == str(x) and OPS[HELDOUT_OP][1](x) != x for x in Xs)
                            / sum(1 for x in Xs if OPS[HELDOUT_OP][1](x) != x), 3)  # output==X where X+3!=X
    placebo_heldout = acc(placebo, HELDOUT_OP, Xs)
    # SAME-INJECTION / DIFFERENT-PROMPT discriminator: for the same X, state->X AND add3->(X+3)
    same_inj = round(sum(out_digit(real, "state", x) == str(x) and out_digit(real, HELDOUT_OP, x) == str((x + 3) % 10)
                         for x in Xs) / len(Xs), 3)
    # causal: inject X vs X' on the held-out op -> output op(X) vs op(X')
    causal = round(sum(out_digit(real, HELDOUT_OP, x) == str((x + 3) % 10)
                       and out_digit(real, HELDOUT_OP, (x + 5) % 10) == str(((x + 5) % 10 + 3) % 10) for x in Xs) / len(Xs), 3)
    real.armed = False
    disabled_eq = all(first_token(real, HELDOUT_OP, x) == first_token(None, HELDOUT_OP, x) for x in Xs[:10])
    real.armed = True

    chance = 1.0 / K
    success = (heldout_acc > 0.6 and heldout_acc > placebo_heldout + 0.4 and same_inj > 0.5
               and echo_on_heldout < 0.3 and disabled_eq)
    verdict = {
        "site": site, "train_ops": TRAIN_OPS, "heldout_op": HELDOUT_OP, "chance": chance,
        "train_op_acc": train_op_acc, "heldout_op_acc": heldout_acc, "unaided_heldout": unaided_heldout,
        "echo_rate_on_heldout(output==X)": echo_on_heldout, "placebo_heldout": placebo_heldout,
        "same_injection_diff_prompt": same_inj, "causal_heldout": causal, "disabled_eq": bool(disabled_eq),
        "VERDICT": ("INTERMEDIATE-FACT INJECTION WORKS: the model reads the injected secret digit and applies the "
                    "prompt's op — incl. a HELD-OUT op — same injection yields different outputs per prompt, not "
                    "echoing, placebo collapses" if success else
                    "PARTIAL/NEGATIVE (honest boundary): see metrics — Mode B may deliver emittable values but not "
                    "yet model-transformable facts"),
    }
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(verdict, open(os.path.join(logdir, "mode_b_north_g3.json"), "w"), indent=2, default=str)
    print("\n===== Mode-B G3 (intermediate-fact injection) =====", flush=True)
    print(f"  train ops: {train_op_acc}", flush=True)
    print(f"  HELD-OUT op (+3) acc = {heldout_acc}   (unaided={unaided_heldout}, placebo={placebo_heldout}, chance={chance:.2f})", flush=True)
    print(f"  echo-rate (output==X on +3) = {echo_on_heldout}  [low => transforming, not echoing]", flush=True)
    print(f"  same-injection/diff-prompt (state->X AND +3->X+3) = {same_inj}  [the discriminator]", flush=True)
    print(f"  causal (X vs X' both transform correctly) = {causal}   disabled=={disabled_eq}", flush=True)
    print(f"VERDICT: {verdict['VERDICT']}", flush=True)
    print("(-> logs/mode_b_north_g3.json)", flush=True)


if __name__ == "__main__":
    main()
