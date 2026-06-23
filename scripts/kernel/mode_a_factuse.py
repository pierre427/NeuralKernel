#!/usr/bin/env python3
"""Mode-A fact-USE control — the clean counterpart to the G3/G3b negative.

G3 delivered a secret digit via a mid-stack RESIDUAL edit (L24) and the model could not apply an operation to it
(held-out op +3 = 0.0; echo rose but never transformed). The diagnosis (see design §M5 discussion): a residual
edit arrives too late / in the wrong basis to become an OPERAND. This control changes ONLY the delivery channel:
the secret digit is given AS A TOKEN in the prompt (processed from layer 0, where early layers turn it into an
operand) — no injector, no training, just the base model. Same ops, same metric.

If token-delivered held-out (+3) accuracy is high (>> G3's 0.0 and >> the unaided baseline), it pins the G3
failure on the mid-stack residual channel, NOT on the model's ability to use an injected fact -> the sound way to
make the kernel's verified result USABLE is token-level (Mode A) / re-prefill, exactly as argued.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_a_factuse.py
"""
from __future__ import annotations
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from north_adapter import NorthAdapter

OPS = {
    "state":  ("State the secret digit.", lambda x: x % 10),
    "add1":   ("Add 1 to the secret digit (mod 10).", lambda x: (x + 1) % 10),
    "add2":   ("Add 2 to the secret digit (mod 10).", lambda x: (x + 2) % 10),
    "double": ("Double the secret digit (mod 10).", lambda x: (2 * x) % 10),
    "add3":   ("Add 3 to the secret digit (mod 10).", lambda x: (x + 3) % 10),   # the G3 HELD-OUT op
}


def _first_digit(text):
    m = re.search(r"\d", text or "")
    return m.group(0) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    print("[load] north ...", flush=True)
    a = NorthAdapter()
    Xs = list(range(10)) * ((args.n // 10) + 1)
    Xs = Xs[:args.n]

    def acc(op, delivered):
        req, fn = OPS[op]
        hit = 0
        for X in Xs:
            if delivered:
                prompt = f"The secret digit is {X}. {req} Reply with only the resulting digit."
            else:
                prompt = f"A secret digit between 0 and 9 was chosen but is NOT shown to you. {req} Reply with only the resulting digit."
            text, _ = a.gen_fast(prompt, maxn=8)
            hit += int(_first_digit(text) == str(fn(X)))
        return round(hit / len(Xs), 3)

    token_acc = {op: acc(op, True) for op in OPS}
    unaided_acc = {op: acc(op, False) for op in OPS}

    # pull G3 (residual-injection) numbers for the side-by-side
    g3 = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "mode_b_north_g3.json")
    if os.path.exists(p):
        d = json.load(open(p)); g3 = {**d.get("train_op_acc", {}), "add3": d.get("heldout_op_acc")}

    heldout_token = token_acc["add3"]
    confirmed = heldout_token > 0.6 and heldout_token > unaided_acc["add3"] + 0.4
    R = {
        "n": args.n,
        "token_delivered_acc": token_acc, "unaided_acc": unaided_acc,
        "g3_residual_injection_acc": g3,
        "heldout_add3": {"token": heldout_token, "unaided": unaided_acc["add3"],
                         "g3_residual": g3.get("add3", "n/a")},
        "VERDICT": ("CONFIRMED: fact delivered AS TOKENS is usable (held-out op works) — the G3 failure was the "
                    "mid-stack RESIDUAL channel, not the model. Sound fact-use = token-level / re-prefill (Mode A)."
                    if confirmed else
                    "INCONCLUSIVE: token-delivered held-out op is not clearly high — base model arithmetic may be the limit; see per-op."),
    }
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "mode_a_factuse.json"), "w"), indent=2, default=str)
    print("\n===== Mode-A fact-USE control (digit delivered AS A TOKEN) =====", flush=True)
    print(f"  {'op':8s} {'token':>7} {'unaided':>8} {'G3-residual':>12}", flush=True)
    for op in OPS:
        print(f"  {op:8s} {token_acc[op]:7} {unaided_acc[op]:8} {str(g3.get(op,'-')):>12}", flush=True)
    print(f"\n  HELD-OUT +3:  token={heldout_token}  unaided={unaided_acc['add3']}  G3-residual={g3.get('add3','n/a')}", flush=True)
    print(f"VERDICT: {R['VERDICT']}", flush=True)
    print("(-> logs/mode_a_factuse.json)", flush=True)


if __name__ == "__main__":
    main()
