#!/usr/bin/env python3
"""Phase-4 M2 rung-3 PARITY: speculate -> rollback -> re-decode is bit-identical to straight-through on North.

gen_spec_rollback parks the KV cache, speculates k tokens (advancing the cache), then rolls back via
rollback_or_restore and continues the real decode. If rollback is exact, the speculate+rollback is a NO-OP on
the output -> result == production gen_fast, token-for-token (==0). This is the deferred M1->M2 loop, exercising
the (previously dormant) kernel/kv_park.py rollback_or_restore on the real model.

Scope: PRE-WRAP (trim fast-path) over several (park_at, k) — fast + the common case. The POST-WRAP snapshot-
restore path (>4096 tokens) is already bit-exact-proven on toy caches (test_kv_park: post-wrap alias-guard +
rollback gate); re-proving it on North needs a >4096-token decode (expensive) and is deferred. Strict ==0.
Loads North; run solo."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.utils import load
from north_gateway_invariants import MODEL, PROMPT
from north_adapter import NorthAdapter
from kernel.trapped_adapter import TrappedNorthAdapter

MAXN = 80
CASES = [(4, 3), (8, 6), (16, 8)]            # (park_at, speculate-k)


def _attach(cls, model, tok, eos):
    a = cls.__new__(cls); a.model, a.tok, a.eos = model, tok, eos
    a.plane = None
    return a


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    base = _attach(NorthAdapter, model, tok, eos)
    ta = _attach(TrappedNorthAdapter, model, tok, eos)

    out_oracle, n = base.gen_fast(PROMPT, maxn=MAXN)              # straight-through oracle
    cases = [(pa, k) for (pa, k) in CASES if pa < n]
    R = {"n_oracle": n, "plane": "disabled(None)", "results": {}}
    allok = True
    for (pa, k) in cases:
        txt, n_sr, path = ta.gen_spec_rollback(PROMPT, maxn=MAXN, park_at=pa, k=k)
        eq = (txt == out_oracle and n_sr == n)
        R["results"][f"park{pa}_k{k}"] = {"eq": eq, "rollback_path": path}
        allok = allok and eq
        print(f"  park_at={pa:3d} k={k}: spec+rollback==straight-through: {eq}  (path={path}, n {n_sr} vs {n})", flush=True)
    R["RESULT"] = "PASS" if allok else "FAIL"

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_speculate_rollback.json"), "w"), indent=2, default=str)
    print(json.dumps(R, default=str), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
