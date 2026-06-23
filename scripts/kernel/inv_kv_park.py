#!/usr/bin/env python3
"""Phase-4 M2 PARITY: single-lane KV park/resume is bit-identical (==0) to straight-through decode on North.

gen_trapped_parkable(park_at) deep-copy-snapshots the MIXED KV cache (KVCache + RotatingKVCache) at a token
boundary and resumes from the restored snapshot; it must equal production gen_fast TOKEN-FOR-TOKEN for park_at
across {1, 8, N//2, N-1}. Strict ==, no tolerance (design §5.1). This is the genuinely-new M2 mechanism the toy
proved on tiny caches (test_kv_park 10/10); here it is proven on the real 30G model end-to-end over a full decode.

Run solo: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/inv_kv_park.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.utils import load
from north_gateway_invariants import MODEL, PROMPT
from north_adapter import NorthAdapter
from kernel.trapped_adapter import TrappedNorthAdapter

MAXN = 96


def _attach(cls, model, tok, eos):
    a = cls.__new__(cls); a.model, a.tok, a.eos = model, tok, eos
    a.plane = None                      # park/resume parity is independent of the plane; plane=None == native fwd
    return a


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    base = _attach(NorthAdapter, model, tok, eos)
    ta = _attach(TrappedNorthAdapter, model, tok, eos)
    assert ta.plane is None, "M2 park/resume is proven in ISOLATION from the M3 trap plane (plane must be None)"

    out_base, n = base.gen_fast(PROMPT, maxn=MAXN)               # straight-through oracle
    park_ats = sorted({x for x in (1, 8, n // 2, n - 1) if 1 <= x < n})
    R = {"n_base": n, "park_ats": park_ats, "plane": "disabled(None): park/resume in isolation from M3", "results": {}}
    allok = True
    for pa in park_ats:
        out_p, n_p = ta.gen_trapped_parkable(PROMPT, maxn=MAXN, park_at=pa)
        eq = (out_p == out_base and n_p == n)
        R["results"][pa] = eq; allok = allok and eq
        print(f"  park_at={pa:3d}: parked==straight-through: {eq}  (n {n_p} vs {n})", flush=True)
    R["RESULT"] = "PASS" if allok else "FAIL"

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_kv_park.json"), "w"), indent=2, default=str)
    print(json.dumps(R, default=str), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
