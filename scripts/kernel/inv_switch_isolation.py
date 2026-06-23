#!/usr/bin/env python3
"""Phase-4 M2 rung-2 PARITY: context-switching is bit-exact on North — interleaved multi-lane decode ==
solo per-lane, token-for-token (==0), AND each lane's decode == production gen_fast. Realizes the toy's
mismatch==0 isolation proof on the real 128-expert MoE, B=1-per-lane (no batched router cross-talk).
Loads North; run solo. Strict ==, no tolerance (design §5.1)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.utils import load
from north_gateway_invariants import MODEL
from north_adapter import NorthAdapter
from kernel.trapped_adapter import TrappedNorthAdapter
from kernel.scheduler import KernelScheduler

PROMPTS = [
    "What is the greatest common divisor of 48 and 36?",
    "Write a short poem about the sea.",
    "Is 91 a prime number?",
    "Explain how photosynthesis works in one sentence.",
]
MAXN = 64
W = 2                                   # budget < N -> oversubscribed, heavy interleaving


def _attach(cls, model, tok, eos):
    a = cls.__new__(cls); a.model, a.tok, a.eos = model, tok, eos
    a.plane = None                       # isolation is independent of the plane; plane=None == native fwd
    return a


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    base = _attach(NorthAdapter, model, tok, eos)
    ta = _attach(TrappedNorthAdapter, model, tok, eos)
    assert ta.plane is None
    sched = KernelScheduler(ta)

    # solo: each lane drained alone
    solo = [sched.solo(sched.make_lane(i, p, maxn=MAXN)) for i, p in enumerate(PROMPTS)]
    # interleaved: all lanes through the budgeted scheduler
    lanes = [sched.make_lane(i, p, maxn=MAXN) for i, p in enumerate(PROMPTS)]
    ticks = sched.run(lanes, W)
    inter = [L.out for L in lanes]

    iso_mism = sum(int(inter[i] != solo[i]) for i in range(len(PROMPTS)))
    # production parity: each solo lane == native gen_fast (text), proving the lane decode is the real decode
    prod_ok = all(tok.decode(solo[i]) == base.gen_fast(p, maxn=MAXN)[0] for i, p in enumerate(PROMPTS))

    R = {"n_lanes": len(PROMPTS), "W": W, "ticks": ticks, "plane": "disabled(None)",
         "lane_lens": [len(o) for o in solo], "isolation_mismatches": iso_mism,
         "production_parity": prod_ok, "RESULT": "PASS" if (iso_mism == 0 and prod_ok) else "FAIL"}
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_switch_isolation.json"), "w"), indent=2, default=str)
    print(f"  lanes={len(PROMPTS)} W={W} ticks={ticks} lens={R['lane_lens']}", flush=True)
    print(f"  interleaved==solo mismatches: {iso_mism}  | each lane==gen_fast: {prod_ok}", flush=True)
    print(json.dumps(R, default=str), flush=True)
    sys.exit(0 if R["RESULT"] == "PASS" else 1)


if __name__ == "__main__":
    main()
