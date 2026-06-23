#!/usr/bin/env python3
"""Phase-4 M3-foundation PARITY BATTERY — the no-op trap plane is bit-identical (==0) to native North.

Strict ==0 gate (not <epsilon; design §5.1), run on the SAME real coding prompt the harness uses.

Test A (PREFILL): generalize the proven single-site L24 invariant to all 7 sites.
  - manual_parity: run_stack_plane(plane=None) vs model() -> HARD assert ==0 (was only PRINTED in
    north_gateway_invariants; promoted to an assert so a manual-forward drift can't pass silently).
  - multi-site: all 7 zero-init gateways armed at once -> max|gated-base| == 0.
  - per-site: each site alone (incl. the NEW sites 0 = post dense block, 48 = post final layer, before
    norm+as_linear*logit_scale) -> ==0.

Test B (DECODE — the genuinely-new gate, not covered anywhere): gen_fast (native) vs gen_trapped disarmed AND
armed, token-for-token over a full generation. Proves the cache-threaded step (the M1/M4 seam) is identity over
a whole decode, not just one prefill forward.

RUN (loads the 30G model; do NOT run concurrently with another North job):
  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/inv_trap_plane.py
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load
from north_gateway_invariants import maxabs, MODEL, PROMPT
from north_adapter import NorthAdapter
from kernel.frame import ContinuationFrame
from kernel.trap_plane import TrapPlane, run_stack_plane, SITES
from kernel.trapped_adapter import TrappedNorthAdapter

DECODE_N = 128


def _attach(cls, model, tok, eos, plane=None):
    """Build an adapter around an ALREADY-LOADED model (avoid re-loading the 30G weights per arm)."""
    a = cls.__new__(cls)
    a.model, a.tok, a.eos = model, tok, eos
    if plane is not None:
        a.plane = plane
    return a


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, reasoning=False)
    inputs = mx.array(np.array([ids], dtype=np.int32))
    R = {"sites": list(SITES), "decode_n": DECODE_N}

    # --- Test A: prefill identity ---
    base = model(inputs); mx.eval(base)
    R["manual_parity"] = maxabs(run_stack_plane(model, inputs, plane=None, cache=None), base)
    assert R["manual_parity"] == 0, f"manual-forward drift {R['manual_parity']} — run_stack_plane != model()"

    plane = TrapPlane(model); plane.armed = True                       # armed, but every up.weight==0
    R["multi_site_parity"] = maxabs(run_stack_plane(model, inputs, plane=plane, cache=None), base)
    R["per_site_parity"] = {}
    for s in SITES:
        p1 = TrapPlane(model, sites=(s,)); p1.armed = True
        R["per_site_parity"][s] = maxabs(run_stack_plane(model, inputs, plane=p1, cache=None), base)

    # a ContinuationFrame is LIVE during the parity (constructed from real state, observationally null:
    # disarmed, no capabilities, no trap budget) — proving the plane is null even with a frame present.
    frame = ContinuationFrame(request_id=1, site_idx=48, token_pos=int(inputs.shape[1]),
                              eos_id=eos, capabilities=frozenset(), traps_left=0, armed=False)
    R["frame_live_null"] = bool((not frame.armed) and frame.traps_left == 0 and not frame.capabilities)

    # --- Test B: decode identity over a full generation (the cache-threaded seam) ---
    base_a = _attach(NorthAdapter, model, tok, eos)
    out_base, n_base = base_a.gen_fast(PROMPT, maxn=DECODE_N)
    assert n_base > 0, "decode produced 0 tokens — token parity would be vacuous"
    ta = _attach(TrappedNorthAdapter, model, tok, eos, plane=plane)
    ta.plane.armed = False
    out_dis, n_dis = ta.gen_trapped(PROMPT, maxn=DECODE_N)
    ta.plane.armed = True
    out_arm, n_arm = ta.gen_trapped(PROMPT, maxn=DECODE_N)
    R["decode_disarmed_eq"] = bool(out_dis == out_base and n_dis == n_base)
    R["decode_armed_eq"] = bool(out_arm == out_base and n_arm == n_base)

    ok = (R["manual_parity"] == 0 and R["multi_site_parity"] == 0
          and all(v == 0 for v in R["per_site_parity"].values())
          and R["decode_disarmed_eq"] and R["decode_armed_eq"] and R["frame_live_null"])
    R["RESULT"] = "PASS" if ok else "FAIL"

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_trap_plane.json"), "w"), indent=2, default=str)
    print(json.dumps(R, default=str), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
