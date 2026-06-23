#!/usr/bin/env python3
"""Phase-4 M2 rung-2: a budgeted FIFO SCHEDULER that interleaves multiple decode lanes, each owning its OWN KV
cache — so context-switching is bit-exact (interleaved output == solo output, per lane). Ports the toy
experiments/neural-scheduler/autoregressive.py run_scheduler/prefill/gen_step onto the real North decode.

ISOLATION INVARIANT: each Lane has its own make_prompt_cache(model); a context switch is a pure pointer change to
a different lane's cache (no shared object, no aliasing) — exactly the toy's per-process cache swap. B=1 PER LANE
(each _step_trapped is one B=1 forward) — lanes are NEVER batched through the shared 128-expert router, so top-k
expert selection cannot shift across lanes; routing depends only on the per-token hidden state, which each lane's
own cache reproduces. A non-selected lane sits parked: its cache list is untouched (no read, no write) until
rescheduled. lane.out matches gen_trapped semantics exactly (EOS ends without being appended).
"""
from __future__ import annotations
from dataclasses import replace
from mlx_lm.models.cache import make_prompt_cache
from kernel.frame import ContinuationFrame


class Lane:
    """A decode lane = a ContinuationFrame + its OWN live KV cache (the suspended process)."""
    def __init__(self, lane_id, prompt_ids, cache, eos, maxn):
        self.frame = ContinuationFrame(request_id=lane_id, site_idx=-1, eos_id=eos)
        self.ids = prompt_ids
        self.cache = cache                 # this lane's private make_prompt_cache(model)
        self.eos = eos
        self.maxn = maxn
        self.out = []
        self.done = False


class KernelScheduler:
    """Interleaves decode lanes on a TrappedNorthAdapter (uses _step_trapped/encode_fast/model/eos)."""

    def __init__(self, adapter):
        self.a = adapter

    def make_lane(self, lane_id, prompt, maxn=96):
        ids = list(self.a.encode_fast([{"role": "user", "content": prompt}]))
        return Lane(lane_id, ids, make_prompt_cache(self.a.model), self.a.eos, maxn)

    def _prefill(self, L):
        L.frame.last_token = self.a._step_trapped(L.ids, L.cache)        # first next-token (unappended)

    def _gen_step(self, L):
        tok = L.frame.last_token
        if tok == L.eos or len(L.out) >= L.maxn:                          # EOS/maxn ends WITHOUT appending (== gen_trapped)
            L.done = True
            return
        L.out.append(tok)
        L.frame.n_emitted = len(L.out)
        L.frame.last_token = self.a._step_trapped([tok], L.cache)         # advance THIS lane's own cache by one

    def run(self, lanes, W):
        """Budgeted FIFO drain: each tick advance up to W not-done lanes by one token; the rest sit parked."""
        for L in lanes:
            self._prefill(L)
        ticks = 0
        while not all(L.done for L in lanes):
            ticks += 1
            for L in [L for L in lanes if not L.done][:W]:
                self._gen_step(L)
        return ticks

    def solo(self, lane):
        """Drain one lane to completion alone (== gen_trapped for its prompt). Returns its token list."""
        self._prefill(lane)
        while not lane.done:
            self._gen_step(lane)
        return list(lane.out)

    def dynamic_W(self, *, policy, cost_source=None, sysinfo=None, n_lanes=None, importance=0.5) -> int:
        """Scheduler-owned concurrency width: cost (kernel proprioception) decides how WIDE to drain. Idle -> wide
        (toward n_lanes), loaded -> narrow (toward 1). Maps cost->k via KPolicy with a k_min=1 floor (always make
        progress) and k_max = n_lanes (can't exceed the lane count). Returns an int W in [1, n_lanes].

        This is M5 scheduler-owned dynamic-k applied to the CONCURRENCY surface (a third continuation-k seam): the
        cost sensor (rung-4 sysinfo proprioception) is the lever, and the SCHEDULER disposes how many lanes to
        admit per tick. Idle host -> widen toward n_lanes; a saturated host -> the cost brake collapses W to 1
        (stay thin / make progress, never starve). We feed idle-as-headroom through `disagreement` so the policy's
        cost brake (which acts on the discretionary band) does the widening AND the narrowing in one shot:
        low cost -> disagreement=(1-cost) is high -> widen; high cost both shrinks (1-cost) and trips the cost
        brake -> collapses to k_min=1. disagreement is the INDEPENDENT (trusted-by-construction) lever, so this is
        NOT a calibration-trap violation — it carries no claim about model confidence, only host headroom.

        SAFETY / ISOLATION: this only changes the DRAIN WIDTH passed to run(lanes, W). The interleaved==solo
        isolation invariant holds for ANY W (W only changes drain ORDER, not per-lane correctness, because every
        lane owns its own KV cache and is stepped B=1), so a dynamically chosen W is always safe — there is no W in
        [1, n_lanes] that can make interleaved output diverge from solo. Callers:
            W = sched.dynamic_W(policy=p, cost_source=src, n_lanes=len(lanes)); sched.run(lanes, W)
        """
        # local imports keep scheduler.py importable without the policy/cost modules on the path until this is used
        from kernel.k_policy import KSignals
        from kernel.k_cost import CostModel
        cost = cost_source() if cost_source is not None else CostModel().cost_from_sysinfo(sysinfo or {})["cost"]
        cost = 0.0 if cost < 0.0 else (1.0 if cost > 1.0 else float(cost))
        n = int(n_lanes) if n_lanes is not None else 1
        n = max(1, n)
        # idle cost -> high (1-cost) on the INDEPENDENT lever -> policy widens; high cost -> brake collapses it.
        # k_min=1 (always advance >=1 lane), k_max=n (can't drain more lanes than exist).
        local_policy = replace(policy, k_min=1, k_max=n)
        dec = local_policy.decide(KSignals(disagreement=(1.0 - cost), cost=cost, importance=importance))
        return max(1, min(n, int(dec.k)))


__all__ = ["Lane", "KernelScheduler"]
