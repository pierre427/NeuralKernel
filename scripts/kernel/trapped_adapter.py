#!/usr/bin/env python3
"""Phase-4: NorthAdapter routed through the no-op TrapPlane, threading the real KV cache.

Subclasses NorthAdapter so the PRODUCTION path (gen_fast / _step / _gen, north_adapter.py) stays byte-for-byte
unchanged — the kernel can always call gen_fast for the native output. `gen_trapped` instead routes generation
through run_stack_plane (the cache-threaded step that is the M1/M4 integration seam). With plane.armed=False
(default) it is token-for-token identical to gen_fast; the parity battery (inv_trap_plane.py) proves it ==0.

Entry points that import this must have the parent scripts/ dir on sys.path (for north_adapter +
north_gateway_invariants) — the runners self-insert it.
"""
from __future__ import annotations
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from north_adapter import NorthAdapter, NORTH
from mlx_lm.models.cache import can_trim_prompt_cache
from kernel.trap_plane import TrapPlane, run_stack_plane, SITES
from kernel.kv_park import snapshot_cache, restore_cache, rollback_or_restore


class TrappedNorthAdapter(NorthAdapter):
    name = "north-mini-code-trapped"

    def __init__(self, model_path=NORTH, sites=SITES):
        super().__init__(model_path)                 # self.model, self.tok, self.eos unchanged
        self.plane = TrapPlane(self.model, sites)    # installed; plane.armed=False (no-op) by default

    def _step_trapped(self, ids, cache):
        """Cache-threaded decode step through the trap plane (the integration seam). With plane.armed=False
        this is the same argmax over the same logits as NorthAdapter._step."""
        logits = run_stack_plane(self.model, mx.array([ids], dtype=mx.int32), plane=self.plane, cache=cache)
        return int(mx.argmax(logits[0, -1, :]).item())

    def gen_trapped(self, prompt, maxn=1800):
        """Mirror of NorthAdapter._gen(reasoning=False) but via the trapped step. Identical tokens to gen_fast
        when the plane is disarmed (and when armed-but-no-op)."""
        ids = self.encode_fast([{"role": "user", "content": prompt}])
        cache = make_prompt_cache(self.model)
        nxt = self._step_trapped(list(ids), cache); out = []
        for _ in range(maxn):
            if nxt == self.eos:
                break
            out.append(nxt); nxt = self._step_trapped([nxt], cache)
        return self.tok.decode(out), len(out)

    def gen_trapped_parkable(self, prompt, maxn=512, park_at=None):
        """M2: identical to gen_trapped, but at the `park_at`-th emitted token it PARKS the KV cache (deep-copied
        snapshot) and immediately RESUMES from the restored snapshot — a single-lane park->resume round-trip
        mid-decode. If restore is bit-exact, the token stream is IDENTICAL to a never-parked decode (== gen_fast)."""
        ids = self.encode_fast([{"role": "user", "content": prompt}])
        cache = make_prompt_cache(self.model)
        nxt = self._step_trapped(list(ids), cache); out = []
        for _ in range(maxn):
            if nxt == self.eos:
                break
            out.append(nxt)
            if park_at is not None and len(out) == park_at:        # park -> resume at this token boundary
                cache = restore_cache(snapshot_cache(cache))
            nxt = self._step_trapped([nxt], cache)
        return self.tok.decode(out), len(out)

    def gen_spec_rollback(self, prompt, maxn=96, park_at=8, k=4):
        """M2 rung-3: at park_at, SNAPSHOT, SPECULATE k tokens (advance the cache, discard them), then ROLL BACK
        via rollback_or_restore (trim pre-wrap / snapshot-restore post-wrap) and continue the REAL decode. If
        rollback is exact, the speculate+rollback is a NO-OP on the output -> result == gen_fast (==0), which
        proves the cache returned to the park state. Returns (text, n, rollback_path)."""
        ids = self.encode_fast([{"role": "user", "content": prompt}])
        cache = make_prompt_cache(self.model)
        nxt = self._step_trapped(list(ids), cache); out = []; did = False; path = "none"
        for _ in range(maxn):
            if nxt == self.eos:
                break
            if not did and len(out) == park_at:
                did = True
                snap = snapshot_cache(cache)
                s = nxt
                for _ in range(k):                                  # speculate k (advance the cache, discard)
                    s = self._step_trapped([s], cache)              # no EOS-guard: tokens are discarded so feeding
                    #                                                  EOS is harmless here; add one if speculated
                    #                                                  tokens are ever KEPT (real speculative decode).
                path = "trim" if can_trim_prompt_cache(cache) else "restore"
                cache = rollback_or_restore(cache, snap, k)          # roll the k speculative steps back
            out.append(nxt); nxt = self._step_trapped([nxt], cache)  # real decode continues from the park point
        return self.tok.decode(out), len(out), path

    def gen_constrained(self, prompt, forbidden=None, maxn=400):
        """M4 deterministic in-decode INTERRUPT (reuses north_det_block's mechanism): before each argmax, mask the
        `forbidden` token ids to -inf, enforcing an exact property on-device in the decode loop. forbidden=None/
        empty -> IDENTITY (== gen_fast, the inert control, zero overhead); a real mask -> the constraint the
        unconstrained decode might violate (e.g. no {import,from}). Routes through run_stack_plane (plane disarmed
        == native), so the inert control is bit-identical to production. (The mid-stack residual-WRITE / Mode-B
        flavor needs a trained delta and is the design's open research gap, deferred.)"""
        ids = self.encode_fast([{"role": "user", "content": prompt}])
        cache = make_prompt_cache(self.model)
        cur = list(ids); out = []
        neg = mx.array(-1e9, dtype=mx.float32)
        for _ in range(maxn):
            logits = run_stack_plane(self.model, mx.array([cur], dtype=mx.int32), plane=self.plane, cache=cache)[0, -1, :]
            if forbidden is not None and forbidden.size > 0:                    # the deterministic interrupt
                logits = mx.where(mx.zeros_like(logits).at[forbidden].add(1.0) > 0, neg, logits)
            nxt = int(mx.argmax(logits).item())
            if nxt == self.eos:
                break
            out.append(nxt); cur = [nxt]
        return self.tok.decode(out), len(out)


__all__ = ["TrappedNorthAdapter"]
