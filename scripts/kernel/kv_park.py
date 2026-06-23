#!/usr/bin/env python3
"""Phase-4 M2: decode-time KV PARK / RESUME — snapshot a continuation's KV cache at a token boundary and
restore it bit-identically, across North's MIXED cache stack (KVCache for the full_attention layers,
RotatingKVCache(max_size=4096) for the 36 sliding layers).

park  = a per-layer DEEP-COPY of (.state, .meta_state).
resume = type(c).from_state(state, meta_state) — exactly load_prompt_cache's reconstruction (cache.py:79-82),
         bit-exact for BOTH cache types (KVCache.meta_state=='' restores offset from keys.shape; RotatingKVCache
         carries keep/max_size/offset/_idx in meta_state, set AFTER .state by from_state).

Why NOT trim as the primitive: trim is forbidden once a RotatingKVCache ring has wrapped (is_trimmable()==False,
cache.py:542-543); snapshot/restore is the universal, ring-wrap-safe park. trim is used ONLY as a fast ROLLBACK,
gated by can_trim_prompt_cache.

The deep-copy (mx.array + mx.eval) is MANDATORY: .state can be a LIVE ALIAS of the buffer the next decode step
mutates in place (KVCache at full capacity, cache.py:362; RotatingKVCache post-wrap, cache.py:524) — without the
copy a still-running decode would scribble the snapshot. Never call _temporal_order on a snapshot (it reorders
the physical layout _idx indexes into). dtype is preserved (no astype) -> no MXFP8/bf16 upcast.
"""
from __future__ import annotations
import mlx.core as mx
import mlx_lm.models.cache as _cache
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache


def snapshot_cache(cache):
    """Deep-copied per-layer (classname, (keys, values), meta_state) snapshot — survives continued decode."""
    snap = []
    for c in cache:
        ks, vs = c.state                          # may be a LIVE alias/view of the underlying buffer
        ks, vs = mx.array(ks), mx.array(vs)        # independent device buffers (copy-on-park, beats aliasing)
        mx.eval(ks, vs)                            # materialize NOW (defeat lazy-eval aliasing)
        snap.append((type(c).__name__, (ks, vs), c.meta_state))
    return snap


def restore_cache(snap):
    """Reconstruct a fresh cache list from a snapshot via from_state (sets .state then .meta_state). Bit-exact for
    KVCache (meta_state=='' -> offset from keys.shape) and RotatingKVCache (meta_state carries offset/_idx)."""
    return [getattr(_cache, clsname).from_state(state, meta) for clsname, state, meta in snap]


def rollback_or_restore(cache, snap, k):
    """Roll a speculative span of k tokens back to the park point. Fast path: trim k on every layer while the
    whole cache is still trimmable (no Rotating layer has wrapped). Otherwise discard the live cache and restore
    the deep-copied snapshot — the only ==0-safe rollback once a ring evicted tokens. Returns the rolled-back cache.
    Uniform-or-snapshot, NEVER trim-only when any sliding layer wrapped (else full layers roll back, sliding
    layers no-op -> mid-model desync).

    STATUS: MODEL-VERIFIED on North (both branches). Trim fast-path: inv_speculate_rollback.py (pre-wrap, ==0).
    Snapshot-restore branch: inv_rollback_ringcross.py — decode past 4096 so the sliding ring WRAPS (can_trim
    False) -> restore -> resume is ==0 over 4260 tokens (0 mismatches). Live via gen_spec_rollback. (Originally
    CPU-proven in test_kv_park; the deferred ring-cross model gate is now closed.)"""
    if can_trim_prompt_cache(cache):
        trim_prompt_cache(cache, k)
        return cache
    return restore_cache(snap)


__all__ = ["snapshot_cache", "restore_cache", "rollback_or_restore"]
