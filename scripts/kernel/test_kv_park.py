#!/usr/bin/env python3
"""CPU-only tests for kv_park snapshot/restore — TOY KVCache/RotatingKVCache with tiny arrays, NO 30G model.
De-risks the whole mechanism before the North decode run: the mx.array deep-copy beats aliasing (incl. the
RotatingKVCache POST-WRAP in-place ring writes), restore is bit-exact for both cache types incl. meta_state
(offset/_idx), and rollback_or_restore gates trim (pre-wrap) vs snapshot-restore (post-wrap).
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_kv_park.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache, can_trim_prompt_cache
from kernel.kv_park import snapshot_cache, restore_cache, rollback_or_restore

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")
def maxabs(a, b): mx.eval(a, b); return float(mx.max(mx.abs(a - b)).item())
def upd(c, n, h=2, d=8, base=0):           # n single-token updates (the decode path)
    for i in range(n):
        t = float(base + i + 1)
        c.update_and_fetch(mx.full((1, h, 1, d), t), mx.full((1, h, 1, d), t * 10))


def test_kvcache():
    print("\n[1] KVCache: snapshot -> continue decoding -> restore is bit-exact + alias-guarded")
    c = KVCache(); upd(c, 5)
    snap = snapshot_cache([c]); snap_k = snap[0][1][0]
    upd(c, 3)                                                      # continue (offset 5 -> 8)
    ok(maxabs(snap[0][1][0], snap_k) == 0, "snapshot NOT scribbled by continued decode (alias-guard)")
    r = restore_cache(snap)[0]
    ok(r.offset == 5, f"restored offset == 5 (got {r.offset})")
    ok(r.state[0].shape[2] == 5 and maxabs(r.state[0], snap_k) == 0, "restored K bit-exact to the park point")


def test_rotating_prewrap():
    print("\n[2] RotatingKVCache pre-wrap: restore bit-exact incl. meta_state(offset,_idx)")
    c = RotatingKVCache(max_size=256, keep=0); upd(c, 5)
    snap = snapshot_cache([c])
    r = restore_cache(snap)[0]
    ok(r.offset == c.offset and r._idx == c._idx, f"offset/_idx restored (off {r.offset}, idx {r._idx})")
    ok(maxabs(r.state[0], snap[0][1][0]) == 0, "restored K bit-exact")


def test_rotating_postwrap_alias_guard():
    print("\n[3] RotatingKVCache POST-WRAP: deep-copy survives in-place ring writes (the real alias risk)")
    c = RotatingKVCache(max_size=4, keep=0); upd(c, 4)            # fill the ring
    snap = snapshot_cache([c]); snap_k = snap[0][1][0]
    upd(c, 5, base=100)                                          # wrap: in-place writes into the ring buffer
    ok(maxabs(snap[0][1][0], snap_k) == 0, "post-wrap snapshot survived in-place ring writes (deep-copy works)")
    r = restore_cache(snap)[0]
    ok(maxabs(r.state[0], snap_k) == 0, "restored ring bit-exact to the parked physical layout")


def test_rollback_gate():
    print("\n[4] rollback_or_restore: trim pre-wrap, snapshot-restore post-wrap")
    c = [RotatingKVCache(max_size=256, keep=0)]; upd(c[0], 10); off = c[0].offset
    snap = snapshot_cache(c); upd(c[0], 3)
    rolled = rollback_or_restore(c, snap, 3)
    ok(can_trim_prompt_cache(c) and rolled[0].offset == off, f"pre-wrap: trimmed back to {off} (got {rolled[0].offset})")
    c2 = [RotatingKVCache(max_size=4, keep=0)]; upd(c2[0], 4); off2 = c2[0].offset
    snap2 = snapshot_cache(c2); upd(c2[0], 3)                    # wrapped
    ok(not can_trim_prompt_cache(c2), "post-wrap cache reports not-trimmable")
    rolled2 = rollback_or_restore(c2, snap2, 3)
    ok(rolled2[0].offset == off2, f"post-wrap: restored snapshot offset == {off2} (got {rolled2[0].offset})")


if __name__ == "__main__":
    test_kvcache(); test_rotating_prewrap(); test_rotating_postwrap_alias_guard(); test_rollback_gate()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} kv-park checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
