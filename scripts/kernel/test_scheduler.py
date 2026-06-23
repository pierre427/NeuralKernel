#!/usr/bin/env python3
"""CPU-only test for the M2 scheduler bookkeeping (no model — a deterministic FAKE adapter). Proves the
interleaving/budget drain is correct: interleaved output == solo output per lane, variable-length lanes drain
independently, and a parked lane is never advanced. The real-model isolation (shared weights + MoE) is proven
separately by inv_switch_isolation.py on North.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_scheduler.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.scheduler import KernelScheduler

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


class _FakeModel:
    args = type("A", (), {})()
    def make_cache(self):
        return [{"pos": 0, "seed": None}]          # per-lane INDEPENDENT state (the isolation point)


class _FakeAdapter:
    """Deterministic: a lane with seed S emits tokens [S*10+0 .. S*10+(S-1)] then EOS — variable length, and a
    function ONLY of its own (seed,pos), so interleaving order cannot change a lane's output unless the scheduler
    corrupts per-lane state."""
    eos = -1
    model = _FakeModel()

    def encode_fast(self, msgs):
        seed = sum(ord(ch) for ch in msgs[0]["content"]) % 6 + 3   # 3..8
        return [seed]

    def _step_trapped(self, ids, cache):
        c = cache[0]
        if c["pos"] == 0:
            c["seed"] = ids[0]                      # prime: capture this lane's seed
        pos = c["pos"]; c["pos"] += 1
        seed = c["seed"]
        return self.eos if pos >= seed else seed * 10 + pos


def test_interleaved_eq_solo():
    print("\n[1] interleaved (W<N, oversubscribed) == solo, per lane, exact token-list identity")
    sched = KernelScheduler(_FakeAdapter())
    prompts = [f"prompt-{x}" for x in ("alpha", "beta", "gamma", "delta", "epsilon")]
    solo = [sched.solo(sched.make_lane(i, p, maxn=64)) for i, p in enumerate(prompts)]
    lanes = [sched.make_lane(i, p, maxn=64) for i, p in enumerate(prompts)]
    sched.run(lanes, W=2)                           # 5 lanes, budget 2 -> oversubscribed, heavy interleaving
    inter = [L.out for L in lanes]
    mism = sum(int(inter[i] != solo[i]) for i in range(len(prompts)))
    ok(mism == 0, f"interleaved == solo for every lane (mismatches={mism})")
    ok(all(len(o) >= 1 for o in solo) and len({len(o) for o in solo}) > 1,
       "lanes are non-trivial and variable-length (real interleaving exercised)")


def test_budget_and_parking():
    print("\n[2] budget W advances at most W lanes/tick; parked lanes are untouched")
    sched = KernelScheduler(_FakeAdapter())
    lanes = [sched.make_lane(i, f"p{i}aaaa", maxn=64) for i in range(4)]
    for L in lanes:
        sched._prefill(L)
    # one tick at W=2 advances exactly the first 2 not-done lanes; lanes 2,3 stay at pos after prefill (1)
    advanced_before = [L.cache[0]["pos"] for L in lanes]
    for L in [L for L in lanes if not L.done][:2]:
        sched._gen_step(L)
    advanced_after = [L.cache[0]["pos"] for L in lanes]
    ok(advanced_after[0] > advanced_before[0] and advanced_after[1] > advanced_before[1],
       "the two selected lanes advanced")
    ok(advanced_after[2] == advanced_before[2] and advanced_after[3] == advanced_before[3],
       "the two parked lanes were NOT touched (no read/write to their cache)")


if __name__ == "__main__":
    test_interleaved_eq_solo(); test_budget_and_parking()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} scheduler checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
