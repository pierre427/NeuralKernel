#!/usr/bin/env python3
"""CPU-only tests for the two NEW M5 scheduler-owned dynamic-k seams (no model):

  TASK 1 — KernelScheduler.dynamic_W: cost (kernel proprioception) disposes the DRAIN WIDTH. Idle (cost~0) widens
           W toward n_lanes; a saturated host (cost=1.0) collapses W to 1 (make progress, never starve). And the
           interleaved==solo isolation invariant survives a dynamically chosen W (W only changes drain ORDER).
  TASK 2 — verify_fanout.governed_verify: KPolicy disposes the VERIFICATION DEPTH (KPolicy.verify_k, finally used).
           Thin by default; the RISK safety-floor mandates >=2 checks COST CANNOT brake away (confidently-wrong
           guard); a failing/raising check fails the verdict; k clamps to len(checks); budget caps k.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_dynamic_k_seams.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.scheduler import KernelScheduler
from kernel.k_policy import KPolicy
from kernel.verify_fanout import governed_verify
from kernel.test_scheduler import _FakeAdapter   # reuse the deterministic fake adapter

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")


# ===================================================================================================
# TASK 1 — cost-governed lane W
# ===================================================================================================
def test_dynamic_W_idle_is_wide():
    print("\n[1] dynamic_W: idle host (cost~0) -> W near n_lanes")
    sched = KernelScheduler(_FakeAdapter())
    p = KPolicy()
    n = 5
    # importance=1.0 lets idle headroom spend the full widening band -> W reaches n_lanes.
    w_idle = sched.dynamic_W(policy=p, cost_source=lambda: 0.0, n_lanes=n, importance=1.0)
    ok(w_idle == n, f"idle cost=0.0 widens W to n_lanes ({w_idle} == {n})")
    # default importance (0.5) still widens well above the thin floor of 1 (idle earns concurrency).
    w_idle_def = sched.dynamic_W(policy=p, cost_source=lambda: 0.0, n_lanes=n)
    ok(1 < w_idle_def <= n, f"idle (default importance) widens above 1, toward n ({w_idle_def} in (1, {n}])")


def test_dynamic_W_busy_is_one():
    print("\n[2] dynamic_W: saturated host (cost=1.0) -> W == 1 (make progress, never starve)")
    sched = KernelScheduler(_FakeAdapter())
    p = KPolicy()
    for imp in (0.5, 1.0):
        w = sched.dynamic_W(policy=p, cost_source=lambda: 1.0, n_lanes=5, importance=imp)
        ok(w == 1, f"cost=1.0 collapses W to the k_min=1 floor (importance={imp}) -> {w}")


def test_dynamic_W_bounds_and_sysinfo():
    print("\n[3] dynamic_W: bounded to [1, n_lanes]; sysinfo path works; never exceeds lane count")
    sched = KernelScheduler(_FakeAdapter())
    p = KPolicy()
    # n_lanes=1 -> always 1 regardless of cost (can't drain more lanes than exist)
    ok(sched.dynamic_W(policy=p, cost_source=lambda: 0.0, n_lanes=1) == 1, "n_lanes=1 -> W==1 even when idle")
    # idle W never exceeds n_lanes for a range of sizes
    monotone_ok = True
    for n in (2, 3, 4, 8):
        w = sched.dynamic_W(policy=p, cost_source=lambda: 0.0, n_lanes=n, importance=1.0)
        if not (1 <= w <= n):
            monotone_ok = False
    ok(monotone_ok, "idle W stays within [1, n_lanes] across n in {2,3,4,8}")
    # sysinfo path (empty sysinfo -> cost 0 -> idle/wide) AND a loaded sysinfo -> narrow
    w_empty = sched.dynamic_W(policy=p, sysinfo={}, n_lanes=4, importance=1.0)
    ok(w_empty == 4, f"empty sysinfo -> cost 0 -> wide W ({w_empty} == 4)")
    loaded = {"host": {"load_avg": [16.0, 16.0, 16.0], "cpu_count": 1}}   # load-per-core >> 1 -> cost 1.0
    w_loaded = sched.dynamic_W(policy=p, sysinfo=loaded, n_lanes=4)
    ok(w_loaded == 1, f"loaded sysinfo (high load) -> narrow W==1 ({w_loaded})")


def test_dynamic_W_does_not_mutate_policy():
    print("\n[4] dynamic_W: does not mutate the caller's policy (uses dataclasses.replace)")
    sched = KernelScheduler(_FakeAdapter())
    p = KPolicy(k_min=1, k_max=4)
    _ = sched.dynamic_W(policy=p, cost_source=lambda: 0.0, n_lanes=10, importance=1.0)
    ok(p.k_min == 1 and p.k_max == 4, f"caller policy unchanged (k_min={p.k_min}, k_max={p.k_max})")


def test_run_with_dynamic_W_isolation():
    print("\n[5] interleaved == solo per lane for run(lanes, dynamic_W(...)) — isolation is W-independent")
    p = KPolicy()
    prompts = [f"prompt-{x}" for x in ("alpha", "beta", "gamma", "delta", "epsilon")]
    overall_mismatches = 0
    # try several cost levels => several different W values; isolation must hold for ALL of them.
    for cost in (0.0, 0.5, 1.0):
        sched = KernelScheduler(_FakeAdapter())
        solo = [sched.solo(sched.make_lane(i, pr, maxn=64)) for i, pr in enumerate(prompts)]
        lanes = [sched.make_lane(i, pr, maxn=64) for i, pr in enumerate(prompts)]
        W = sched.dynamic_W(policy=p, cost_source=lambda c=cost: c, n_lanes=len(lanes), importance=1.0)
        sched.run(lanes, W)
        inter = [L.out for L in lanes]
        mism = sum(int(inter[i] != solo[i]) for i in range(len(prompts)))
        overall_mismatches += mism
        ok(mism == 0, f"cost={cost} -> W={W}: interleaved == solo for every lane (mismatches={mism})")
    ok(overall_mismatches == 0, "isolation invariant held across all dynamic W values")


# ===================================================================================================
# TASK 2 — governed verification depth (verify_fanout)
# ===================================================================================================
def _passing(item): return True
def _failing(item): return False
def _raising(item): raise RuntimeError("boom")


def test_verify_thin_by_default():
    print("\n[6] governed_verify: low risk + no uncertainty -> THIN (n_checks_run == 1)")
    p = KPolicy()
    checks = [_passing, _passing, _passing, _passing]
    r = governed_verify("x", checks, p, risk=0.0)
    ok(r.n_checks_run == 1, f"thin default runs exactly 1 check ({r.n_checks_run})")
    ok(r.verdict is True and r.passed == 1, f"verdict True, passed=={r.passed}")
    ok(r.audit["used_signal"] == "thin_default", f"audit used_signal == thin_default ({r.audit['used_signal']})")


def test_verify_risk_floor_widens():
    print("\n[7] governed_verify: high risk -> k>=2 (risk safety-floor; more checks)")
    p = KPolicy()
    checks = [_passing] * 4
    r = governed_verify("x", checks, p, risk=1.0)
    ok(r.k >= 2, f"high risk mandates k>=2 ({r.k})")
    ok(r.n_checks_run == r.k and r.verdict is True, f"runs all k and they pass (run={r.n_checks_run}, k={r.k})")
    ok(r.audit["used_signal"] == "risk_floor", f"audit attributes to risk_floor ({r.audit['used_signal']})")


def test_verify_risk_floor_beats_cost():
    print("\n[8] governed_verify: cost=1.0 CANNOT brake the risk floor (the safety property)")
    p = KPolicy()
    checks = [_passing] * 4
    r_busy = governed_verify("x", checks, p, risk=1.0, cost_source=lambda: 1.0)
    ok(r_busy.k >= 2, f"risk-mandated verification survives cost=1.0 ({r_busy.k} >= 2)")
    ok(r_busy.audit["verify_k"] >= 1, f"verify_k floor present despite cost ({r_busy.audit['verify_k']})")
    # contrast: a DISCRETIONARY widen (independent disagreement, no risk) IS braked away by cost=1.0 -> thin.
    r_disc = governed_verify("x", checks, p, disagreement=1.0, risk=0.0, cost_source=lambda: 1.0)
    ok(r_disc.n_checks_run == 1, f"discretionary widen IS cost-braked to 1 ({r_disc.n_checks_run})")


def test_verify_discretionary_widen_when_idle():
    print("\n[9] governed_verify: independent disagreement + idle cost -> discretionary widen (k>1)")
    p = KPolicy()
    checks = [_passing] * 4
    r = governed_verify("x", checks, p, disagreement=1.0, risk=0.0, cost_source=lambda: 0.0, importance=1.0)
    ok(r.k > 1, f"idle + independent disagreement widens verification ({r.k} > 1)")
    ok(r.audit["used_signal"] == "independent_disagreement", f"attributed to independent signal ({r.audit['used_signal']})")


def test_verify_uncalibrated_held():
    print("\n[10] governed_verify: raw UNCALIBRATED uncertainty does NOT widen (calibration trap)")
    p = KPolicy()
    checks = [_passing] * 4
    r = governed_verify("x", checks, p, uncertainty=0.95, uncertainty_calibrated=False, risk=0.0, cost_source=lambda: 0.0)
    ok(r.n_checks_run == 1, f"uncalibrated hunch stays thin ({r.n_checks_run})")
    ok(r.audit["used_signal"] == "uncalibrated_held", f"audit flags the held hunch ({r.audit['used_signal']})")


def test_verify_failing_check_fails_verdict():
    print("\n[11] governed_verify: a failing check makes verdict False")
    p = KPolicy()
    checks = [_passing, _failing, _passing, _passing]
    r = governed_verify("x", checks, p, risk=1.0)              # risk widens so the failing check (index 1) runs
    ok(r.k >= 2, f"k widened enough to include the failing check ({r.k})")
    ok(r.verdict is False, "verdict is False when a run check fails")
    ok(r.passed < r.n_checks_run, f"passed < run ({r.passed} < {r.n_checks_run})")


def test_verify_raising_check_is_failed():
    print("\n[12] governed_verify: a RAISING check counts as failed (fail-safe, no crash)")
    p = KPolicy()
    checks = [_passing, _raising, _passing, _passing]
    r = governed_verify("x", checks, p, risk=1.0)
    ok(r.verdict is False, "a raised exception does NOT count as a pass (verdict False)")
    ok(r.passed < r.n_checks_run, f"the raising check is not counted as passed ({r.passed} < {r.n_checks_run})")


def test_verify_clamped_to_len_checks():
    print("\n[13] governed_verify: k clamped to len(checks) (can't run more checks than exist)")
    p = KPolicy(k_max=8)
    checks = [_passing, _passing]                              # only 2 available
    r = governed_verify("x", checks, p, risk=1.0, disagreement=1.0, importance=1.0, cost_source=lambda: 0.0)
    ok(r.n_checks_avail == 2, "two checks available")
    ok(r.k == 2 and r.n_checks_run == 2, f"k clamped to len(checks)=2 ({r.k}, run {r.n_checks_run})")
    ok(r.verdict is True, "both available checks pass -> verdict True")
    # zero checks -> k 0, verdict False (a gate with nothing to run does not pass)
    r0 = governed_verify("x", [], p, risk=1.0)
    ok(r0.k == 0 and r0.verdict is False, f"no checks -> k 0, verdict False ({r0.k}, {r0.verdict})")


def test_verify_budget_caps_k():
    print("\n[14] governed_verify: budget caps k (anti-runaway-via-k at the budget level)")
    p = KPolicy(k_max=8)
    checks = [_passing] * 8
    # idle + high disagreement + importance would widen large, but a budget of 2 caps admitted k to 2.
    r = governed_verify("x", checks, p, disagreement=1.0, importance=1.0, cost_source=lambda: 0.0, budget=2.0)
    ok(r.k <= 2, f"budget=2 caps k to <=2 ({r.k})")
    ok(r.audit["budget_note"] in ("budget-clamped", "ok"), f"budget note recorded ({r.audit['budget_note']})")
    ok(r.n_checks_run == r.k, f"runs exactly the budgeted k ({r.n_checks_run} == {r.k})")
    # contrast unbudgeted: same signals widen further (proves the budget is what capped it)
    r_free = governed_verify("x", checks, p, disagreement=1.0, importance=1.0, cost_source=lambda: 0.0)
    ok(r_free.k > r.k, f"unbudgeted k ({r_free.k}) > budgeted k ({r.k}) -> budget is the cap")


if __name__ == "__main__":
    # Task 1
    test_dynamic_W_idle_is_wide()
    test_dynamic_W_busy_is_one()
    test_dynamic_W_bounds_and_sysinfo()
    test_dynamic_W_does_not_mutate_policy()
    test_run_with_dynamic_W_isolation()
    # Task 2
    test_verify_thin_by_default()
    test_verify_risk_floor_widens()
    test_verify_risk_floor_beats_cost()
    test_verify_discretionary_widen_when_idle()
    test_verify_uncalibrated_held()
    test_verify_failing_check_fails_verdict()
    test_verify_raising_check_is_failed()
    test_verify_clamped_to_len_checks()
    test_verify_budget_caps_k()

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} dynamic-k seam checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
