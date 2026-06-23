#!/usr/bin/env python3
"""CPU-only tests for the M5 rung-1 KPolicy (scheduler-owned dynamic-k). No model — pure Python.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_k_policy.py

Covers: the entropy-signal normalisation; thin-by-default; calibrated widening; the CALIBRATION TRAP (an
uncalibrated raw signal must NOT widen k); independent disagreement rescues widening; cost brakes DISCRETIONARY
exploration; the RISK SAFETY FLOOR cost cannot brake away (the confidently-wrong guard); anti-runaway clamping;
the degenerate syscall-head policy; monotonicities; and a property sweep of the hard invariants.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.k_policy import normalized_entropy, KSignals, KDecision, KPolicy

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")


def test_entropy():
    print("\n[1] normalized_entropy: the MoE-gate uncertainty signal in [0,1]")
    ok(normalized_entropy([1.0, 0.0, 0.0, 0.0]) == 0.0, "one-hot gate -> 0.0 (maximally confident routing)")
    ok(abs(normalized_entropy([0.25, 0.25, 0.25, 0.25]) - 1.0) < 1e-9, "uniform over n -> 1.0 (max uncertain)")
    ok(normalized_entropy([1.0]) == 0.0 and normalized_entropy([]) == 0.0, "n<=1 / empty -> 0.0 (no choice)")
    ok(abs(normalized_entropy([2, 2, 2, 2]) - 1.0) < 1e-9, "renormalises unnormalised input")
    v = normalized_entropy([0.7, 0.1, 0.1, 0.1])
    ok(0.6 < v < 0.72, f"peaked-but-not-onehot is mid-range ({v:.3f})")
    # 128-expert (North) sanity: top-8 concentrated vs near-uniform
    conc = [0.9] + [0.1 / 127] * 127
    ok(normalized_entropy(conc) < 0.2, f"North-shape concentrated gate -> low entropy ({normalized_entropy(conc):.3f})")


def test_thin_by_default():
    print("\n[2] thin by default: low uncertainty -> k == k_min, no spend")
    p = KPolicy()
    d = p.decide(KSignals(uncertainty=0.2, uncertainty_calibrated=True, importance=1.0))
    ok(d.k == p.k_min, f"low calibrated uncertainty -> k=k_min ({d.k})")
    ok(d.used_signal == "thin_default" and d.explore_k == 0 and d.verify_k == 0, "audited as thin_default, no branches")


def test_calibrated_widening():
    print("\n[3] high CALIBRATED uncertainty + low cost + high importance -> widen toward k_max")
    p = KPolicy()
    d = p.decide(KSignals(uncertainty=0.9, uncertainty_calibrated=True, cost=0.0, importance=1.0))
    ok(d.k == 4 and d.explore_k == 3, f"widens to k_max ({d.k}, explore_k={d.explore_k})")
    ok(d.used_signal == "calibrated_uncertainty", f"audited as calibrated_uncertainty ({d.used_signal})")


def test_calibration_trap():
    print("\n[4] THE CALIBRATION TRAP: an UNCALIBRATED raw signal must NOT widen k (validate the branch instead)")
    p = KPolicy()
    d = p.decide(KSignals(uncertainty=0.95, uncertainty_calibrated=False, cost=0.0, importance=1.0))
    ok(d.k == p.k_min, f"uncalibrated high uncertainty -> stays thin k=k_min ({d.k})")
    ok(d.explore_k == 0, "no discretionary branches on an untrusted hunch")
    ok(d.used_signal == "uncalibrated_held", f"audited as uncalibrated_held ({d.used_signal})")
    # and require_calibrated=False (a trusted source) DOES widen on the same number
    p2 = KPolicy(require_calibrated=False)
    d2 = p2.decide(KSignals(uncertainty=0.95, uncertainty_calibrated=False, cost=0.0, importance=1.0))
    ok(d2.k > p2.k_min, f"with require_calibrated=False the same signal widens ({d2.k})")


def test_disagreement_rescues():
    print("\n[5] an INDEPENDENT disagreement signal widens k even when the raw signal is uncalibrated")
    p = KPolicy()
    d = p.decide(KSignals(uncertainty=0.95, uncertainty_calibrated=False, disagreement=0.9, importance=1.0))
    ok(d.k > p.k_min and d.explore_k > 0, f"independent disagreement -> widen ({d.k})")
    ok(d.used_signal == "independent_disagreement", f"audited as independent_disagreement ({d.used_signal})")


def test_cost_brake():
    print("\n[6] cost brakes DISCRETIONARY exploration (stay thin when busy)")
    p = KPolicy()
    base = p.decide(KSignals(uncertainty=0.9, uncertainty_calibrated=True, cost=0.0, importance=1.0))
    busy = p.decide(KSignals(uncertainty=0.9, uncertainty_calibrated=True, cost=1.0, importance=1.0))
    ok(busy.k < base.k, f"high cost reduces k ({busy.k} < {base.k})")
    ok(busy.k == p.k_min and busy.used_signal == "cost_braked", f"fully braked -> thin, audited cost_braked ({busy.used_signal})")


def test_risk_floor_beats_cost():
    print("\n[7] SAFETY: risk floors a MANDATORY verify count that cost CANNOT brake away (confidently-wrong guard)")
    p = KPolicy()
    # confident (low, even uncalibrated) + maximally busy + high risk: verification must still happen
    d = p.decide(KSignals(uncertainty=0.05, uncertainty_calibrated=False, cost=1.0, risk=0.9, importance=0.5))
    ok(d.verify_k > 0, f"risk mandates verification despite max cost + confidence (verify_k={d.verify_k})")
    ok(d.k >= min(p.k_max, p.k_min + d.verify_k), "safety invariant: k >= min(k_max, k_min+verify_k)")
    ok(d.used_signal == "risk_floor", f"audited as risk_floor ({d.used_signal})")
    # a miscalibrated-confident model cannot talk the scheduler out of it: same risk, claimed-certain
    d2 = p.decide(KSignals(uncertainty=0.0, uncertainty_calibrated=True, cost=1.0, risk=0.9))
    ok(d2.verify_k == d.verify_k, "even a calibrated-confident claim does not remove the risk-mandated verify")


def test_clamp_anti_runaway():
    print("\n[8] anti-runaway-via-k: k is hard-clamped to [k_min, k_max] under extreme signals")
    p = KPolicy(k_min=1, k_max=4)
    d = p.decide(KSignals(uncertainty=1.0, uncertainty_calibrated=True, disagreement=1.0, risk=1.0, importance=1.0, cost=0.0))
    ok(d.k == p.k_max, f"everything maxed -> k clamped to k_max ({d.k})")
    d2 = p.decide(KSignals(uncertainty=0.0, uncertainty_calibrated=True, cost=1.0))
    ok(d2.k >= p.k_min, f"nothing -> k never below k_min ({d2.k})")


def test_syscall_head_degenerate():
    print("\n[9] degenerate syscall-head policy: k_min=0,k_max=1,thr=0.5 -> {below:k=0 continue, above:k=1 dispatch}")
    p = KPolicy(k_min=0, k_max=1, widen_threshold=0.5)
    below = p.decide(KSignals(uncertainty=0.3, uncertainty_calibrated=True, importance=1.0))
    above = p.decide(KSignals(uncertainty=0.8, uncertainty_calibrated=True, importance=1.0))
    ok(below.k == 0, f"below threshold -> k=0 (continue in-model) ({below.k})")
    ok(above.k == 1, f"above threshold -> k=1 (dispatch one) ({above.k})")


def test_monotonicity_and_cost_accounting():
    print("\n[10] monotonic in calibrated uncertainty (up) and cost (down); est_cost monotonic in k; importance scales")
    p = KPolicy()
    ks = [p.decide(KSignals(uncertainty=u, uncertainty_calibrated=True, importance=1.0)).k for u in (0.4, 0.6, 0.9)]
    ok(ks[0] <= ks[1] <= ks[2], f"k non-decreasing in uncertainty ({ks})")
    cs = [p.decide(KSignals(uncertainty=0.9, uncertainty_calibrated=True, importance=1.0, cost=c)).k for c in (0.0, 0.8, 1.0)]
    ok(cs[0] >= cs[1] >= cs[2], f"k non-increasing in cost ({cs})")
    d_lo = p.decide(KSignals(uncertainty=0.7, uncertainty_calibrated=True, importance=0.2))
    d_hi = p.decide(KSignals(uncertainty=0.7, uncertainty_calibrated=True, importance=1.0))
    ok(d_hi.k > d_lo.k, f"higher importance -> more spend ({d_hi.k} > {d_lo.k})")
    d0 = p.decide(KSignals(uncertainty=0.9, uncertainty_calibrated=True, importance=0.0))
    ok(d0.k == p.k_min and d0.explore_k == 0, "importance=0 -> no discretionary spend even at high uncertainty")
    ok(d_hi.est_cost == d_hi.k * p.unit_cost and d_lo.est_cost == d_lo.k * p.unit_cost, "est_cost == k*unit_cost (accountable)")


def test_property_sweep():
    print("\n[11] property sweep: hard invariants hold across the signal grid")
    p = KPolicy()
    bad_clamp = bad_safety = bad_cost = 0
    n = 0
    for unc in (0.0, 0.3, 0.5, 0.7, 1.0):
        for calib in (True, False):
            for dis in (None, 0.8):
                for cost in (0.0, 0.5, 1.0):
                    for risk in (0.0, 0.5, 1.0):
                        for imp in (0.0, 0.5, 1.0):
                            n += 1
                            d = p.decide(KSignals(unc, calib, dis, cost, risk, imp))
                            if not (p.k_min <= d.k <= p.k_max):
                                bad_clamp += 1
                            if d.k < min(p.k_max, p.k_min + d.verify_k):
                                bad_safety += 1
                            if abs(d.est_cost - d.k * p.unit_cost) > 1e-9:
                                bad_cost += 1
    ok(bad_clamp == 0, f"k in [k_min,k_max] for all {n} cases")
    ok(bad_safety == 0, f"safety floor k>=min(k_max,k_min+verify_k) for all {n} cases")
    ok(bad_cost == 0, f"est_cost==k*unit_cost for all {n} cases")


if __name__ == "__main__":
    test_entropy(); test_thin_by_default(); test_calibrated_widening(); test_calibration_trap()
    test_disagreement_rescues(); test_cost_brake(); test_risk_floor_beats_cost(); test_clamp_anti_runaway()
    test_syscall_head_degenerate(); test_monotonicity_and_cost_accounting(); test_property_sweep()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} k_policy checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
