#!/usr/bin/env python3
"""CPU-only test for M5 rung-5 integration: scheduler-owned k wired into the EscalatingSolver Tier-2 width.
No model — drives _t1_severity / _t2_width directly with a fake Tier-1 gate result + injected cost.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_k_policy_ladder.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.escalate_solve import EscalatingSolver
from kernel.k_policy import KPolicy

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")
_NOLOG = lambda *a, **k: None
_FAIL_ALL = {"gate": {"passed": 0, "n": 1000}}      # T1 wrong on every held-out instance -> severity 1.0


def _solver(policy=None, cost=0.0, importance=0.5, budget=None):
    return EscalatingSolver(model=None, web=False, k_policy=policy, importance=importance,
                            trap_budget=budget, cost_source=(lambda: cost))


def test_default_preserved():
    print("\n[1] no policy -> fixed k=3 (existing ladder behavior byte-unchanged)")
    k, entry = _solver(policy=None)._t2_width(_FAIL_ALL, _NOLOG)
    ok(k == 3 and entry is None, f"no policy -> fixed k=3, no kdecision ({k})")


def test_severity():
    print("\n[2] _t1_severity = independent (validate-the-branch) uncertainty from the hold-out gate")
    sev = EscalatingSolver._t1_severity
    ok(sev({"gate": {"passed": 0, "n": 1000}}) == 1.0, "wrong on all held-out -> 1.0")
    ok(abs(sev({"gate": {"passed": 900, "n": 1000}}) - 0.1) < 1e-9, "wrong on 10% -> 0.1")
    ok(sev({}) == 1.0, "no gate (failed before it) -> 1.0 (maximally uncertain)")


def test_idle_widens():
    print("\n[3] idle + high severity -> widen (driven by the INDEPENDENT signal, not entropy)")
    p = KPolicy(k_min=1, k_max=5, widen_threshold=0.5)
    k, entry = _solver(p, cost=0.0, importance=1.0)._t2_width(_FAIL_ALL, _NOLOG)
    ok(k >= 3, f"idle -> wide k ({k})")
    ok(entry["used"] == "independent_disagreement", f"driven by validate-the-branch signal ({entry['used']})")


def test_busy_thins():
    print("\n[4] busy -> thin k=1 (cost brake) — the thin/heavy dissociation, scheduler-owned")
    p = KPolicy(k_min=1, k_max=5, widen_threshold=0.5)
    k, entry = _solver(p, cost=1.0, importance=1.0)._t2_width(_FAIL_ALL, _NOLOG)
    ok(k == 1, f"fully busy -> thin k=1 ({k})")
    ok(entry["used"] == "cost_braked", f"audited cost_braked ({entry['used']})")


def test_budget_caps():
    print("\n[5] trap budget caps k (anti-runaway-via-k at the budget level)")
    p = KPolicy(k_min=1, k_max=5, widen_threshold=0.5)
    k, entry = _solver(p, cost=0.0, importance=1.0, budget=2.0)._t2_width(_FAIL_ALL, _NOLOG)
    ok(k <= 2, f"budget=2 caps k<=2 ({k})")
    ok(entry["budget_note"] in ("budget-clamped", "ok"), f"budget accounted ({entry['budget_note']})")


def test_importance_scales():
    print("\n[6] importance scales the spend (idle, same severity)")
    p = KPolicy(k_min=1, k_max=5, widen_threshold=0.5)
    lo = _solver(p, cost=0.0, importance=0.2)._t2_width(_FAIL_ALL, _NOLOG)[0]
    hi = _solver(p, cost=0.0, importance=1.0)._t2_width(_FAIL_ALL, _NOLOG)[0]
    ok(hi >= lo and hi > 1, f"higher importance -> more spend (hi={hi} >= lo={lo})")


if __name__ == "__main__":
    test_default_preserved(); test_severity(); test_idle_widens(); test_busy_thins()
    test_budget_caps(); test_importance_scales()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} k_policy_ladder checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
