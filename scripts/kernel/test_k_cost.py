#!/usr/bin/env python3
"""CPU-only tests for M5 rung-4: the sysinfo->cost normalizer + budget accounting. No model.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_k_cost.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.k_policy import KPolicy, KDecision
from kernel.k_cost import CostModel, signals_from, charge_budget, decide_with_cost

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")


def _sys(load=None, ncpu=10, mempct=None, ready=0, held=0):
    return {"host": {"load_avg": ([load, load, load] if load is not None else None), "cpu_count": ncpu,
                     "memory": ({"percent": mempct} if mempct is not None else {})},
            "scheduler": {"ready_depth": ready, "held_depth": held}}


def test_idle_vs_saturated():
    print("\n[1] idle -> cost~0; saturated -> cost~1")
    cm = CostModel()
    idle = cm.cost_from_sysinfo(_sys(load=0.1, ncpu=10, mempct=30, ready=0, held=0))
    sat = cm.cost_from_sysinfo(_sys(load=20, ncpu=10, mempct=95, ready=8, held=4))
    ok(idle["cost"] < 0.1, f"idle host -> cost~0 ({idle['cost']:.3f}, {idle['breakdown']})")
    ok(sat["cost"] >= 0.99, f"saturated host -> cost~1 ({sat['cost']:.3f}, {sat['breakdown']})")


def test_binding_constraint():
    print("\n[2] binding-constraint: ANY one saturated resource drives cost to ~1 (the scarcest governs)")
    cm = CostModel()
    load_only = cm.cost_from_sysinfo(_sys(load=10, ncpu=10, mempct=20, ready=0))   # load==ncpu
    mem_only = cm.cost_from_sysinfo(_sys(load=0.1, ncpu=10, mempct=100, ready=0))
    queue_only = cm.cost_from_sysinfo(_sys(load=0.1, ncpu=10, mempct=20, ready=8))
    ok(load_only["cost"] >= 0.99 and load_only["breakdown"]["load"] >= 0.99, "load alone binds cost")
    ok(mem_only["cost"] >= 0.99 and mem_only["breakdown"]["memory"] >= 0.99, "memory alone binds cost")
    ok(queue_only["cost"] >= 0.99 and queue_only["breakdown"]["queue"] >= 0.99, "queue depth alone binds cost")


def test_graceful_degradation():
    print("\n[3] missing probes contribute 0, never raise")
    cm = CostModel()
    none = cm.cost_from_sysinfo({"host": {}, "scheduler": {}})
    empty = cm.cost_from_sysinfo({})
    noload = cm.cost_from_sysinfo(_sys(load=None, ncpu=10, mempct=None, ready=0))
    ok(none["cost"] == 0.0 and empty["cost"] == 0.0, "empty/missing sysinfo -> cost 0 (no crash)")
    ok(noload["cost"] == 0.0, "missing load_avg + memory -> cost 0")


def test_budget_charge():
    print("\n[4] budget accounting clamps k to what's affordable (anti-runaway-via-k at budget level)")
    d = KDecision(k=4, explore_k=3, verify_k=0, used_signal="x", reason="x", est_cost=4.0)
    full = charge_budget(d, remaining=10.0, unit_cost=1.0)
    clamp = charge_budget(d, remaining=2.0, unit_cost=1.0)
    exh = charge_budget(d, remaining=0.0, unit_cost=1.0)
    nob = charge_budget(d, remaining=None)
    ok(full.admitted_k == 4 and full.remaining == 6.0 and full.note == "ok", "affordable -> full k, budget debited")
    ok(clamp.admitted_k == 2 and clamp.note == "budget-clamped", f"tight budget -> k clamped to 2 ({clamp.admitted_k})")
    ok(exh.admitted_k == 0 and exh.note == "budget-exhausted", "exhausted budget -> k=0")
    ok(nob.admitted_k == 4 and nob.note == "no-budget", "no budget tracking -> policy k as-is")


def test_end_to_end_seam():
    print("\n[5] decide_with_cost: idle widens, saturated stays thin (cost brake), budget clamps")
    p = KPolicy()
    # robust lever = a calibrated/independent signal (NOT raw entropy); here a strong calibrated uncertainty
    wide = decide_with_cost(p, _sys(load=0.1, ncpu=10, mempct=20), uncertainty=0.9, uncertainty_calibrated=True, importance=1.0)
    thin = decide_with_cost(p, _sys(load=20, ncpu=10, mempct=95, ready=8), uncertainty=0.9, uncertainty_calibrated=True, importance=1.0)
    ok(wide["k"] == 4, f"idle + high calibrated uncertainty -> wide k ({wide['k']}, cost={wide['cost']:.2f})")
    ok(thin["k"] == p.k_min, f"saturated + same uncertainty -> thin k ({thin['k']}, cost={thin['cost']:.2f}) [cost brake]")
    ok(thin["decision"].used_signal == "cost_braked", "audited as cost_braked")
    clamped = decide_with_cost(p, _sys(load=0.1, ncpu=10, mempct=20), remaining=2.0,
                               uncertainty=0.9, uncertainty_calibrated=True, importance=1.0)
    ok(clamped["k"] == 2 and clamped["charge"].note == "budget-clamped", f"wide decision clamped by budget ({clamped['k']})")


if __name__ == "__main__":
    test_idle_vs_saturated(); test_binding_constraint(); test_graceful_degradation()
    test_budget_charge(); test_end_to_end_seam()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} k_cost checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
