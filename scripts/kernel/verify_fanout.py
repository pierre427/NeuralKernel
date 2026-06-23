#!/usr/bin/env python3
"""M5 scheduler-owned dynamic-k applied to VERIFICATION DEPTH — the fourth continuation-k seam, and the place
where KPolicy.verify_k (built in k_policy.py but until now unused) finally drives a real decision.

The branching factor here is "how MANY independent checks do we run on this item?". Same architecture spine as
the other seams (project_scheduler_owned_k): the proposer offers a candidate + signals, the SCHEDULER disposes k.
THIN by default (low risk + low/uncalibrated uncertainty -> run ONE check; a heavy verification protocol HURTS a
capable model, per project_grounding_result / project_kernel_intent_reconciliation), and WIDEN verification only
when it is earned:

  * RISK floors a MANDATORY verify count that COST CANNOT brake away — the confidently-wrong guard. A
    miscalibrated-confident model (gpt-oss conf=1.0 on a FAILED grade) cannot talk the scheduler out of running
    extra checks on a risky item. This is the risk safety-floor from KPolicy, applied: verify_k is safety, not
    discretion.
  * INDEPENDENT / calibrated uncertainty earns DISCRETIONARY extra checks (the explore band), and COST brakes
    only that discretionary part (stay thin when the host is busy). Raw uncalibrated uncertainty does NOT widen
    (the calibration trap — validate, don't speculate on a hunch).
  * BUDGET caps the total (anti-runaway-via-k at the budget level), and k is hard-clamped to len(checks) (can't
    run more checks than exist).

Wiring mirrors kernel/escalate_solve.py and kernel/fact_loop.py exactly: KPolicy.decide(KSignals(...)) for the
disposal, charge_budget(...) for the budget clamp, independent uncertainty as the robust lever (NOT raw entropy).

FAIL-SAFE semantics: a check is callable(item) -> bool. A check that RAISES counts as FAILED (we never let an
exception masquerade as a pass). verdict = ALL of the k checks that ran passed. We run EXACTLY k checks in list
order (no short-circuit) so the audit is honest about how many ran and how many passed.

Pure Python (no MLX) so it stays importable/testable without a model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Any

from kernel.k_policy import KSignals, KPolicy, KDecision
from kernel.k_cost import charge_budget


@dataclass
class VerifyResult:
    verdict: bool          # all run checks passed (a verification gate)
    n_checks_run: int
    n_checks_avail: int
    k: int
    passed: int
    audit: dict            # used_signal, reason, cost, risk, budget_note


def governed_verify(item: Any, checks: Sequence[Callable[[Any], bool]], policy: KPolicy, *,
                    risk: float = 0.0, uncertainty: float = 0.0, uncertainty_calibrated: bool = False,
                    disagreement: Optional[float] = None, cost_source=None, sysinfo=None,
                    budget=None, importance: float = 0.5) -> VerifyResult:
    """Run k of the N independent `checks` (each callable(item)->bool) where k is KPolicy-disposed from
    risk/uncertainty/cost: THIN by default (low risk+uncertainty -> 1 check), WIDEN verification when risk or
    calibrated/independent uncertainty earns it (the risk safety-floor mandates a minimum verify count that COST
    CANNOT brake away — the confidently-wrong guard), cost brakes the discretionary part, budget caps total. k is
    clamped to len(checks). verdict = all of the k run checks pass. A check that raises counts as FAILED (fail-safe).
    """
    checks = list(checks)
    n_avail = len(checks)

    # --- cost (kernel proprioception): explicit source > sysinfo snapshot > none (0). ---
    if cost_source is not None:
        cost = cost_source()
    elif sysinfo is not None:
        from kernel.k_cost import CostModel
        cost = CostModel().cost_from_sysinfo(sysinfo)["cost"]
    else:
        cost = 0.0
    cost = 0.0 if cost < 0.0 else (1.0 if cost > 1.0 else float(cost))

    # --- dispose k from the signals (independent uncertainty is the robust lever; risk floors the safety count). ---
    sig = KSignals(uncertainty=uncertainty, uncertainty_calibrated=uncertainty_calibrated,
                   disagreement=disagreement, cost=cost, risk=risk, importance=importance)
    decision: KDecision = policy.decide(sig)

    # --- budget cap (anti-runaway-via-k at the budget level; mirrors charge_budget use in escalate_solve/fact_loop).
    charge = charge_budget(decision, budget, unit_cost=policy.unit_cost)
    k = charge.admitted_k

    # --- hard-clamp to what exists: can't run more checks than we have (and never negative). ---
    if n_avail <= 0:
        k = 0
    else:
        k = max(0, min(k, n_avail))

    # --- run EXACTLY k checks in list order (no short-circuit) so the audit honestly reports run/passed. A check
    #     that raises is FAILED (fail-safe: an exception never counts as a pass). ---
    passed = 0
    for i in range(k):
        try:
            if bool(checks[i](item)):
                passed += 1
        except Exception:
            pass  # raised -> counts as failed
    verdict = (k > 0) and (passed == k)

    audit = {
        "used_signal": decision.used_signal,
        "reason": decision.reason,
        "cost": round(cost, 3),
        "risk": round(float(risk), 3),
        "verify_k": decision.verify_k,
        "explore_k": decision.explore_k,
        "budget_note": charge.note,
        "budget_remaining": charge.remaining,
        "requested_k": charge.requested_k,
    }
    return VerifyResult(verdict=verdict, n_checks_run=k, n_checks_avail=n_avail, k=k, passed=passed, audit=audit)


__all__ = ["VerifyResult", "governed_verify"]
