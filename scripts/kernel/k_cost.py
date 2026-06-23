#!/usr/bin/env python3
"""M5 rung-4: COST ACCOUNTING + the sysinfo SENSOR for scheduler-owned-k.

The k-policy's cost input is the kernel's PROPRIOCEPTION: it reads its OWN stat (host load + per-TaskState queue
depths + held-process count + memory) and widens k when idle/low-load, stays thin when queues are deep / memory
is low (project_scheduler_owned_k: "the sysinfo tool supplies the cost/latency inputs of the k-decision"). sysinfo
(kernel/tools_builtin.py) is the sensor; this module is the NORMALIZER (sysinfo dict -> cost in [0,1]) + the
ACCOUNTANT (charge KDecision.est_cost against a budget so widening k cannot run away).

BINDING-CONSTRAINT model: cost = the MAX normalized pressure across {host load-per-core, memory, queue depth, GPU}.
Any single saturated resource should brake discretionary widening — the scarcest resource governs, exactly as a
scheduler stays thin under its tightest constraint. Each dimension degrades gracefully (a missing probe
contributes 0, never raises), so cost is always defined.

Two seams it serves: (1) `signals_from` builds a KSignals with cost filled, leaving uncertainty/disagreement to
the caller (the robust lever, per the rung-3 calibration verdict — NOT raw entropy); (2) `charge_budget` clamps a
decision's k to what the budget affords (anti-runaway-via-k at the budget level, complementing KPolicy's k_max).
"""
from __future__ import annotations
from dataclasses import dataclass
from kernel.k_policy import KSignals, KPolicy, KDecision


def _clamp01(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass
class CostModel:
    """Maps a sysinfo dict -> cost in [0,1]. Thresholds are the points at which each resource hits full pressure."""
    queue_soft_cap: int = 8        # (ready+held) depth at which queue pressure == 1.0
    load_saturate: float = 1.0     # load-average-per-core at which host pressure == 1.0 (1.0 == fully subscribed)
    mem_floor: float = 0.5         # memory %used below which there is NO pressure (e.g. <50% used -> 0)
    gpu_saturate: float = 0.85     # GPU 'Device Utilization %'/100 at which GPU pressure == 1.0 (the contention brake
                                   # the CPU/mem/queue sensors miss — a co-resident inference process pins the GPU)

    def cost_from_sysinfo(self, info) -> dict:
        info = info or {}
        host = info.get("host", {}) or {}
        sch = info.get("scheduler", {}) or {}

        # host load per core (None if os.getloadavg unavailable -> 0 pressure)
        la = host.get("load_avg")
        ncpu = host.get("cpu_count") or 1
        load_p = _clamp01((la[0] / max(ncpu, 1)) / max(self.load_saturate, 1e-9)) if (la and la[0] is not None) else 0.0

        # memory %used above a floor (None -> 0 pressure)
        pct = (host.get("memory", {}) or {}).get("percent")
        mem_p = _clamp01(((pct / 100.0) - self.mem_floor) / max(1.0 - self.mem_floor, 1e-9)) if pct is not None else 0.0

        # scheduler queue: ready + held depth, normalized by a soft cap
        qd = (sch.get("ready_depth", 0) or 0) + (sch.get("held_depth", 0) or 0)
        queue_p = _clamp01(qd / max(self.queue_soft_cap, 1))

        # GPU utilization (Apple-Silicon, via sysinfo's ioreg probe). Sampled at a decision point it reflects OTHER
        # GPU consumers -> contention. None (off-mac / unavailable) contributes 0 pressure, never raises.
        gpu_pct = (host.get("gpu") or {}).get("util_pct")
        gpu_p = _clamp01((gpu_pct / 100.0) / max(self.gpu_saturate, 1e-9)) if gpu_pct is not None else 0.0

        breakdown = {"load": round(load_p, 3), "memory": round(mem_p, 3), "queue": round(queue_p, 3),
                     "gpu": round(gpu_p, 3)}
        return {"cost": max(breakdown.values()), "breakdown": breakdown}


def signals_from(sysinfo_info, *, uncertainty=0.0, uncertainty_calibrated=False, disagreement=None,
                 risk=0.0, importance=0.5, cost_model=None) -> KSignals:
    """Build a KSignals with `cost` filled from a sysinfo snapshot; the caller supplies the uncertainty lever
    (validate-the-branch / seed-disagreement — the robust signal, not raw entropy per the rung-3 verdict)."""
    cm = cost_model or CostModel()
    cost = cm.cost_from_sysinfo(sysinfo_info)["cost"]
    return KSignals(uncertainty=uncertainty, uncertainty_calibrated=uncertainty_calibrated,
                    disagreement=disagreement, cost=cost, risk=risk, importance=importance)


@dataclass
class BudgetCharge:
    admitted_k: int        # k after clamping to what the budget affords
    requested_k: int       # the policy's pre-budget k
    spent: float           # cost charged
    remaining: float       # budget left after the charge (None if untracked)
    note: str              # ok | budget-clamped | budget-exhausted | no-budget


def charge_budget(decision: KDecision, remaining, unit_cost: float = 1.0) -> BudgetCharge:
    """Clamp a KDecision's k to what `remaining` budget affords and charge it — anti-runaway-via-k at the budget
    level (complements KPolicy's hard k_max). remaining=None == no budget tracking (admit the policy's k as-is)."""
    if remaining is None:
        return BudgetCharge(decision.k, decision.k, decision.est_cost, None, "no-budget")
    affordable = max(0, int(remaining // max(unit_cost, 1e-9)))
    admitted = min(decision.k, affordable)
    spent = admitted * unit_cost
    note = "ok" if admitted == decision.k else ("budget-clamped" if admitted > 0 else "budget-exhausted")
    return BudgetCharge(admitted, decision.k, spent, remaining - spent, note)


def decide_with_cost(policy: KPolicy, sysinfo_info, *, remaining=None, unit_cost=1.0, cost_model=None,
                     **lever) -> dict:
    """End-to-end rung-4 seam: read cost from sysinfo -> KPolicy.decide -> charge the budget. `lever` carries the
    robust uncertainty inputs (uncertainty/uncertainty_calibrated/disagreement/risk/importance)."""
    sig = signals_from(sysinfo_info, cost_model=cost_model, **lever)
    decision = policy.decide(sig)
    charge = charge_budget(decision, remaining, unit_cost=unit_cost)
    return {"signals": sig, "decision": decision, "charge": charge,
            "k": charge.admitted_k, "cost": sig.cost, "est_cost": decision.est_cost}


__all__ = ["CostModel", "signals_from", "BudgetCharge", "charge_budget", "decide_with_cost"]
