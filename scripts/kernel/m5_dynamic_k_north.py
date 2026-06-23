#!/usr/bin/env python3
"""M5 rung-5 DEMONSTRATION: scheduler-owned dynamic-k wired into the real EscalatingSolver ladder, on North.

Shows the Tier-2 width (the old fixed k=3) become a SCHEDULER decision that adapts to cost/severity/budget, live
on the 30G model. The uncertainty lever is validate-the-branch (Tier-1's hold-out-gate severity) + cost from the
sysinfo sensor — NOT raw gate entropy (rung-3 found it inverted+single-suite, so it stays unwired).

c156 (resp_parser) fails Tier-1 -> reaches the T2 width decision; we run it 4 ways and show the disposed k adapt:
  fixed-3        no policy           -> k=3 (the original ladder, byte-unchanged)
  dyn-idle       cost=0.0            -> wide k (idle: the scheduler can afford to branch)
  dyn-busy       cost=1.0            -> thin k=1 (loaded: stay thin — the thin/heavy dissociation)
  dyn-budget2    cost=0.0, budget=2  -> k capped to 2 (anti-runaway-via-k at the budget level)
w47 (to_snake_case) passes Tier-1 -> the ladder is thin BY DEFAULT (the T2 decision is never reached).

Reports the disposed k, the audit reason, model-gens actually spent, and solved? per condition. Honest scope: the
deterministic, strong result is that the DECISION is wired + adapts live on North; a broad statistical
"matches fixed-k at lower cost" A/B needs more fixtures and is future work.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/m5_dynamic_k_north.py
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.self_extend import Gap
from kernel.escalate_solve import EscalatingSolver
from kernel.k_policy import KPolicy
from kernel.solve_failed_challenges import TASKS
from north_adapter import NorthAdapter

N_GATE = 1000
MAXN = 1200
POLICY = dict(k_min=1, k_max=5, widen_threshold=0.5)     # the ladder-seam policy: floor 1 once escalating, cap 5


def _gap(tid):
    s = TASKS[tid]
    return Gap(entry=s["entry"], description=s["prompt"], sample_cases=s["sample_cases"],
               generate_src=s["generate_src"], oracle_src=s["oracle_src"], n=N_GATE, seed=0)


def _gens(log):
    """count model generations actually spent (each step() = 1 gen; the t2-kpolicy entry is meta, not a gen)."""
    return sum(1 for e in log if e.get("tier", "").startswith(("t1", "t2-div", "t3", "t4")))


def _t2k(log):
    kd = next((e for e in log if e.get("tier") == "t2-kpolicy"), None)
    if kd:
        return kd["k"], kd.get("used"), kd.get("reason")
    n_div = sum(1 for e in log if e.get("tier", "").startswith("t2-div"))
    return (n_div if n_div else None), "fixed", "no policy (fixed k=3)"


def run(model, tid, label, *, policy=None, cost=0.0, budget=None, importance=1.0):
    solver = EscalatingSolver(model, web=False, n=N_GATE, maxn=MAXN, k_policy=policy, importance=importance,
                              trap_budget=budget, cost_source=(lambda: cost))
    res = solver.solve(_gap(tid), on=lambda *a, **k: None)
    log = res.get("log", [])
    k, used, reason = _t2k(log)
    row = {"task": tid, "condition": label, "solved": bool(res.get("solved")), "tier": res.get("tier"),
           "t2_k": k, "k_audit": used, "k_reason": reason, "model_gens": _gens(log)}
    print(f"  {tid:5s} {label:14s} solved={str(row['solved']):5s} tier={str(row['tier']):4} "
          f"t2_k={str(k):4} gens={row['model_gens']:2d}  [{used}] {reason}", flush=True)
    return row


def main():
    print("[load] north ...", flush=True)
    model = NorthAdapter()
    p = KPolicy(**POLICY)
    rows = []
    print("\n=== c156 (resp_parser): Tier-1 FAILS -> T2 width decision fires ===", flush=True)
    rows.append(run(model, "c156", "fixed-3", policy=None))
    rows.append(run(model, "c156", "dyn-idle", policy=p, cost=0.0))
    rows.append(run(model, "c156", "dyn-busy", policy=p, cost=1.0))
    rows.append(run(model, "c156", "dyn-budget2", policy=p, cost=0.0, budget=2.0))
    print("\n=== w47 (to_snake_case): Tier-1 PASSES -> thin by default (T2 never reached) ===", flush=True)
    rows.append(run(model, "w47", "dyn-idle", policy=p, cost=0.0))

    # the deterministic claim: the disposed T2 k adapts to cost/budget on the real ladder
    by = {r["condition"]: r for r in rows if r["task"] == "c156"}
    adapts = (by["dyn-idle"]["t2_k"] is not None and by["dyn-busy"]["t2_k"] == 1
              and by["dyn-idle"]["t2_k"] > by["dyn-busy"]["t2_k"]
              and by["dyn-budget2"]["t2_k"] <= 2)
    w47 = next(r for r in rows if r["task"] == "w47")
    thin_default = (w47["tier"] == 1 and w47["t2_k"] is None)   # solved at T1, T2 decision never reached
    verdict = {
        "rows": rows,
        "k_adapts_to_cost_and_budget": bool(adapts),
        "thin_by_default_on_t1_pass": bool(thin_default),
        "VERDICT": ("scheduler-owned k WIRED + ADAPTS live on North (idle wide / busy thin / budget-capped); "
                    "thin by default when T1 passes" if (adapts and thin_default) else "see rows"),
    }
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(verdict, open(os.path.join(logdir, "m5_dynamic_k_north.json"), "w"), indent=2, default=str)
    print("\n===== M5 rung-5 verdict =====", flush=True)
    print(f"  k adapts to cost+budget (c156): {adapts}", flush=True)
    print(f"  thin by default on T1-pass (w47): {thin_default}", flush=True)
    print(f"  VERDICT: {verdict['VERDICT']}", flush=True)
    print(f"  (-> logs/m5_dynamic_k_north.json)", flush=True)


if __name__ == "__main__":
    main()
