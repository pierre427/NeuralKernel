#!/usr/bin/env python3
"""Analyze supervisor telemetry across a run directory: reconstruct each run's flow and CHECK
WORKFLOW-CORRECTNESS INVARIANTS — the point is to verify the supervisor's flows are correct, not the
pass rate. Reports the flow shape, validation-level distribution, repair efficacy, calibration, and any
invariant VIOLATIONS (a committed run whose oracle never passed, a proof that didn't verify, orphan states,
an inert repair, mis-ordered transitions, etc.)."""
from __future__ import annotations
import sys, os, json, glob
from collections import Counter


def _load(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        if os.path.basename(f) == "index.jsonl":
            continue
        runs[os.path.basename(f)[:-6]] = [json.loads(l) for l in open(f) if l.strip()]
    return runs


def _num(x) -> float:
    """Parse a model-reported confidence (float, '0.95', or words like 'high') to a number for comparison."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return {"high": 0.9, "very high": 0.95, "certain": 1.0, "medium": 0.6, "low": 0.3}.get(str(x).strip().lower(), 0.0)


def check_invariants(jid, evs) -> list:
    """Workflow-correctness invariants over ONE run's event list -> list of violation strings (empty = clean).
    SHARED by the CLI analyzer AND the runtime `trace` self-check tool, so both judge by IDENTICAL rules:
    well-formed flow, no unsafe commit (committed without an ORACLE pass), ledger verified at commit, every PASSED
    node carries a proof, no inert repair (repair reproduced the original failed proof), no miscalibration."""
    if not evs:
        return [f"{jid}: empty trace"]
    evs = [e for e in evs if isinstance(e, dict)]              # tolerate garbled JSONL (bare scalars / non-dicts)
    if not evs:
        return [f"{jid}: empty/garbled trace (no well-formed dict events)"]
    viol = []
    kinds = [e.get("kind") for e in evs]                       # .get -> a missing 'kind' can't KeyError-crash
    tasks = [e for e in evs if e.get("kind") == "task"]
    is_committed = any(e.get("kind") == "commit" for e in evs)
    # 1. flow shape: classify -> plan present; run bracketed by start/end
    if "classify" not in kinds or "plan" not in kinds:
        viol.append(f"{jid}: missing classify/plan")
    if kinds[0] != "run.start" or kinds[-1] != "run.end":
        viol.append(f"{jid}: run not bracketed by start/end ({kinds[0]}..{kinds[-1]})")
    # 2. no commit without an ORACLE-level pass (the safety invariant)
    if is_committed and not any(e.get("to_state") in ("PASSED", "COMMITTED") and e.get("level") == "ORACLE" for e in tasks):
        viol.append(f"{jid}: COMMITTED but no ORACLE-level pass in the chain (unsafe commit)")
    # 3. proof ledger verified at commit
    ce = next((e for e in evs if e.get("kind") == "commit"), None)
    if ce and not ce.get("ledger_verified", False):
        viol.append(f"{jid}: committed with ledger_verified=False (tamper/uniqueness broken)")
    # 4. every PASSED node carries a proof hash
    for e in tasks:
        if e.get("to_state") == "PASSED" and not e.get("proof"):
            viol.append(f"{jid}/{e.get('task')}: PASSED without a proof hash")
    # 5. inert repair: a repair reproduced the ORIGINAL (non-repair) failed node's proof = no real work
    if any(e.get("kind") == "repair" for e in evs):
        orig_failed = {e["proof"] for e in tasks if e.get("to_state") == "FAILED"
                       and not str(e.get("op", "")).endswith("_repair") and e.get("proof")}
        rep_proofs = {e["proof"] for e in tasks if str(e.get("op", "")).endswith("_repair") and e.get("proof")}
        if orig_failed & rep_proofs:
            viol.append(f"{jid}: INERT repair (repair reproduced the original failed proof — no real change)")
    # 6. miscalibration: model claimed high while the validated verdict supports low
    mis = [e.get("task") for e in tasks if e.get("conf_model") is not None and e.get("conf_cal") is not None
           and _num(e["conf_model"]) >= 0.7 and _num(e["conf_cal"]) <= 0.5]   # coerce BOTH (string conf can't crash)
    if mis:
        viol.append(f"{jid}: miscalibration — model claimed high but validated low on {mis}")
    return viol


def analyze(d):
    runs = _load(d)
    print(f"=== telemetry flow analysis: {len(runs)} runs in {d} ===\n")
    viol = []
    committed = 0
    levels = Counter()
    flow_shapes = Counter()
    repair_runs = 0; repair_flipped = 0
    gate_fails = 0; aborts = 0
    conf_root, conf_best = [], []
    durs = []

    for jid, evs in sorted(runs.items()):
        kinds = [e["kind"] for e in evs]
        tasks = [e for e in evs if e["kind"] == "task"]
        is_committed = any(e["kind"] == "commit" for e in evs)
        committed += int(is_committed)
        durs.append(next((e.get("dur_ms", 0) for e in evs if e["kind"] == "run.end"), 0))

        # ---- INVARIANT CHECKS (shared with the runtime `trace` self-check tool) ----
        viol.extend(check_invariants(jid, evs))
        # repair efficacy counting for the summary line (the inert-repair VIOLATION is in check_invariants)
        reps = [e for e in evs if e["kind"] == "repair"]
        if reps:
            repair_runs += 1
            if any(str(e.get("op", "")).endswith("_repair") and e.get("to_state") == "PASSED"
                   and e.get("level") == "ORACLE" for e in tasks):
                repair_flipped += 1
        gate_fails += sum(1 for e in evs if e["kind"] == "gate_fail")
        aborts += sum(1 for e in evs if e["kind"] == "abort")

        # ---- distributions ----
        for e in tasks:
            if e.get("to_state") in ("PASSED", "COMMITTED") and e.get("level"):
                levels[e["level"]] += 1
        flow_shapes[" → ".join(e["task"] for e in tasks if e.get("to_state") in ("PASSED", "COMMITTED", "FAILED"))[:80]] += 1
        # calibration: root (commit) conf vs the BEST validated conf in the chain
        rc = next((e.get("conf_cal") for e in reversed(tasks) if e.get("to_state") in ("COMMITTED", "PASSED")), None)
        bc = max([e.get("conf_cal") for e in tasks if e.get("conf_cal") is not None] or [None])
        if is_committed:
            conf_root.append(rc); conf_best.append(bc)

    n = len(runs)
    # draft routing (recovery-first design: det-block vs model) + pass rate by route
    droute = Counter(); droute_pass = Counter()
    for jid, evs in runs.items():
        d = next((e.get("decision", {}).get("draft_route") for e in evs
                  if e["kind"] == "task" and e.get("op") == "draft" and isinstance(e.get("decision"), dict)), None)
        if d:
            droute[d] += 1
            if any(e["kind"] == "commit" for e in evs):
                droute_pass[d] += 1
    print(f"committed: {committed}/{n}")
    if droute:
        print(f"draft routing: " + ", ".join(f"{k}={droute_pass[k]}/{droute[k]} pass" for k in droute))
    print(f"validation levels reached (PASSED nodes): {dict(levels)}")
    print(f"repairs: {repair_runs} runs had a repair; {repair_flipped} flipped fail→ORACLE-pass "
          f"({'effective' if repair_flipped else 'INEFFECTIVE'})")
    print(f"gate_fails: {gate_fails} | aborts: {aborts}")
    print(f"avg run dur: {sum(durs)/max(1,len(durs)):.0f}ms")
    print(f"\ncalibration (committed runs):")
    if conf_root:
        print(f"  run-level confidence (strongest gating validation): {dict(Counter(conf_best))}   <- ORACLE-verified results")
        print(f"  per-node commit-gate conf: {dict(Counter(conf_root))}   <- now reflects the evidence backing the commit (honest commit verification); RUN conf uses the gated validation")
    print(f"\nflow shapes: {dict(flow_shapes)}")
    print(f"\n=== INVARIANT VIOLATIONS: {len(viol)} ===")
    for v in viol[:40]:
        print("  ✗", v)
    if not viol:
        print("  ✓ none — every run is well-formed, no unsafe commits, all proofs verified, no inert repairs")


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "reports/telemetry/expert50")
