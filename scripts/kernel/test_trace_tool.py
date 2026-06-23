#!/usr/bin/env python3
"""CPU test for the `trace` tool + the shared check_invariants self-check (no model). Builds synthetic traces
that each trip exactly one invariant, and exercises live/file sourcing + the tool registration."""
import sys, os, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/ (so `kernel` imports as a package)
from kernel.analyze_telemetry import check_invariants
from kernel.trace_tool import trace_handler
from kernel.telemetry import RunTrace
from kernel.task_graph import Scheduler
from kernel.tools import register_builtins, TOOL_CATALOG

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


_SEQ = [0]
def ev(kind, **kw):                                          # match emit()'s shape (seq/t_ms/run/job present)
    _SEQ[0] += 1
    return {"seq": _SEQ[0], "t_ms": float(_SEQ[0]), "run": "r", "job": "j", "kind": kind, **kw}
def task_ev(task, to_state="PASSED", level="ORACLE", proof="P", op="draft", conf_model=None, conf_cal=None):
    return ev("task", task=task, op=op, from_state="RUNNING", to_state=to_state, level=level,
              verification="verified", proof=proof, conf_model=conf_model, conf_cal=conf_cal, dur_ms=1.0)

def clean_run():
    return [ev("run.start", job="j", input_keys=["x"]), ev("classify", job_type="code_gen", template="code_gen"),
            ev("plan", nodes=1), task_ev("n1", "PASSED", "ORACLE", "P1"),
            ev("commit", ledger_verified=True, ledger_entries=1),
            ev("run.end", committed=True, events=6, dur_ms=6.0)]

def unsafe_commit_run():                                     # committed but NO ORACLE pass
    r = clean_run(); r[3] = task_ev("n1", "PASSED", "SCHEMA", "P1"); return r

def miscal_run():                                            # model claimed high, validated low
    r = clean_run(); r[3] = task_ev("n1", "PASSED", "ORACLE", "P1", conf_model=0.9, conf_cal=0.1); return r

def inert_repair_run():                                      # repair reproduced the original failed proof
    return [ev("run.start"), ev("classify"), ev("plan"),
            task_ev("n1", "FAILED", "SCHEMA", "P1", op="draft"), ev("repair", task="n1", retry=1),
            task_ev("n1r", "FAILED", "SCHEMA", "P1", op="draft_repair"), ev("run.end")]

def noproof_run():                                           # PASSED node with no proof hash
    return [ev("run.start"), ev("classify"), ev("plan"), task_ev("n1", "PASSED", "ORACLE", proof=None),
            ev("run.end")]


def main():
    # 1. check_invariants: clean run is clean; each bad run trips exactly its invariant
    ok(check_invariants("j", clean_run()) == [], "clean run -> no violations")
    ok(any("unsafe commit" in v for v in check_invariants("j", unsafe_commit_run())), "unsafe commit detected")
    ok(any("miscalibration" in v for v in check_invariants("j", miscal_run())), "miscalibration detected")
    ok(any("INERT repair" in v for v in check_invariants("j", inert_repair_run())), "inert repair detected")
    ok(any("without a proof" in v for v in check_invariants("j", noproof_run())), "PASSED-without-proof detected")
    ok(check_invariants("j", []) == ["j: empty trace"], "empty trace -> reported, no crash")
    # BUG-SWEEP HIGH: garbled JSONL (bare scalars / missing 'kind') and string-typed conf must NOT crash the check
    garbled = check_invariants("g", [42, "x", {"no_kind": 1},
                                     {"kind": "task", "conf_model": "high", "conf_cal": "low", "to_state": "PASSED", "proof": "p"}])
    ok(isinstance(garbled, list), "check_invariants tolerates garbled/non-dict events + string conf (no crash)")

    # 2. trace tool — LIVE source reads sched.telemetry
    s = Scheduler()
    rt = RunTrace("r1", "jlive"); rt.events = clean_run(); s.telemetry = rt
    live = trace_handler(s, {"source": "live"})
    ok(live["runs"] == 1 and live["selfcheck"]["passed"], "trace(live): reads sched.telemetry, clean -> passed")
    s2 = Scheduler()                                          # no telemetry set
    ok(trace_handler(s2, {"source": "live"})["selfcheck"]["passed"], "trace(live) with no telemetry -> no crash, passed")

    # 3. trace tool — FILE source flags a bad run
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.jsonl")
        with open(p, "w") as f:
            for e in unsafe_commit_run():
                f.write(json.dumps(e) + "\n")
        r = trace_handler(None, {"source": "file", "path": p})
        ok((not r["selfcheck"]["passed"]) and any("unsafe commit" in v for v in r["selfcheck"]["violations"]),
           "trace(file): self-check flags the unsafe-commit run")

    # 4. facets: summary/lines/all are added on top of the always-on self-check
    allv = trace_handler(s, {"source": "live", "facet": "all"})
    ok("summaries" in allv and "lines" in allv and "events" in allv and "selfcheck" in allv,
       "trace(facet=all): summaries + lines + events + selfcheck all present")
    ok(allv["summaries"]["jlive"]["committed"] is True, "summary facet reflects the committed run")

    # 4b. defensive: a MALFORMED trace (events missing t_ms) must not crash the tool or lose the self-check
    sm = Scheduler(); rtm = RunTrace("r", "jm")
    rtm.events = [{"kind": "run.start"}, {"kind": "task", "to_state": "PASSED"}]   # no t_ms/job -> lines() would KeyError
    sm.telemetry = rtm
    rm = trace_handler(sm, {"source": "live", "facet": "all"})
    ok("selfcheck" in rm and isinstance(rm.get("lines", {}).get("jm"), list),
       "malformed trace: tool degrades gracefully (self-check preserved, lines render-note, no crash)")

    # 5. registration: trace ships as a default tool, gated op:trace
    Scheduler()  # fresh
    wirings = register_builtins(Scheduler(), grant=True)
    names = {w["kind"] for w in wirings}
    ok("trace" in names and "trace" in TOOL_CATALOG, "trace is registered as a default tool (op:trace)")
    ok(next(w for w in wirings if w["kind"] == "trace")["capability"] == "op:trace", "trace gated behind op:trace")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} trace-tool checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
