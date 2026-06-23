#!/usr/bin/env python3
"""Supervisor-level telemetry — a structured, persisted, per-run trace of everything the kernel decided AT
THE SUPERVISOR LEVEL: intake + classification (why this job_type/template/budget), the plan (the DAG), every
node's lifecycle (execute → validate → commit/repair/abort with validation level, verdict, proof hash,
duration, confidence), budget charges, and any routing decision an executor annotates on task.meta["telemetry"].

The scheduler already kept an in-memory TaskEvent log; this turns it into a rich, on-disk audit trail so
EVERY run can be inspected after the fact (reports/telemetry/<job>.jsonl + an index.jsonl rollup). Pure
stdlib; no LLM, no network.
"""
from __future__ import annotations
import json, os, time, hashlib


def sha12(x) -> str:
    try:
        return hashlib.sha256(json.dumps(x, default=str, sort_keys=True).encode()).hexdigest()[:12]
    except Exception:
        return hashlib.sha256(str(x).encode()).hexdigest()[:12]


class TelemetrySink:
    """Persists a run's events to <dir>/<job_id>.jsonl and a one-line-per-run summary to <dir>/index.jsonl."""
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def write_event(self, run: "RunTrace", ev: dict):
        with open(os.path.join(self.dir, f"{_safe(run.job_id)}.jsonl"), "a") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def write_summary(self, summary: dict):
        with open(os.path.join(self.dir, "index.jsonl"), "a") as f:
            f.write(json.dumps(summary, default=str) + "\n")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))[:80]


def _num(x) -> float:
    """Parse a model-reported confidence (float, "0.95", or words like "high") to a number for comparison."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return {"high": 0.9, "very high": 0.95, "certain": 1.0, "medium": 0.6, "low": 0.3}.get(str(x).strip().lower(), 0.0)


class RunTrace:
    """One run's event stream. emit(kind, **fields) appends an ordered, timestamped event."""
    def __init__(self, run_id: str, job_id: str, sink: "TelemetrySink" = None):
        self.run_id, self.job_id, self.sink = run_id, job_id, sink
        self.events: list[dict] = []
        self._t0 = time.perf_counter()

    def emit(self, kind: str, **fields) -> dict:
        ev = {"seq": len(self.events) + 1, "t_ms": round((time.perf_counter() - self._t0) * 1000, 1),
              "run": self.run_id, "job": self.job_id, "kind": kind}
        ev.update({k: v for k, v in fields.items() if v is not None})
        self.events.append(ev)
        if self.sink:
            try:
                self.sink.write_event(self, ev)
            except Exception:
                pass
        return ev

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self.events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        tdone = [e for e in self.events if e["kind"] == "task" and e.get("to_state") in ("PASSED", "COMMITTED")]
        levels: dict[str, int] = {}
        for e in tdone:
            if e.get("level"):
                levels[e["level"]] = levels.get(e["level"], 0) + 1
        committed = any(e["kind"] == "commit" for e in self.events)
        # honest, validation-grounded run confidence: for a COMMITTED run it's the STRONGEST validation that
        # gated the committed result (an ORACLE-verified result committed through a judgment pass-through gate
        # is still oracle-verified — don't let the gate's judgment(0.5) undersell it). For a non-committed run
        # it's the LOWEST (a failure drags it down).
        cals = [e["conf_cal"] for e in self.events if e["kind"] == "task" and e.get("conf_cal") is not None]
        # miscalibration: model claimed high while the verdict supports low
        miscal = sum(1 for e in self.events if e["kind"] == "task"
                     and e.get("conf_model") is not None and e.get("conf_cal") is not None
                     and _num(e["conf_model"]) >= 0.7 and e["conf_cal"] <= 0.5)
        return {
            "run": self.run_id, "job": self.job_id,
            "dur_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "events": len(self.events), "event_kinds": kinds,
            "committed": committed,
            "confidence_calibrated": (None if not cals else (max(cals) if committed else min(cals))),
            "miscalibrations": miscal,
            "repairs": sum(1 for e in self.events if e["kind"] == "repair"),
            "aborts": sum(1 for e in self.events if e["kind"] == "abort"),
            "gate_fails": sum(1 for e in self.events if e["kind"] == "gate_fail"),
            "validation_levels": levels,
            "job_type": next((e.get("job_type") for e in self.events if e["kind"] == "classify"), None),
            "template": next((e.get("template") for e in self.events if e["kind"] == "classify"), None),
        }

    def lines(self) -> list[str]:
        """Human-readable one-line-per-event view of a run."""
        out = []
        for e in self.events:
            t = f"{e['t_ms']:>8.1f}ms"
            k = e["kind"]
            if k == "run.start":
                out.append(f"{t}  ▶ RUN {e['job']}  inputs={e.get('input_keys')}")
            elif k == "classify":
                out.append(f"{t}  ⊟ classify → {e.get('job_type')}/{e.get('template')}  budget={e.get('budget')} caps={e.get('capabilities')}")
            elif k == "recall":
                out.append(f"{t}  ⊟ recall  learnings={e.get('learnings')} prior={e.get('has_prior')}")
            elif k == "plan":
                out.append(f"{t}  ⊟ plan  nodes={e.get('nodes')}")
            elif k == "task":
                cm, cc = e.get("conf_model"), e.get("conf_cal")
                conf = ""
                if cc is not None:
                    conf = f" conf={cc}"                                  # validation-grounded (honest)
                    if cm is not None and str(cm) != str(cc):
                        conf += f"(model claimed {cm})"                   # expose the miscalibration gap
                routed = f" routed={e['decision'].get('routed')}" if isinstance(e.get("decision"), dict) and e["decision"].get("routed") else ""
                out.append(f"{t}  • {e['task']:<12}[{e.get('op')}] {e.get('from_state')}→{e.get('to_state')} "
                           f"{e.get('level') or '-'}/{e.get('verification')} {e.get('dur_ms')}ms proof={e.get('proof')}{conf}{routed}")
            elif k == "gate_fail":
                out.append(f"{t}  ⊘ GATE-FAIL {e.get('task')} reason={e.get('reason')} ({e.get('note')})")
            elif k == "repair":
                out.append(f"{t}  ↻ REPAIR {e.get('task')} (retry {e.get('retry')}) reason={e.get('reason')}")
            elif k == "abort":
                out.append(f"{t}  ✗ ABORT {e.get('task')} reason={e.get('reason')}")
            elif k == "budget":
                out.append(f"{t}  $ budget {e.get('verdict')} for {e.get('task')} ({e.get('detail')})")
            elif k == "commit":
                out.append(f"{t}  ✓ COMMIT root  ledger={e.get('ledger_entries')} verified={e.get('ledger_verified')}")
            elif k == "run.end":
                out.append(f"{t}  ■ END committed={e.get('committed')} events={e.get('events')} {e.get('dur_ms')}ms")
            else:
                out.append(f"{t}  · {k} {json.dumps({kk: vv for kk, vv in e.items() if kk not in ('seq','t_ms','run','job','kind')}, default=str)[:120]}")
        return out


def render(jsonl_path: str) -> str:
    """Pretty-print a persisted run JSONL (for after-the-fact inspection)."""
    rt = RunTrace("?", os.path.basename(jsonl_path).replace(".jsonl", ""))
    rt.events = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    return "\n".join(rt.lines())


if __name__ == "__main__":
    import sys
    print(render(sys.argv[1]))
