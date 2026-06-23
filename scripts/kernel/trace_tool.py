#!/usr/bin/env python3
"""kernel/trace_tool.py — the `trace` SYSCALL-tier tool: read scheduler execution traces (LIVE telemetry or
PERSISTED reports/telemetry/*.jsonl) and run the kernel's SELF-CHECKS over them. The historical twin of sysinfo
(sysinfo = live state; trace = what happened + whether it was correct).

Reuses the existing telemetry stack — RunTrace (events/summary/lines) for reading, and the SHARED
analyze_telemetry.check_invariants for the self-check — so a runtime self-check judges by the SAME rules as the
CLI flow analyzer. The self-check verdict is ALWAYS computed (that's the point of the tool); the requested facet
(summary/events/lines) is added on top. op:trace, syscall tier (reads I/O + live state; nondeterministic)."""
from __future__ import annotations
import json, os, glob

try:
    from .analyze_telemetry import check_invariants
except ImportError:
    from analyze_telemetry import check_invariants


def _RunTrace():
    try:
        from .telemetry import RunTrace
    except ImportError:
        from telemetry import RunTrace
    return RunTrace


def _events_from_file(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _runs_from_dir(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        if os.path.basename(f) == "index.jsonl":
            continue
        try:
            runs[os.path.basename(f)[:-6]] = _events_from_file(f)
        except Exception:
            continue
    return runs


def _view(evs, kind):
    """Render summary/lines via RunTrace, but DEFENSIVELY — a malformed/partial external trace must not lose the
    (already-computed) self-check verdict. A render failure degrades to an error note, never raises."""
    rt = _RunTrace()("?", "?")
    rt.events = evs
    try:
        return rt.summary() if kind == "summary" else rt.lines()
    except Exception as e:
        msg = f"{kind} render failed: {type(e).__name__}: {e}"
        return {"error": msg} if kind == "summary" else [msg]


def trace_handler(sched, inputs):
    """inputs: {source:'live'|'file'|'dir' (default live), path?, job?, facet:'selfcheck'|'summary'|'events'|
    'lines'|'all' (default selfcheck), limit?}. Returns the self-check verdict for the selected run(s) (always)
    plus the requested facet. The self-check is the kernel auditing its OWN execution history."""
    inputs = inputs or {}
    source = inputs.get("source", "live")
    facet = inputs.get("facet", "selfcheck")
    limit = int(inputs.get("limit", 0) or 0)

    # gather runs: {job_id -> events}
    if source == "dir":
        d = inputs.get("path") or os.path.join("reports", "telemetry")
        if not os.path.isdir(d):
            return {"error": f"trace dir not found: {d}"}
        runs = _runs_from_dir(d)
    elif source == "file":
        p = inputs.get("path")
        if not p or not os.path.isfile(p):
            return {"error": f"trace file not found: {p!r}"}
        jid = os.path.basename(p)[:-6] if p.endswith(".jsonl") else os.path.basename(p)
        runs = {jid: _events_from_file(p)}
    else:                                                    # live
        rt = getattr(sched, "telemetry", None)
        evs = list(getattr(rt, "events", []) or []) if rt is not None else []
        if not evs:
            return {"source": "live", "note": "no live telemetry on this scheduler (sched.telemetry unset/empty)",
                    "selfcheck": {"runs": 0, "passed": True, "violation_count": 0, "violations": []}}
        runs = {getattr(rt, "job_id", "live"): evs}

    if inputs.get("job"):
        runs = {k: v for k, v in runs.items() if k == inputs["job"]}

    # SELF-CHECK is always computed — the kernel auditing its own trace
    all_viol = []
    per_run = {}
    for jid, evs in sorted(runs.items()):
        v = check_invariants(jid, evs)
        per_run[jid] = {"events": len(evs), "committed": any(e.get("kind") == "commit" for e in evs),
                        "violations": v}
        all_viol.extend(v)
    out = {"source": source, "runs": len(runs),
           "selfcheck": {"passed": not all_viol, "violation_count": len(all_viol),
                         "violations": all_viol[:50], "per_run": per_run}}

    if facet in ("summary", "all"):
        out["summaries"] = {jid: _view(evs, "summary") for jid, evs in runs.items()}
    if facet in ("events", "all"):
        out["events"] = {jid: (evs[-limit:] if limit else evs) for jid, evs in runs.items()}
    if facet in ("lines", "all"):
        out["lines"] = {jid: (_view(evs, "lines")[-limit:] if limit else _view(evs, "lines"))
                        for jid, evs in runs.items()}
    return out


__all__ = ["trace_handler"]
