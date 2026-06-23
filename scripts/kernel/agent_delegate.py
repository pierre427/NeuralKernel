#!/usr/bin/env python3
"""kernel/agent_delegate.py — TRUSTLESS agent delegation (Step 3.5): a requestor asks the scheduler to fire a
clean-context AGENT to run a tool, the scheduler runs + RECORDS it (content-addressed artifact + telemetry +
proof), switches back to the requestor, and the requestor RECONCILES the returned result against the kernel's own
trace — never trusting the agent's word.

"agent claims, trace proves, requestor reconciles" — learned-proposes / kernel-disposes applied to delegation.
A fabricated/tampered result won't hash-match the recorded proof; an unbacked result has no trace event; a result
from an unsafe/ill-formed run fails the self-check. Pairs the self-extending tool (the agent runs a freshly-minted
primitive) with the trace tool (the requestor's verification). Reuses Scheduler artifacts + RunTrace +
analyze_telemetry.check_invariants.
"""
from __future__ import annotations

try:
    from .task_graph import Task
    from .commit_trap import _sha
    from .telemetry import RunTrace
    from .analyze_telemetry import check_invariants
except ImportError:
    from task_graph import Task
    from commit_trap import _sha
    from telemetry import RunTrace
    from analyze_telemetry import check_invariants


def delegate_run(sched, tool, inputs, *, agent_id="a1", requestor="requestor", run_id="d1"):
    """Fire a clean-context AGENT sub-task to run `tool(inputs)` through the scheduler's registered executor; the
    scheduler RECORDS the run (content-addressed artifact + a well-formed delegation trace whose task event's
    proof IS the result's content hash). Returns the run record the requestor reconciles against. The result is
    GENUINELY produced by the registered tool — not fabricated — so the trace is real."""
    if tool not in sched.executors:
        return {"ok": False, "error": f"unknown tool {tool!r} (not a registered executor)", "agent_task_id": None}
    # Resolve the tool's REGISTERED wiring (validator + capability) so the agent task goes through the SAME kernel
    # gates as any other task — fixes the admit-BYPASS (CRITICAL): delegation must not call the executor directly.
    vkey = f"{tool}_ok" if f"{tool}_ok" in sched.validators else None
    sop = getattr(sched, "_syscall_ops", {}).get(tool)
    cap = sop["cap"] if sop else (f"prim:{tool}" if f"prim:{tool}" in getattr(sched, "capabilities", set()) else None)
    level = (sop["max_level"] if sop else "PROVEN") if vkey else "NONE"
    agent_task_id = f"agent:{agent_id}/{tool}"
    atask = Task(task_id=agent_task_id, kind=tool, owner=f"agent:{agent_id}",
                 validator=({"fn": vkey, "level": level} if vkey else None),
                 capabilities=([cap] if cap else []))
    try:
        sched.admit(atask)                                   # KERNEL GATES: cap mandatory + ceiling (syscall), validator registered
    except Exception as e:
        return {"ok": False, "error": f"admit refused: {e}", "agent_task_id": agent_task_id}
    # the agent's clean-context tool-run (executor sees ONLY its inputs)
    try:
        result, ex_ok, ex_level = sched.executors[tool](sched, atask, inputs or {})
    except Exception as e:
        return {"ok": False, "error": f"agent tool-run raised: {type(e).__name__}: {e}", "agent_task_id": agent_task_id}
    result_ref = sched.artifacts.put(result, kind=f"agent_result:{tool}")    # content-addressed (refs not blobs)
    content_hash = sched.artifacts.meta(result_ref)["hash"]
    # KERNEL-SIDE verdict: the REGISTERED validator decides PASS/FAIL — NOT the executor's self-reported ex_ok
    # (fixes the SELF-GRADED-verdict CRITICAL). A raising/absent validator fails CLOSED.
    vfn = sched.validators.get(vkey) if vkey else None
    try:
        kernel_ok = bool(vfn(result, atask)) if vfn else False
    except BaseException:
        kernel_ok = False
    state = "PASSED" if kernel_ok else "FAILED"
    # the task event records the KERNEL verdict (to_state) + provenance proof (content_hash); no commit (an
    # intermediate delegation), so check_invariants' unsafe-commit/ledger checks don't apply.
    rt = RunTrace(run_id, f"delegate:{tool}")
    rt.emit("run.start", input_keys=sorted((inputs or {}).keys()), requestor=requestor)
    rt.emit("classify", job_type="delegate", template="delegate")
    rt.emit("plan", nodes=1)
    rt.emit("task", task=agent_task_id, op=tool, owner=f"agent:{agent_id}", from_state="READY", to_state=state,
            level=(level if kernel_ok else "NONE"), verification=("validated" if kernel_ok else "unverified"),
            proof=content_hash, dur_ms=0.0)
    rt.emit("run.end", committed=False, events=5)
    sched.telemetry = rt                                     # the requestor reads this to reconcile
    return {"ok": kernel_ok, "agent_task_id": agent_task_id, "tool": tool, "result": result,
            "result_ref": result_ref, "content_hash": content_hash, "run_id": run_id, "trace": rt,
            "kernel_validated": kernel_ok}


def reconcile_with_trace(sched, agent_task_id, claimed_result, *, trace=None):
    """The requestor's verification: does `claimed_result` LINE UP with the kernel's recorded trace for
    `agent_task_id`? Consistent iff (1) the run's self-checks pass (well-formed, no unsafe commit / inert repair /
    proofless pass / miscalibration), (2) the trace shows the agent task PASSED — where the verdict is the KERNEL'S
    registered validator, NOT the agent's self-reported ex_ok (so a self-grading executor can't fake it), and
    (3) the claimed result HASHES to the recorded proof (PROVENANCE: it's the kernel-recorded result, not a
    fabrication). NOTE: the STRENGTH of 'consistent' equals the strength of the tool's VALIDATOR — for a hold-out-
    gated PROVEN primitive it's correctness; for a generic syscall tool it's CHECKED (the call ran + returned a
    well-formed result). Untrusted agent code never runs here — the trace is data; the hashes are the kernel's."""
    rt = trace or getattr(sched, "telemetry", None)
    if rt is None or not getattr(rt, "events", None):
        return {"consistent": False, "reason": "no trace to reconcile against", "checks": {}}
    events = rt.events
    jid = getattr(rt, "job_id", "delegate")
    viol = check_invariants(jid, events)
    ev = next((e for e in events if e.get("kind") == "task" and e.get("task") == agent_task_id), None)
    claimed_hash = _sha(claimed_result)
    recorded = ev.get("proof") if ev else None
    checks = {
        "self_check_passed": (viol == []),
        "violations": viol,
        "task_event_found": ev is not None,
        "task_passed": bool(ev and ev.get("to_state") == "PASSED"),
        "hash_match": (recorded is not None and claimed_hash == recorded),
        "claimed_hash": claimed_hash,
        "recorded_proof": recorded,
    }
    consistent = checks["self_check_passed"] and checks["task_passed"] and checks["hash_match"]
    return {"consistent": consistent, "checks": checks,
            "reason": None if consistent else "result does not line up with the trace"}


def delegate_and_verify(sched, tool, inputs, **kw):
    """Fire the agent, switch back, reconcile. Returns {verified, result, reconcile, run}. `verified` is True ONLY
    if the returned result is backed by a sound, hash-matching trace — the requestor's trust comes from the
    kernel's record, not the agent."""
    run = delegate_run(sched, tool, inputs, **kw)
    if not run.get("agent_task_id"):
        return {"verified": False, "result": None, "run": run, "reconcile": {"consistent": False, "reason": run.get("error")}}
    rec = reconcile_with_trace(sched, run["agent_task_id"], run.get("result"), trace=run.get("trace"))
    return {"verified": bool(rec["consistent"] and run.get("ok")), "result": run.get("result"),
            "reconcile": rec, "run": {k: run[k] for k in ("agent_task_id", "tool", "content_hash", "result_ref") if k in run}}


__all__ = ["delegate_run", "reconcile_with_trace", "delegate_and_verify"]
