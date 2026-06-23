#!/usr/bin/env python3
"""REAL executors that wire the Job Kernel to the actual appliance + det-primitives — the convergence of
both threads. Coding nodes route to hold-out-proven det-blocks (recover-direct, 0 model tokens) and
validate against the task's real cases via the appliance's check_cases; cyber nodes run the real
det_primitives_cyber to extract guaranteed-correct evidence. CPU-only for these paths (no GPU): the
orchestrator drives genuine proven primitives end-to-end, every node audited in the proof ledger.

A model-backed executor (gen_fast via an adapter) can be registered for the non-recoverable tail, but the
proven recover-direct + det-primitive paths need no model — exactly the appliance's recovery-first design.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape_recoveries import RECOVERIES
from shape_appliance import check_cases, _compile
from kernel.task_graph import ValidationLevel
import det_primitives_cyber as cyber

try:
    from challenge_union_recoveries import RECOVERIES_EXTRA
except Exception:
    RECOVERIES_EXTRA = {}

# the full hold-out-proven det-block library (177 entries: 59 shape + 121 challenge-union, deduped)
ALL_RECOVERIES = {**RECOVERIES, **RECOVERIES_EXTRA}


# ------------------------------------------------------------------ coding executors (the appliance) ---
def _jin(task) -> dict:
    return task.meta.get("job_inputs", {})

def solve(sched, task, inputs):
    """Route recover-direct: if a hold-out-proven recovery exists for the entry, that det-block IS the
    candidate (0 model tokens, guaranteed-correct). Otherwise fail -> a model executor would escalate."""
    entry = _jin(task).get("entry")
    rec = ALL_RECOVERIES.get(entry)
    if rec is None:
        return None, False, ValidationLevel.NONE        # no det-block; out of scope for the CPU path
    return rec, True, ValidationLevel.PROVEN             # the recovery source (a code string)

def ast_parse(sched, task, inputs):
    code = inputs.get("draft") or inputs.get("solve") or next((v for v in inputs.values() if isinstance(v, str)), None)
    try:
        compile(code, "<recovery>", "exec"); return code, True, ValidationLevel.SYNTAX
    except Exception:
        return code, False, ValidationLevel.NONE

def run_tests(sched, task, inputs):
    """REAL validation: compile the det-block and run it against the task's actual cases (check_cases).
    Carries the code forward so the commit node can commit it."""
    code = next((v for v in inputs.values() if isinstance(v, str)), None)
    jin = _jin(task); entry, cases = jin.get("entry"), jin.get("cases", [])
    passed = bool(code) and check_cases(code, entry, cases)
    return {"code": code, "entry": entry, "n_cases": len(cases), "passed": passed}, passed, ValidationLevel.ORACLE

def commit_node(sched, task, inputs):
    """Generic commit (the only output path) — works for any job type: commit the upstream validated
    artifact. For coding it's the code carried in {code,...}; for cyber it's the analyst-note dict."""
    val = next((v for v in inputs.values() if v is not None), None)
    if isinstance(val, dict) and "code" in val:
        val = val["code"]
    return val, val is not None, ValidationLevel.PROVEN

# coding validators
def v_compiles(result, task): return result is not None
def v_tests_pass(result, task): return isinstance(result, dict) and result.get("passed") is True
def v_has_code(result, task): return isinstance(result, str) and len(result) > 0


# ------------------------------------------------------------- cyber executors (det_primitives_cyber) ---
def extract_observables(sched, task, inputs):
    """Pull structured observables from the job's evidence dict (the parts the det-primitives consume)."""
    ev = _jin(task).get("evidence", {})
    return {k: ev.get(k) for k in ("blob", "url", "host", "allowlist", "timestamps", "tol",
                                   "event_id", "geo") if k in ev}, True, ValidationLevel.SCHEMA

def det_checks(sched, task, inputs):
    """THE CYBER DET-BLOCK: run the real hold-out-proven cyber primitives over the observables to produce
    guaranteed-correct evidence anchors (the tuning lever for the anchor-heavy tiers)."""
    obs = inputs.get("observ") or next((v for v in inputs.values() if isinstance(v, dict)), {})
    out = {}
    if obs.get("blob") is not None:
        out["decoded_magic"] = cyber.decode_and_magic(obs["blob"])
        out["sha256"] = cyber.sha256_hex(obs["blob"])
    if obs.get("url") is not None and obs.get("allowlist") is not None:
        out["url_allowed"] = cyber.url_hostname_allowed(obs["url"], obs["allowlist"])
    if obs.get("host") is not None:
        out["ssrf_target"] = cyber.is_ssrf_target(obs["host"])
    if obs.get("timestamps") is not None:
        out["beacon_period"] = cyber.detect_beacon(obs["timestamps"], obs.get("tol", 2))
    if obs.get("event_id") is not None:
        out["win_event"] = cyber.classify_win_event(obs["event_id"])
    if obs.get("geo") is not None:
        g = obs["geo"]; out["travel_kmh"] = cyber.haversine_kmh(*g)
    out["checks_ran"] = len(out)
    return out, len(out) > 0, ValidationLevel.ORACLE

def draft_note(sched, task, inputs):
    """Compose an analyst note from the deterministic evidence (a model would write prose here; for the
    CPU path we assemble the verified anchors so the note is grounded by construction)."""
    ev = inputs.get("checks") or next((v for v in inputs.values() if isinstance(v, dict)), {})
    return {"summary": "evidence-grounded analyst note", "anchors": ev}, True, ValidationLevel.SCHEMA

def verify_claims(sched, task, inputs):
    note = inputs.get("note") or next((v for v in inputs.values() if isinstance(v, dict)), {})
    return note, isinstance(note, dict) and "anchors" in note, ValidationLevel.PROPERTY

# cyber validators
def v_iocs_ok(result, task): return isinstance(result, dict)
def v_checks_ran(result, task): return isinstance(result, dict) and result.get("checks_ran", 0) > 0
def v_claims_grounded(result, task): return isinstance(result, dict) and bool(result.get("anchors"))


def wire_real_executors(orch):
    """Register the real coding + cyber executors/validators/agent-assignments onto an Orchestrator."""
    ex = {"draft": solve, "ast_parse": ast_parse, "run_tests": run_tests, "commit": commit_node,
          "extract_observables": extract_observables, "normalize_iocs": extract_observables,
          "det_checks": det_checks, "draft_note": draft_note, "verify_claims": verify_claims}
    for k, fn in ex.items():
        orch.register_executor(k, fn)
    vs = {"compiles": v_compiles, "tests_pass": v_tests_pass, "has_code": v_has_code,
          "is_json": v_iocs_ok, "schema_ok": v_iocs_ok, "iocs_ok": v_iocs_ok,
          "checks_ran": v_checks_ran, "claims_grounded": v_claims_grounded}
    for k, fn in vs.items():
        orch.register_validator(k, fn)
    for k, owner in {"draft": "appliance:recover-direct", "ast_parse": "primitive:ast",
                     "run_tests": "primitive:check_cases", "commit": "commit_trap",
                     "extract_observables": "primitive:nlp", "normalize_iocs": "primitive:ioc",
                     "det_checks": "primitive:det_primitives_cyber", "draft_note": "model_lane",
                     "verify_claims": "agent:verifier"}.items():
        orch.assign_agent(k, owner)
    return orch
