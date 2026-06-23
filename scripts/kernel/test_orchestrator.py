#!/usr/bin/env python3
"""CPU-only tests for the Job Kernel (kernel/orchestrator.py).

Demonstrates: ONE orchestrator assesses incoming jobs of DIFFERENT types (coding / cyber / structured-
output), classifies each, builds an agent-assigned work plan from a template, runs it on the task graph,
and rolls up a master tracker view; the PlanMemory stores + recalls details, learns which template worked,
and persists across runs."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.orchestrator import Orchestrator, Job, PlanMemory, default_classifier_rules
from kernel.task_graph import ValidationLevel

CHECKS = []
def ok(c, label):
    CHECKS.append((c, label)); print(f"  {'OK ' if c else 'XX '} {label}")


# ---- deterministic stub executors across ALL job-type templates ----
def _draft(s, t, i):            return {"order_id": 1, "items": [{"sku": "A", "price": 9.99}]}, True, ValidationLevel.SCHEMA
def _vsyntax(s, t, i):          return i.get("draft"), True, ValidationLevel.SYNTAX
def _vschema(s, t, i):          return i.get("syntax"), True, ValidationLevel.SCHEMA
def _ast_parse(s, t, i):        return i.get("draft"), True, ValidationLevel.SYNTAX
def _run_tests(s, t, i):        return {"passed": 5, "failed": 0}, True, ValidationLevel.ORACLE
def _extract_obs(s, t, i):      return {"ips": ["10.0.0.1"], "domains": ["evil.test"]}, True, ValidationLevel.SCHEMA
def _normalize(s, t, i):        return {"ips": ["10.0.0.1"], "private": True}, True, ValidationLevel.PROPERTY
def _det_checks(s, t, i):       return {"ssrf": True, "beacon_ms": 60000, "checks_ran": 3}, True, ValidationLevel.ORACLE
def _draft_note(s, t, i):       return {"summary": "C2 beacon to evil.test from 10.0.0.1"}, True, ValidationLevel.SCHEMA
def _verify_claims(s, t, i):    return (i.get("note") or next((v for v in i.values() if v is not None), None)), True, ValidationLevel.PROPERTY
def _commit(s, t, i):           # commit returns the most-validated upstream artifact
    return i.get("schema") or i.get("tests") or i.get("verify") or next(iter(i.values()), None), True, ValidationLevel.PROVEN

# validators
def _is_json(r, t):      return isinstance(r, dict)
def _schema_ok(r, t):    return isinstance(r, dict)
def _compiles(r, t):     return r is not None
def _tests_pass(r, t):   return isinstance(r, dict) and r.get("failed") == 0
def _iocs_ok(r, t):      return isinstance(r, dict) and "ips" in r
def _checks_ran(r, t):   return isinstance(r, dict) and r.get("checks_ran", 0) > 0
def _claims_ok(r, t):    return r is not None


def build_orch(mem=None):
    o = Orchestrator(memory=mem)
    default_classifier_rules(o)
    for k, fn in {"draft": _draft, "validate_syntax": _vsyntax, "validate_schema": _vschema,
                  "ast_parse": _ast_parse, "run_tests": _run_tests, "extract_observables": _extract_obs,
                  "normalize_iocs": _normalize, "det_checks": _det_checks, "draft_note": _draft_note,
                  "verify_claims": _verify_claims, "commit": _commit}.items():
        o.register_executor(k, fn)
    for k, fn in {"is_json": _is_json, "schema_ok": _schema_ok, "compiles": _compiles,
                  "tests_pass": _tests_pass, "iocs_ok": _iocs_ok, "checks_ran": _checks_ran,
                  "claims_grounded": _claims_ok}.items():
        o.register_validator(k, fn)
    # AGENT ASSIGNMENT — map each task kind to the agent/primitive that owns it
    for k, owner in {"draft": "model_lane", "draft_note": "model_lane",
                     "validate_syntax": "primitive:json", "validate_schema": "primitive:schema",
                     "ast_parse": "primitive:ast", "run_tests": "tool:test_runner",
                     "extract_observables": "primitive:nlp", "normalize_iocs": "primitive:ioc",
                     "det_checks": "primitive:cyber", "verify_claims": "agent:verifier",
                     "commit": "commit_trap"}.items():
        o.assign_agent(k, owner)
    return o


def test_generalizes_across_job_types():
    print("\n[1] one orchestrator handles coding / cyber / structured-output jobs")
    o = build_orch()
    jobs = [
        Job("j-code", prompt="Write a Python function `gcd(a,b)` that returns the gcd.", inputs={"entry": "gcd"}),
        Job("j-cyber", prompt="Triage this SOC alert: possible C2 beacon and malware IOC.", inputs={}),
        Job("j-struct", prompt="Return a JSON object matching the order schema.", inputs={}),
    ]
    rollups = {j.job_id: o.submit(j) for j in jobs}
    ok(rollups["j-code"]["job_type"] == "coding", "coding job classified as coding (-> code_gen template)")
    ok(rollups["j-cyber"]["job_type"] == "cyber", "cyber job classified as cyber (-> cyber_alert template)")
    ok(rollups["j-struct"]["job_type"] == "structured_output", "JSON job classified as structured_output")
    ok(all(r["committed"] for r in rollups.values()), "all three jobs committed via their plans")
    ok(all(r["ledger_verified"] for r in rollups.values()), "every job's proof ledger is hash-verified")
    # agents were assigned per node
    owners = {row["owner"] for row in rollups["j-cyber"]["tasks"]}
    ok("primitive:cyber" in owners and "commit_trap" in owners, "cyber plan nodes were ASSIGNED to agents/primitives")
    print("    cyber checklist:")
    for line in rollups["j-cyber"]["checklist"]:
        print("      ", line)


def test_master_rollup_dashboard():
    print("\n[2] master rollup across all jobs (the dashboard)")
    o = build_orch()
    for j in [Job("a", prompt="def f(): pass"), Job("b", prompt="C2 beacon alert IOC"),
              Job("c", prompt="emit json schema"), Job("d", prompt="another malware incident")]:
        o.submit(j)
    dash = o.dashboard()
    ok(dash["jobs_total"] == 4, "dashboard counts all 4 jobs")
    ok(dash["committed_total"] == 4, "dashboard shows all committed")
    ok(dash["by_type"]["cyber"]["jobs"] == 2, "dashboard groups by job type (2 cyber)")
    ok("cyber" in dash["proven_templates"], "dashboard surfaces the proven template per type")
    print("    dashboard:", {k: dash[k] for k in ("jobs_total", "committed_total", "by_type")})


def test_memory_store_recall_learn_persist():
    print("\n[3] plan memory: store/recall details, learn proven templates, persist across runs")
    path = os.path.join(tempfile.gettempdir(), "test_plan_memory.json")
    if os.path.exists(path): os.remove(path)
    mem = PlanMemory(path)
    # store an arbitrary detail (used like a memory) + a learning
    mem.remember("coding", "user_pref", "wants type hints + docstrings")
    mem.note_learning("cyber", "det_checks catches missing SSRF evidence")
    o = build_orch(mem)
    o.submit(Job("m1", prompt="def add(a,b): return a+b"))    # a coding run -> records template stats
    ok(mem.best_template("coding") == "code_gen", "memory learned code_gen as the coding template")
    ok(mem.recall("coding", "user_pref") == "wants type hints + docstrings", "stored detail recalled")
    # the recalled details are threaded onto the plan's task nodes (available to executors)
    r = o.submit(Job("m2", prompt="def sub(a,b): return a-b"))
    node_meta = next(t for t in o.history[-1]["tasks"])
    ok(o.classifier.classify(Job("m3", prompt="def z(): pass"), mem).template == "code_gen",
       "classifier reuses the memory's proven template")
    # persist + reload
    mem.save()
    mem2 = PlanMemory(path)
    ok(mem2.recall("coding", "user_pref") == "wants type hints + docstrings", "memory persisted + reloaded")
    ok("det_checks catches missing SSRF evidence" in mem2.learnings_for("cyber"), "learnings persisted")
    os.remove(path)


def test_failure_is_learned():
    print("\n[4] a job that fails its plan is recorded as a learning (memory grows from failure)")
    o = build_orch()
    o.register_executor("run_tests", lambda s, t, i: ({"passed": 2, "failed": 3}, True, ValidationLevel.ORACLE))  # tests fail
    r = o.submit(Job("bad", prompt="def broken(): pass"))   # coding -> run_tests validator fails
    ok(not r["committed"], "the failing coding job did NOT commit")
    ok(any("failed at" in L for L in o.memory.learnings_for("coding")), "the failure was recorded as a learning")


if __name__ == "__main__":
    test_generalizes_across_job_types()
    test_master_rollup_dashboard()
    test_memory_store_recall_learn_persist()
    test_failure_is_learned()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} job-kernel checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
