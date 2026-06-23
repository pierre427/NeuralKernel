#!/usr/bin/env python3
"""CPU-only tests for the Proof-Carrying Task Graph (kernel/task_graph.py).

Demonstrates: model proposes / scheduler admits; the DAG executes with the scheduler owning state; a
validation FAILURE spawns a clean-context repair worker that fixes only the failing slice; the root
commits via early bailout; every transition is recorded in a hash-verified proof ledger; the verification
taxonomy is explicit (not collapsed to "done"); unsupported model-proposed ops are rejected."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.task_graph import (
    Scheduler, Task, TaskState, Verification, Budget, DeterministicPlanner,
    ModelAssistedPlanner, SchedulerReject, ValidationLevel,
)

CHECKS = []
def ok(cond, label):
    CHECKS.append((cond, label))
    print(f"  {'OK ' if cond else 'XX '} {label}")


# ----- stub executors (deterministic; stand in for primitives / model lane / repair agent) -----
def _draft(sched, task, inputs):
    # the model "drafts" a structured object with a SCHEMA BUG: price is a string, not a number
    return {"order_id": 7, "items": [{"sku": "A", "price": "19.99"}]}, True, ValidationLevel.SYNTAX

def _validate_syntax(sched, task, inputs):
    return inputs.get("draft"), True, ValidationLevel.SYNTAX        # passes the dict through

def _validate_schema(sched, task, inputs):
    return inputs.get("syntax"), True, ValidationLevel.SCHEMA       # passes through; validator judges it

def _validate_schema_repair(sched, task, inputs):
    # CLEAN CONTEXT: the repair worker sees only the failing object + fixes the one bad field
    bad = inputs.get("_failed") or inputs.get("schema") or inputs.get("draft")
    fixed = {**bad, "items": [{**it, "price": float(it["price"])} for it in bad["items"]]}
    return fixed, True, ValidationLevel.SCHEMA

def _commit(sched, task, inputs):
    return inputs.get("schema"), True, ValidationLevel.PROVEN       # the validated object

# ----- validators -----
def _is_json(result, task): return isinstance(result, dict)
def _schema_ok(result, task):
    try: return all(isinstance(it["price"], (int, float)) for it in result["items"])
    except Exception: return False


def build_sched():
    s = Scheduler()
    s.register_executor("draft", _draft)
    s.register_executor("validate_syntax", _validate_syntax)
    s.register_executor("validate_schema", _validate_schema)
    s.register_executor("validate_schema_repair", _validate_schema_repair)
    s.register_executor("commit", _commit)
    s.register_validator("is_json", _is_json)
    s.register_validator("schema_ok", _schema_ok)
    return s


def test_structured_output_repair():
    print("\n[1] structured-output with mid-plan repair (draft bad -> validate -> repair -> commit)")
    s = build_sched()
    plan = [
        Task("draft", "draft"),
        Task("syntax", "validate_syntax", dependencies=["draft"], validator={"fn": "is_json", "level": "SYNTAX"}),
        Task("schema", "validate_schema", dependencies=["syntax"], validator={"fn": "schema_ok", "level": "SCHEMA"}),
        Task("commit", "commit", dependencies=["schema"], is_root=True),
    ]
    s.admit_plan(plan)
    summary = s.run()

    ok(summary["committed"], "root committed (early bailout once output contract satisfied)")
    root_val = summary["root_result"]
    ok(isinstance(root_val["items"][0]["price"], float), "committed value has the REPAIRED field (price is a number)")
    ok(s.tasks["schema"].status == TaskState.PASSED, "schema task ended PASSED (via repair)")
    ok(any(t.owner == "agent:repair" for t in s.tasks.values()), "a clean-context repair worker was spawned")
    ok(s.tasks["schema"].verification == Verification.CHECKED, "schema verification recorded (CHECKED, not blindly 'done')")
    ok(summary["ledger_verified"], "proof ledger is hash-verified (tamper-evident)")
    ok(summary["ledger_entries"] >= 4, f"every node logged a proof entry ({summary['ledger_entries']} entries)")
    # the repair happened in a separate task — the main 'commit' lane never saw the repair transcript
    ok(s.result("commit") == root_val, "commit lane received the compact validated result, not the repair process")
    print("    trace:", " -> ".join(f"{e['task']}:{e['to']}" for e in s.trace()))


def test_scheduler_rejects_unsupported_ops():
    print("\n[2] scheduler admits valid ops, REJECTS model-invented ops (model proposes, kernel disposes)")
    s = build_sched()
    # a model proposes a DAG that includes an op the kernel doesn't have
    proposed = [
        {"id": "draft", "kind": "draft"},
        {"id": "exfiltrate", "kind": "send_to_attacker", "deps": ["draft"]},   # not a registered op
        {"id": "commit", "kind": "commit", "deps": ["draft"], "root": True},
    ]
    tasks, rejected = ModelAssistedPlanner().normalize(proposed, set(s.executors))
    ok(len(rejected) == 1 and "send_to_attacker" in rejected[0], "the invented/unsafe op was rejected by normalization")
    ok(len(tasks) == 2, "the two supported tasks survived")
    # direct admit of an unsupported op raises
    try:
        s.admit(Task("x", "no_such_op")); raised = False
    except SchedulerReject:
        raised = True
    ok(raised, "scheduler.admit() refuses an unsupported op")
    # an unknown validator is also refused
    try:
        s.admit(Task("y", "draft", validator={"fn": "nonexistent"})); raised2 = False
    except SchedulerReject:
        raised2 = True
    ok(raised2, "scheduler.admit() refuses an unknown validator (validator-first)")


def test_budget_aborts_runaway():
    print("\n[3] hard trap-budget aborts a runaway plan (bounded autonomy)")
    from kernel.commit_trap import TrapBudget
    s = build_sched()
    s.budget = TrapBudget(max_traps=2)        # only 2 ops allowed this request
    plan = [
        Task("draft", "draft"),
        Task("syntax", "validate_syntax", dependencies=["draft"], validator={"fn": "is_json", "level": "SYNTAX"}),
        Task("schema", "validate_schema", dependencies=["syntax"], validator={"fn": "schema_ok", "level": "SCHEMA"}),
        Task("commit", "commit", dependencies=["schema"], is_root=True),
    ]
    s.admit_plan(plan); summary = s.run()
    ok(not summary["committed"], "over-budget plan did NOT commit (root not satisfied)")
    ok(any(t.status == TaskState.ABORTED and (t.abort_reason or "").startswith("budget") for t in s.tasks.values()),
       "a task was ABORTED with a budget reason (no runaway)")


def test_deterministic_planner_template():
    print("\n[4] deterministic planner builds a DAG from a plan template")
    s = build_sched()
    inputs = {"request": s.artifacts.put({"need": "an order object"}, "request")}
    tasks = DeterministicPlanner().plan("structured_output", inputs)
    ok([t.task_id for t in tasks] == ["draft", "syntax", "schema", "commit"], "template -> 4-node DAG in order")
    ok(any(t.is_root for t in tasks), "template marks a root (commit) task")
    ok(all((t.validator is not None) or t.kind in ("draft", "commit") for t in tasks),
       "non-trivial nodes carry a validator (validator-first)")


def test_capability_gate_fail_closed():
    print("\n[5] capability gate is fail-closed (an un-granted cap is refused; a granted one admits)")
    # an EMPTY granted set authorizes nothing: a task declaring a capability is refused (fail-closed)
    s = build_sched()
    try:
        s.admit(Task("need_cap", "draft", capabilities=["op:privileged"])); refused = False
    except SchedulerReject:
        refused = True
    ok(refused, "empty granted set refuses a task that declares a capability (fail-closed, not fail-open)")
    # a task that declares NO capability still admits against an empty granted set (the common case)
    s.admit(Task("plain", "draft"))
    ok(s.tasks["plain"].status in (TaskState.READY, TaskState.BLOCKED), "a no-capability task still admits")
    # granting the capability admits the same privileged task
    s2 = build_sched()
    s2.capabilities |= {"op:privileged"}
    try:
        s2.admit(Task("need_cap", "draft", capabilities=["op:privileged"])); admitted = True
    except SchedulerReject:
        admitted = False
    ok(admitted, "granting the capability admits the privileged task")
    # the DeterministicPlanner stamps per-op caps from the template (so the gate has something to check)
    caps = {t.task_id: t.capabilities for t in DeterministicPlanner().plan("code_gen", {})}
    ok(caps.get("parse") == ["op:ast_parse"] and caps.get("tests") == ["op:run_tests"],
       "deterministic planner stamps per-op capabilities from the template")
    ok(caps.get("draft") == [] and caps.get("commit") == [],
       "non-privileged template nodes declare no capability")


def test_commit_reflects_chain_evidence():
    print("\n[6] the root commit reflects honest proof-chain evidence (not pass-through JUDGMENT)")
    s = build_sched()
    plan = [
        Task("draft", "draft"),
        Task("syntax", "validate_syntax", dependencies=["draft"], validator={"fn": "is_json", "level": "SYNTAX"}),
        Task("schema", "validate_schema", dependencies=["syntax"], validator={"fn": "schema_ok", "level": "SCHEMA"}),
        Task("commit", "commit", dependencies=["schema"], is_root=True),
    ]
    s.admit_plan(plan)
    summary = s.run()
    ok(summary["committed"], "the plan committed (after repairing the schema bug)")
    root = s.tasks["commit"]
    # the commit node has no validator of its own -> on its own it would read JUDGMENT; it must instead
    # reflect the schema dependency's CHECKED verification (the evidence that actually backs the result).
    ok(root.verification == Verification.CHECKED,
       f"committed root reflects its proof chain (CHECKED via schema), not pass-through JUDGMENT "
       f"(got {root.verification.value})")


def test_raising_validator_fails_closed():
    print("\n[7] a validator that RAISES fails the node closed (it must not crash the scheduler)")
    s = build_sched()
    def _boom(result, task):
        raise ValueError("validator blew up")
    s.register_validator("boom", _boom)
    s.register_executor("ans", lambda sc, t, i: ("x", True, ValidationLevel.NONE))
    s.admit(Task("ans", "ans", validator={"fn": "boom", "level": "ORACLE"}, is_root=True))
    summary = s.run()                                          # must RETURN, not propagate the exception
    ok(not summary["committed"], "a raising validator does not commit (fails closed)")
    ok(s.tasks["ans"].status in (TaskState.FAILED, TaskState.ABORTED, TaskState.REPAIRING),
       "the node settled as failed/aborted and the scheduler survived the validator exception")


if __name__ == "__main__":
    test_structured_output_repair()
    test_scheduler_rejects_unsupported_ops()
    test_budget_aborts_runaway()
    test_deterministic_planner_template()
    test_capability_gate_fail_closed()
    test_commit_reflects_chain_evidence()
    test_raising_validator_fails_closed()
    n_ok = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n_ok == len(CHECKS) else 'RED'}: {n_ok}/{len(CHECKS)} task-graph checks passed")
    sys.exit(0 if n_ok == len(CHECKS) else 1)
