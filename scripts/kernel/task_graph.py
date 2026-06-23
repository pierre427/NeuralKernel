#!/usr/bin/env python3
"""Proof-Carrying Task Graph — the kernel-owned process table (the microkernel's source of truth).

Above hold/switch/resume, clean-context agents, the deterministic primitive ABI, and the M1 commit trap
sits one thing: a kernel-owned task graph that tracks intent, subgoals, continuations, evidence,
validators, budgets, and completion state. The model PROPOSES a plan; the SCHEDULER admits (type-checks),
executes the DAG, validates each node, records every transition in the proof ledger, spawns clean-context
repair workers on failure, and bails the moment the root's output contract is satisfied.

Design rule: the model is NOT the task tracker — the scheduler owns task state, dependencies, budgets,
capabilities, artifacts, validation, commit, and history. The model receives only the clean slice it needs.

OS correspondence: task graph = process table; task = continuation; state = thread state; primitive ABI =
syscall table; artifact refs = file descriptors; capabilities = permissions; validation = exit code;
event log = kernel log. Pure-Python, no LLM, no network — executors/validators are injected.
"""
from __future__ import annotations
import itertools, time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Optional

from .commit_trap import ProofLedger, TrapBudget, ValidationLevel, Contract, _sha


# ---------------------------------------------------------------------------
# Task state machine (explicit states, not "think/act/observe")
# ---------------------------------------------------------------------------
class TaskState(str, Enum):
    NEW = "NEW"; READY = "READY"; RUNNING = "RUNNING"; BLOCKED = "BLOCKED"
    WAITING_PRIMITIVE = "WAITING_FOR_PRIMITIVE"; WAITING_AGENT = "WAITING_FOR_AGENT"
    WAITING_TOOL = "WAITING_FOR_TOOL"; VALIDATING = "VALIDATING"
    PASSED = "PASSED"; FAILED = "FAILED"; REPAIRING = "REPAIRING"
    ABORTED = "ABORTED"; COMMITTED = "COMMITTED"


_SETTLED = {TaskState.PASSED, TaskState.COMMITTED, TaskState.ABORTED}


# Verification taxonomy — do NOT collapse into "done" (design §"distinguish verified/validated/...").
class Verification(str, Enum):
    UNVERIFIED = "unverified"   # no validator cleared
    CHECKED = "checked"         # a cheap form/syntax check passed
    VALIDATED = "validated"     # schema/property validator passed
    VERIFIED = "verified"       # oracle/proven validator passed
    JUDGMENT = "judgment"       # no deterministic validator exists; needs human/judgment


# Calibrated, VALIDATION-GROUNDED confidence: the system's honest confidence is what the VERDICT supports,
# NOT the model's self-reported number. A failed/unverified node is ~0.1 regardless of what the model claimed;
# a judgment-only node is capped at 0.5 (genuinely unverifiable); only an oracle pass earns high confidence.
_CALIBRATED_CONF = {
    Verification.VERIFIED: 0.95, Verification.VALIDATED: 0.8, Verification.CHECKED: 0.6,
    Verification.JUDGMENT: 0.5, Verification.UNVERIFIED: 0.1,
}


# Repair-efficiency band (the explicitly-open gate, see project_gptoss_kernel_first_run): a bare same-kind
# {kind}_repair is a T=0 near-deterministic RE-RUN — in the cyber runner it fired on ~40 fails to save ~2.
# Spend a retry only when a fresh attempt can plausibly flip the verdict: the BORDERLINE band, calibrated
# conf in [0.5, 0.8] — i.e. JUDGMENT(0.5) / CHECKED(0.6) / VALIDATED(0.8). Excluded are the two extremes the
# memo calls out: VERIFIED(0.95) = an ORACLE gave a definitive verdict (confidently-failed; a deterministic
# re-run won't flip it) and UNVERIFIED(0.1) = no real validation path (hopeless).
_REPAIR_BAND = (0.5, 0.8)   # inclusive [lo, hi] over _CALIBRATED_CONF


def _intended_verification(task: "Task") -> Verification:
    """The verification level this node is TRYING to clear — its DECLARED validator strength.

    A node that just failed always carries verification=UNVERIFIED (conf 0.1), so the post-hoc value can't
    tell a borderline miss from a hopeless one — both read 0.1. The repair-worthiness signal is therefore the
    node's declared check strength, mapped through the SAME ladder _validate() uses on a pass (so the gate and
    the calibration table stay in lock-step)."""
    if task.validator is None:
        return Verification.JUDGMENT              # no deterministic validator — genuinely borderline
    try:
        lvl = ValidationLevel[task.validator.get("level", "SCHEMA").upper()]
    except KeyError:
        return Verification.UNVERIFIED            # malformed declared level — treat as hopeless
    if lvl >= ValidationLevel.ORACLE:
        return Verification.VERIFIED
    if lvl >= ValidationLevel.PROPERTY:
        return Verification.VALIDATED
    return Verification.CHECKED


# ---------------------------------------------------------------------------
# Artifacts live OUTSIDE context; tasks pass references, never blobs.
# ---------------------------------------------------------------------------
@dataclass
class Artifact:
    ref: str
    kind: str
    value: Any
    content_hash: str


class ArtifactStore:
    """Content-addressed store so the planner passes refs, not blobs (no context pollution)."""
    def __init__(self):
        self._a: dict[str, Artifact] = {}
        self._c = itertools.count(1)

    def put(self, value: Any, kind: str = "blob") -> str:
        ref = f"artifact:a-{next(self._c)}"
        self._a[ref] = Artifact(ref, kind, value, _sha(value))
        return ref

    def has(self, ref: str) -> bool: return ref in self._a
    def get(self, ref: str) -> Any: return self._a[ref].value
    def meta(self, ref: str) -> dict:
        a = self._a[ref]; return {"ref": a.ref, "kind": a.kind, "hash": a.content_hash}


@dataclass
class Budget:
    """Per-task bounded autonomy: budget + validator + max_retries make runaway impossible."""
    tokens: int = 0
    wall_ms: int = 0
    traps: int = 1
    max_retries: int = 1


# ---------------------------------------------------------------------------
# A typed task node (a unit of work with a contract)
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: str
    kind: str                                    # executor key (the op)
    owner: str = "scheduler"                     # scheduler | model_lane | agent:* | primitive:* | tool:*
    parent_id: Optional[str] = None
    status: TaskState = TaskState.NEW
    input_refs: dict = field(default_factory=dict)        # name -> artifact ref
    output_contract: dict = field(default_factory=dict)   # {type, required_fields}
    validator: Optional[dict] = None             # {fn, level} or None => judgment-based
    capabilities: list = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    dependencies: list = field(default_factory=list)      # task_ids that must PASS first
    result_ref: Optional[str] = None
    proof_ref: Optional[str] = None
    verification: Verification = Verification.UNVERIFIED
    retries: int = 0
    abort_reason: Optional[str] = None
    is_root: bool = False
    reopen: Optional[str] = None                 # repair tasks point back at the task they fix
    meta: dict = field(default_factory=dict)     # arbitrary details (notes/evidence/memory hooks)


@dataclass
class TaskEvent:
    """A task state transition, linked to a proof-ledger entry (replay/audit/debug)."""
    event_id: str
    task_id: str
    ts: Optional[float]
    frm: str
    to: str
    executor: str
    artifact_in: list
    artifact_out: list
    proof: Optional[dict]


class SchedulerReject(Exception):
    """Raised when the scheduler refuses to admit a model-proposed task (unsupported op / missing cap)."""


# ---------------------------------------------------------------------------
# The scheduler: owns the graph, artifacts, ledger, budget, capabilities, history.
# ---------------------------------------------------------------------------
class Scheduler:
    def __init__(self, budget: Optional[TrapBudget] = None, capabilities: Optional[set] = None):
        self.tasks: dict[str, Task] = {}
        self.artifacts = ArtifactStore()
        self.ledger = ProofLedger()
        self.budget = budget or TrapBudget()
        self.capabilities: set = capabilities if capabilities is not None else set()
        self.events: list[TaskEvent] = []
        self.executors: dict[str, Callable] = {}   # kind -> fn(sched, task, inputs) -> (result, ok, ValidationLevel)
        self.validators: dict[str, Callable] = {}   # key -> fn(result, task) -> bool
        self._syscall_ops: dict = {}   # name -> {cap, max_level}: syscall-tier tools whose cap+ceiling admit() ENFORCES
        self._ec = itertools.count(1)
        self._root: Optional[str] = None
        self.telemetry = None                            # optional RunTrace (supervisor telemetry); set by the orchestrator

    # --- registration ---
    def register_executor(self, kind: str, fn: Callable): self.executors[kind] = fn
    def register_validator(self, key: str, fn: Callable): self.validators[key] = fn

    # --- the scheduler ADMITS tasks (model proposes, scheduler disposes) ---
    def admit(self, task: Task) -> Task:
        if task.kind not in self.executors:
            raise SchedulerReject(f"unsupported op: {task.kind!r}")
        # Capability gate — FAIL-CLOSED (design §5.3): a task may only invoke an op whose required
        # capability is in the kernel-granted set. An EMPTY granted set authorizes nothing, so a task
        # that declares a capability is rejected unless it was explicitly granted. (Tasks that declare
        # no capabilities are unaffected — the common case.) The trusted DeterministicPlanner's caps are
        # granted by the orchestrator; an injected/model-proposed op requesting an un-granted cap is refused.
        for cap in task.capabilities:
            if cap not in self.capabilities:
                raise SchedulerReject(f"missing capability: {cap!r}")
        # validator-first: a non-judgment contract with no validator is a planning error -> mark judgment
        if task.validator is not None and task.validator.get("fn") not in self.validators:
            raise SchedulerReject(f"unknown validator: {task.validator.get('fn')!r}")
        # Syscall-tier tools (registered via kernel/tools.register_tool): the capability is MANDATORY — not the
        # opt-in "common case" — and the verification ceiling is enforced HERE, kernel-side, so a plan cannot call
        # a syscall op ungated (capabilities=[]) nor over-claim its trust level. Non-syscall ops are unaffected.
        sop = self._syscall_ops.get(task.kind)
        if sop is not None:
            if sop["cap"] not in task.capabilities:
                raise SchedulerReject(f"syscall op {task.kind!r} requires capability {sop['cap']!r}")
            # A syscall op MUST carry a validator: otherwise the ceiling check below could be skipped by simply
            # omitting the validator (validator=None -> JUDGMENT pass-through), evading the CHECKED ceiling.
            if task.validator is None:
                raise SchedulerReject(f"syscall op {task.kind!r} requires a result validator (ceiling enforced)")
            try:
                declared = ValidationLevel[task.validator.get("level", "SCHEMA").upper()]
                ceiling = ValidationLevel[str(sop["max_level"]).upper()]
            except KeyError:
                raise SchedulerReject(f"syscall op {task.kind!r}: malformed validator level")
            if declared > ceiling:
                raise SchedulerReject(
                    f"syscall op {task.kind!r} cannot exceed {sop['max_level']} (declared {task.validator.get('level')})")
        self.tasks[task.task_id] = task
        if task.is_root:
            self._root = task.task_id
        self._recompute_state(task)
        return task

    def admit_plan(self, tasks: list[Task]) -> list[Task]:
        return [self.admit(t) for t in tasks]

    # --- state helpers ---
    def _deps_passed(self, task: Task) -> bool:
        return all(self.tasks[d].status in (TaskState.PASSED, TaskState.COMMITTED)
                   for d in task.dependencies if d in self.tasks)

    def _recompute_state(self, task: Task):
        if task.status in _SETTLED or task.status in (TaskState.FAILED, TaskState.REPAIRING):
            return
        task.status = TaskState.READY if self._deps_passed(task) else TaskState.BLOCKED

    def _ready(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskState.READY]

    def _unblock_dependents(self, task: Task):
        for t in self.tasks.values():
            if task.task_id in t.dependencies and t.status == TaskState.BLOCKED:
                self._recompute_state(t)

    def _emit(self, task, frm, to, executor, ain, aout, proof) -> TaskEvent:
        ev = TaskEvent(f"e-{next(self._ec)}", task.task_id, None, frm, to, executor, ain, aout, proof)
        self.events.append(ev)
        if self.telemetry is not None:                   # forward the supervisor-level transition to telemetry
            t0 = task.meta.get("_t0")
            cm = (task.meta.get("telemetry") or {}).get("confidence")   # what the MODEL claimed
            self.telemetry.emit("task", task=task.task_id, op=task.kind, owner=task.owner,
                            from_state=frm, to_state=to, executor=executor,
                            verification=task.verification.value,
                            level=(proof or {}).get("level"), proof=(proof or {}).get("hash"),
                            retries=task.retries,
                            dur_ms=(round((time.perf_counter() - t0) * 1000, 1) if t0 else None),
                            conf_model=cm,                                # model's self-reported (the vibe)
                            conf_cal=_CALIBRATED_CONF.get(task.verification),  # validation-grounded (honest)
                            decision=(task.meta.get("telemetry") or None))
        return ev

    # --- run one task through its allowed executor ---
    def _execute(self, task: Task, ts):
        frm = task.status.value
        ok_b, why = self.budget.can_charge(task.kind)        # budget gate FIRST (anti-runaway)
        if not ok_b:
            task.status = TaskState.ABORTED; task.abort_reason = f"budget:{why}"
            self._emit(task, frm, task.status.value, "budget", [], [], None); return
        self.budget.charge(task.kind)
        task.status = TaskState.RUNNING
        task.meta["_t0"] = time.perf_counter()               # telemetry: per-node wall-clock
        inputs = {k: self.artifacts.get(v) for k, v in task.input_refs.items() if self.artifacts.has(v)}
        for d in task.dependencies:                           # wire each dependency's result in by id
            dep = self.tasks.get(d)
            if dep and dep.result_ref and self.artifacts.has(dep.result_ref):
                inputs.setdefault(d, self.artifacts.get(dep.result_ref))
        try:
            result, ex_ok, _ = self.executors[task.kind](self, task, inputs)
        except Exception as e:                                # executor crash -> fail -> repair/abort
            task.status = TaskState.FAILED
            self._emit(task, frm, "FAILED", task.kind, list(task.input_refs.values()), [], None)
            self._maybe_repair(task, f"executor_error:{e}"); return

        out_ref = self.artifacts.put(result, kind=task.output_contract.get("type", "blob"))
        task.result_ref = out_ref
        if isinstance(result, str) and '"confidence"' in result:   # telemetry: surface a carried confidence
            import re as _re
            m = _re.search(r'"confidence"\s*:\s*("?[\w.\-]+"?)', result)
            if m:
                task.meta.setdefault("telemetry", {}).setdefault("confidence", m.group(1).strip('"'))
        task.status = TaskState.VALIDATING
        v_ok, verification, vlevel = self._validate(task, result)
        # Bind the node's CHECK identity (its validator fn) into the proof contract, so the ledger entry
        # records WHAT verified this result — not just the opcode (reclaim missed-opp #1: a real contract,
        # not a bare stub). The achieved level is already recorded; this binds the check that produced it.
        entry = self.ledger.append(                            # proof for THIS transition
            Contract(opcode=task.kind, op_version="tg1", entry=(task.validator or {}).get("fn") or task.kind),
            [self.artifacts.meta(r) for r in task.input_refs.values() if self.artifacts.has(r)],
            result, vlevel, "pass" if v_ok else "fail", ts=ts)
        task.proof_ref = entry.entry_hash
        task.verification = verification
        proof = {"hash": entry.entry_hash, "level": vlevel.name, "verification": verification.value}
        if v_ok:
            task.status = TaskState.PASSED
            self._emit(task, frm, "PASSED", task.owner, list(task.input_refs.values()), [out_ref], proof)
            if task.reopen and task.reopen in self.tasks:      # a repair worker fixed its parent
                parent = self.tasks[task.reopen]
                parent.result_ref = out_ref; parent.status = TaskState.PASSED
                parent.verification = verification; parent.proof_ref = entry.entry_hash
                self._emit(parent, "REPAIRING", "PASSED", "agent:repair", [out_ref], [out_ref], proof)
                self._unblock_dependents(parent)
            self._unblock_dependents(task)
        else:
            task.status = TaskState.FAILED
            self._emit(task, frm, "FAILED", task.kind, list(task.input_refs.values()), [out_ref], proof)
            self._maybe_repair(task, "validation_failed")

    def _validate(self, task: Task, result: Any):
        if task.validator is None:                             # judgment-based: uncertainty made explicit
            return True, Verification.JUDGMENT, ValidationLevel.NONE
        vfn = self.validators.get(task.validator["fn"])
        if vfn is None:
            return False, Verification.UNVERIFIED, ValidationLevel.NONE
        try:
            ok = bool(vfn(result, task))
        except Exception:
            # a validator that raises (e.g. a malformed contract) must FAIL THE NODE CLOSED, never crash
            # the scheduler — _validate runs OUTSIDE the executor try/except, so an unguarded raise here
            # would abort the whole run() (a model-proposed plan could weaponize that into a DoS).
            return False, Verification.UNVERIFIED, ValidationLevel.NONE
        lvl = ValidationLevel[task.validator.get("level", "SCHEMA").upper()] if ok else ValidationLevel.NONE
        if not ok:
            ver = Verification.UNVERIFIED
        elif lvl >= ValidationLevel.ORACLE:
            ver = Verification.VERIFIED
        elif lvl >= ValidationLevel.PROPERTY:
            ver = Verification.VALIDATED
        else:
            ver = Verification.CHECKED
        return ok, ver, lvl

    def _maybe_repair(self, task: Task, reason: str):
        """Spawn a CLEAN-CONTEXT repair worker that gets only the failing slice; bounded by max_retries."""
        # The ROOT COMMIT GATE, when it's a pure pass-through (its result IS an upstream result) AND there is
        # no dedicated {kind}_repair executor, cannot be fixed by re-running itself — the content lives
        # upstream, so re-committing the same artifact just fails the grade again and aborts (telemetry-
        # surfaced bug). Fail it cleanly. If a real {kind}_repair IS registered (e.g. a re-synthesis repair),
        # it does genuine work, so DON'T short-circuit — let the normal repair path run it. Mid-graph validate
        # nodes are never gated here. Compare CONTENT HASHES (the store assigns a fresh ref per put).
        if (task.is_root and (task.kind + "_repair") not in self.executors
                and task.result_ref and self.artifacts.has(task.result_ref)):
            rh = self.artifacts.meta(task.result_ref)["hash"]
            passthrough = any(self.artifacts.has(self.tasks[d].result_ref or "")
                              and self.artifacts.meta(self.tasks[d].result_ref)["hash"] == rh
                              for d in task.dependencies if d in self.tasks and self.tasks[d].result_ref)
            if passthrough:
                task.status = TaskState.FAILED; task.abort_reason = f"gate_failed:{reason}"
                if self.telemetry is not None:
                    self.telemetry.emit("gate_fail", task=task.task_id, op=task.kind, reason=reason,
                                        note="pass-through gate — content is upstream; no self-repair")
                return
        # EFFICIENCY GATE (explicitly-open, project_gptoss_kernel_first_run): a bare same-kind retry (no
        # dedicated {kind}_repair executor) is a T=0 deterministic re-run that just reproduces the same failed
        # proof. Spend the trap only on the BORDERLINE band where a fresh attempt can realistically pass; let
        # confidently-failed (oracle) and hopeless nodes fail cleanly instead of burning a retry to save ~1-in-20.
        # A DEDICATED {kind}_repair does genuine, non-deterministic work (re-synthesis, not a re-run), so it is
        # EXEMPT and always allowed — mirrors the gate_fail "defer to a real {kind}_repair" rule above.
        if (task.kind + "_repair") not in self.executors:
            conf = _CALIBRATED_CONF.get(_intended_verification(task), 0.0)
            if not (_REPAIR_BAND[0] <= conf <= _REPAIR_BAND[1]):
                task.status = TaskState.ABORTED; task.abort_reason = f"repair_skipped:{reason}"
                if self.telemetry is not None:
                    self.telemetry.emit("repair_skipped", task=task.task_id, op=task.kind, reason=reason,
                                        conf_cal=conf, band=list(_REPAIR_BAND),
                                        note="outside borderline band (confidently-failed/hopeless); "
                                             "T=0 same-kind re-run is low-yield")
                return
        if task.retries >= task.budget.max_retries:
            task.status = TaskState.ABORTED; task.abort_reason = reason
            if self.telemetry is not None:
                self.telemetry.emit("abort", task=task.task_id, op=task.kind, reason=reason, retries=task.retries)
            return
        task.retries += 1; task.status = TaskState.REPAIRING
        if self.telemetry is not None:
            self.telemetry.emit("repair", task=task.task_id, op=task.kind, retry=task.retries, reason=reason)
        rkind = task.kind + "_repair" if (task.kind + "_repair") in self.executors else task.kind
        rid = f"{task.task_id}.repair{task.retries}"
        rep = Task(task_id=rid, kind=rkind, owner="agent:repair", parent_id=task.task_id,
                   input_refs=dict(task.input_refs, _failed=task.result_ref) if task.result_ref else dict(task.input_refs),
                   output_contract=task.output_contract, validator=task.validator,
                   budget=Budget(max_retries=0), reopen=task.task_id, status=TaskState.READY,
                   meta=dict(task.meta))      # inherit job_inputs/recalled details so validators/executors work
        self.tasks[rid] = rep

    # --- the scheduler loop: pick READY -> execute -> validate -> unblock -> bail when root satisfied ---
    def run(self, ts: Optional[float] = None, max_steps: int = 1000):
        steps = 0
        while steps < max_steps:
            steps += 1
            if self._root and self.tasks[self._root].status == TaskState.COMMITTED:
                break
            ready = self._ready()
            if not ready:
                # root passed but not yet committed -> commit it (the only output path)
                if self._root and self.tasks[self._root].status == TaskState.PASSED:
                    self._commit_root(ts); continue
                break                                          # done or deadlocked (no runnable task)
            self._execute(ready[0], ts)
            if self._root and self.tasks[self._root].status == TaskState.PASSED:
                self._commit_root(ts)                          # EARLY BAILOUT: root contract satisfied
        return self.summary()

    def _commit_evidence(self, root: Task) -> tuple[Verification, Optional[str]]:
        """The verification (+ proof_ref) that actually backs the committed artifact: the stronger of the
        commit node's OWN verification (when it ran a validator) and the verification of the dependency that
        PRODUCED this result. The commit executor re-emits a dependency's value as a fresh artifact, so the
        producer is the dep whose result content-hash equals the committed result's — binding to that producer
        (rather than to the strongest sibling) avoids over-attributing a sibling's proof to a result it never
        made. A commit that carries its own ORACLE validator keeps that level (no dep is stronger)."""
        best, best_ref = root.verification, root.proof_ref
        rh = (self.artifacts.meta(root.result_ref)["hash"]
              if root.result_ref and self.artifacts.has(root.result_ref) else None)
        for d in root.dependencies:
            dep = self.tasks.get(d)
            if not (dep and dep.status in (TaskState.PASSED, TaskState.COMMITTED)):
                continue
            is_producer = bool(rh and dep.result_ref and self.artifacts.has(dep.result_ref)
                               and self.artifacts.meta(dep.result_ref)["hash"] == rh)
            if is_producer and _CALIBRATED_CONF.get(dep.verification, 0.0) > _CALIBRATED_CONF.get(best, 0.0):
                best, best_ref = dep.verification, dep.proof_ref
        return best, best_ref

    def _commit_root(self, ts):
        root = self.tasks[self._root]
        # HONEST COMMIT VERIFICATION (design §5.4; reclaim missed-opp #1/#6): when the commit node has no
        # validator of its own (structured_output/code_gen/cyber_alert templates) it reads JUDGMENT (conf 0.5)
        # and would commit even an ORACLE-proven upstream result at the wrong confidence. Reflect the evidence
        # that actually backs the committed artifact so its recorded verification + calibrated confidence are
        # honest. (Templates whose commit carries its own ORACLE validator already read VERIFIED — no dep is
        # stronger, so this is a no-op there.) A commit with no real validator anywhere stays JUDGMENT and is
        # surfaced as evidence_backed=False — an honest low-confidence boundary, not silently treated as proven.
        backing, backing_ref = self._commit_evidence(root)
        if _CALIBRATED_CONF.get(backing, 0.0) > _CALIBRATED_CONF.get(root.verification, 0.0):
            root.verification = backing
        evidence_backed = root.verification in (Verification.CHECKED, Verification.VALIDATED, Verification.VERIFIED)
        root.status = TaskState.COMMITTED
        self._emit(root, "PASSED", "COMMITTED", "commit_trap", [root.result_ref] if root.result_ref else [], [], None)
        if self.telemetry is not None:
            self.telemetry.emit("commit", task=root.task_id, ledger_entries=len(self.ledger),
                            ledger_verified=self.ledger.verify_all(), result_sha=(root.result_ref or None),
                            verification=root.verification.value, conf_cal=_CALIBRATED_CONF.get(root.verification),
                            evidence_backed=evidence_backed, backing_proof=backing_ref)

    # --- introspection ---
    def result(self, task_id: str) -> Any:
        t = self.tasks[task_id]
        return self.artifacts.get(t.result_ref) if t.result_ref else None

    def summary(self) -> dict:
        states: dict[str, int] = {}
        for t in self.tasks.values():
            states[t.status.value] = states.get(t.status.value, 0) + 1
        root = self.tasks.get(self._root) if self._root else None
        return {
            "committed": bool(root and root.status == TaskState.COMMITTED),
            "root_result": self.result(self._root) if root and root.result_ref else None,
            "states": states,
            "ledger_entries": len(self.ledger),
            "ledger_verified": self.ledger.verify_all(),
            "events": len(self.events),
            "verification": {t.task_id: t.verification.value for t in self.tasks.values()},
        }

    def trace(self) -> list[dict]:
        return [{"event": e.event_id, "task": e.task_id, "from": e.frm, "to": e.to,
                 "executor": e.executor, "proof": e.proof} for e in self.events]


# ---------------------------------------------------------------------------
# Planners — deterministic (template-driven) + model-assisted (proposes, scheduler normalizes)
# ---------------------------------------------------------------------------
# A plan template is a deterministic scaffold; model-filled steps are the ambiguous parts.
PLAN_TEMPLATES: dict[str, list[dict]] = {
    "structured_output": [
        {"id": "draft",    "kind": "draft",          "deps": []},
        {"id": "syntax",   "kind": "validate_syntax", "deps": ["draft"], "validator": {"fn": "is_json", "level": "SYNTAX"}},
        {"id": "schema",   "kind": "validate_schema", "deps": ["syntax"], "validator": {"fn": "schema_ok", "level": "SCHEMA"}, "cap": "op:validate_schema"},
        {"id": "commit",   "kind": "commit",          "deps": ["schema"], "root": True},
    ],
    "code_gen": [
        {"id": "draft",  "kind": "draft",        "deps": []},
        {"id": "parse",  "kind": "ast_parse",    "deps": ["draft"], "validator": {"fn": "compiles", "level": "SYNTAX"}, "cap": "op:ast_parse"},
        {"id": "tests",  "kind": "run_tests",    "deps": ["parse"], "validator": {"fn": "tests_pass", "level": "ORACLE"}, "cap": "op:run_tests"},
        {"id": "commit", "kind": "commit",       "deps": ["tests"], "root": True},
    ],
    "cyber_alert": [
        {"id": "observ",  "kind": "extract_observables", "deps": []},
        {"id": "norm",    "kind": "normalize_iocs",      "deps": ["observ"], "validator": {"fn": "iocs_ok", "level": "PROPERTY"}},
        {"id": "checks",  "kind": "det_checks",          "deps": ["norm"],   "validator": {"fn": "checks_ran", "level": "ORACLE"}, "cap": "op:det_checks"},
        {"id": "note",    "kind": "draft_note",          "deps": ["checks"]},
        {"id": "verify",  "kind": "verify_claims",       "deps": ["note"],   "validator": {"fn": "claims_grounded", "level": "PROPERTY"}},
        {"id": "commit",  "kind": "commit",              "deps": ["verify"], "root": True},
    ],
}


class DeterministicPlanner:
    """Rule-driven: a known shape -> a fixed DAG. Fast, reliable, auditable."""
    def plan(self, template: str, inputs: dict) -> list[Task]:
        spec = PLAN_TEMPLATES[template]
        tasks = []
        for s in spec:
            tasks.append(Task(
                task_id=s["id"], kind=s["kind"], owner=s.get("owner", "scheduler"),
                dependencies=s.get("deps", []), validator=s.get("validator"),
                output_contract=s.get("contract", {}), is_root=s.get("root", False),
                capabilities=[s["cap"]] if s.get("cap") else [],
                input_refs=inputs if not s.get("deps") else {},
            ))
        return tasks


class ModelAssistedPlanner:
    """The model proposes a DAG (list of dicts); the scheduler NORMALIZES + validates it: unknown ops are
    rejected, missing validators are flagged judgment, budgets/ids are filled. The model can't invent ops."""
    def normalize(self, proposed: list[dict], known_ops: set) -> tuple[list[Task], list[str]]:
        tasks, rejected = [], []
        for i, p in enumerate(proposed):
            kind = p.get("kind")
            if kind not in known_ops:
                rejected.append(f"{p.get('id', i)}:{kind} (unsupported op)"); continue
            tasks.append(Task(
                task_id=p.get("id", f"t{i}"), kind=kind, owner=p.get("owner", "model_lane"),
                dependencies=p.get("deps", []), validator=p.get("validator"),
                output_contract=p.get("contract", {}), is_root=p.get("root", False),
                capabilities=p.get("capabilities", []),
                input_refs=p.get("input_refs", {})))
        return tasks, rejected
