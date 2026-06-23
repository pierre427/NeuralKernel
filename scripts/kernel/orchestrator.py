#!/usr/bin/env python3
"""Job Kernel — assess incoming jobs, build agent-assigned work plans, run them on the Proof-Carrying
Task Graph, and roll up a master view. Generalized across job TYPES (coding, cyber, structured-output, …)
with a persistent PLAN MEMORY so the planner accumulates operational knowledge and reuses what worked.

Pipeline:  Job ─▶ JobClassifier (assess type)
                 ─▶ PlanMemory.recall (template + prior learnings + stored details)
                 ─▶ Planner builds a DAG, AGENTS ASSIGNED per node (model_lane / primitive:* / agent:* / tool:*)
                 ─▶ Scheduler runs the task graph (validate ▸ proof-ledger ▸ repair ▸ early-bail)
                 ─▶ master Rollup (the user-facing tracker view)
                 ─▶ PlanMemory.learn (record outcome + new details)

The model is NOT the orchestrator — the Job Kernel owns intake, classification, planning, assignment,
budgets, rollup, and memory. Pure-Python; executors/validators are injected per job type.
"""
from __future__ import annotations
import json, os, itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from .telemetry import RunTrace, TelemetrySink, sha12 as _sha12
from .task_graph import (
    Scheduler, Task, TaskState, Verification, Budget, DeterministicPlanner, PLAN_TEMPLATES,
)
from .commit_trap import TrapBudget


# ---------------------------------------------------------------------------
# Plan memory — the planner as a memory system (operational knowledge, persisted)
# ---------------------------------------------------------------------------
class PlanMemory:
    """Stores + recalls operational knowledge so planning improves over time:
      - details   : arbitrary stored facts, keyed by scope (job-type / job-id / "global")
      - templates : per-job-type template stats (runs, passes) -> a proven-template library
      - learnings : per-job-type lessons ("this validator catches this mistake")
    Persistable to JSON so it accumulates across runs (the moat: a library of proven plans)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.details: dict = {}      # scope -> {key: value}
        self.templates: dict = {}    # job_type -> {template: {"runs": n, "passes": m}}
        self.learnings: dict = {}    # job_type -> [lesson, ...]
        if path and os.path.exists(path):
            self.load()

    # --- arbitrary details (used like a memory) ---
    def remember(self, scope: str, key: str, value: Any):
        self.details.setdefault(scope, {})[key] = value

    def recall(self, scope: str, key: Optional[str] = None, default=None):
        d = self.details.get(scope, {})
        return d if key is None else d.get(key, default)

    # --- proven-template library ---
    def record_run(self, job_type: str, template: str, passed: bool):
        t = self.templates.setdefault(job_type, {}).setdefault(template, {"runs": 0, "passes": 0})
        t["runs"] += 1; t["passes"] += int(passed)

    def best_template(self, job_type: str, fallback: Optional[str] = None) -> Optional[str]:
        cands = self.templates.get(job_type, {})
        if not cands:
            return fallback
        # highest pass-rate, then most-run (proven)
        return max(cands.items(), key=lambda kv: (kv[1]["passes"] / max(1, kv[1]["runs"]), kv[1]["runs"]))[0]

    def note_learning(self, job_type: str, lesson: str):
        ls = self.learnings.setdefault(job_type, [])
        if lesson not in ls:
            ls.append(lesson)

    def learnings_for(self, job_type: str) -> list:
        return self.learnings.get(job_type, [])

    # --- persistence ---
    def save(self, path: Optional[str] = None):
        p = path or self.path
        if p:
            json.dump({"details": self.details, "templates": self.templates, "learnings": self.learnings},
                      open(p, "w"), indent=2, default=str)

    def load(self, path: Optional[str] = None):
        d = json.load(open(path or self.path))
        self.details = d.get("details", {}); self.templates = d.get("templates", {}); self.learnings = d.get("learnings", {})


# ---------------------------------------------------------------------------
# Job + classification
# ---------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    prompt: str = ""
    inputs: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)     # additional details stored + used (memory hook)
    job_type: Optional[str] = None               # set by the classifier (or pre-declared)


@dataclass
class JobClass:
    job_type: str
    template: str
    capabilities: list = field(default_factory=list)
    budget: int = 64                              # max_traps for the request
    reason: str = ""


class JobClassifier:
    """Assess an incoming job -> {job_type, template, capabilities, budget}. Rule-driven + extensible;
    a job may also pre-declare its type. New job types register a matcher + a default template."""

    def __init__(self):
        self.rules: list[tuple[str, Callable[[Job], bool], str, list]] = []

    def register(self, job_type: str, matcher: Callable[[Job], bool], template: str, capabilities=None):
        self.rules.append((job_type, matcher, template, capabilities or []))

    def classify(self, job: Job, memory: Optional[PlanMemory] = None) -> JobClass:
        # pre-declared type wins
        for jt, matcher, template, caps in self.rules:
            if job.job_type == jt or (job.job_type is None and matcher(job)):
                # memory may know a better-proven template for this type
                tmpl = (memory.best_template(jt, template) if memory else template)
                return JobClass(jt, tmpl, caps, job.meta.get("budget", 64), reason=f"matched {jt}")
        return JobClass("generic", "structured_output", [], 64, reason="fallthrough -> structured_output")


# ---------------------------------------------------------------------------
# The Job Kernel / Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """Owns intake, classification, planning, agent-assignment, scheduling, rollup, and memory.
    Executors/validators are registered per job type (the syscall surface); agent_policy maps a task
    kind -> an owner (which agent/primitive runs it)."""

    def __init__(self, memory: Optional[PlanMemory] = None, telemetry=None):
        self.memory = memory or PlanMemory()
        self.classifier = JobClassifier()
        self.executors: dict[str, Callable] = {}
        self.validators: dict[str, Callable] = {}
        self.agent_policy: dict[str, str] = {}    # task kind -> owner (agent assignment)
        self.history: list[dict] = []             # rollups of every submitted job (master rollup feed)
        # telemetry sink: pass a dir (persist per-run JSONL) or a TelemetrySink; None -> in-memory trace only
        self.sink = TelemetrySink(telemetry) if isinstance(telemetry, str) else telemetry
        self.traces: list[RunTrace] = []          # the last in-memory trace per run (inspect after the fact)
        self._rc = itertools.count(1)

    # --- registration ---
    def register_job_type(self, job_type, matcher, template, capabilities=None):
        self.classifier.register(job_type, matcher, template, capabilities)

    def register_executor(self, kind, fn): self.executors[kind] = fn
    def register_validator(self, key, fn): self.validators[key] = fn
    def assign_agent(self, kind, owner): self.agent_policy[kind] = owner

    # --- submit a job: classify -> recall -> plan (assign agents) -> run -> rollup -> learn ---
    def submit(self, job: Job) -> dict:
        trace = RunTrace(f"r{next(self._rc)}", job.job_id, sink=self.sink)   # supervisor telemetry for THIS run
        self.traces.append(trace)
        trace.emit("run.start", input_keys=sorted(job.inputs.keys()), prompt_sha=_sha12(job.prompt))

        jc = self.classifier.classify(job, self.memory)
        job.job_type = jc.job_type
        learnings = self.memory.learnings_for(jc.job_type)
        prior = self.memory.recall(jc.job_type)          # any stored details for this job type
        trace.emit("classify", job_type=jc.job_type, template=jc.template, budget=jc.budget,
                   capabilities=list(jc.capabilities or []))
        trace.emit("recall", learnings=len(learnings or []), has_prior=bool(prior))

        sched = Scheduler(budget=TrapBudget(max_traps=jc.budget), capabilities=set(jc.capabilities) or None)
        sched.telemetry = trace                              # attach: scheduler forwards node lifecycle to telemetry
        for k, fn in self.executors.items(): sched.register_executor(k, fn)
        for k, fn in self.validators.items(): sched.register_validator(k, fn)

        input_refs = {k: sched.artifacts.put(v, kind=k) for k, v in job.inputs.items()}   # inputs are artifacts (refs)
        tasks = DeterministicPlanner().plan(jc.template, input_refs)
        for t in tasks:                                  # AGENT ASSIGNMENT + thread the job's details onto nodes
            t.owner = self.agent_policy.get(t.kind, t.owner)
            t.meta["job_id"] = job.job_id
            t.meta["job_inputs"] = job.inputs            # raw inputs (entry/cases/evidence) reachable by executors
            t.meta["recalled"] = {"learnings": learnings, "prior_details": prior}
        # ARM the fail-closed capability gate: the DeterministicPlanner's plan is the kernel's OWN trusted
        # scaffold, so its required caps are authorized — grant exactly what the plan declares.
        # NOTE: only the DETERMINISTIC path is wired today. When ModelAssistedPlanner is wired into submit(),
        # its proposed caps MUST be checked against jc.capabilities (the job-type policy) and NOT unioned in
        # here — unioning a model-proposed plan's own caps would self-authorize and defeat the gate.
        sched.capabilities |= {c for t in tasks for c in t.capabilities}
        trace.emit("plan", nodes=[{"id": t.task_id, "kind": t.kind, "owner": t.owner,
                                   "deps": list(t.dependencies), "root": t.is_root,
                                   "caps": list(t.capabilities)} for t in tasks])
        sched.admit_plan(tasks)
        summary = sched.run(ts=job.meta.get("ts"))
        tsum = trace.summary()
        trace.emit("run.end", committed=summary["committed"], events=tsum["events"], dur_ms=tsum["dur_ms"])
        if self.sink:
            self.sink.write_summary(tsum)

        rollup = self._rollup(job, jc, sched, summary)
        rollup["telemetry"] = tsum                       # the supervisor trace summary travels with the rollup
        self.history.append(rollup)
        # LEARN: record the run + store useful details back into memory
        self.memory.record_run(jc.job_type, jc.template, rollup["committed"])
        self.memory.remember(jc.job_type, "last_job", job.job_id)
        if not rollup["committed"]:
            failed_kinds = [s["kind"] for s in rollup["tasks"] if s["status"] in ("FAILED", "ABORTED")]
            if failed_kinds:
                self.memory.note_learning(jc.job_type, f"job {job.job_id} failed at: {failed_kinds}")
        return rollup

    # --- master rollup: the user-facing tracker view of a job ---
    def _rollup(self, job: Job, jc: JobClass, sched: Scheduler, summary: dict) -> dict:
        rows = []
        for t in sched.tasks.values():
            rows.append({"task": t.task_id, "kind": t.kind, "owner": t.owner,
                         "status": t.status.value, "verification": t.verification.value,
                         "proof": t.proof_ref})
        return {
            "job_id": job.job_id, "job_type": jc.job_type, "template": jc.template,
            "committed": summary["committed"], "ledger_verified": summary["ledger_verified"],
            "ledger_entries": summary["ledger_entries"],
            "result": summary["root_result"],
            "tasks": rows,
            "checklist": [f"{'✓' if r['status'] in ('PASSED','COMMITTED') else '✗'} {r['kind']}"
                          f" [{r['owner']}] ({r['verification']})" for r in rows],
        }

    # --- master rollup ACROSS all jobs (the dashboard) ---
    def dashboard(self) -> dict:
        by_type: dict = {}
        for r in self.history:
            b = by_type.setdefault(r["job_type"], {"jobs": 0, "committed": 0})
            b["jobs"] += 1; b["committed"] += int(r["committed"])
        return {
            "jobs_total": len(self.history),
            "committed_total": sum(int(r["committed"]) for r in self.history),
            "by_type": by_type,
            "proven_templates": {jt: self.memory.best_template(jt) for jt in by_type},
            "learnings": {jt: self.memory.learnings_for(jt) for jt in by_type},
        }


# ---------------------------------------------------------------------------
# Built-in classifier rules (generalize across the job types we have)
# ---------------------------------------------------------------------------
def default_classifier_rules(orch: Orchestrator):
    """Register the job types this project actually has: coding, cyber, structured-output."""
    def is_coding(j: Job) -> bool:
        s = (j.prompt or "").lower()
        return ("def " in s) or ("function" in s and "return" in s) or bool(j.inputs.get("entry")) or bool(j.inputs.get("cases"))
    def is_cyber(j: Job) -> bool:
        s = (j.prompt or "").lower()
        return any(k in s for k in ("alert", "ioc", "beacon", "malware", "incident", "mitre", "soc", "forensic", "cve"))
    def is_structured(j: Job) -> bool:
        s = (j.prompt or "").lower()
        return "json" in s or "schema" in s or bool(j.inputs.get("schema"))
    orch.register_job_type("coding", is_coding, "code_gen", capabilities=["op:ast_parse", "op:run_tests"])
    orch.register_job_type("cyber", is_cyber, "cyber_alert", capabilities=["op:det_checks", "read:fixture"])
    orch.register_job_type("structured_output", is_structured, "structured_output", capabilities=["op:validate_schema"])
