"""Neural-microkernel package: the scheduler plane (M1 commit trap, and later
the entry trap / dispatch chokepoint). Pure-Python, importable, no LLM, no network.

M1 — the post-last-layer COMMIT TRAP (`commit_trap.py`) is the only output path:
a candidate is validated against its contract (syntax<schema<property<oracle<proven)
before commit, every commit is recorded in a hash-replayable ProofLedger, and a
hard kernel-owned TrapBudget rejects work over budget (the anti-runaway invariant).

Proof-Carrying Task Graph (`task_graph.py`) — the kernel-owned process table that
tracks intent, subgoals, continuations, evidence, validators, budgets, and completion
state. The model proposes a plan; the Scheduler admits (type-checks), executes the DAG,
validates each node, logs every transition to the ProofLedger, spawns clean-context
repair workers on failure, and bails when the root's contract is satisfied. The model
is NOT the task tracker — the scheduler owns task state, deps, budgets, caps, artifacts.
"""
from .commit_trap import (
    CommitTrap,
    Decision,
    ProofLedger,
    LedgerEntry,
    TrapBudget,
    BudgetExceeded,
    ValidationLevel,
    Contract,
)
from .task_graph import (
    Scheduler,
    Task,
    TaskState,
    TaskEvent,
    Verification,
    Budget,
    Artifact,
    ArtifactStore,
    DeterministicPlanner,
    ModelAssistedPlanner,
    SchedulerReject,
    PLAN_TEMPLATES,
)
from .orchestrator import (
    Orchestrator,
    Job,
    JobClass,
    JobClassifier,
    PlanMemory,
    default_classifier_rules,
)

__all__ = [
    # M1 — commit trap
    "CommitTrap", "Decision", "ProofLedger", "LedgerEntry", "TrapBudget",
    "BudgetExceeded", "ValidationLevel", "Contract",
    # Proof-Carrying Task Graph (the kernel process table)
    "Scheduler", "Task", "TaskState", "TaskEvent", "Verification", "Budget",
    "Artifact", "ArtifactStore", "DeterministicPlanner", "ModelAssistedPlanner",
    "SchedulerReject", "PLAN_TEMPLATES",
    # Job Kernel (intake -> classify -> plan -> rollup, with plan memory)
    "Orchestrator", "Job", "JobClass", "JobClassifier", "PlanMemory", "default_classifier_rules",
]
