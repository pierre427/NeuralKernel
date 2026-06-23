#!/usr/bin/env python3
"""kernel/tools.py — the SYSCALL-TIER tool registry (Step 2 of the self-extending det plane).

The scheduler's op space has TWO trust tiers:
  PROVEN det primitives — pure, deterministic, hold-out-gated (count_byte, c_gcd, ...; + self-extended ones).
  SYSCALL-tier TOOLS    — I/O / nondeterministic / introspective (venv_exec, web_search, fetch_page, datetime,
                          sysinfo). Capability-gated + sandboxed; NOT "proven" (no fixed ground truth to gate,
                          so a result earns at most CHECKED — "the call ran and returned a well-formed result").

A ToolSpec wraps a tool-native handler(sched, inputs)->result into the Scheduler ABI (executor
fn(sched,task,inputs)->(result,ok,ValidationLevel) + a result validator fn(result,task)->bool) and registers it
FAIL-CLOSED: register_tool installs the executor + validator, and grants the capability token ONLY if grant=True.
A tool the kernel has not granted is inert — a model/injected plan can NAME it but admit() refuses it (the cap
gate), so adding tools never widens the attack surface unless the kernel explicitly enables each one.

The spec also carries description + signature so the Step-4 lexical search can index tools alongside primitives.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Any

try:                                             # works both as a package (kernel.tools) and flat (tests/runners)
    from .commit_trap import ValidationLevel
except ImportError:
    from commit_trap import ValidationLevel

_LEVEL = {lvl.name: lvl for lvl in ValidationLevel}


@dataclass
class ToolSpec:
    name: str
    handler: Callable                      # handler(sched, inputs: dict) -> result  (tool-native)
    description: str = ""                  # indexed by the lexical proposer (Step 4)
    signature: str = ""                    # human/JSON arg signature shown to the model
    capability: Optional[str] = None       # default f"op:{name}"; the fail-closed gate token
    validator_level: str = "SCHEMA"        # declared level for the node's validator -> CHECKED (I/O is not ORACLE)
    result_ok: Optional[Callable] = None   # validator(result, task) -> bool; default = well-formed, no error
    tier: str = "syscall"

    def cap_token(self) -> str:
        return self.capability or f"op:{self.name}"


def _default_ok(result, task) -> bool:
    """A syscall tool 'passed' iff it returned a well-formed result that did not report an error. (We can check
    that the call RAN; we cannot ORACLE-verify that an I/O result is the truth — hence CHECKED, never PROVEN.)
    NOTE: this only understands the {"error": ...} failure convention — it treats ANY other non-None result
    (including 0 / "" / [] / False) as a pass. A tool that signals failure some other way MUST supply its own
    result_ok; the default is intentionally permissive about result shape."""
    if result is None:
        return False
    if isinstance(result, dict) and result.get("error"):
        return False
    return True


# searchable catalog of registered tools (name -> ToolSpec); the Step-4 lexical proposer indexes this.
TOOL_CATALOG: dict[str, "ToolSpec"] = {}


def register_tool(sched, spec: ToolSpec, grant: bool = True) -> dict:
    """Register a syscall-tier tool on a live Scheduler. Installs the executor (wrapping the handler into the ABI,
    so a crashing tool fails the NODE — never the run) + a result validator; grants the capability only if
    grant=True (fail-closed otherwise). Returns the wiring a planner uses to build an admit()-able Task."""
    cap = spec.cap_token()
    vkey = f"{spec.name}_ok"
    lvl = _LEVEL.get(spec.validator_level.upper(), ValidationLevel.SCHEMA)

    def _executor(s, task, inputs):
        try:
            return (spec.handler(s, inputs or {}), True, lvl)
        except Exception as e:                      # fail the node (validator sees the error), never crash run()
            return ({"error": f"{type(e).__name__}: {e}"}, False, ValidationLevel.NONE)

    sched.register_executor(spec.name, _executor)
    sched.register_validator(vkey, spec.result_ok or _default_ok)
    # Mark this op syscall-tier so admit() ENFORCES (not just documents) that any Task invoking it declares the
    # cap and does not over-claim its level — closes the capabilities=[] bypass and the level-overclaim hole.
    if not hasattr(sched, "_syscall_ops"):
        sched._syscall_ops = {}
    sched._syscall_ops[spec.name] = {"cap": cap, "max_level": spec.validator_level}
    if grant:
        sched.capabilities.add(cap)
    TOOL_CATALOG[spec.name] = spec
    return {"kind": spec.name, "validator": vkey, "level": spec.validator_level, "capability": cap,
            "granted": bool(grant)}


def tool_task(task_id: str, wiring: dict, input_refs: Optional[dict] = None, **kw):
    """Build a correctly-gated Task for a registered tool straight from register_tool's wiring dict — declares the
    capability and the admit-allowed level so the syscall-op gate passes. The kernel-side admit() enforcement is
    the backstop; this is the sanctioned way to construct a tool Task so callers don't hand-roll it wrong."""
    try:
        from .task_graph import Task
    except ImportError:
        from task_graph import Task
    return Task(task_id=task_id, kind=wiring["kind"],
                validator={"fn": wiring["validator"], "level": wiring["level"]},
                capabilities=[wiring["capability"]], input_refs=input_refs or {}, **kw)


def builtin_tool_specs() -> list:
    """The default tools shipped with the kernel. web_search/fetch_page are bridged separately (they live in
    harness/tool-bridge); venv_exec is registered by the venv tool. datetime + sysinfo + trace are native here."""
    try:
        from .tools_builtin import datetime_handler, sysinfo_handler
        from .trace_tool import trace_handler
        from .venv_tool import venv_tool_spec
    except ImportError:
        from tools_builtin import datetime_handler, sysinfo_handler
        from trace_tool import trace_handler
        from venv_tool import venv_tool_spec
    return [
        ToolSpec(name="datetime", handler=datetime_handler,
                 description="Current real-world date and time: UTC + local ISO timestamps, epoch seconds, "
                             "timezone, weekday. Use when the task depends on the actual current date/time.",
                 signature="datetime() -> {utc_iso, local_iso, epoch, tz, weekday}"),
        ToolSpec(name="sysinfo", handler=sysinfo_handler,
                 description="System and kernel introspection: host vitals (memory, load average, cpu count, "
                             "processor type, free disk on the running partition) and live scheduler state "
                             "(process table, per-state queue depths, held/waiting processes, memory slots, "
                             "granted capabilities, trap budget). Use to observe system/kernel load.",
                 signature="sysinfo(facet='all'|'host'|'scheduler', expose_all=False, verbose=False)"),
        ToolSpec(name="trace", handler=trace_handler,
                 description="Read scheduler execution TRACES (live telemetry or persisted reports/telemetry/"
                             "*.jsonl) and run kernel SELF-CHECKS over them: unsafe commit (committed with no "
                             "ORACLE pass), ledger not verified, PASSED node missing a proof, inert repair, "
                             "malformed flow, miscalibration. Use to audit whether a run behaved correctly.",
                 signature="trace(source='live'|'file'|'dir', path?, facet='selfcheck'|'summary'|'events'|'lines'|'all', job?, limit?)"),
        venv_tool_spec(),     # op:venv_exec — run candidate code out-of-process in a hermetic sandboxed venv
    ]


def register_builtins(sched, grant: bool = True) -> list:
    """Register the default native syscall-tier tools (datetime, sysinfo) on a live Scheduler."""
    return [register_tool(sched, s, grant=grant) for s in builtin_tool_specs()]


__all__ = ["ToolSpec", "register_tool", "register_builtins", "builtin_tool_specs", "tool_task", "TOOL_CATALOG",
           "_default_ok"]
