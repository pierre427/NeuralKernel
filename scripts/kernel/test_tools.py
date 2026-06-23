#!/usr/bin/env python3
"""CPU test for the syscall-tier tool registry + native tools (no model). Proves: registration wires the
Scheduler ABI; FAIL-CLOSED capability gating (ungranted tool is refused at admit); a crashing tool fails the node
not the run; datetime + sysinfo return well-formed results; sysinfo host is partition-scoped (no all-mounts unless
expose_all); sysinfo scheduler facet reflects the live process table over a real Scheduler."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))           # kernel/
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/ (so `kernel` is an importable package)
from kernel.task_graph import Scheduler, Task, SchedulerReject
from kernel.tools import register_tool, register_builtins, ToolSpec, TOOL_CATALOG, _default_ok, tool_task

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


def main():
    # 1. registration wires the ABI + catalog
    s = Scheduler()
    wiring = register_tool(s, ToolSpec(name="echo", handler=(lambda sched, inp: {"echo": inp.get("x")}),
                                       description="echo back x"), grant=True)
    ok("echo" in s.executors and wiring["validator"] in s.validators and wiring["capability"] in s.capabilities,
       "register_tool installs executor + validator + grants capability")
    ok("echo" in TOOL_CATALOG, "tool is added to the searchable TOOL_CATALOG")

    # 2. the wrapped executor returns (result, ok, level); a crashing handler fails the NODE, never raises
    res, ex_ok, _ = s.executors["echo"](s, None, {"x": 5})
    ok(res == {"echo": 5} and ex_ok, "executor runs the handler and returns its result")
    register_tool(s, ToolSpec(name="boom", handler=(lambda sched, inp: 1 / 0)), grant=True)
    bres, bok, _ = s.executors["boom"](s, None, {})
    ok((not bok) and isinstance(bres, dict) and "error" in bres, "crashing tool -> (error, False), no raise")
    ok((not _default_ok(bres, None)) and _default_ok({"echo": 5}, None), "default validator: error fails, well-formed passes")

    # 3. FAIL-CLOSED capability gate: an ungranted tool is REFUSED at admit; granting it admits
    s2 = Scheduler()
    register_tool(s2, ToolSpec(name="locked", handler=(lambda sched, inp: {"ok": 1})), grant=False)  # NOT granted
    t = Task(task_id="t1", kind="locked", validator={"fn": "locked_ok", "level": "SCHEMA"},
             capabilities=["op:locked"])
    refused = False
    try:
        s2.admit(t)
    except SchedulerReject:
        refused = True
    ok(refused, "ungranted tool with declared cap -> admit() raises SchedulerReject (fail-closed)")
    s2.capabilities.add("op:locked")                         # kernel grants it
    s2.admit(Task(task_id="t2", kind="locked", validator={"fn": "locked_ok", "level": "SCHEMA"},
                  capabilities=["op:locked"]))
    ok("t2" in s2.tasks, "once the kernel grants the cap, the same tool task admits")

    # 4. default tools register + run well-formed
    s3 = Scheduler()
    register_builtins(s3, grant=True)
    ok("datetime" in s3.executors and "sysinfo" in s3.executors, "register_builtins installs datetime + sysinfo")
    dt, _, _ = s3.executors["datetime"](s3, None, {})
    ok(isinstance(dt, dict) and "utc_iso" in dt and "epoch" in dt, f"datetime returns well-formed result ({dt.get('utc_iso','')[:19]})")

    # 5. sysinfo HOST facet — partition-scoped by default, all-mounts gated behind expose_all
    h, _, _ = s3.executors["sysinfo"](s3, None, {"facet": "host"})
    host = h["host"]
    ok(all(k in host for k in ("memory", "load_avg", "cpu_count", "processor", "disk_running_partition")),
       f"sysinfo host has memory/load/cpu/processor/disk ({host['processor'][:28]!r}, {host['cpu_count']} cpu)")
    ok("disk_all_partitions" not in host, "sysinfo host: NO all-partitions by default (least-exposure)")
    h2, _, _ = s3.executors["sysinfo"](s3, None, {"facet": "host", "expose_all": True})
    ok("disk_all_partitions" in h2["host"], "sysinfo host: all-partitions appears ONLY with expose_all=True")

    # 6. sysinfo SCHEDULER facet reflects the live process table
    sc, _, _ = s3.executors["sysinfo"](s3, None, {"facet": "scheduler"})
    sch = sc["scheduler"]
    ok("queue_depths" in sch and "process_table" in sch and "memory_slots" in sch and "capabilities" in sch,
       f"sysinfo scheduler: queue_depths/process_table/slots/caps present ({len(sch['process_table'])} procs)")
    ok("op:sysinfo" in sch["capabilities"], "sysinfo scheduler reports the granted capabilities (op:sysinfo)")

    # 7. HARDENING (review FIX-FIRST): syscall cap is MANDATORY — a Task with capabilities=[] is REFUSED at admit
    refused_nocap = False
    try:
        s3.admit(Task(task_id="nocap", kind="sysinfo", validator={"fn": "sysinfo_ok", "level": "SCHEMA"},
                      capabilities=[]))                               # omits the cap -> must be rejected now
    except SchedulerReject:
        refused_nocap = True
    ok(refused_nocap, "syscall op with capabilities=[] -> admit REFUSES (cap is mandatory, not opt-in)")

    # 8. HARDENING: a syscall op cannot OVER-CLAIM its trust level (declared ORACLE > SCHEMA ceiling -> refused)
    refused_lvl = False
    try:
        s3.admit(Task(task_id="overclaim", kind="sysinfo", validator={"fn": "sysinfo_ok", "level": "ORACLE"},
                      capabilities=["op:sysinfo"]))
    except SchedulerReject:
        refused_lvl = True
    ok(refused_lvl, "syscall op declaring level ORACLE -> admit REFUSES (ceiling is SCHEMA/CHECKED)")

    # 8b. BUG-SWEEP HIGH: a syscall op with NO validator is refused (can't skip the ceiling by omitting it)
    refused_noval = False
    try:
        s3.admit(Task(task_id="noval", kind="sysinfo", validator=None, capabilities=["op:sysinfo"]))
    except SchedulerReject:
        refused_noval = True
    ok(refused_noval, "syscall op with validator=None -> admit REFUSES (validator mandatory; ceiling not skippable)")

    # 9. the sanctioned tool_task builder produces an admit-able, correctly-gated Task
    built = tool_task("dt1", wiring={"kind": "datetime", "validator": "datetime_ok", "level": "SCHEMA",
                                     "capability": "op:datetime"})
    s3.admit(built)
    ok("dt1" in s3.tasks, "tool_task() builds a correctly-gated Task that admits")

    # 10. HARDENING: secrets in artifact slots are REDACTED in the scheduler facet, even with verbose=True
    s3.artifacts.put("config: api_key=sk-ABCD1234EFGH5678IJKL and token=ghp_0123456789abcdef0123", "blob")
    scv, _, _ = s3.executors["sysinfo"](s3, None, {"facet": "scheduler", "verbose": True})
    blob = repr(scv["scheduler"]["memory_slots"])
    ok(("sk-ABCD1234EFGH5678" not in blob) and ("ghp_0123456789abcdef" not in blob) and ("«redacted»" in blob),
       "sysinfo scheduler REDACTS secret-like tokens in slot contents (even verbose)")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} tool-registry checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
