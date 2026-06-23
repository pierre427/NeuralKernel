#!/usr/bin/env python3
"""CPU test for the web-tool bridge structure (NO network — the live web_search/fetch_page are verified manually):
handlers fail-soft on missing args, the specs exist, and registration wires op:web_search / op:fetch_page
fail-closed on a Scheduler. (Actual network calls are exercised by the escalation-solver research tier.)"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kernel.web_tools import web_available, web_search_handler, fetch_page_handler, web_tool_specs, register_web_tools
from kernel.task_graph import Scheduler, Task, SchedulerReject

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


def main():
    ok(isinstance(web_available(), bool), f"web_available() -> bool ({web_available()})")

    # handlers fail SOFT on missing args (return an error dict, never raise)
    ok(web_search_handler(None, {}).get("error"), "web_search_handler with no query -> {error}")
    ok(fetch_page_handler(None, {}).get("error"), "fetch_page_handler with no url -> {error}")

    specs = web_tool_specs()
    names = {s.name for s in specs}
    ok(names == {"web_search", "fetch_page"}, f"web_tool_specs -> web_search + fetch_page ({names})")

    # registration wires both as syscall-tier ops, capability-gated (fail-closed: ungranted -> admit refuses)
    s = Scheduler()
    wiring = register_web_tools(s, grant=True)
    ok("web_search" in s.executors and "fetch_page" in s.executors, "register_web_tools installs both executors")
    ok("op:web_search" in s.capabilities and "op:fetch_page" in s.capabilities, "both capabilities granted")
    caps = {w["kind"]: w["capability"] for w in wiring}
    ok(caps.get("web_search") == "op:web_search", "web_search gated behind op:web_search")

    s2 = Scheduler(); register_web_tools(s2, grant=False)               # NOT granted
    refused = False
    try:
        s2.admit(Task(task_id="w", kind="web_search", validator={"fn": "web_search_ok", "level": "SCHEMA"},
                      capabilities=["op:web_search"]))
    except SchedulerReject:
        refused = True
    ok(refused, "ungranted web tool -> admit REFUSES (fail-closed, syscall-tier)")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} web-tool checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
