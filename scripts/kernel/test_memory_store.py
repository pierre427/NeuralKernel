#!/usr/bin/env python3
"""CPU test for the persistent, hard-bounded kernel memory store. Uses a SMALL cap to exercise eviction fast.
Proves: round-trip + persistence; the governance object is pinned + always consulted; the HARD cap is never
exceeded (LRU eviction of non-pinned; oversized writes rejected); pinned/governance survive eviction; and the
op:memory scheduler tool dispatches + is fail-closed."""
import sys, os, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kernel.memory_store import MemoryStore, GOV_KEY, register_memory_tool
from kernel.task_graph import Scheduler, Task, SchedulerReject

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.sqlite")

        # 1. round-trip + persistence across reopen
        m = MemoryStore(path, max_bytes=4000)
        m.put("fact1", "the kernel runs North via MLX", kind="fact")
        ok(m.get("fact1") == "the kernel runs North via MLX", "put/get round-trip")
        m.close()
        m = MemoryStore(path, max_bytes=4000)                    # reopen
        ok(m.get("fact1") == "the kernel runs North via MLX", "persists across reopen (durable)")

        # 2. governance object: present, pinned, always first in consult()
        c = m.consult()
        ok(c["governance"] and "ALWAYS CONSULT" in c["governance"], "consult() returns the governance policy first")
        gov_pinned = any(e["key"] == GOV_KEY and e["pinned"] for e in m.list())
        ok(gov_pinned, "governance object is pinned")
        ok(m.delete(GOV_KEY)["ok"] is False, "governance object cannot be deleted")
        ok(m.put(GOV_KEY, "HACKED")["ok"] is False and "ALWAYS CONSULT" in m.get(GOV_KEY),
           "governance object is immutable via put() (no policy self-tampering)")

        # 3. HARD cap: fill past the cap -> LRU non-pinned evicted, cap NEVER exceeded
        m.put("keep", "x" * 300, kind="note", pin=True)          # a PINNED entry (must survive eviction)
        m.get("fact1")                                            # touch fact1 so it's NOT the LRU
        for i in range(40):                                       # write enough to force eviction
            m.put(f"e{i}", "y" * 200, kind="note")
        st = m.stats()
        ok(st["total_bytes"] <= 4000, f"hard cap NEVER exceeded ({st['total_bytes']} <= {st['max_bytes']})")
        ok(m.get(GOV_KEY, touch=False) is not None and m.get("keep") is not None,
           "pinned governance + pinned 'keep' SURVIVED eviction")
        ok(m.get("e0") is None, "oldest non-pinned entry (e0) was evicted (LRU)")

        # 4. an oversized single entry is REJECTED (never exceeds the cap)
        r = m.put("huge", "z" * 5000)
        ok(r["ok"] is False and "cap" in r["error"], "oversized entry REJECTED (hard bound)")
        ok(m.stats()["total_bytes"] <= 4000, "cap still intact after a rejected oversized write")

        # 5. op:memory scheduler tool: dispatch + fail-closed
        s = Scheduler()
        wiring = register_memory_tool(s, m, grant=True)
        ok("memory" in s.executors and wiring["capability"] == "op:memory", "op:memory registered + gated")
        res, vok, _ = s.executors["memory"](s, None, {"action": "consult"})
        ok(vok and "ALWAYS CONSULT" in (res.get("governance") or ""), "op:memory consult -> governance policy")
        pr, _, _ = s.executors["memory"](s, None, {"action": "put", "key": "k2", "value": "v2"})
        gr, _, _ = s.executors["memory"](s, None, {"action": "get", "key": "k2"})
        ok(pr.get("ok") and gr.get("value") == "v2", "op:memory put + get round-trip via the scheduler")

        s2 = Scheduler(); register_memory_tool(s2, m, grant=False)   # NOT granted
        refused = False
        try:
            s2.admit(Task(task_id="t", kind="memory", validator={"fn": "memory_ok", "level": "SCHEMA"},
                          capabilities=["op:memory"]))
        except SchedulerReject:
            refused = True
        ok(refused, "ungranted op:memory -> admit REFUSES (fail-closed, syscall tier)")

        # 6. hardening: a non-serializable value fails soft (no crash); a store too small for governance is rejected
        circ = {}; circ["self"] = circ
        ok(m.put("circ", circ)["ok"] is False, "non-serializable value -> fail-soft (no crash)")
        raised = False
        try:
            MemoryStore(os.path.join(d, "tiny.sqlite"), max_bytes=10)
        except ValueError:
            raised = True
        ok(raised, "max_bytes too small to hold governance -> ValueError (clear misconfig)")

        # 7. THREAD-SAFETY (regression for the observed SIGSEGV: a shared sqlite connection under concurrent access
        # corrupts SQLite internals). 8 threads hammering put/get/consult/list/stats; a segfault would crash the
        # whole process, so simply REACHING the assertion proves the lock fixed it.
        import threading as _thr
        mc = MemoryStore(os.path.join(d, "concurrent.sqlite"), max_bytes=200_000)
        errs = []
        def _hammer(tid):
            try:
                for i in range(250):
                    mc.put(f"t{tid}_{i}", "v" * 60, kind="note")
                    mc.get(f"t{tid}_{i % 10}")
                    mc.consult(5)
                    if i % 25 == 0:
                        mc.list(20); mc.stats()
            except Exception as e:                            # pragma: no cover
                errs.append(repr(e))
        ts = [_thr.Thread(target=_hammer, args=(k,)) for k in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        ok(not errs, f"8 threads x 250 ops: no crash/exception (was SIGSEGV pre-lock) {errs[:1]}")
        ok(mc.stats()["total_bytes"] <= 200_000, "hard cap still respected under concurrent writers")
        ok("ALWAYS CONSULT" in (mc.get(GOV_KEY) or ""), "governance object intact after concurrent hammering")
        mc.close()
        m.close()

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} memory-store checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
