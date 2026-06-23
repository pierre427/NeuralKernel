#!/usr/bin/env python3
"""CPU test for trustless agent delegation. Proves: a delegated tool-run is recorded + reconciled against the
trace (result matches -> verified); a TAMPERED result (hash mismatch), an UNBACKED result (no trace event), and
an ill-formed run (self-check fails) are all flagged; unknown tools are refused. Capstone: self-extend a
primitive, then delegate-and-verify it end to end (mint -> delegate -> reconcile)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/ (so `kernel` imports as a package)
from kernel.task_graph import Scheduler
from kernel.tools import register_tool, ToolSpec
from kernel.telemetry import RunTrace
from kernel.commit_trap import _sha
from kernel.agent_delegate import delegate_run, reconcile_with_trace, delegate_and_verify
from kernel.self_extend import Gap, SelfExtender

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


def main():
    # an echo tool to delegate (fast, no venv)
    s = Scheduler()
    register_tool(s, ToolSpec(name="echo", handler=lambda sched, inp: {"echo": inp.get("x")}), grant=True)

    # 1. HAPPY: fire agent -> record -> reconcile -> result lines up with the trace
    dv = delegate_and_verify(s, "echo", {"x": 42})
    ok(dv["verified"] and dv["result"] == {"echo": 42}, f"delegate_and_verify: result backed by a sound, hash-matching trace ({dv['result']})")
    ck = dv["reconcile"]["checks"]
    ok(ck["self_check_passed"] and ck["task_passed"] and ck["hash_match"], "reconcile: self-check + task PASSED + hash match all hold")

    # 2. TAMPER: a fabricated result does NOT hash-match the recorded proof -> flagged
    run = delegate_run(s, "echo", {"x": 7})
    rec_t = reconcile_with_trace(s, run["agent_task_id"], {"echo": 999}, trace=run["trace"])   # claimed != recorded
    ok((not rec_t["consistent"]) and (not rec_t["checks"]["hash_match"]),
       "tampered result -> hash mismatch -> NOT consistent (agent's word distrusted)")

    # 3. UNBACKED: a result for a task with no trace event -> flagged
    rec_u = reconcile_with_trace(s, "agent:a1/ghost", {"echo": 1}, trace=run["trace"])
    ok((not rec_u["consistent"]) and (not rec_u["checks"]["task_event_found"]), "unbacked result (no trace event) -> NOT consistent")

    # 4. SELF-CHECK FAIL: an ill-formed run (missing classify/plan) is rejected EVEN IF the hash matches
    res = {"echo": 1}
    bad = RunTrace("d", "delegate:echo")
    bad.events = [{"kind": "run.start"},
                  {"kind": "task", "task": "agent:a1/echo", "op": "echo", "to_state": "PASSED", "proof": _sha(res)},
                  {"kind": "run.end"}]                        # NO classify/plan -> check_invariants flags it
    rec_b = reconcile_with_trace(s, "agent:a1/echo", res, trace=bad)
    ok((not rec_b["consistent"]) and rec_b["checks"]["hash_match"] and (not rec_b["checks"]["self_check_passed"]),
       "ill-formed run -> self-check FAILS -> NOT consistent (even though the hash matched)")

    # 5. unknown tool refused
    ok(delegate_run(s, "nope", {}).get("ok") is False, "delegating an unregistered tool is refused")

    # 6. CAPSTONE (real venv): self-extend a primitive, then delegate-and-verify it end to end
    s2 = Scheduler()
    GOOD = "def popcount(n):\n    return bin(n).count('1')"
    GEN = "def generate(rng):\n    return (rng.randint(0, 2**31 - 1),)"
    ORACLE = "def oracle(n):\n    c = 0\n    while n:\n        c += n & 1\n        n >>= 1\n    return c"
    gap = Gap(entry="popcount", sample_cases=[{"args": [7], "expected": 3}, {"args": [0], "expected": 0}],
              generate_src=GEN, oracle_src=ORACLE, n=400, seed=7)
    tr = SelfExtender(s2, synthesize=lambda g, fb: GOOD).extend(gap)
    ok(tr.get("registered"), "capstone: primitive self-extended + registered")
    dv2 = delegate_and_verify(s2, "popcount", {"args": [7]})
    ok(dv2["verified"] and dv2["result"].get("result") == 3,
       f"capstone: agent ran the minted primitive, requestor reconciled vs trace -> verified (popcount(7)={dv2['result'].get('result')})")

    # 7. BUG-SWEEP CRITICAL (self-graded verdict): an executor reporting ex_ok=True but whose KERNEL validator
    #    rejects the result is recorded FAILED -> NOT verified (the verdict is the kernel's, not the agent's).
    s3 = Scheduler()
    register_tool(s3, ToolSpec(name="liar", handler=(lambda sched, inp: {"answer": "I solved it"}),
                               result_ok=(lambda r, t: False)), grant=True)   # validator ALWAYS rejects
    dv_l = delegate_and_verify(s3, "liar", {})
    ok(not dv_l["verified"], "self-grading executor (ex_ok ignored): the kernel validator rejects -> NOT verified")

    # 8. BUG-SWEEP CRITICAL (admit bypass): delegation routes through admit(); an ungranted-cap tool is REFUSED
    s4 = Scheduler()
    register_tool(s4, ToolSpec(name="locked", handler=(lambda sched, inp: {"ok2": 1})), grant=False)  # cap NOT granted
    run_lk = delegate_run(s4, "locked", {})
    ok((not run_lk.get("ok")) and ("admit refused" in str(run_lk.get("error", ""))),
       "delegation goes through admit(): ungranted-cap tool REFUSED (executor never called directly)")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} agent-delegate checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
