#!/usr/bin/env python3
"""CPU test for the self-extending pipeline (no model — stub synthesizers stand in for the model). Proves the MVP
loop end to end: a gap for a MISSING primitive -> synthesize -> sandbox-test -> hold-out-gate (vs an INDEPENDENT
oracle) -> placebo control -> register -> the primitive is callable and correct. Plus: the gate rejects a wrong
impl (never registers), the feedback loop repairs a first-attempt miss, and a non-discriminating oracle is caught
by the placebo guard (refuses to register)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/ (so `kernel` imports as a package)
from kernel.self_extend import Gap, SelfExtender
from kernel.task_graph import Scheduler

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")

# An INDEPENDENT oracle (bit-shift count) — a DIFFERENT algorithm from the candidate (bin().count), so the gate
# is meaningful, not a tautology.
GEN = "def generate(rng):\n    return (rng.randint(0, 2**31 - 1),)"
ORACLE = "def oracle(n):\n    c = 0\n    while n:\n        c += n & 1\n        n >>= 1\n    return c"
GOOD = "def popcount(n):\n    return bin(n).count('1')"        # correct, different algo from the oracle
# A MEMORIZED LOOKUP: passes the 3 visible sample cases but returns 0 on novel inputs -> the hold-out gate's
# signature catch ("a memorized lookup scores ~0% held-out"). Sample-pre-filter passes; the GATE rejects it.
WRONG = "def popcount(n):\n    return {7: 3, 255: 8, 0: 0}.get(n, 0)"


def popcount_gap():
    return Gap(entry="popcount", description="count set bits",
               sample_cases=[{"args": [7], "expected": 3}, {"args": [255], "expected": 8}, {"args": [0], "expected": 0}],
               generate_src=GEN, oracle_src=ORACLE, n=800, seed=7)


def main():
    # 1. HAPPY PATH: gap -> synth -> gate -> register -> callable + correct
    s = Scheduler()
    ext = SelfExtender(s, synthesize=lambda gap, fb: GOOD)
    tr = ext.extend(popcount_gap())
    ok(tr.get("registered") is True, f"missing primitive synthesized + gated + REGISTERED (gate {tr.get('gate_passed')}/800)")
    ok("popcount" in s.executors and "popcount_ok" in s.validators and "prim:popcount" in s.capabilities,
       "registration wired the executor + validator + granted prim:popcount")
    ok(ext.call("popcount", [7]).get("result") == 3 and ext.call("popcount", [255]).get("result") == 8,
       "the registered primitive is callable and CORRECT (sandboxed dispatch)")
    res, vok, _ = s.executors["popcount"](s, None, {"args": [0]})         # dispatch through the scheduler executor
    ok(vok and res.get("result") == 0, "scheduler executor dispatches to the new primitive (popcount(0)=0)")
    ok(any(a["result"] == "registered" for a in tr["attempts"]), "trace records the registration")
    ok(tr["attempts"][-1].get("regate", {}).get("ok"), "happy path also cleared the 2nd-seed determinism re-gate")

    # 2. GATE REJECT: a wrong impl never registers (gate fails every attempt)
    s2 = Scheduler()
    tr2 = SelfExtender(s2, synthesize=lambda gap, fb: WRONG, max_attempts=2).extend(popcount_gap())
    ok((tr2.get("registered") is False) and ("popcount" not in s2.executors),
       f"memorized-lookup REJECTED, never registered (reason: {tr2.get('reason')})")
    ok(all(a.get("result") == "gate_fail" and a["sample"]["ok"] for a in tr2["attempts"]),
       "memorized lookup PASSES the visible samples but the HOLD-OUT GATE rejects it (the methodology's core catch)")

    # 3. REPAIR: first attempt fails its sample cases, feedback drives a correct second attempt
    s3 = Scheduler()
    calls = {"n": 0}
    def flaky(gap, fb):
        calls["n"] += 1
        return WRONG if fb is None else GOOD                  # fb is set only after a failed attempt
    tr3 = SelfExtender(s3, synthesize=flaky, max_attempts=3).extend(popcount_gap())
    ok(tr3.get("registered") is True and len(tr3["attempts"]) == 2 and calls["n"] == 2,
       "feedback loop: attempt 1 fails, attempt 2 (with feedback) registers")

    # 4. PLACEBO GUARD: a non-discriminating oracle (returns None) — a None-returning candidate would pass the
    #    gate, but so would the placebo, so the pipeline REFUSES to register (fail->pass is not proof).
    s4 = Scheduler()
    bad_gap = Gap(entry="trivial", sample_cases=[{"args": [1], "expected": None}, {"args": [2], "expected": None}],
                  generate_src=GEN, oracle_src="def oracle(n):\n    return None", n=200, seed=1)
    tr4 = SelfExtender(s4, synthesize=lambda gap, fb: "def trivial(n):\n    return None").extend(bad_gap)
    ok((tr4.get("registered") is False) and ("discriminating" in tr4.get("reason", "")),
       f"placebo guard: non-discriminating oracle -> REFUSE to register ({tr4.get('reason')})")
    ok("trivial" not in s4.executors, "the unsound primitive was NOT registered")

    # 5. ADVERSARIAL (review CRITICAL): a candidate that tries to PATCH the gate comparator cannot — comparison
    #    runs in the trusted PARENT now, so __main__._eq doesn't exist in the child to patch. It passes the visible
    #    samples (memorized lookup) but the parent-judged gate rejects it. MUST NOT register.
    ATTACK_EQ = ("import sys\ntry:\n    sys.modules['__main__']._eq = lambda a, b: True\nexcept Exception:\n    pass\n"
                 "def popcount(n):\n    return {7: 3, 255: 8, 0: 0}.get(n, 999)")
    s5 = Scheduler()
    tr5 = SelfExtender(s5, synthesize=lambda gap, fb: ATTACK_EQ, max_attempts=1).extend(popcount_gap())
    ok((tr5.get("registered") is False) and ("popcount" not in s5.executors),
       "adversarial _eq-PATCH candidate NOT registered (comparator lives in the trusted parent)")

    # 6. ADVERSARIAL: a candidate that FORGES a passing output line + os._exit cannot spoof the verdict — it can
    #    forge the 3 known sample answers, but the gate's novel instances are judged against the parent's oracle
    #    (and the output count is validated), so it fails the gate. MUST NOT register.
    ATTACK_FORGE = ("import os, sys, json\n"
                    "sys.stdout.write(json.dumps({'outs': [{'v': 3}, {'v': 8}, {'v': 0}]}) + '\\n')\n"
                    "sys.stdout.flush()\n"
                    "os._exit(0)\n")
    s6 = Scheduler()
    tr6 = SelfExtender(s6, synthesize=lambda gap, fb: ATTACK_FORGE, max_attempts=1).extend(popcount_gap())
    ok(tr6.get("registered") is False, "adversarial FORGED-output candidate NOT registered (parent judges novel instances)")

    # 5. BUG-SWEEP HIGH: a candidate OVERFIT to one seed's instances (memorized lookup of exactly the gate-1
    #    instances) passes gate 1 but FAILS the 2nd-seed re-gate -> NOT registered.
    import random as _r
    rng = _r.Random(7)
    insts = [rng.randint(0, 2**31 - 1) for _ in range(20)]    # must match GEN+seed 7 exactly
    lookup = {x: bin(x).count("1") for x in insts}
    OVERFIT = "_L = %r\ndef popcount(n):\n    return _L.get(n, 999)" % (lookup,)
    s5 = Scheduler()
    gap5 = Gap(entry="popcount", sample_cases=[{"args": [insts[0]], "expected": lookup[insts[0]]}],
               generate_src=GEN, oracle_src=ORACLE, n=20, seed=7)
    tr5 = SelfExtender(s5, synthesize=lambda g, fb: OVERFIT, max_attempts=1).extend(gap5)
    ok((tr5.get("registered") is False) and tr5["attempts"][0].get("gate", {}).get("ok") and not tr5["attempts"][0].get("regate", {}).get("ok"),
       "seed-overfit candidate passes gate 1 but FAILS the 2nd-seed re-gate -> NOT registered")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} self-extend checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
