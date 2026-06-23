#!/usr/bin/env python3
"""CPU-only tests for the M1 commit-trap-as-output-path LOGIC (no LLM — the decode needs the model, the
validation/commit does not). Proves the trap EMITs ORACLE-correct code and CATCHES wrong code (escalate,
don't fabricate). Run with the mlx venv (imports mlx via the kernel chain; loads NO model):
  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_commit_kernel.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commit_trap import CommitTrap, TrapBudget, ValidationLevel
from kernel.commit_kernel import coding_contract

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
def _ladder_cases():
    # bundle-first (kernel/fixtures/), then the original repo path — works both standalone and in-tree.
    for p in (os.path.join(_HERE, "fixtures", "ladder_all_cases.json"),
              os.path.join(_REPO, "reports", "north-m0", "ladder_all_cases.json")):
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError("ladder_all_cases.json not found (looked in kernel/fixtures and reports/north-m0)")
TASK = next(t for t in _ladder_cases() if t["entry"] == "gcd")
GOOD = "```python\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a\n```"
BAD  = "```python\ndef gcd(a: int, b: int) -> int:\n    return a + b\n```"   # compiles, but wrong (returns the sum)


def test_contract():
    print("\n[1] coding_contract builds entry + cases from the task")
    c = coding_contract(TASK)
    ok(c.entry == "gcd" and len(c.cases) == len(TASK["cases"]), "contract carries the entry + the task's real cases")


def test_emit_correct():
    print("\n[2] correct code clears ORACLE and EMITs (the only output path)")
    d = CommitTrap(TrapBudget()).commit(GOOD, coding_contract(TASK), ts=0.0, min_level=ValidationLevel.ORACLE)
    ok(d.action == "emit", f"correct gcd -> emit (got {d.action})")
    ok(d.validation_level == ValidationLevel.ORACLE, f"reached ORACLE (got {d.validation_level.name})")


def test_catch_wrong():
    print("\n[3] wrong code is CAUGHT, not emitted — escalate, don't fabricate")
    t = CommitTrap(TrapBudget())
    d = t.commit(BAD, coding_contract(TASK), ts=0.0, min_level=ValidationLevel.ORACLE)
    ok(d.action != "emit", f"wrong gcd -> NOT emitted (got {d.action})")
    ok(d.validation_level < ValidationLevel.ORACLE, f"did not reach ORACLE (got {d.validation_level.name})")
    ok(t.ledger.verify_all(), "the failed validation is sealed in a hash-verified ProofLedger")


def test_always_accept_floor():
    print("\n[4] at a low trust bar (SYNTAX) the compiling answer emits unchanged (the always-accept control)")
    d = CommitTrap(TrapBudget()).commit(BAD, coding_contract(TASK), ts=0.0, min_level=ValidationLevel.SYNTAX)
    ok(d.action == "emit", "min_level=SYNTAX -> even the (compiling) wrong code emits == no-trap control (==gen_fast)")


if __name__ == "__main__":
    test_contract(); test_emit_correct(); test_catch_wrong(); test_always_accept_floor()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} commit-kernel checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
