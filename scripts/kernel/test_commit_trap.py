#!/usr/bin/env python3
"""CPU-only unit tests for the M1 commit trap. No LLM, no network.

Run:
  PYTHONPATH=finetune/gpt-oss-unsloth/scripts \
    /Users/pierrelamy/nm-mlx-venv/bin/python \
    finetune/gpt-oss-unsloth/scripts/kernel/test_commit_trap.py
"""
from __future__ import annotations

import sys

from commit_trap import (
    CommitTrap,
    Decision,
    ProofLedger,
    LedgerEntry,
    TrapBudget,
    ValidationLevel,
    Contract,
)

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    flag = "OK  " if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"  {flag} {name}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# 1. A valid JSON candidate commits at schema level.
# ---------------------------------------------------------------------------
def test_valid_json_commits_schema():
    trap = CommitTrap(TrapBudget())
    c = Contract(opcode="to_json", otype="json", required_keys=["a", "b"], op_version="v1")
    cand = '```json\n{"a": 1, "b": [2, 3]}\n```'
    # No oracle cases for the structured output -> trust threshold is schema.
    d = trap.commit(cand, c, ts=100.0, min_level=ValidationLevel.SCHEMA)
    check("valid json -> emit", d.action == "emit", d.action)
    check("valid json -> SCHEMA level", d.validation_level == ValidationLevel.SCHEMA,
          d.validation_level.name)
    check("valid json -> committed flag", d.committed)
    check("valid json -> ledger entry exists", d.ledger_entry is not None)


# ---------------------------------------------------------------------------
# 2. A valid Python candidate with passing cases commits at ORACLE level.
# ---------------------------------------------------------------------------
def test_valid_python_commits_oracle():
    trap = CommitTrap(TrapBudget())
    c = Contract(opcode="add", entry="add",
                 prompt="Write a Python function `add(a, b)`",
                 cases=({"args": [2, 3], "expected": 5},
                        {"args": [10, -4], "expected": 6}))
    cand = "```python\ndef add(a, b):\n    return a + b\n```"
    d = trap.commit(cand, c, ts=101.0)  # default min_level = ORACLE
    check("python passing -> emit", d.action == "emit", d.action)
    check("python passing -> ORACLE level", d.validation_level == ValidationLevel.ORACLE,
          d.validation_level.name)


# ---------------------------------------------------------------------------
# 3. An invalid candidate routes to repair/reject (not emit).
# ---------------------------------------------------------------------------
def test_invalid_candidate_repairs():
    trap = CommitTrap(TrapBudget())
    # 3a. Syntactically broken python -> form invalid -> repair, output None.
    c = Contract(opcode="f", entry="f", prompt="Write a Python function `f`")
    d = trap.commit("```python\ndef f(x):\n    return x +\n```", c, ts=200.0)
    check("broken python -> not emit", d.action != "emit", d.action)
    check("broken python -> repair", d.action == "repair", d.action)
    check("broken python -> no output", d.output is None)
    check("broken python -> NONE level", d.validation_level == ValidationLevel.NONE,
          d.validation_level.name)

    # 3b. Valid python but FAILS the provided cases -> repair at property level
    #     (parses + signature matches, just wrong) -> NOT emitted.
    c2 = Contract(opcode="add", entry="add",
                  prompt="Write a Python function `add(a, b)`",
                  cases=({"args": [2, 3], "expected": 5},))
    d2 = trap.commit("```python\ndef add(a, b):\n    return a - b\n```", c2, ts=201.0)
    check("wrong python -> repair", d2.action == "repair", d2.action)
    check("wrong python -> not emitted", not d2.committed)
    check("wrong python -> PROPERTY level", d2.validation_level == ValidationLevel.PROPERTY,
          d2.validation_level.name)

    # 3c. Invalid JSON -> form invalid -> repair.
    c3 = Contract(opcode="j", otype="json")
    d3 = trap.commit('{"a": 1,}', c3, ts=202.0)
    check("broken json -> not emit", d3.action != "emit", d3.action)


# ---------------------------------------------------------------------------
# 4. A proven (hold-out-gated) candidate commits at PROVEN level.
# ---------------------------------------------------------------------------
def test_proven_commits():
    trap = CommitTrap(TrapBudget())
    c = Contract(opcode="rabin_karp_search", entry="rabin_karp_search", proven=True,
                 prompt="def rabin_karp_search(text, pattern):")
    # Candidate content is irrelevant at PROVEN level (correctness is structural).
    d = trap.commit("<proven primitive output>", c, ts=300.0)
    check("proven -> emit", d.action == "emit", d.action)
    check("proven -> PROVEN level", d.validation_level == ValidationLevel.PROVEN,
          d.validation_level.name)


# ---------------------------------------------------------------------------
# 5. Budget exhaustion rejects (anti-runaway hard counter).
# ---------------------------------------------------------------------------
def test_budget_exhaustion_rejects():
    trap = CommitTrap(TrapBudget(max_traps=2, max_same_op=99))
    c = Contract(opcode="to_json", otype="json", required_keys=["a"])
    cand = '```json\n{"a": 1}\n```'
    d1 = trap.commit(cand, c, ts=400.0, min_level=ValidationLevel.SCHEMA)
    d2 = trap.commit(cand, c, ts=401.0, min_level=ValidationLevel.SCHEMA)
    d3 = trap.commit(cand, c, ts=402.0, min_level=ValidationLevel.SCHEMA)  # over budget
    check("budget: first two emit", d1.action == "emit" and d2.action == "emit",
          f"{d1.action},{d2.action}")
    check("budget: third rejected", d3.action == "reject", d3.action)
    check("budget: reject reason names cap", "max_traps" in d3.reason, d3.reason)
    check("budget: n_traps did not over-count", trap.budget.n_traps == 2,
          str(trap.budget.n_traps))

    # per-op cap (same-op-once generalized)
    trap2 = CommitTrap(TrapBudget(max_traps=99, max_same_op=1))
    e1 = trap2.commit(cand, c, ts=410.0, min_level=ValidationLevel.SCHEMA)
    e2 = trap2.commit(cand, c, ts=411.0, min_level=ValidationLevel.SCHEMA)  # same op again
    check("same_op: first emit", e1.action == "emit", e1.action)
    check("same_op: second rejected", e2.action == "reject", e2.action)
    check("same_op: reject reason names cap", "max_same_op" in e2.reason, e2.reason)


# ---------------------------------------------------------------------------
# 6. Ledger entries are well-formed and hash-replayable.
# ---------------------------------------------------------------------------
def test_ledger_replayable():
    trap = CommitTrap(TrapBudget())
    c = Contract(opcode="add", entry="add",
                 prompt="Write a Python function `add(a, b)`",
                 cases=({"args": [1, 1], "expected": 2},), op_version="v3")
    d = trap.commit("```python\ndef add(a, b):\n    return a + b\n```", c, ts=500.0)
    e = d.ledger_entry

    check("ledger: entry sealed", bool(e.entry_hash))
    check("ledger: opcode recorded", e.opcode == "add", e.opcode)
    check("ledger: op_version recorded", e.op_version == "v3", e.op_version)
    check("ledger: level recorded", e.validation_level == "ORACLE", e.validation_level)
    check("ledger: action recorded", e.action == "emit", e.action)
    check("ledger: ts is caller-stamped", e.ts == 500.0, str(e.ts))
    check("ledger: input/result hashes present",
          bool(e.input_hash) and bool(e.result_hash))

    # Replay: recompute the sealed hash and confirm it matches (tamper-evident).
    check("ledger: entry verifies", e.verify())
    check("ledger: verify_all green", trap.ledger.verify_all())

    # Tamper detection: mutate a field, the seal no longer matches.
    tampered = LedgerEntry(**{k: v for k, v in _entry_fields(e).items()})
    tampered.validation_level = "PROVEN"  # forge a stronger claim
    check("ledger: tamper detected", not tampered.verify())

    # Determinism: same inputs + ts -> identical entry hash (replayable bit-for-bit).
    trap_b = CommitTrap(TrapBudget())
    d_b = trap_b.commit("```python\ndef add(a, b):\n    return a + b\n```", c, ts=500.0)
    check("ledger: replay deterministic", d_b.ledger_entry.entry_hash == e.entry_hash,
          f"{d_b.ledger_entry.entry_hash[:12]} vs {e.entry_hash[:12]}")


def _entry_fields(e: LedgerEntry) -> dict:
    return dict(opcode=e.opcode, op_version=e.op_version, input_hash=e.input_hash,
                result_hash=e.result_hash, contract_hash=e.contract_hash,
                validation_level=e.validation_level, action=e.action, ts=e.ts, entry_hash=e.entry_hash)


def test_ledger_binds_contract():
    """A ledger entry binds the EXACT Contract checked: same opcode/IO/level/action/ts but a different
    contract (e.g. different oracle cases) -> a different entry hash; an identical contract -> identical."""
    L = ProofLedger()
    c_a = Contract(opcode="x", op_version="v1", cases=({"args": [1], "expected": 1},))
    c_b = Contract(opcode="x", op_version="v1", cases=({"args": [2], "expected": 2},))  # differs only in cases
    e1 = L.append(c_a, "IN", "OUT", ValidationLevel.ORACLE, "emit", ts=1.0)
    e2 = L.append(c_b, "IN", "OUT", ValidationLevel.ORACLE, "emit", ts=1.0)
    e3 = L.append(c_a, "IN", "OUT", ValidationLevel.ORACLE, "emit", ts=1.0)            # identical to e1's contract
    check("ledger: contract_hash recorded", bool(e1.contract_hash))
    check("ledger: binds the contract (different cases -> different entry, same opcode/IO/level/action/ts)",
          e1.entry_hash != e2.entry_hash, f"{e1.entry_hash[:12]} vs {e2.entry_hash[:12]}")
    check("ledger: identical contract -> identical entry (replayable)", e1.entry_hash == e3.entry_hash)
    check("ledger: verify_all green with bound contracts", L.verify_all())
    from commit_trap import _canon
    check("ledger: a set inside a contract hashes deterministically (replay guard)",
          _canon({"s": {3, 1, 2}}) == _canon({"s": {1, 2, 3}}))


# ---------------------------------------------------------------------------
# 7. failed-op-disables: an executor crash disarms the opcode for the frame.
# ---------------------------------------------------------------------------
def test_failed_op_disables():
    # We can't easily crash check_cases from valid inputs, so exercise the budget
    # API directly: disabling an op makes subsequent commits reject without work.
    trap = CommitTrap(TrapBudget())
    trap.budget.disable("danger")
    c = Contract(opcode="danger", otype="json", required_keys=["a"])
    d = trap.commit('```json\n{"a": 1}\n```', c, ts=600.0, min_level=ValidationLevel.SCHEMA)
    check("disabled op -> reject", d.action == "reject", d.action)
    check("disabled op -> reason names disable", "op_disabled" in d.reason, d.reason)


# ---------------------------------------------------------------------------
# 8. Honest boundary: parses but unverifiable -> bail (escalate, don't fabricate).
# ---------------------------------------------------------------------------
def test_bail_when_unverifiable():
    trap = CommitTrap(TrapBudget())
    # Text output, no cases, no must_include -> only SYNTAX, below ORACLE bar.
    c = Contract(opcode="freeform", otype="text")
    d = trap.commit("the answer is 42", c, ts=700.0)  # default min_level ORACLE
    check("unverifiable text -> bail", d.action == "bail", d.action)
    check("unverifiable text -> not committed", not d.committed)


def main() -> int:
    print("commit_trap unit tests (CPU-only):")
    test_valid_json_commits_schema()
    test_valid_python_commits_oracle()
    test_invalid_candidate_repairs()
    test_proven_commits()
    test_budget_exhaustion_rejects()
    test_ledger_replayable()
    test_ledger_binds_contract()
    test_failed_op_disables()
    test_bail_when_unverifiable()
    print()
    if _FAILS:
        print(f"RED: {len(_FAILS)} failing checks: {_FAILS}")
        return 1
    print("GREEN: all commit_trap checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
