#!/usr/bin/env python3
"""M1 — the post-last-layer COMMIT TRAP: the only output path to the user.

Design: docs/neural-microkernel-design.md §"M1 — Post-last-layer commit trap",
§"Cross-cutting: validation levels", §5.2 "Trap-budget / anti-runaway rules",
§5.4 "Validation levels ... proof-ledger / replayability".

This module promotes the existing appliance's validate/bail/escalate logic
(`shape_appliance.run_appliance`) into a reusable, always-on trap class. It is the
*single commit-to-user channel*: a candidate cannot reach the user without clearing
`CommitTrap.commit(...)`. The trap

  1. validates the candidate against its Contract along an escalating ladder
     syntax < schema < property < oracle < proven  (ValidationLevel),
     reusing `output_validators.is_complete_valid` (syntax/schema),
     `shape_appliance._struct_eq`/`_contract_matches` (property), and
     `shape_appliance.check_cases` (oracle — SIGALRM-sandboxed execution);
  2. returns a Decision{action: emit|repair|bail|reject, ...} — emit on success,
     repair on a recoverable-but-invalid candidate, bail when validation is too
     weak to trust, reject when the kernel refuses (over budget);
  3. records every commit in a ProofLedger (hash-replayable, caller-stamped ts);
  4. decrements a hard, kernel-owned TrapBudget that REJECTS over budget — the
     anti-runaway invariant (hard counter, not sampling pressure).

Pure-Python, importable, no LLM, no network. `ts` is a caller-stamped field; this
module never calls time/date (so commits are replayable bit-for-bit at temp=0).

Reused production code (cite file:line):
  - output_validators.is_complete_valid          (output_validators.py:81)  -> syntax/schema
  - output_validators.detect_output_type         (output_validators.py:17)
  - shape_appliance.check_cases                   (shape_appliance.py:125)   -> oracle (SIGALRM sandbox)
  - shape_appliance._struct_eq                    (shape_appliance.py:93)    -> property
  - shape_appliance._contract_matches             (shape_appliance.py:74)    -> property (signature)
  - shape_appliance._extract / RECOVERIES         (shape_appliance.py:23,19) -> proven (hold-out-gated)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Optional

# Import the production deterministic plane. The trap REUSES these validators as
# its commit syscall; it does not reimplement them.
from output_validators import is_complete_valid, detect_output_type
import shape_appliance as _appliance
from shape_appliance import check_cases, _struct_eq, _contract_matches, _extract
from shape_recoveries import RECOVERIES


# ---------------------------------------------------------------------------
# Validation ladder (syntax < schema < property < oracle < proven)
# ---------------------------------------------------------------------------
class ValidationLevel(IntEnum):
    """The proof-ledger ladder, weakest -> strongest (design §"validation levels").

    A committed output names the highest level it cleared; ordering is what makes
    one commit stronger evidence than another. NONE = nothing cleared.
    """
    NONE = 0
    SYNTAX = 1     # parses / compiles  (is_complete_valid form check)
    SCHEMA = 2     # required keys / entry symbol present  (is_complete_valid)
    PROPERTY = 3   # structural / signature contract match  (_struct_eq / _contract_matches)
    ORACLE = 4     # passes provided cases in the SIGALRM sandbox  (check_cases)
    PROVEN = 5     # hold-out-gated primitive — correctness is structural, no per-call check


# ---------------------------------------------------------------------------
# Contract: the per-candidate commit request (the typed thing the trap validates)
# ---------------------------------------------------------------------------
@dataclass
class Contract:
    """What the candidate is contracted to satisfy. Mirrors a harness task plus the
    routing identity (opcode/entry + signature) the appliance already keys on.

    opcode        : the op/contract identity (entry name for code; e.g. "to_json").
    entry         : entry symbol the candidate must define (code path); may equal opcode.
    otype         : output type override; inferred from `prompt` when None.
    prompt        : the task prompt (used for type detection + signature/contract match).
    cases         : provided oracle cases [{args|input, expected}, ...] (shape_appliance form).
    required_keys : JSON schema keys (schema level).
    must_include  : substrings the text must contain (schema level for text).
    proven        : if True, the candidate IS a hold-out-gated primitive (PROVEN level).
    op_version    : handler/contract version, recorded in the ledger for replay.
    """
    opcode: str
    entry: Optional[str] = None
    otype: Optional[str] = None
    prompt: Optional[str] = None
    cases: tuple = ()
    required_keys: Optional[list] = None
    must_include: Optional[list] = None
    proven: bool = False
    op_version: str = "v1"

    def __post_init__(self):
        if self.entry is None:
            self.entry = self.opcode


# ---------------------------------------------------------------------------
# Proof ledger (per-commit audit record, hash-replayable)
# ---------------------------------------------------------------------------
def _canon_default(o: Any):
    """Stable fallback for non-JSON values. Sets/frozensets are order-unstable under repr (their iteration
    order varies by hash seed), which would make a ledger hash non-replayable — sort them to a list first."""
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    return repr(o)


def _canon(obj: Any) -> str:
    """Canonical, stable JSON for hashing. Non-JSON values fall back to a deterministic encoder so any
    Python candidate (code str, dict, list, tuple, set) hashes replayably bit-for-bit at temp=0."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_canon_default)
    except (TypeError, ValueError):
        return repr(obj)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


@dataclass
class LedgerEntry:
    """One commit's audit record. `entry_hash` binds all fields so the ledger is
    tamper-evident and replayable: recompute from {opcode, hashes, level, version, ts}
    and compare. `ts` is caller-stamped (this module never reads the clock)."""
    opcode: str
    op_version: str
    input_hash: str
    result_hash: str
    validation_level: str       # ValidationLevel.name
    action: str                 # Decision.action
    contract_hash: str = ""     # binds the FULL Contract checked (entry/cases/required_keys/...) for replay
    ts: Optional[float] = None  # caller-stamped; None until the kernel stamps it
    entry_hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "opcode": self.opcode,
            "op_version": self.op_version,
            "input_hash": self.input_hash,
            "result_hash": self.result_hash,
            "contract_hash": self.contract_hash,
            "validation_level": self.validation_level,
            "action": self.action,
            "ts": self.ts,
        }
        return _sha(payload)

    def seal(self) -> "LedgerEntry":
        self.entry_hash = self.compute_hash()
        return self

    def verify(self) -> bool:
        """Replay gate: True iff the sealed hash matches a recompute of the fields."""
        return bool(self.entry_hash) and self.entry_hash == self.compute_hash()


class ProofLedger:
    """Append-only log of commit entries. One entry per commit (design §M1: "op,
    validation-level reached, pass/fail"). Hash-replayable; no clock access."""

    def __init__(self):
        self.entries: list[LedgerEntry] = []

    def append(
        self,
        contract: Contract,
        input_obj: Any,
        result_obj: Any,
        level: ValidationLevel,
        action: str,
        ts: Optional[float] = None,
    ) -> LedgerEntry:
        e = LedgerEntry(
            opcode=contract.opcode,
            op_version=contract.op_version,
            input_hash=_sha(input_obj),
            result_hash=_sha(result_obj),
            contract_hash=_sha(asdict(contract)),   # bind the exact Contract (entry/cases/required_keys/...) checked
            validation_level=level.name,
            action=action,
            ts=ts,
        ).seal()
        self.entries.append(e)
        return e

    def verify_all(self) -> bool:
        return all(e.verify() for e in self.entries)

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Trap budget (hard, kernel-owned anti-runaway counters)
# ---------------------------------------------------------------------------
class BudgetExceeded(Exception):
    """Raised internally when a charge would exceed a hard cap; surfaced as a
    reject Decision (the trap never throws out of .commit())."""


@dataclass
class TrapBudget:
    """Hard global counters bounding scheduler-mediated work for one request
    (design §5.2: "Budgets are hard, kernel-owned counters ... not sampling pressure").

    max_traps    : total commit attempts charged this request.
    max_same_op  : per-opcode cap (same-op-once generalized to a small N).
    Over budget -> charge() raises BudgetExceeded -> CommitTrap emits a reject.
    `failed-op-disables`: an opcode that errors at the oracle gate is disarmed for
    the rest of the request (design §5.2 row "failed-op-disables")."""
    max_traps: int = 64
    max_same_op: int = 8
    n_traps: int = 0
    per_op: dict = field(default_factory=dict)
    disabled_ops: set = field(default_factory=set)

    def can_charge(self, opcode: str) -> tuple[bool, str]:
        if opcode in self.disabled_ops:
            return False, f"op_disabled:{opcode}"
        if self.n_traps >= self.max_traps:
            return False, f"max_traps:{self.max_traps}"
        if self.per_op.get(opcode, 0) >= self.max_same_op:
            return False, f"max_same_op:{opcode}:{self.max_same_op}"
        return True, ""

    def charge(self, opcode: str) -> None:
        ok, why = self.can_charge(opcode)
        if not ok:
            raise BudgetExceeded(why)
        self.n_traps += 1
        self.per_op[opcode] = self.per_op.get(opcode, 0) + 1

    def disable(self, opcode: str) -> None:
        """failed-op-disables: a handler that errors disarms that opcode for the frame."""
        self.disabled_ops.add(opcode)

    def remaining(self) -> int:
        return max(0, self.max_traps - self.n_traps)


# ---------------------------------------------------------------------------
# Decision (the trap's return value — the sole output object)
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """The trap's verdict. `action` is one of emit|repair|bail|reject:
      emit   — validated; `output` is safe to deliver to the user.
      repair — recoverable but not yet valid; route to a deterministic repair/escalate.
      bail   — validation too weak to trust at temp=0; escalate (don't fabricate).
      reject — kernel refuses (budget exhausted / op disabled). Anti-runaway stop.
    """
    action: str
    output: Any
    validation_level: ValidationLevel
    ledger_entry: Optional[LedgerEntry]
    reason: str = ""

    @property
    def committed(self) -> bool:
        return self.action == "emit"


# ---------------------------------------------------------------------------
# The commit trap
# ---------------------------------------------------------------------------
class CommitTrap:
    """The only output path. `commit(candidate, contract)` validates along the
    escalating ladder and returns a Decision. Every commit is charged against a
    hard TrapBudget and recorded in the ProofLedger.

    Construct with a budget (and optionally a shared ledger). One CommitTrap is one
    request frame: the budget is global to the frame, the anti-runaway invariant.
    """

    def __init__(self, budget: Optional[TrapBudget] = None, ledger: Optional[ProofLedger] = None):
        self.budget = budget if budget is not None else TrapBudget()
        self.ledger = ledger if ledger is not None else ProofLedger()

    # -- validation ladder --------------------------------------------------
    def _validate(self, candidate: Any, contract: Contract) -> tuple[ValidationLevel, str]:
        """Return the highest ValidationLevel `candidate` clears for `contract`,
        plus a short reason. Each rung is gated by an existing production validator.
        """
        # PROVEN: the candidate IS a hold-out-gated primitive (design §"proven").
        # Correctness is structural — but we still confirm it routes by contract
        # identity (entry-name + signature) so a proven claim can't be spoofed.
        if contract.proven:
            rec = RECOVERIES.get(contract.entry)
            # PROVEN requires POSITIVE evidence: a hold-out-gated recovery must actually exist for this
            # entry AND (no prompt given, or its signature matches). Any other case (no recovery, or a
            # signature disagreement) is NOT proven -> fall through to the real validation rungs below.
            if rec is not None and (not contract.prompt or _contract_matches(contract.prompt, rec, contract.entry)):
                return ValidationLevel.PROVEN, "hold-out-gated primitive"

        otype = contract.otype or detect_output_type(contract.prompt or (candidate if isinstance(candidate, str) else ""))

        # SYNTAX/SCHEMA: form-level validity (parses/compiles, keys/entry present).
        text = candidate if isinstance(candidate, str) else _canon(candidate)
        form_ok, reason = is_complete_valid(
            text, otype=otype, prompt=contract.prompt, entry=contract.entry,
            required_keys=contract.required_keys, must_include=contract.must_include,
        )
        if not form_ok:
            return ValidationLevel.NONE, f"form_invalid:{reason}"

        # schema vs syntax: a clearing of required_keys / must_include is a schema-level
        # guarantee; otherwise it is a syntax-level guarantee.
        has_schema = bool(contract.required_keys) or bool(contract.must_include)
        form_level = ValidationLevel.SCHEMA if has_schema else ValidationLevel.SYNTAX

        # For code, only oracle execution is meaningful above schema; without cases we
        # cannot reach ORACLE. PROPERTY for code = signature/contract match.
        if otype == "python":
            code = _extract(text) if isinstance(text, str) else text
            # PROPERTY: signature/contract match (route-on-contract — never regress).
            if contract.prompt is not None:
                rec_code = code if isinstance(code, str) else ""
                if not _contract_matches(contract.prompt, rec_code, contract.entry):
                    return form_level, "signature_mismatch"
            level = max(form_level, ValidationLevel.PROPERTY)
            # ORACLE: run against provided cases in the SIGALRM sandbox. check_cases
            # internally compiles + runs each case under _timed (SIGALRM) and returns
            # False on any exception/timeout/mismatch; a crash of the executor itself
            # (not the candidate) propagates and commit() turns it into op-disable.
            if contract.cases:
                if check_cases(code, contract.entry, list(contract.cases)):
                    return ValidationLevel.ORACLE, "passes provided cases"
                return level, "fails provided cases"
            return level, "signature_ok_no_cases"

        # Non-code (json/xml/text): schema/syntax is the strongest available form gate.
        # PROPERTY for structured data = a structural-equality check against an expected
        # value supplied as the single oracle case.
        if contract.cases:
            for c in contract.cases:
                exp = c.get("expected")
                got = candidate
                if not _struct_eq(got, exp):
                    return form_level, "struct_mismatch"
            return ValidationLevel.ORACLE, "matches expected structure"
        return form_level, "form_ok"

    # -- the only output path ----------------------------------------------
    def commit(self, candidate: Any, contract: Contract, ts: Optional[float] = None,
               min_level: ValidationLevel = ValidationLevel.ORACLE) -> Decision:
        """Validate `candidate` against `contract` and return a Decision.

        `min_level` is the trust threshold for emit (default ORACLE — the design's
        rule that the commit trap "must reach oracle/proven for opcode results, and
        accept the honest boundary (escalate, don't fabricate)" otherwise). For pure
        structured-output contracts with no cases, a caller may lower this to SCHEMA.

        `ts` is caller-stamped and recorded verbatim in the ledger (no clock here).
        """
        # Hard budget gate FIRST — over budget -> reject, before any validation work.
        ok, why = self.budget.can_charge(contract.opcode)
        if not ok:
            entry = self.ledger.append(contract, candidate, None, ValidationLevel.NONE, "reject", ts)
            return Decision("reject", None, ValidationLevel.NONE, entry, reason=f"budget:{why}")
        self.budget.charge(contract.opcode)

        # Validate along the ladder. An executor crash at the oracle gate disarms the op.
        try:
            level, reason = self._validate(candidate, contract)
        except Exception as ex:  # failed-op-disables (design §5.2)
            self.budget.disable(contract.opcode)
            entry = self.ledger.append(contract, candidate, None, ValidationLevel.NONE, "reject", ts)
            return Decision("reject", None, ValidationLevel.NONE, entry,
                            reason=f"handler_error:{type(ex).__name__}")

        # Decide the action from the level reached vs the trust threshold.
        if level >= min_level:
            action = "emit"
            output = candidate
        elif level >= ValidationLevel.PROPERTY:
            # Form + structure are fine but we couldn't reach the trust bar (e.g. no
            # cases, or cases failed): hand to a deterministic repair/escalate path.
            action = "repair"
            output = candidate
        elif level >= ValidationLevel.SYNTAX:
            # Parses but unverified beyond form: honest boundary — escalate, don't emit.
            action = "bail"
            output = candidate
        else:
            action = "repair"   # form-invalid: recoverable by repair/escalation
            output = None

        committed_output = output if action == "emit" else None
        entry = self.ledger.append(contract, candidate, committed_output, level, action, ts)
        return Decision(action, output, level, entry, reason=reason)


__all__ = [
    "CommitTrap",
    "Decision",
    "ProofLedger",
    "LedgerEntry",
    "TrapBudget",
    "BudgetExceeded",
    "ValidationLevel",
    "Contract",
]


if __name__ == "__main__":  # pragma: no cover
    # Smoke demo (full suite lives in test_commit_trap.py).
    trap = CommitTrap(TrapBudget(max_traps=4, max_same_op=2))
    c = Contract(opcode="add", entry="add", prompt="Write a Python function `add`",
                 cases=({"args": [2, 3], "expected": 5},))
    d = trap.commit("```python\ndef add(a, b):\n    return a + b\n```", c, ts=0.0)
    print(d.action, d.validation_level.name, d.reason, "| ledger ok:", trap.ledger.verify_all())
