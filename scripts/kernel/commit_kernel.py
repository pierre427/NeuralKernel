#!/usr/bin/env python3
"""Phase-4 M1: the COMMIT TRAP as the only output path of the forward-pass-trapped North decode.

Wires the proven CommitTrap (validate -> emit|repair|bail|reject ladder + ProofLedger + TrapBudget,
commit_trap.py) as the sole commit-to-user channel of the TrappedNorthAdapter. A generated candidate is
validated against its task Contract along syntax < schema < property < ORACLE (check_cases runs the code
against the task's real cases); an ORACLE-clearing answer EMITs, an invalid one is CAUGHT (repair/bail) and
NEVER emitted blindly, and the validation level reached is sealed in the ProofLedger. At T=0 a "repair"
re-decodes with the failure fed back into the prompt (a real input change, not a futile re-roll, per the
no-retries-at-T=0 discipline). This is design §M1: "no token reaches the user without passing the commit trap."

NOTE: the decode goes through run_stack_plane (the trapped path); with plane.armed=False it is token-identical
to production gen_fast (proven ==0 by inv_trap_plane), so the commit trap validates exactly the answer the model
would have emitted — and emits it unchanged when it commits. Speculative KV trim-rollback is M2 (hold/resume);
M1 escalates honestly instead of re-rolling.
"""
from __future__ import annotations
from mlx_lm.models.cache import make_prompt_cache
from commit_trap import CommitTrap, TrapBudget, Contract, ValidationLevel
from kernel.trapped_adapter import TrappedNorthAdapter


def coding_contract(task) -> Contract:
    """Build the per-task Contract from a ladder-style coding task {entry, prompt, cases}.

    The cases are load-bearing: for a code contract, ORACLE is reachable ONLY by check_cases actually executing
    the candidate against them (commit_trap._validate). So "an ORACLE emit == the code ran and passed" holds
    precisely because every coding task carries real cases — a caseless contract caps at PROPERTY, never emits at
    min_level=ORACLE.
    """
    return Contract(opcode=task["entry"], entry=task["entry"], prompt=task.get("prompt"),
                    cases=tuple(task.get("cases", ())))


class CommitKernel(TrappedNorthAdapter):
    """A TrappedNorthAdapter whose output passes through the M1 commit trap (the only output path)."""

    def _decode(self, ids, cache, maxn):
        nxt = self._step_trapped(list(ids), cache); out = []
        for _ in range(maxn):
            if nxt == self.eos:
                break
            out.append(nxt); nxt = self._step_trapped([nxt], cache)
        return self.tok.decode(out)

    def gen_committed(self, task, max_repairs=1, min_level=ValidationLevel.ORACLE, maxn=512, ts=0.0):
        """Decode -> CommitTrap.commit (the only output path). Returns the verdict + the answer it cleared.

        action: emit (cleared min_level), repair/bail (caught invalid — escalated, not emitted), reject (over budget).
        On a non-emit verdict (and budget allowing), re-decode once with the failure fed back into the prompt.
        """
        contract = coding_contract(task)
        # size the budget so the bounded repairs can't be budget-pre-empted (initial + max_repairs commits of
        # the same opcode); otherwise a tight max_same_op would reject a recoverable fail before re-validating it.
        trap = CommitTrap(TrapBudget(max_same_op=max_repairs + 2))
        prompt = task["prompt"]
        answer = self._decode(self.encode_fast([{"role": "user", "content": prompt}]),
                              make_prompt_cache(self.model), maxn)
        d = trap.commit(answer, contract, ts=ts, min_level=min_level)
        attempt = 0
        while d.action != "emit" and attempt < max_repairs:
            attempt += 1
            rp = (prompt + f"\n\n# A previous attempt did NOT pass validation ({d.reason}). "
                  "Return a corrected, complete function.")
            answer = self._decode(self.encode_fast([{"role": "user", "content": rp}]),
                                  make_prompt_cache(self.model), maxn)
            d = trap.commit(answer, contract, ts=ts, min_level=min_level)
        return {"entry": task.get("entry"), "answer": answer, "action": d.action,
                "level": d.validation_level.name, "reason": d.reason, "attempts": attempt,
                "committed": d.action == "emit",
                "proof": d.ledger_entry.entry_hash if d.ledger_entry else None,
                "ledger_verified": trap.ledger.verify_all()}


__all__ = ["CommitKernel", "coding_contract"]
