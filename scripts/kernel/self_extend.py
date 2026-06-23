#!/usr/bin/env python3
"""kernel/self_extend.py — Step 3: the SELF-EXTENDING pipeline (the MVP loop).

When the kernel lacks a deterministic primitive for a case, it: SYNTHESIZES a candidate (the model proposes code),
SANDBOX-TESTS it on cheap sample cases (venv_exec), GATES it against an INDEPENDENT oracle on 1000s of novel
instances (the hold-out gate, run inside the sandbox), runs a PLACEBO control (the gate must reject a deliberately-
wrong impl, else the oracle isn't discriminating), and only then REGISTERS it on the scheduler as a callable,
PROVEN primitive. Every untrusted run is out-of-process; nothing is trusted until it clears the gate + placebo.

This assembles already-verified parts: holdout_gate (the trust rule), venv_tool (sandboxed exec/gate/call),
task_graph.Scheduler (fail-closed registration). The `synthesize` callable is the model (or a stub for tests) —
gap -> code string. TRUST BOUNDARY: the GAP supplies the generator + INDEPENDENT oracle (the source of truth);
the model supplies only the CANDIDATE. (The harder variant — model proposes its own oracle — is the open
"independent-oracle crux"; not the MVP.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from .venv_tool import run_candidate, run_gate_in_venv, call_in_venv
    from .commit_trap import ValidationLevel
except ImportError:
    from venv_tool import run_candidate, run_gate_in_venv, call_in_venv
    from commit_trap import ValidationLevel

PLACEBO_TMPL = "def {entry}(*args, **kwargs):\n    return None\n"   # a deliberately-wrong impl the gate must reject


@dataclass
class Gap:
    """A capability gap: what to synthesize + how to gate it. The generator + oracle are the kernel's trusted
    source of truth (a DIFFERENT-algorithm reference); the model only fills `code`."""
    entry: str                                          # the function name to synthesize
    description: str = ""                               # for the synthesizer prompt
    signature: str = ""
    sample_cases: list = field(default_factory=list)    # cheap pre-filter [{args|input, expected}]
    generate_src: str = ""                              # code defining generate(rng) -> instance (tuple)
    oracle_src: str = ""                                # code defining oracle(*inst) -> truth (INDEPENDENT)
    n: int = 2000
    seed: int = 0


class SelfExtender:
    def __init__(self, sched, synthesize: Callable, *, max_attempts: int = 3, placebo_n: int = 300,
                 cap_prefix: str = "prim", grant: bool = True):
        self.sched = sched
        self.synthesize = synthesize                    # synthesize(gap, feedback) -> code string
        self.max_attempts = max_attempts
        self.placebo_n = placebo_n
        self.cap_prefix = cap_prefix
        self.grant = grant
        self.registry: dict = {}                        # entry -> trusted code string (what this extender admitted)

    def gate_candidate(self, gap: Gap, code: str) -> dict:
        """Full per-candidate gating (shared by extend() AND the escalation ladder) — NO registration. Returns
        {passed, reason, sample, gate, regate, placebo_passed, feedback, code}. Stages: cheap sandbox sample-test
        -> hold-out gate (n novel vs independent oracle) -> fail-closed-if-unsandboxed -> 2nd-seed determinism
        re-gate -> placebo (gate must reject a wrong impl). passed=True only if ALL clear."""
        d = {"code": code, "passed": False}
        if not code:
            return {**d, "reason": "no_code", "feedback": {"stage": "synthesize", "error": "empty"}}
        vt = run_candidate(code, gap.entry, gap.sample_cases)
        d["sample"] = {"ok": vt.get("ok"), "passed": vt.get("passed"), "total": vt.get("total")}
        if not vt.get("ok"):
            return {**d, "reason": "sample_fail",
                    "feedback": {"stage": "sample_cases", "fails": vt.get("fails"), "error": vt.get("error")}}
        g = run_gate_in_venv(code, gap.entry, gap.generate_src, gap.oracle_src, n=gap.n, seed=gap.seed)
        d["gate"] = {"ok": g.get("ok"), "passed": g.get("passed"), "n": g.get("n")}
        if not g.get("ok"):
            return {**d, "reason": "gate_fail",
                    "feedback": {"stage": "holdout_gate", "first_fail": g.get("first_fail"), "error": g.get("error")}}
        if g.get("sandboxed_nonet") is not True:
            return {**d, "reason": "unsandboxed"}            # fail-closed: don't promote a non-sandboxed run
        seed2 = (gap.seed ^ 0x9E3779B9) or 1
        g2 = run_gate_in_venv(code, gap.entry, gap.generate_src, gap.oracle_src, n=gap.n, seed=seed2)
        d["regate"] = {"ok": g2.get("ok"), "passed": g2.get("passed"), "seed2": seed2}
        if not g2.get("ok"):
            return {**d, "reason": "regate_fail",
                    "feedback": {"stage": "determinism_regate", "first_fail": g2.get("first_fail")}}
        pl = run_gate_in_venv(PLACEBO_TMPL.format(entry=gap.entry), gap.entry, gap.generate_src,
                              gap.oracle_src, n=self.placebo_n, seed=gap.seed)
        d["placebo_passed"] = bool(pl.get("ok"))
        if pl.get("ok"):
            return {**d, "reason": "placebo_unsound"}        # gate not discriminating -> a pass proves nothing
        return {**d, "passed": True, "reason": "ok"}

    def extend(self, gap: Gap) -> dict:
        """Synthesize -> gate_candidate -> register, up to max_attempts (the model iterates on the gate feedback)."""
        trace = {"entry": gap.entry, "attempts": []}
        feedback = None
        for attempt in range(1, self.max_attempts + 1):
            code = self.synthesize(gap, feedback)
            r = self.gate_candidate(gap, code)
            a = {"attempt": attempt, "result": "registered" if r["passed"] else r["reason"],
                 "sample": r.get("sample"), "gate": r.get("gate"), "regate": r.get("regate"),
                 "placebo_passed": r.get("placebo_passed")}
            trace["attempts"].append(a)
            if r["passed"]:
                trace.update({"registered": True, "wiring": self._register(gap, code), "code": code,
                              "gate_passed": r["gate"]["passed"]})
                return trace
            if r["reason"] in ("unsandboxed", "placebo_unsound"):
                trace.update({"registered": False, "reason": (
                    "gate run was not sandboxed (no-net unavailable) — refusing to register"
                    if r["reason"] == "unsandboxed" else "gate not discriminating (placebo passed)")})
                return trace
            feedback = r.get("feedback")
        trace.update({"registered": False, "reason": f"no candidate passed in {self.max_attempts} attempts"})
        return trace

    def _register(self, gap: Gap, code: str) -> dict:
        """Install the gate-passed primitive as a scheduler executor (dispatch sandboxed via call_in_venv) + a
        result validator + grant its capability. A proven primitive's executor returns ValidationLevel.PROVEN."""
        entry, cap = gap.entry, f"{self.cap_prefix}:{gap.entry}"
        self.registry[entry] = code

        def _exec(sched, task, inputs):
            r = call_in_venv(code, entry, (inputs or {}).get("args", []))
            return (r, bool(r.get("ok")), ValidationLevel.PROVEN if r.get("ok") else ValidationLevel.NONE)

        self.sched.register_executor(entry, _exec)
        self.sched.register_validator(f"{entry}_ok", lambda result, task: isinstance(result, dict) and result.get("ok") is True)
        if self.grant:
            self.sched.capabilities.add(cap)
        try:                                              # make the gate-passed primitive ROUTABLE by the Step-4
            try:                                          # lexical proposer (synthesize -> register -> reuse next time)
                from .lexical_proposer import register_proven
            except ImportError:
                from lexical_proposer import register_proven
            register_proven(entry, gap.description, gap.signature)
        except Exception:
            pass                                          # proposer is optional; indexing must never fail the gate
        return {"kind": entry, "validator": f"{entry}_ok", "capability": cap, "granted": self.grant}

    def call(self, entry: str, args) -> dict:
        """Call a registered primitive on args (sandboxed). Returns {ok, result}."""
        if entry not in self.registry:
            return {"ok": False, "error": f"no registered primitive {entry!r}"}
        return call_in_venv(self.registry[entry], entry, args)


__all__ = ["Gap", "SelfExtender", "PLACEBO_TMPL"]
