#!/usr/bin/env python3
"""CPU test for the escalation LADDER orchestration (no model, no venv — stub the model + stub the gate so we
test the tier logic itself): tier order (greedy -> 3 diverse@0.8 -> 3 repairs -> research -> abstain), the abstain
message, web-tier gating, and persistence of gate-proven helpers."""
import sys, os, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kernel.escalate_solve import EscalatingSolver, load_learned
from kernel.self_extend import Gap

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


class StubModel:
    """Returns canned outputs in order (default 'BAD'); records each call's mode so we can assert tier behavior."""
    def __init__(self, outputs): self.outputs = list(outputs); self.calls = []
    def _next(self): return self.outputs.pop(0) if self.outputs else "def f():\n    return 'BAD'"
    def gen_fast(self, p, maxn=1600): self.calls.append(("greedy", None)); return self._next(), 1
    def gen_sample(self, p, temp=0.8, seed=0, maxn=1600): self.calls.append(("sample", (temp, seed))); return self._next(), 1


def _gap():
    return Gap(entry="f", description="d", sample_cases=[], generate_src="def generate(r): return (1,)",
               oracle_src="def oracle(x): return x", n=10, seed=0)


def _solver(outputs, web=False, tmp=None):
    s = EscalatingSolver(StubModel(outputs), web=web, n=10, persist_path=tmp or "/tmp/_le.json")
    # stub the gate: a candidate "passes" iff its code contains GOOD (isolates the LADDER logic from the venv gate)
    s.ext.gate_candidate = lambda gap, code: {
        "passed": ("GOOD" in (code or "")), "reason": "ok" if "GOOD" in (code or "") else "gate_fail",
        "gate": {"passed": gap.n, "n": gap.n}, "regate": {"passed": gap.n}, "feedback": {"first_fail": "novel"}, "code": code}
    s.ext._register = lambda gap, code: {"kind": gap.entry}      # no real scheduler registration in the unit test
    return s


def main():
    nop = lambda *a, **k: None

    # 1. tier 1: greedy passes immediately
    r = _solver(["def f():\n GOOD1"]).solve(_gap(), on=nop)
    ok(r.get("solved") and r["tier"] == 1, f"tier 1: greedy pass -> solved at tier 1 ({r.get('tier')})")

    # 2. tier 2: greedy fails, a diverse temp-0.8 attempt passes (and it's at temp 0.8)
    s = _solver(["BAD", "BAD", "GOOD2", "BAD"]); r = s.solve(_gap(), on=nop)
    ok(r.get("solved") and r["tier"] == 2, f"tier 2: diverse @0.8 pass -> solved at tier 2 ({r.get('tier')})")
    ok(any(m == "sample" and a[0] == 0.8 for m, a in s.model.calls), "tier 2 used temperature 0.8 sampling")

    # 3. tier 3: greedy + 3 diverse fail, a repair passes
    r = _solver(["BAD", "BAD", "BAD", "BAD", "GOOD3"]).solve(_gap(), on=nop)
    ok(r.get("solved") and r["tier"] == 3, f"tier 3: repair pass -> solved at tier 3 ({r.get('tier')})")

    # 4. tier 4: tiers 1-3 fail, web research passes (web enabled, _research stubbed)
    s = _solver(["BAD"] * 7 + ["GOOD4"], web=True); s._research = lambda gap: "RESEARCHED REFS"
    r = s.solve(_gap(), on=nop)
    ok(r.get("solved") and r["tier"] == 4, f"tier 4: research pass -> solved at tier 4 ({r.get('tier')})")

    # 5. tier 5: everything fails + web disabled -> abstain with the exact message
    r = _solver(["BAD"] * 12, web=False).solve(_gap(), on=nop)
    ok((not r.get("solved")) and r.get("abstain") == "I'm sorry, I don't know how to do that.",
       "tier 5: all fail + no web -> abstain 'I'm sorry, I don't know how to do that.'")

    # 6. persistence: a solved helper is written to the registry (+ reusable via load_learned)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "learned.json")
        _solver(["def f():\n GOODP"], tmp=p).solve(_gap(), on=nop)
        reg = load_learned(p)
        ok("f" in reg and reg["f"]["gated_n"] == 10 and "GOODP" in reg["f"]["code"],
           "gate-proven helper PERSISTED to learned_primitives registry (+ load_learned reads it)")

    # 7. web-tier is genuinely skipped when web disabled (no research attempts)
    s = _solver(["BAD"] * 12, web=False); s._research = lambda gap: (_ for _ in ()).throw(AssertionError("research ran!"))
    r = s.solve(_gap(), on=nop)
    ok(not r.get("solved"), "web disabled -> research tier skipped (no _research call), abstains")

    # 8. BUG-SWEEP MEDIUM: load_learned tolerates a malformed/corrupt registry (non-dict top level) -> {}
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.json"); open(bad, "w").write('[1, 2, 3]')      # array, not a dict
        ok(load_learned(bad) == {}, "load_learned on a malformed (non-dict) registry -> {} (no crash/corruption)")
        open(bad, "w").write('{"f": {"code": "x", "gated_n": 1, "tier": 1}}')
        ok(load_learned(bad).get("f", {}).get("code") == "x", "load_learned reads a well-formed registry")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} escalation-ladder checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
