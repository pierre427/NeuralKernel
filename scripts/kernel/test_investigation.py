#!/usr/bin/env python3
"""CPU-only tests for the structured-investigation protocol (kernel/investigation.py). No LLM.

A deterministic StubModel scripts each stage so we can prove the G->C back-edge: an insufficient round
RE-PLANS, the next round's NEW evidence converges, evidence accumulates+dedups across rounds, and the loop
stops the instant the standard of proof is met. Also covers FRAME routing and enforce_contract shape.

Run:  python3 finetune/gpt-oss-unsloth/scripts/kernel/test_investigation.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import investigation as inv

CHECKS = []
def ok(cond, label):
    CHECKS.append((cond, label)); print(f"  {'OK ' if cond else 'XX '} {label}")


FRAME = {"task_kind": "diagnostic", "question": "which cause?", "decision": "act",
         "standard_of_proof": "balance-of-probabilities", "output_contract": "json"}
HYPS = {"hypotheses": [{"id": "H1", "statement": "cause one", "predicts": "p1"},
                       {"id": "H2", "statement": "cause two", "predicts": "p2"}]}

# evidence the stub source yields; the [C:Hk]/[I:Hk] tags drive the deterministic ASSESS below
AMBIG  = "baseline observation [C:H1 C:H2]"     # fits both hypotheses -> non-diagnostic (a tie)
DISCR  = "smoking-gun observation [C:H1 I:H2]"  # rules H2 out
DISCR2 = "corroborating detail [C:H1 I:H2]"     # a second, distinct discriminator


def _matrix_from_prompt(prompt):
    """Deterministic ASSESS: read the evidence lines and mark C/I per the tags the source embeds."""
    rows = {"H1": {}, "H2": {}}
    for m in re.finditer(r"(E\d+) \([^)]*\): (.+)", prompt):
        eid, text = m.group(1), m.group(2)
        for h in ("H1", "H2"):
            if f"I:{h}" in text:   rows[h][eid] = "I"
            elif f"C:{h}" in text: rows[h][eid] = "C"
    return {"matrix": rows}


class StubModel:
    """Scripts each stage by keyword. `r1_queries` = what PLAN returns on round 1; a re-plan (the PLAN prompt
    then contains 'ALREADY COLLECTED') returns the discriminator query 'find B'."""
    def __init__(self, r1_queries, frame=None):
        self.r1_queries = r1_queries
        self.frame = frame or FRAME

    def gen_fast(self, prompt, maxn=0, **kw):
        if "Stage FRAME" in prompt:        return (json.dumps(self.frame),)
        if "Stage HYPOTHESIZE" in prompt:  return (json.dumps(HYPS),)
        if "Stage PLAN" in prompt:
            qs = ["find B"] if "ALREADY COLLECTED" in prompt else self.r1_queries
            return (json.dumps({"queries": qs}),)
        if "Stage ASSESS" in prompt:       return (json.dumps(_matrix_from_prompt(prompt)),)
        if "Stage COLLECT" in prompt:      return (json.dumps({"facts": ["stray fact"]}),)
        if "Stage REPORT" in prompt:       return (json.dumps({"answer": "cause one", "confidence": 0.7}),)
        return ("{}",)   # _prescriptive_answer / fallbacks


class StubSource(inv.EvidenceSource):
    def __init__(self, table): self.table = table
    def manifest(self): return "stub"
    def collect(self, query): return self.table.get(query, "")
    def reliability(self, t): return "A1"


def test_backedge_fires_on_insufficiency():
    print("\n[1] back-edge G->C: an insufficient round 1 re-plans; round 2's new evidence converges")
    model = StubModel(r1_queries=["find A"])
    src = StubSource({"find A": AMBIG, "find B": DISCR})
    rep, st = inv.investigate(model, src, "which cause?", max_rounds=2)
    ok(st.rounds == 2, f"ran a SECOND round (back-edge fired); rounds={st.rounds}")
    ok(len(st.evidence) == 2, f"evidence accumulated across rounds (E1+E2); got {len(st.evidence)}")
    ok(st.sufficiency.get("sufficient"), "round 2 reached sufficiency (corroborated + margin)")
    ok(st.ranking[:1] == ["H1"], f"H1 leads after the discriminator; ranking={st.ranking}")


def test_sufficient_round1_skips_backedge():
    print("\n[2] a round that already meets the bar does NOT re-plan (back-edge skipped)")
    model = StubModel(r1_queries=["find B", "find C"])
    src = StubSource({"find B": DISCR, "find C": DISCR2})
    rep, st = inv.investigate(model, src, "which cause?", max_rounds=2)
    ok(st.rounds == 1, f"stopped after round 1 (sufficient); rounds={st.rounds}")
    ok(st.sufficiency.get("sufficient"), "round 1 was sufficient")


def test_collect_dedups_across_rounds():
    print("\n[3] _collect skips evidence already gathered (a re-plan adds only NEW evidence)")
    src = StubSource({"q": DISCR})
    first = inv._collect(StubModel([]), src, ["q"], "?", seen=set())
    again = inv._collect(StubModel([]), src, ["q"], "?", seen={DISCR})
    ok(len(first) == 1 and len(again) == 0, f"identical hit deduped; first={len(first)} again={len(again)}")


def test_frame_routes_prescriptive_away():
    print("\n[4] a prescriptive task is routed AWAY from ACH (the hypotheses loop never runs)")
    model = StubModel(r1_queries=["x"], frame=dict(FRAME, task_kind="prescriptive"))
    rep, st = inv.investigate(model, StubSource({}), "list the standard families", max_rounds=2)
    ok(st.frame.get("routed") == "prescriptive", f"frame routed=prescriptive; got {st.frame.get('routed')}")
    ok(st.rounds == 0, f"the ACH round loop never ran; rounds={st.rounds}")


def test_enforce_contract_guarantees_shape():
    print("\n[5] enforce_contract guarantees required keys, non-empty arrays, computed confidence")
    iv = inv.Investigation(question="q")
    iv.hypotheses = [{"id": "H1", "statement": "lead"}, {"id": "H2", "statement": "alt"}]
    iv.ranking = ["H1"]; iv.matrix = {"H1": {"E1": "C"}}
    iv.evidence = [{"id": "E1", "text": "a fact"}]
    iv.sufficiency = {"confidence": 0.72, "leader": "lead", "missing": None}
    spec = {"keys": ["answer", "confidence", "evidence", "alternatives", "uncertainties"],
            "array_keys": ["evidence", "alternatives", "uncertainties"]}
    out = json.loads(inv.enforce_contract('{"answer":"lead"}', iv, spec))
    ok(all(k in out for k in spec["keys"]), "all required keys present")
    ok(all(isinstance(out[k], list) and out[k] for k in spec["array_keys"]), "array keys are non-empty lists")
    ok(out["confidence"] == 0.72, f"confidence is the computed structural value; got {out['confidence']}")
    # with NO investigation state, array keys default to the honest 'insufficient evidence'
    out2 = json.loads(inv.enforce_contract("no json here", inv.Investigation(question="q"), spec))
    ok(any(out2[k] == ["insufficient evidence"] for k in spec["array_keys"]),
       "no-state path defaults under-evidenced arrays to the honest 'insufficient evidence'")


if __name__ == "__main__":
    test_backedge_fires_on_insufficiency()
    test_sufficient_round1_skips_backedge()
    test_collect_dedups_across_rounds()
    test_frame_routes_prescriptive_away()
    test_enforce_contract_guarantees_shape()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} investigation checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
