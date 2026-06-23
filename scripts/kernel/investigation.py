#!/usr/bin/env python3
"""Domain-agnostic STRUCTURED-INVESTIGATION protocol — the common reasoning loop that detectives,
diagnosticians, intelligence analysts, scientists, and auditors all share. Only the EvidenceSource is
domain-specific; the protocol is universal.

Grounded in: Analysis of Competing Hypotheses (Heuer/CIA), medical hypothetico-deductive differential
diagnosis, the criminal "investigative mindset" + ABC (Assume/Believe/Challenge nothing), the scientific
method (falsification), audit evidence sufficiency+appropriateness, and source-reliability grading
(Admiralty A-F/1-6, GRADE).

The 8 stages (a state machine with one back-edge G->C):
  A FRAME       fix the question, the decision it serves, and the STANDARD OF PROOF
  B HYPOTHESIZE the differential — competing answers incl. the null/benign + any can't-miss
  C PLAN        what evidence would DISCRIMINATE the hypotheses (diagnosticity), what counts as reliable
  D COLLECT     gather via the domain EvidenceSource, recording provenance
  E ASSESS      ACH matrix: score each evidence x hypothesis Consistent/Inconsistent/NA, reliability-weighted
  F CONVERGE    rank by LEAST inconsistency (not most confirmation); require corroboration; rule out
  G SUFFICIENCY is the proof standard met / residual uncertainty acceptable? else loop back to C
  H REPORT      conclusion + evidence chain + alternatives ruled out + residual uncertainty + calibrated confidence

Anti-bias disciplines baked in (recur across all the source fields):
  1. seek DISCONFIRMATION, not confirmation (rank by fewest inconsistencies)
  2. NO single-source conclusion — require corroboration
  3. reliability-weight every input (provenance before use)
  4. hold the differential OPEN (anti premature-closure/anchoring; keep null + can't-miss alive)
  5. match effort to the STANDARD OF PROOF
  6. surface assumptions ("assume nothing")
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field


def _aslist(o, key):
    """Robust to weak-model shape drift: accept {key:[...]}, a bare [...], or any dict with a list value."""
    if isinstance(o, list):
        return o
    if isinstance(o, dict):
        v = o.get(key)
        if isinstance(v, list):
            return v
        for vv in o.values():
            if isinstance(vv, list):
                return vv
    return []


def _xjson(s: str):
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", s or "", re.I)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    a, b = (s or "").find("{"), (s or "").rfind("}")
    if 0 <= a < b:
        try: return json.loads(s[a:b + 1])
        except Exception: pass
    a, b = (s or "").find("["), (s or "").rfind("]")
    if 0 <= a < b:
        try: return json.loads(s[a:b + 1])
        except Exception: pass
    return None


# ---------------------------------------------------------------------------
# The ONLY domain-specific seam. Subclass per domain; the protocol never changes.
# ---------------------------------------------------------------------------
class EvidenceSource:
    """A read-only evidence surface. cyber -> fixture grep/read/det; clinical -> chart/labs; the prompt
    itself is a valid source when no external corpus exists (then COLLECT just qualifies stated facts)."""
    def manifest(self) -> str:
        """One-line description of what is available to collect."""
        return "(no external evidence corpus; the prompt's stated facts are the evidence)"

    def collect(self, query: str) -> str:
        """Gather evidence relevant to a query. Returns text (cite-able). Default: nothing external."""
        return ""

    def reliability(self, evidence_text: str) -> str:
        """Admiralty-style grade for a piece of evidence. Default: retrieved-from-source is reliable."""
        return "B2"  # usually reliable / probably true


@dataclass
class Investigation:
    question: str
    frame: dict = field(default_factory=dict)
    hypotheses: list = field(default_factory=list)      # [{id, statement, predicts}]
    evidence: list = field(default_factory=list)         # [{id, text, source, reliability}]
    matrix: dict = field(default_factory=dict)           # {hyp_id: {ev_id: 'C'|'I'|'NA'}}
    ranking: list = field(default_factory=list)          # hyp_ids best-first
    leader: dict = field(default_factory=dict)
    sufficiency: dict = field(default_factory=dict)
    rounds: int = 0


# ---------------------------------------------------------------------------
# Stages (each is one focused model call; model = any adapter with .gen_fast)
# ---------------------------------------------------------------------------
def _frame(model, question, contract_hint=""):
    p = ("Stage FRAME of a structured investigation. Do NOT answer yet. Classify the task and state the bar.\n"
         "task_kind is one of:\n"
         "  diagnostic   = decide WHICH explanation/answer is true among competing possibilities (whodunit, "
         "what-happened, differential diagnosis, root cause) — the investigation protocol applies.\n"
         "  prescriptive = produce/enumerate required items (list the standard X, write a detection/plan) — "
         "completeness matters, not competing hypotheses.\n"
         "  computational = a determinate calculation/transform with one correct output.\n"
         "JSON:\n"
         '{"task_kind":"diagnostic|prescriptive|computational",'
         ' "question":"<the exact question>",'
         ' "decision":"<what the answer serves>",'
         ' "standard_of_proof":"<triage-hunch | balance-of-probabilities | beyond-reasonable-doubt>",'
         ' "output_contract":"<exact output shape/tokens demanded>"}\n\n'
         f"REQUEST:\n{question}\n{contract_hint}")
    return _xjson(model.gen_fast(p, maxn=400)[0]) or {}


def _prescriptive_answer(model, question, source, contract_hint):
    """Non-diagnostic path: enumerate the STANDARD complete set across all distinct dimensions, grounded in
    any collected evidence, with canonical terms and no fabrication. (The completeness discipline, not ACH.)"""
    ev = source.collect(question) if hasattr(source, "collect") else ""
    eb = f"\n\nEVIDENCE:\n{ev}" if ev and ev.strip() else ""
    p = (f"REQUEST:\n{question}{eb}\n\nProduce the answer. If asked to name N items or the families/categories, "
         "enumerate the STANDARD complete set spanning every distinct dimension (do not give several flavors of "
         "one dimension). Use the plain canonical term for each. Do not fabricate identifiers. "
         f"Conform exactly to the output contract.\nOUTPUT CONTRACT: {contract_hint}")
    return model.gen_fast(p, maxn=900)[0]


def _hypothesize(model, question, frame):
    p = ("Stage HYPOTHESIZE (the differential). Before examining evidence, enumerate the COMPETING answers/"
         "explanations for the question. Rules: make them mutually distinct; INCLUDE the benign/null one and "
         "any high-cost 'can't-miss' one; for each note what it would PREDICT (its tell-tale signature). JSON:\n"
         '{"hypotheses":[{"id":"H1","statement":"...","predicts":"..."}]}\n\n'
         f"QUESTION: {frame.get('question', question)}")
    o = _xjson(model.gen_fast(p, maxn=600)[0])
    hs = [h for h in _aslist(o, "hypotheses") if isinstance(h, dict)]
    for i, h in enumerate(hs):
        h.setdefault("id", f"H{i+1}"); h.setdefault("statement", str(h))
    return hs


def _plan(model, question, hyps, have=None):
    """PLAN the next collection. On a re-plan (back-edge G->C) `have`=(already_collected, still_tied) so the
    round seeks NEW discriminators instead of re-requesting evidence that already failed to separate the leaders."""
    hb = "\n".join(f"  {h['id']}: {h['statement']} (predicts: {h.get('predicts','')})" for h in hyps)
    hav = ""
    if have:
        collected, tied = have
        hav = ("\n\nALREADY COLLECTED (do NOT re-request these — they did NOT separate the leaders):\n"
               + "\n".join(f"  - {c}" for c in collected)
               + (f"\nSTILL AMBIGUOUS between: {tied}" if tied else "")
               + "\nPlan NEW evidence that would DISCRIMINATE the still-ambiguous hypotheses.")
    p = ("Stage PLAN. Choose evidence to collect, prioritizing DISCRIMINATORS — evidence that would SEPARATE "
         "the hypotheses (especially evidence that could DISCONFIRM the leading ones), not merely confirm a "
         "favorite. List concrete collection queries. JSON:\n"
         '{"queries":["<specific thing to look for / retrieve>"], "most_discriminating":"<the single pivotal one>"}\n\n'
         f"QUESTION: {question}\nHYPOTHESES:\n{hb}{hav}")
    o = _xjson(model.gen_fast(p, maxn=500)[0])
    qs = [q.get("query", q) if isinstance(q, dict) else q for q in _aslist(o, "queries")]
    return [str(q) for q in qs][:6]


def _collect(model, source: EvidenceSource, queries, question, seen=None):
    """COLLECT via the domain source. If the source returns nothing (no external corpus), fall back to
    qualifying the facts STATED in the prompt (ABC: record only what's actually there, no fabrication).
    `seen` = evidence text already gathered in prior rounds; identical hits are skipped so a re-plan
    (back-edge) adds only NEW evidence and the prompt-fact fallback never re-extracts what we already have."""
    seen = set(seen or ())
    ev = []
    for q in queries:
        got = source.collect(q)
        if got and got.strip():
            txt = got.strip()[:700]
            if txt in seen:                       # already collected in a prior round — don't double-count
                continue
            seen.add(txt)
            ev.append({"id": f"E{len(ev)+1}", "text": txt, "source": "collected", "reliability": source.reliability(got)})
    if not ev and not seen:
        # no external corpus AND nothing gathered yet -> the prompt's stated facts ARE the evidence
        # (qualify, don't invent). Guarded by `not seen` so a later round never re-extracts the same facts.
        p = ("Stage COLLECT (no external corpus). Extract ONLY the concrete facts STATED in the request as "
             "discrete evidence items. Do NOT invent identifiers/values. JSON: {\"facts\":[\"...\"]}\n\n"
             f"REQUEST:\n{question}")
        o = _xjson(model.gen_fast(p, maxn=500)[0])
        for f in _aslist(o, "facts")[:10]:
            ev.append({"id": f"E{len(ev)+1}", "text": str(f)[:300], "source": "prompt", "reliability": "A1"})
    return ev


def _assess(model, hyps, evidence):
    """ACH matrix: score each (evidence, hypothesis) Consistent/Inconsistent/NA. The core discipline."""
    hb = "\n".join(f"  {h['id']}: {h['statement']}" for h in hyps)
    eb = "\n".join(f"  {e['id']} ({e['reliability']}): {e['text']}" for e in evidence)
    p = ("Stage ASSESS — Analysis of Competing Hypotheses. For EACH evidence item against EACH hypothesis, mark "
         "C (consistent), I (inconsistent), or NA (not applicable). Be strict: an item that fits ALL hypotheses "
         "is non-diagnostic; look hard for INCONSISTENCIES that would rule a hypothesis out. JSON:\n"
         '{"matrix":{"H1":{"E1":"C","E2":"I"}, ...}}\n\n'
         f"HYPOTHESES:\n{hb}\n\nEVIDENCE:\n{eb}")
    o = _xjson(model.gen_fast(p, maxn=700)[0]) or {}
    if isinstance(o, dict):
        m = o.get("matrix")
        if isinstance(m, dict):
            return m
        if o and all(isinstance(v, dict) for v in o.values()):  # model returned the bare matrix
            return o
    return {}


def _converge(hyps, evidence, matrix):
    """Deterministic CONVERGE: rank by FEWEST inconsistencies (Heuer); corroboration = >=2 consistent items."""
    rel_w = {"A": 1.0, "B": 0.85, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.1}
    scores = []
    for h in hyps:
        row = matrix.get(h["id"], {})
        inc = sum(1 for v in row.values() if str(v).upper().startswith("I"))
        con = [ek for ek, v in row.items() if str(v).upper().startswith("C")]
        # weight corroboration by source reliability
        relmap = {e["id"]: rel_w.get(str(e.get("reliability", "B"))[0].upper(), 0.6) for e in evidence}
        corrob = sum(relmap.get(ek, 0.6) for ek in con)
        scores.append({"id": h["id"], "statement": h["statement"], "inconsistencies": inc,
                       "consistent": con, "corroboration": round(corrob, 2)})
    # rank: fewest inconsistencies first, then most corroboration
    scores.sort(key=lambda s: (s["inconsistencies"], -s["corroboration"]))
    ranking = [s["id"] for s in scores]
    leader = scores[0] if scores else {}
    leader = dict(leader)
    leader["corroborated"] = leader.get("corroboration", 0) >= 1.5  # >= ~2 reliable consistent items
    return ranking, leader, scores


def _sufficiency(model, frame, leader, scores):
    """SUFFICIENCY gate vs the standard of proof. Returns dict with sufficient + confidence + what's missing."""
    margin = 0
    if len(scores) >= 2:
        margin = scores[1]["inconsistencies"] - scores[0]["inconsistencies"]
    standard = (frame.get("standard_of_proof") or "balance-of-probabilities").lower()
    need = {"triage-hunch": 0, "balance-of-probabilities": 1, "beyond-reasonable-doubt": 2}.get(standard, 1)
    sufficient = bool(leader) and leader.get("corroborated", False) and margin >= need
    # confidence from the structural state (calibrated to the investigation, not a vibe)
    conf = 0.0
    if leader:
        conf = min(1.0, 0.4 + 0.15 * margin + (0.3 if leader.get("corroborated") else 0.0))
    return {"sufficient": sufficient, "margin": margin, "standard": standard, "confidence": round(conf, 2),
            "leader": leader.get("statement", ""), "missing": None if sufficient else "more discriminating/corroborating evidence"}


def _report(model, question, frame, scores, evidence, suff, contract_hint=""):
    elim = "; ".join(f"{s['statement']} (ruled out: {s['inconsistencies']} inconsistencies)"
                     for s in scores[1:] if s["inconsistencies"] > scores[0]["inconsistencies"]) or "none firmly eliminated"
    chain = "; ".join(e["text"][:120] for e in evidence[:6])
    p = ("Stage REPORT. Produce the final answer. Lead with the surviving hypothesis as the conclusion; ground it "
         "in the evidence chain; list the alternatives you ruled out and why; state residual uncertainty. Conform "
         "EXACTLY to the output contract.\n"
         f"OUTPUT CONTRACT: {frame.get('output_contract','')}\n{contract_hint}\n"
         f"CONCLUSION (leading hypothesis): {suff.get('leader','')}\n"
         f"EVIDENCE CHAIN: {chain}\n"
         f"ALTERNATIVES RULED OUT: {elim}\n"
         f"RESIDUAL UNCERTAINTY: {suff.get('missing') or 'acceptable for the standard of proof'}\n"
         f"CALIBRATED CONFIDENCE: {suff.get('confidence')}\n\n"
         f"QUESTION:\n{question}")
    return model.gen_fast(p, maxn=900)[0]


def enforce_contract(answer: str, inv: "Investigation", spec: dict) -> str:
    """Output discipline: GUARANTEE the result is valid JSON honoring `spec` — every required key present,
    every array_key NON-EMPTY, confidence = the COMPUTED structural value. Gaps are filled from the
    investigation STATE (alternatives <- ruled-out hypotheses, evidence/known_facts <- collected evidence,
    uncertainties <- residual, next_steps <- plan), i.e. assembled from work actually done — NOT invented
    answers. The model's reasoned content is preserved; the kernel only guarantees the shape.

    spec = {"keys": [...all required keys...], "array_keys": [...keys that must be non-empty lists...]}.
    """
    obj = _xjson(answer)
    if not isinstance(obj, dict):
        obj = {"answer": (answer or "").strip()[:1400]}
    keys = spec.get("keys", [])
    arr = spec.get("array_keys", [])
    leader = inv.sufficiency.get("leader") or (inv.leader.get("statement") if inv.leader else "")
    ev = [e.get("text", "") for e in inv.evidence if e.get("text")]
    hyp = [h.get("statement", "") for h in inv.hypotheses if h.get("statement")]
    alts = hyp[1:] if len(hyp) > 1 else hyp                      # the non-leading hypotheses = the alternatives
    lead_row = inv.matrix.get(inv.ranking[0], {}) if inv.ranking else {}
    links = [f"{ek} supports the leading hypothesis" for ek, v in lead_row.items() if str(v).upper().startswith("C")]
    resid = inv.sufficiency.get("missing")
    fill = {
        "answer": obj.get("answer") or leader or (answer or "").strip()[:400],
        "confidence": inv.sufficiency.get("confidence", obj.get("confidence", 0.5)),
        "known_facts": ev[:8],
        "evidence": ev[:8],
        "inferred_links": links or (["the leading hypothesis best fits the evidence"] if leader else []),
        "alternatives": alts[:6],
        "uncertainties": ([resid] if resid else ["residual uncertainty acceptable for the standard of proof"]),
        "next_steps": ["corroborate with an independent source", "collect the most discriminating missing evidence"],
    }
    if "confidence" in keys:
        obj["confidence"] = fill["confidence"]                   # always the computed value, not the vibe
    for k in keys:
        if k == "confidence":
            continue
        if k not in obj or obj.get(k) in (None, "", [], {}):
            obj[k] = fill.get(k, "" if k not in arr else [])
    for k in arr:                                                # final guarantee: non-empty list
        v = obj.get(k)
        if not isinstance(v, list) or len(v) == 0:
            obj[k] = fill.get(k) or ["insufficient evidence"]
    return json.dumps(obj)


def investigate(model, source: EvidenceSource, question: str, contract_hint: str = "",
                max_rounds: int = 2, contract_spec: dict = None, log=lambda *a: None) -> tuple:
    """Run the full protocol. Returns (final_answer_text, Investigation state). If contract_spec is given,
    the final answer is run through enforce_contract() to guarantee the required JSON shape."""
    def _final(ans):
        return (enforce_contract(ans, inv, contract_spec) if contract_spec else ans), inv

    inv = Investigation(question=question)
    inv.frame = _frame(model, question, contract_hint)
    kind = (inv.frame.get("task_kind") or "diagnostic").lower()
    log("frame", {"kind": kind, "standard": inv.frame.get("standard_of_proof")})
    if kind.startswith("prescriptive") or kind.startswith("comput"):
        # ROUTE AWAY: ACH/differential mis-frames non-diagnostic tasks. Use the completeness discipline.
        inv.frame["routed"] = kind
        return _final(_prescriptive_answer(model, question, source, contract_hint))
    inv.hypotheses = _hypothesize(model, question, inv.frame)
    log("hypotheses", [h["id"] + ":" + h["statement"][:50] for h in inv.hypotheses])
    if len(inv.hypotheses) < 2:
        # thin differential -> not a genuine competing-hypotheses problem; don't force empty ACH
        inv.frame["routed"] = "thin-differential"
        return _final(_prescriptive_answer(model, question, source, contract_hint))
    scores = []
    for rnd in range(max_rounds):
        inv.rounds = rnd + 1
        # back-edge G->C: a re-plan tells PLAN what we already have + which hypotheses are still tied, so it
        # seeks NEW discriminating evidence rather than re-collecting what already failed to separate them.
        have = None
        if rnd > 0 and scores:
            tied = ", ".join(s["statement"][:40] for s in scores[:2])
            have = ([e["text"][:120] for e in inv.evidence], tied)
        queries = _plan(model, question, inv.hypotheses, have=have)
        new = _collect(model, source, queries, question, seen={e["text"] for e in inv.evidence})
        # accumulate evidence across rounds (re-id to stay unique)
        for e in new:
            e["id"] = f"E{len(inv.evidence)+1}"; inv.evidence.append(e)
        inv.matrix = _assess(model, inv.hypotheses, inv.evidence)
        inv.ranking, inv.leader, scores = _converge(inv.hypotheses, inv.evidence, inv.matrix)
        inv.sufficiency = _sufficiency(model, inv.frame, inv.leader, scores)
        log(f"round{rnd+1}", {"leader": inv.leader.get("statement", "")[:50],
                              "inc": inv.leader.get("inconsistencies"), "suff": inv.sufficiency["sufficient"]})
        if inv.sufficiency["sufficient"]:
            break  # standard of proof met -> close (back-edge skipped)
    inv._scores = scores
    report = _report(model, question, inv.frame, scores, inv.evidence, inv.sufficiency, contract_hint)
    return _final(report)
