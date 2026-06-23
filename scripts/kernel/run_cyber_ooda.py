#!/usr/bin/env python3
"""OODA flow on the Job Kernel — Observe, Orient, Decide, Act. Generalized intent-driven planning, NOT
per-task skills. The model reads the prose, extracts intent + the output contract (which the ask usually
states), the planner decides the solution paths + checks alignment, then we act conforming to the
extracted contract. det-blocks/det-primitives handle the deterministic sub-problems; the model handles
intent + judgment. This is general (any request); cyber is one instantiation.

  Observe : scheduler — no det-block solves the whole task -> prose path
  Orient  : [agent:orient]   model extracts {output_contract, deliverables, must_cite} from the request
  Decide  : [scheduler]      build the plan + ALIGN-check (does the plan cover the ask's deliverables?)
  Act     : [model_lane]     produce the answer conforming EXACTLY to the extracted contract
  Commit  : [commit_trap]    graded against the task's real checks

A/B: --mode baseline (evidence->act) vs --mode ooda (evidence->orient->decide->act) vs ooda-correct
(+ deterministic contract self-check -> one repair pass on the gap)."""
from __future__ import annotations
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.task_graph import PLAN_TEMPLATES, ValidationLevel
from kernel.orchestrator import Orchestrator, Job, PlanMemory
from kernel.run_cyber_mellum import det_evidence, grade_checks, _extract_json, FIX_ROOT  # reuse evidence + grader
from kernel.fixture_tools import FixtureSandbox, investigate  # grounding: read-only fixture investigation

ADAPTERS = {"mellum": "mellum_adapter.MellumAdapter", "gemma": "gemma_adapter.GemmaAdapter",
            "gptoss": "gptoss_adapter.GptOssAdapter", "north": "north_adapter.NorthAdapter",
            "apertus": "apertus_adapter.ApertusAdapter"}

PLAN_TEMPLATES["baseline"] = [
    {"id": "evidence", "kind": "det_evidence", "deps": []},
    {"id": "act",      "kind": "direct_act",   "deps": ["evidence"]},
    {"id": "commit",   "kind": "commit",       "deps": ["act"], "validator": {"fn": "cyber_grade", "level": "ORACLE"}, "root": True},
]
# OODA with clean-agent fan-out: MAIN does Observe/Orient/Decide; CLEAN AGENTS do the Act (one per
# deliverable, isolated context) and write findings to plan-memory; MAIN synthesizes all threads.
PLAN_TEMPLATES["ooda"] = [
    {"id": "evidence",   "kind": "det_evidence", "deps": []},
    {"id": "orient",     "kind": "orient",       "deps": ["evidence"]},                 # MAIN: extract intent
    {"id": "act",        "kind": "act",          "deps": ["orient", "evidence"]},        # CLEAN AGENTS -> plan-memory
    {"id": "synthesize", "kind": "synthesize",   "deps": ["act", "orient", "evidence"]}, # MAIN: pull threads -> response
    {"id": "commit",     "kind": "commit",       "deps": ["synthesize"], "validator": {"fn": "cyber_grade", "level": "ORACLE"}, "root": True},
]
# GROUNDED OODA: orient extracts the questions; a clean agent INVESTIGATES the fixture with a read-only
# toolset (grep/read/det) until each fact is found in the evidence (or honestly 'insufficient'); synthesize
# assembles from grounded findings. Attacks the EVIDENCE_GAP bucket + the confabulation-confidence floor.
PLAN_TEMPLATES["ooda_grounded"] = [
    {"id": "evidence",    "kind": "det_evidence", "deps": []},
    {"id": "orient",      "kind": "orient",       "deps": ["evidence"]},
    {"id": "investigate", "kind": "investigate",  "deps": ["orient", "evidence"]},        # CLEAN AGENT drives tools
    {"id": "synthesize",  "kind": "synthesize",   "deps": ["investigate", "orient", "evidence"]},
    {"id": "commit",      "kind": "commit",       "deps": ["synthesize"], "validator": {"fn": "cyber_grade", "level": "ORACLE"}, "root": True},
]


def _confidence(answer: str, intent_obj: dict, must_tokens: list) -> tuple:
    """EVIDENCE-GROUNDED confidence: computed from COVERAGE of the work, not self-reported (the traces'
    verify.evidence lane). Start at 0; accrue only what's actually present in the answer. Returns
    (score 0..1, gaps[])."""
    obj = _extract_json(answer)
    a = (answer or "").lower()
    gaps = []
    # 1) contract: emit the JSON shape, with the stated array keys NON-EMPTY + a confidence field
    #    (the prompt states "use arrays for known_facts, evidence, alternatives, ..." — reading the
    #    contract, not the answer key). Empty arrays => contract not met.
    fmt = str(intent_obj.get("required_output_format", "")).lower()
    want_keys = [k for k in ("answer", "confidence", "known_facts", "evidence", "alternatives",
                             "inferred_links", "uncertainties", "next_steps") if k in fmt]
    arr_keys = [k for k in ("known_facts", "evidence", "alternatives") if k in fmt or k in (obj or {})]
    contract_ok = obj is not None and (not want_keys or all(k in obj for k in want_keys))
    if contract_ok and arr_keys:
        contract_ok = all(isinstance((obj or {}).get(k), list) and len((obj or {}).get(k, [])) >= 1 for k in arr_keys)
    contract = 1.0 if contract_ok else 0.0
    if contract < 1.0:
        gaps.append("required JSON not emitted with all keys present and the list fields (known_facts/evidence/alternatives) non-empty")
    # 2) deliverables: did each planned sub-question get addressed (its head words appear)?
    delivs = [str(d) for d in (intent_obj.get("deliverables") or [])]
    cov_d = []
    for d in delivs:
        words = [w for w in re.findall(r"[a-z0-9]{4,}", d.lower())][:4]
        hit = (not words) or any(w in a for w in words)
        cov_d.append(1.0 if hit else 0.0)
        if not hit:
            gaps.append(f"deliverable not addressed: {d}")
    deliv = sum(cov_d) / len(cov_d) if cov_d else 1.0
    # 3) must-cite tokens actually echoed verbatim (the strongest, fully-deterministic signal)
    toks = list(dict.fromkeys([str(t) for t in (intent_obj.get("must_cite") or [])] + must_tokens))
    miss = [t for t in toks if t.lower() not in a]
    tok = (len(toks) - len(miss)) / len(toks) if toks else 1.0
    if miss:
        gaps.append(f"missing required tokens: {miss[:8]}")
    score = 0.34 * contract + 0.33 * deliv + 0.33 * tok    # weighted coverage
    return round(score, 3), gaps


_KNOWLEDGE_CRITIQUE = (
    "Self-critique your draft answer to the analyst question, then output the FULL corrected JSON. Apply "
    "three GENERAL checks (do not invent content to satisfy them):\n"
    "  1. CANONICAL TERMS — name each item with the plain, standard term a defender would search for, not an "
    "elaborate synonym, an abbreviation, or a specific value standing in for the concept. Prefer the simplest "
    "common word for the artifact/field/category.\n"
    "  2. COMPLETENESS — if the question asks to name N items, or for the 'families/categories/pivots', make "
    "sure you have enumerated the STANDARD complete set across all distinct dimensions (not several flavors of "
    "the same one). If you are missing a whole category an analyst would expect, add it.\n"
    "  3. NO FABRICATION — every entry in evidence/known_facts must reference only facts actually stated in the "
    "prompt. Delete any invented identifiers, hashes, IPs, or file paths.\n"
    "Also ensure the most SPECIFIC correct label for the scenario (not a broader near-synonym). "
    "Return ONLY the corrected JSON object with all required keys.")


def _knowledge_critique(model, prompt: str, answer: str):
    """General reasoning/output-discipline pass for no-fixture knowledge tasks. Hold-out-safe: it names no
    task-specific answer — it makes the model re-derive using its OWN knowledge with canonical wording,
    complete enumeration, and no fabrication. Returns the critiqued answer."""
    cp = f"QUESTION:\n{prompt}\n\nYOUR DRAFT:\n{answer}\n\n{_KNOWLEDGE_CRITIQUE}"
    out, _ = model.gen_fast(cp, maxn=900)
    return out


def _set_conf(answer: str, score: float) -> str:
    """Overwrite the answer's confidence field with the COMPUTED (calibrated) value, not the model's vibe."""
    obj = _extract_json(answer)
    if obj is None:
        return answer
    obj["confidence"] = score
    return json.dumps(obj)


def reconcile_confidence(self_conf, committed: bool):
    """FIX 1 — blend the deterministic commit grade into the REPORTED confidence. The model never sees
    this; it is the post-hoc honest value surfaced to the rollup/dashboard. A self-estimate cannot exceed
    'shape-only' (the contract weight, 0.34) once the ORACLE grade rejected the answer — so a failed task
    can no longer read 1.0."""
    if not isinstance(self_conf, (int, float)):
        return self_conf
    return self_conf if committed else round(min(self_conf, 0.34), 3)


def resynth_canonical(model, jin: dict, prior_answer: str, intent_obj: dict, salient: list):
    """FIX 2 — commit-failure repair that RE-ENTERS synthesis (not a pass-through). Applies the canonical-
    terms + completeness + no-fabrication critique (leakage-free: names no answer-key content) to the prior
    answer and returns the corrected, calibrated JSON. This is the reasoning lever that can actually flip a
    grade failure (e.g. re-deriving the canonical term a defender would search for)."""
    crit = _knowledge_critique(model, jin["prompt"], prior_answer)
    ans = crit if _extract_json(crit) is not None else prior_answer   # keep only if still valid JSON
    conf, _ = _confidence(ans, intent_obj or {}, salient)
    return _set_conf(ans, conf), conf


def _salient_tokens(prompt: str) -> list:
    """LEVER: deterministically pull the exact tokens a faithful answer must echo — quoted indicator
    names / action labels, usernames (j.smith, CORP\\user), event IDs, hashes, IPs. Reads the PROMPT
    only (the scenario states them) — NOT the checks (no answer-key leakage). Entity extraction is
    deterministic, so this is a det-block applied to the prose, not a skill."""
    toks = []
    toks += re.findall(r'"([^"\n]{2,48})"', prompt)                 # quoted strings
    toks += re.findall(r"'([^'\n]{2,48})'", prompt)
    toks += re.findall(r"\b[a-z][a-z0-9]*\.[a-z][a-z0-9_.]{1,}\b", prompt)  # j.smith / file.exe-ish
    toks += re.findall(r"\b[A-Z]{2,}\\[A-Za-z0-9._]+\b", prompt)     # CORP\helpdesk.tan
    toks += re.findall(r"\b4\d{3}\b", prompt)                        # Windows event IDs
    toks += re.findall(r"\b[a-fA-F0-9]{32,64}\b", prompt)            # hashes
    toks += re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", prompt)       # IPs
    seen, out = set(), []
    for t in toks:
        t = t.strip()
        if t and t.lower() not in seen and not t.isspace():
            seen.add(t.lower()); out.append(t)
    return out[:24]


def _ev_text(ev: dict) -> str:
    parts = []
    ff = ev.get("fixture_files") or {}
    if ff:
        parts.append("=== EVIDENCE FILES ===\n" + "\n".join(f"--- {k} ---\n{v}" for k, v in ff.items()))
    if ev.get("det"):
        parts.append("=== VERIFIED DETERMINISTIC FACTS (trust exactly) ===\n" + json.dumps(ev["det"], default=str))
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--tasks", default="/tmp/cyber50.json")
    ap.add_argument("--model", default="mellum", choices=list(ADAPTERS))
    ap.add_argument("--mode", default="ooda", choices=["baseline", "ooda", "grounded"])
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--budget", type=int, default=6, help="tool-call budget per investigation question")
    ap.add_argument("--telemetry", default=None, help="dir to persist per-run supervisor telemetry JSONL")
    ap.add_argument("--critique", action=argparse.BooleanOptionalAction, default=False, help="knowledge-task discipline pass (no-fixture)")
    args = ap.parse_args()
    tasks = json.load(open(args.tasks))[:args.n]
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    print(f"[load] {args.model} | mode={args.mode}", flush=True)
    model = getattr(__import__(mod), cls)()

    o = Orchestrator(memory=PlanMemory(), telemetry=args.telemetry)   # define first so executors can close over o.memory (plan-memory)

    # ---- BASELINE: a single direct model call (no orientation) ----
    def direct_act(sched, task, inputs):
        jin = task.meta["job_inputs"]; ev = _ev_text(inputs.get("evidence") or {})
        ans, _ = model.gen_fast(f"REQUEST:\n{jin['prompt']}\n\nEVIDENCE (cite exactly):\n{ev}", maxn=900)
        return ans, True, ValidationLevel.SCHEMA

    # ---- OODA: MAIN orients; CLEAN AGENTS act -> plan-memory; MAIN synthesizes ----
    def orient(sched, task, inputs):
        # MAIN process: read the prose, extract intent (contract + deliverables + must-cite). Do NOT answer.
        jin = task.meta["job_inputs"]; ev = _ev_text(inputs.get("evidence") or {})
        op = ("You are ORIENTING on a request before answering. Read the REQUEST + EVIDENCE and output ONE "
              "compact JSON object describing your PLAN (do NOT answer yet):\n"
              '{"required_output_format": "<exact output shape/keys the request asks for, quoted>",'
              ' "deliverables": ["<each distinct sub-question/section to produce>"],'
              ' "must_cite": ["<exact tokens/IPs/IDs/labels the answer MUST contain>"]}\n\n'
              f"REQUEST:\n{jin['prompt']}\n\nEVIDENCE:\n{ev}")
        intent, _ = model.gen_fast(op, maxn=500)
        o.memory.remember(task.meta["job_id"], "_orient", intent)   # so commit_repair can re-derive intent
        return intent, True, ValidationLevel.SCHEMA

    def act(sched, task, inputs):
        # CLEAN AGENTS do the Act: one isolated agent per deliverable; each finding -> plan-memory
        jid = task.meta["job_id"]; ev = _ev_text(inputs.get("evidence") or {})
        obj = _extract_json(inputs.get("orient") or "") or {}
        delivs = [str(d) for d in (obj.get("deliverables") or [])][:6] or ["the complete analysis"]
        for d in delivs:
            ap = ("You are a focused sub-analyst with a CLEAN context. Using ONLY the evidence below, produce "
                  f"JUST this deliverable, concise + factual:\n\nDELIVERABLE: {d}\n\nEVIDENCE:\n{ev}")
            finding, _ = model.gen_fast(ap, maxn=320)
            o.memory.remember(jid, d, finding)           # store each thread's finding in plan-memory
        return {"deliverables": delivs, "n": len(delivs)}, True, ValidationLevel.PROPERTY

    def synthesize(sched, task, inputs):
        # MAIN pulls all threads back from plan-memory and assembles; then VERIFY (evidence-grounded
        # confidence) + "check yourself before you wreck yourself" -> verify+repair agent if < 100%.
        jid = task.meta["job_id"]; jin = task.meta["job_inputs"]
        intent_obj = _extract_json(inputs.get("orient") or "") or {}
        findings = o.memory.recall(jid) or {}
        fb = "\n".join(f"- [{k}]: {v}" for k, v in findings.items())
        salient = _salient_tokens(jin["prompt"])
        tok_block = (f"\n\nVERBATIM TOKENS — the answer MUST contain each exactly: {salient}" if salient else "")
        sp = (f"REQUEST:\n{jin['prompt']}\n\nYOUR ORIENTATION PLAN:\n{inputs.get('orient') or ''}\n\n"
              f"FINDINGS FROM YOUR SUB-ANALYSES:\n{fb}{tok_block}\n\n"
              "Assemble the SINGLE final answer: conform EXACTLY to the requested output format "
              "(ONE JSON object with all requested keys) and include every verbatim token above.")
        ans, _ = model.gen_fast(sp, maxn=900)
        conf, gaps = _confidence(ans, intent_obj, salient)        # VERIFY: confidence from coverage (starts 0)
        if args.verify and conf < 0.999 and gaps:                 # check ourselves before we wreck ourselves
            rp = (sp + f"\n\nSELF-CHECK (verify+repair) — your draft did NOT fully satisfy the request. GAPS:\n"
                  + "\n".join(f"  - {g}" for g in gaps[:10])
                  + "\n\nProduce the FULL corrected answer that closes every gap: emit the required JSON with "
                    "all keys, address every deliverable, and include every missing token verbatim.")
            ans2, _ = model.gen_fast(rp, maxn=950)
            c2, g2 = _confidence(ans2, intent_obj, salient)
            if c2 >= conf:                                        # keep the repair only if it raised confidence
                ans, conf, gaps = ans2, c2, g2
        if args.critique and not jin.get("fixture"):             # knowledge-task discipline pass (no fixture)
            ac = _knowledge_critique(model, jin["prompt"], ans)
            if _extract_json(ac) is not None:                    # keep only if it stayed valid JSON
                cc, _ = _confidence(ac, intent_obj, salient)
                ans, conf = ac, cc
        ans = _set_conf(ans, conf)                                # emit the CALIBRATED confidence, not the vibe
        o.memory.remember(task.meta["job_id"], "_confidence", conf)  # surface it for the rollup
        return ans, True, ValidationLevel.SCHEMA

    def investigate_node(sched, task, inputs):
        # CLEAN AGENT does grounded investigation: drive the read-only fixture toolset until each
        # question is answered FROM the evidence (or honestly 'insufficient'). Findings -> plan-memory.
        jid = task.meta["job_id"]; jin = task.meta["job_inputs"]
        sb = FixtureSandbox(os.path.join(FIX_ROOT, jin["fixture"]) if jin.get("fixture") else "")
        det_facts = (inputs.get("evidence") or {}).get("det") or {}
        obj = _extract_json(inputs.get("orient") or "") or {}
        # investigate the deliverables AND the must-cite targets as forensic questions (cap to bound cost)
        questions = [str(d) for d in (obj.get("deliverables") or [])][:3] or ["the key indicators, accounts, and IOCs in this incident"]
        grounded = 0
        for q in questions:
            ans, tr = investigate(model, sb, q, budget=args.budget, det_facts=det_facts)
            ev = ("  EVIDENCE:\n" + "\n".join(tr[-3:])) if tr else ""
            o.memory.remember(jid, q, f"{ans}\n{ev}")            # grounded finding + the lines it rests on
            if "insufficient evidence" not in (ans or "").lower():
                grounded += 1
        return {"questions": len(questions), "grounded": grounded, "tool_calls": sb.calls}, True, ValidationLevel.PROPERTY

    def cyber_grade(result, task):
        ok, _ = grade_checks(result, task.meta["job_inputs"]["checks"]); return ok

    def commit_repair(sched, task, inputs):
        # FIX 2: clean-context repair worker re-enters SYNTHESIS with a canonical/completeness critique.
        jid = task.meta["job_id"]; jin = task.meta["job_inputs"]
        prior = inputs.get("_failed") or next((v for v in inputs.values() if isinstance(v, str)), "")
        intent_obj = _extract_json(o.memory.recall(jid, "_orient") or "") or {}
        salient = _salient_tokens(jin["prompt"])
        ans, conf = resynth_canonical(model, jin, prior, intent_obj, salient)
        o.memory.remember(jid, "_confidence", conf); o.memory.remember(jid, "_repaired", True)
        return ans, True, ValidationLevel.PROVEN

    template = {"baseline": "baseline", "ooda": "ooda", "grounded": "ooda_grounded"}[args.mode]
    o.register_job_type("cyber", lambda j: True, template)
    o.register_executor("det_evidence", det_evidence)
    o.register_executor("direct_act", direct_act)
    o.register_executor("orient", orient); o.register_executor("act", act); o.register_executor("synthesize", synthesize)
    o.register_executor("investigate", investigate_node)
    o.register_executor("commit", lambda s, t, i: (next((v for v in i.values() if isinstance(v, str)), None), True, ValidationLevel.PROVEN))
    o.register_executor("commit_repair", commit_repair)
    o.register_validator("cyber_grade", cyber_grade)
    for k, owner in {"det_evidence": "primitive:det_primitives_cyber", "direct_act": "model_lane",
                     "orient": "main:OOD", "act": "agent:clean", "investigate": "agent:investigator",
                     "synthesize": "main:synthesize", "commit": "commit_trap", "commit_repair": "agent:repair"}.items():
        o.assign_agent(k, owner)

    npass = 0
    print(f"[run] {len(tasks)} cyber jobs | {args.model} | mode={args.mode}", flush=True)
    for t in tasks:
        p = f"{t.get('system','')}\n\n{t['prompt']}"
        r = o.submit(Job(t["id"], prompt=p, inputs={"prompt": p, "checks": t["checks"], "fixture": t.get("fixture")}))
        npass += int(r["committed"])
        cf = o.memory.recall(t["id"], "_confidence")
        rep = " (repaired)" if o.memory.recall(t["id"], "_repaired") else ""
        honest = reconcile_confidence(cf, r["committed"])              # FIX 1: oracle-blended reported conf
        print(f"  {t['id']:8s} diff{t.get('difficulty','?'):<2} {'PASS' if r['committed'] else 'fail '} "
              f"self_conf={cf} honest_conf={honest}{rep}", flush=True)
    print(f"\n===== {args.mode} | {args.model}: {npass}/{len(tasks)} committed =====")
    print("  dashboard:", json.dumps(o.dashboard(), default=str)[:200])


if __name__ == "__main__":
    main()
