#!/usr/bin/env python3
"""Failure forensics for cyber-under-kernel: re-run FAILING tasks through OODA with FULL CAPTURE
(orient intent, clean-agent findings, final answer, claimed confidence, per-check verdicts), then
CLASSIFY each failure so we can say how much is formal-problem-solving-solvable vs miscalibrated
confidence vs token-fidelity vs format vs evidence-gap vs eval-artifact.

Categories (per failed check, rolled up to a primary cause):
  FORMAT          only the structured-JSON shape failed (analysis fine, contract not emitted)
  TOKEN_FIDELITY  a required exact token WAS present in the evidence/prompt but the answer didn't echo it
  DERIVATION      a required token must be DERIVED (not verbatim in the input) and the model got it wrong
                  -> the formal-problem-solving / det-block-addressable bucket
  EVIDENCE_GAP    required token is nowhere in the provided evidence (needs tools/multi-turn)
  EVAL_ARTIFACT   answer is substantively right but phrased so the strict check misses it
Plus a miscalibration flag: claimed confidence HIGH while content checks failed (confident-but-wrong).
"""
from __future__ import annotations
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.run_cyber_mellum import det_evidence, grade_checks, _extract_json
from kernel.run_cyber_ooda import _ev_text, _salient_tokens, ADAPTERS
from kernel.task_graph import Task


def _conf_high(obj) -> bool:
    c = (obj or {}).get("confidence")
    if isinstance(c, str): return c.strip().lower() in ("high", "very high", "certain")
    if isinstance(c, (int, float)): return c >= 0.7
    return False


def _required_tokens(check) -> list:
    if check["type"] == "contains": return [check["value"]]
    if check["type"] == "contains_all": return list(check["value"])
    return []


def classify(task, answer, ev_text):
    obj = _extract_json(answer)
    ok, per = grade_checks(answer, task["checks"])
    blob = (answer or "").lower(); evb = ev_text.lower()
    cats = []
    for chk, (label, cok) in zip(task["checks"], per):
        if cok:
            continue
        if chk["type"] == "cyber_structured_json":
            cats.append("FORMAT")
        elif chk["type"] == "regex":
            # a regex miss is usually a derived/structural phrasing gap
            cats.append("DERIVATION")
        else:
            for tok in _required_tokens(chk):
                t = tok.lower()
                if t in blob:
                    continue                                   # actually present (shouldn't happen if check failed)
                in_ev = t in evb or re.sub(r"[^a-z0-9]", "", t) in re.sub(r"[^a-z0-9]", "", evb)
                cats.append("TOKEN_FIDELITY" if in_ev else "EVIDENCE_GAP")
    # primary cause: prefer the most actionable signal
    order = ["DERIVATION", "EVIDENCE_GAP", "TOKEN_FIDELITY", "FORMAT", "EVAL_ARTIFACT"]
    # ok==True here means the re-run PASSED -> the original failure was MARGINAL (non-robust, flips on tiny diffs)
    primary = "MARGINAL" if ok else next((c for c in order if c in cats), "EVAL_ARTIFACT")
    return {"primary": primary, "all": sorted(set(cats)), "ok": ok,
            "confidence": (obj or {}).get("confidence"), "miscalibrated": (_conf_high(obj) and not ok),
            "failed_checks": [label for chk, (label, cok) in zip(task["checks"], per) if not cok]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mellum", choices=list(ADAPTERS))
    ap.add_argument("--ids", required=True, help="comma-separated failing task ids")
    ap.add_argument("--tasks", default="/tmp/cyber50.json")
    args = ap.parse_args()
    want = set(args.ids.split(","))
    tasks = [t for t in json.load(open(args.tasks)) if t["id"] in want]
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    print(f"[load] {args.model} | examining {len(tasks)} failures", flush=True)
    model = getattr(__import__(mod), cls)()

    from collections import Counter
    tally = Counter(); miscal = 0; rows = []
    for t in tasks:
        prompt = f"{t.get('system','')}\n\n{t['prompt']}"
        tk = Task("e", "det_evidence"); tk.meta["job_inputs"] = {"prompt": prompt, "fixture": t.get("fixture")}
        ev, _, _ = det_evidence(None, tk, {}); evt = _ev_text(ev)
        # OODA with capture: orient -> clean agents -> synthesize
        op = ("You are ORIENTING on a request. Output ONE JSON: {\"required_output_format\":\"...\","
              "\"deliverables\":[...],\"must_cite\":[...]}. Do NOT answer yet.\n\n"
              f"REQUEST:\n{prompt}\n\nEVIDENCE:\n{evt}")
        intent, _ = model.gen_fast(op, maxn=500)
        delivs = [str(d) for d in (_extract_json(intent) or {}).get("deliverables", [])][:6] or ["the complete analysis"]
        findings = []
        for d in delivs:
            f, _ = model.gen_fast(f"Clean-context sub-analyst. Using ONLY this evidence, produce JUST: {d}\n\nEVIDENCE:\n{evt}", maxn=320)
            findings.append(f"- [{d}]: {f}")
        sal = _salient_tokens(prompt)
        sp = (f"REQUEST:\n{prompt}\n\nPLAN:\n{intent}\n\nFINDINGS:\n" + "\n".join(findings)
              + (f"\n\nMUST contain verbatim: {sal}" if sal else "")
              + "\n\nAssemble ONE JSON object answer conforming to the requested format with every verbatim token.")
        answer, _ = model.gen_fast(sp, maxn=900)
        c = classify(t, answer, evt + " " + prompt)
        tally[c["primary"]] += 1; miscal += int(c["miscalibrated"]); rows.append((t["id"], t["difficulty"], c))
        print(f"  {t['id']:8s} d{t['difficulty']:<2} primary={c['primary']:<14} conf={str(c['confidence'])[:8]:8} "
              f"miscal={c['miscalibrated']!s:5} failed={c['failed_checks']}", flush=True)

    n = len(tasks)
    print(f"\n===== FAILURE TAXONOMY ({args.model}, n={n}) =====")
    for cat, k in tally.most_common():
        print(f"  {cat:14s} {k:2d}  ({100*k/n:.0f}%)")
    print(f"  miscalibrated (confident-but-wrong): {miscal}/{n} ({100*miscal/n:.0f}%)")
    print(f"\n  MARGINAL (passed on re-run; original failure non-robust): {tally['MARGINAL']}/{n}")
    print(f"  formal-problem-solving-addressable (DERIVATION): {tally['DERIVATION']}/{n}")
    print(f"  data-was-there (TOKEN_FIDELITY, det-extraction lever): {tally['TOKEN_FIDELITY']}/{n}")
    print(f"  contract/shape (FORMAT): {tally['FORMAT']}/{n}")
    print(f"  needs tools/evidence (EVIDENCE_GAP): {tally['EVIDENCE_GAP']}/{n}")
    print(f"  miscalibrated confident-but-wrong: {miscal}/{n} ({100*miscal/n:.0f}%)")


if __name__ == "__main__":
    main()
