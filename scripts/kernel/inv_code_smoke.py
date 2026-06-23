#!/usr/bin/env python3
"""Probe the investigation methodology on the CODING expert-50 suite (a different domain shape).
Two questions: (1) how does the FRAME router CLASSIFY code-synthesis tasks? (2) what pass rate does the
full protocol get vs the det-block shape-appliance (which already gets 50/50)?"""
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernel.investigation as inv
from shape_appliance import check_cases

ADAPTERS = {"mellum": "mellum_adapter.MellumAdapter", "gptoss": "gptoss_adapter.GptOssAdapter",
            "gemma": "gemma_adapter.GemmaAdapter", "north": "north_adapter.NorthAdapter"}
CODE_HINT = "Return ONLY the complete Python function definition inside a ```python code block. No commentary."


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text or "", re.I)
    if m:
        return m.group(1)
    # fallback: from first 'def ' to end
    i = (text or "").find("def ")
    return text[i:] if i >= 0 else (text or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptoss", choices=list(ADAPTERS))
    ap.add_argument("--frame_n", type=int, default=50, help="how many tasks to FRAME-classify")
    ap.add_argument("--solve_n", type=int, default=6, help="how many to run through the FULL protocol + grade")
    ap.add_argument("--tasks", default="/tmp/expert50.json")
    args = ap.parse_args()
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    model = getattr(__import__(mod), cls)()
    tasks = json.load(open(args.tasks))

    from collections import Counter
    print(f"=== FRAME routing of code-synthesis tasks ({args.model}, n={args.frame_n}) ===", flush=True)
    kinds = Counter()
    for t in tasks[:args.frame_n]:
        fr = inv._frame(model, t["prompt"], CODE_HINT)
        k = (fr.get("task_kind") or "?").lower().split()[0]
        kinds[k] += 1
        print(f"  {t['id']} [{t['topic']:11s}] -> {k}", flush=True)
    print("  ROUTING:", dict(kinds), flush=True)

    print(f"\n=== FULL protocol pass rate on a slice ({args.solve_n} tasks, real check_cases) ===", flush=True)
    npass = 0
    for t in tasks[:args.solve_n]:
        rep, st = inv.investigate(model, inv.EvidenceSource(), t["prompt"], contract_hint=CODE_HINT, max_rounds=2)
        code = _extract_code(rep)
        ok = False
        try:
            ok = bool(code) and check_cases(code, t["entry"], t["cases"])
        except Exception:
            ok = False
        npass += int(ok)
        print(f"  {t['id']} routed={st.frame.get('routed','diagnostic-full'):16s} {'PASS' if ok else 'fail'}", flush=True)
    print(f"\n  investigation-protocol: {npass}/{args.solve_n}  [det-block shape-appliance gets 50/50 on this suite]", flush=True)


if __name__ == "__main__":
    main()
