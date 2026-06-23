#!/usr/bin/env python3
"""Raw single-shot baseline — the CAUSAL CONTROL for "does the investigation kernel help?".

The model answers each cyber task DIRECTLY in one shot: no investigation protocol (FRAME/ACH/back-edge),
no enforce_contract, no grounded ReAct. Same model, same tasks, same grader (grade_checks) as inv_smoke.
To isolate the kernel's STRUCTURE rather than evidence access, the task's fixture (when present) is dumped
into the prompt with the same per-file/max-file caps the kernel's read_fixture uses — so both arms see the
same evidence; only the structure differs.

Caveat: the kernel's grounded source does TARGETED, iterative grep (capped ~900 chars); this baseline dumps
the fixture wholesale. If anything that gives the RAW arm an evidence edge, so a kernel win here is conservative.

Run (from repo root, so FIX_ROOT resolves):
  PYTHONPATH unneeded (self-inserts); use the mlx venv python:
  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/raw_baseline.py --model gptoss
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.run_cyber_mellum import grade_checks, read_fixture

ADAPTERS = {"mellum": "mellum_adapter.MellumAdapter", "gptoss": "gptoss_adapter.GptOssAdapter",
            "gemma": "gemma_adapter.GemmaAdapter", "north": "north_adapter.NorthAdapter"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptoss", choices=list(ADAPTERS))
    ap.add_argument("--tasks", default="/tmp/cyber50.json")
    ap.add_argument("--ids", default="")            # empty = all tasks in the file
    ap.add_argument("--maxn", type=int, default=1200)  # one shot does everything -> a touch more headroom (fair)
    args = ap.parse_args()
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    print(f"[load] {args.model} adapter ...", flush=True)
    model = getattr(__import__(mod), cls)()
    tasks = json.load(open(args.tasks))
    if args.ids:
        keep = set(args.ids.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    print(f"[run] {len(tasks)} cyber jobs on {args.model} RAW single-shot (no kernel) ...", flush=True)
    npass = 0
    for t in tasks:
        ev = ""
        if t.get("fixture"):
            d = read_fixture(t["fixture"])
            if d:
                ev = "\n\nEVIDENCE (fixture files):\n" + "\n".join(f"--- {k} ---\n{v}" for k, v in d.items())
        prompt = f"{t.get('system','')}\n\n{t['prompt']}{ev}".strip()
        ans = model.gen_fast(prompt, maxn=args.maxn)[0]
        ok, per = grade_checks(ans, t["checks"]); npass += int(ok)
        print(f"  {t['id']:7s} d{t['difficulty']:<2} {'PASS' if ok else 'fail'} "
              f"failed={[l for l, o in per if not o]}", flush=True)
    print(f"\n===== {args.model} RAW single-shot (no kernel): {npass}/{len(tasks)} =====", flush=True)


if __name__ == "__main__":
    main()
