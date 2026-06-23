#!/usr/bin/env python3
"""Run the coding expert-50 suite THROUGH the Job Kernel (Orchestrator -> code_gen DAG) with a model-backed
draft, so the SUPERVISOR TELEMETRY captures the full generate -> parse -> test -> commit workflow per task.
The point here is NOT the pass rate — it's to inspect the supervisor's flows (state transitions, validation
levels, repairs, gate-fails, calibrated confidence) and verify the workflows are correct.

  draft  [model_lane: gpt-oss] : generate the function from the prompt (judgment — no validator)
  parse  [primitive:ast]       : compiles? (SYNTAX validator)
  tests  [primitive:check_cases]: run the REAL cases (ORACLE validator: tests_pass)
  commit [commit_trap]         : the only output path (root)
"""
from __future__ import annotations
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.task_graph import PLAN_TEMPLATES, ValidationLevel
from kernel.orchestrator import Orchestrator, Job, PlanMemory
from kernel.executors import ast_parse, run_tests, commit_node, v_compiles, v_tests_pass, ALL_RECOVERIES
from kernel.code_fixups import auto_import, diagnose       # model-tail: det-block auto-import + leakage-safe repair feedback
from shape_appliance import check_cases

ADAPTERS = {"gptoss": "gptoss_adapter.GptOssAdapter", "mellum": "mellum_adapter.MellumAdapter",
            "gemma": "gemma_adapter.GemmaAdapter", "north": "north_adapter.NorthAdapter"}


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text or "", re.I)
    if m:
        return m.group(1)
    i = (text or "").find("def ")
    return text[i:] if i >= 0 else (text or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptoss", choices=list(ADAPTERS))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tasks", default="/tmp/expert50.json")
    ap.add_argument("--telemetry", default="reports/telemetry/expert50")
    ap.add_argument("--draft", default="recovery", choices=["recovery", "model"],
                    help="recovery = recovery-first (det-block if proven, else model) [the proper workflow]; "
                         "model = always generate (exercises the model lane only)")
    args = ap.parse_args()
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    print(f"[load] {args.model} ...", flush=True)
    model = getattr(__import__(mod), cls)()
    tasks = json.load(open(args.tasks))[:args.n]

    o = Orchestrator(memory=PlanMemory(), telemetry=args.telemetry)
    o.register_job_type("coding", lambda j: True, "code_gen")

    def draft_model(sched, task, inputs):                    # the draft node
        jin = task.meta["job_inputs"]
        if args.draft == "recovery":
            rec = ALL_RECOVERIES.get(jin.get("entry"))
            if rec is not None:                              # RECOVERY-FIRST: proven det-block, 0 model tokens
                task.meta.setdefault("telemetry", {})["draft_route"] = "det-block"
                return rec, True, ValidationLevel.PROVEN
        task.meta.setdefault("telemetry", {})["draft_route"] = "model"   # tail (or --draft model): generate
        out, _ = model.gen_fast(jin["prompt"], maxn=1400)
        code = auto_import(_extract_code(out))                           # det-block: fix missing stdlib imports
        return code, bool(code and "def " in code), ValidationLevel.NONE

    def run_tests_repair(sched, task, inputs):               # tests-fail -> REGENERATE w/ ERROR-SPECIFIC feedback
        jin = task.meta["job_inputs"]
        failed = inputs.get("_failed") if isinstance(inputs.get("_failed"), dict) else {}
        prior = failed.get("code") or next((v for v in inputs.values() if isinstance(v, str)), "")
        diag = diagnose(prior or "", jin["entry"], jin.get("cases", []))   # leakage-safe: own behavior, not expected
        task.meta.setdefault("telemetry", {})["repair_diag"] = diag[:120]
        fb = (f"{jin['prompt']}\n\nYour previous attempt was incorrect — {diag}. Fix the specific problem above "
              f"(and any other edge cases). Previous attempt:\n```python\n{(prior or '')[:1200]}\n```")
        out, _ = model.gen_fast(fb, maxn=1400)
        code = auto_import(_extract_code(out))                            # det-block auto-import on the repair too
        passed = bool(code) and check_cases(code, jin["entry"], jin.get("cases", []))
        return {"code": code, "entry": jin["entry"], "passed": passed}, passed, ValidationLevel.ORACLE

    o.register_executor("draft", draft_model)
    o.register_executor("run_tests_repair", run_tests_repair)   # the PROPER repair: regenerate, don't re-run inertly
    o.register_executor("ast_parse", ast_parse)
    o.register_executor("run_tests", run_tests)
    o.register_executor("commit", commit_node)
    o.register_validator("compiles", v_compiles)
    o.register_validator("tests_pass", v_tests_pass)
    for k, owner in {"draft": "model_lane:" + args.model, "ast_parse": "primitive:ast",
                     "run_tests": "primitive:check_cases", "commit": "commit_trap"}.items():
        o.assign_agent(k, owner)

    npass = 0
    print(f"[run] {len(tasks)} coding jobs on {args.model} under the kernel | telemetry -> {args.telemetry}", flush=True)
    for t in tasks:
        job = Job(t["id"], prompt=t["prompt"], inputs={"prompt": t["prompt"], "entry": t["entry"], "cases": t["cases"]})
        r = o.submit(job)
        npass += int(r["committed"])
        tel = r["telemetry"]
        print(f"  {t['id']:5s} {'PASS' if r['committed'] else 'fail'}  conf_cal={tel['confidence_calibrated']} "
              f"repairs={tel['repairs']} gate_fails={tel['gate_fails']} events={tel['events']} {tel['dur_ms']:.0f}ms", flush=True)
    print(f"\n===== coding-expert-{len(tasks)} on {args.model} under kernel: {npass}/{len(tasks)} committed =====", flush=True)
    print(f"  telemetry persisted to {args.telemetry}/ (per-task JSONL + index.jsonl)", flush=True)


if __name__ == "__main__":
    main()
