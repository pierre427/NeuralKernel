#!/usr/bin/env python3
"""Observe the microkernel end-to-end on gpt-oss: scheduler loop, agent assignment, clean-agent acts,
proof ledger. Runs a few cyber jobs through the OODA template and dumps the FULL rollup + event trace
(not just pass/fail) so we can see what each subsystem actually did."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# reuse the fully-wired OODA orchestrator builder by importing run_cyber_ooda's main pieces
import kernel.run_cyber_ooda as ro
from kernel.task_graph import PLAN_TEMPLATES, ValidationLevel
from kernel.orchestrator import Orchestrator, Job, PlanMemory
from kernel.run_cyber_mellum import det_evidence, grade_checks, _extract_json


def build(model, mode="ooda", verify=True, budget=6):
    o = Orchestrator(memory=PlanMemory())
    ev_text = ro._ev_text

    def direct_act(sched, task, inputs):
        jin = task.meta["job_inputs"]; ev = ev_text(inputs.get("evidence") or {})
        ans, _ = model.gen_fast(f"REQUEST:\n{jin['prompt']}\n\nEVIDENCE (cite exactly):\n{ev}", maxn=900)
        return ans, True, ValidationLevel.SCHEMA

    def orient(sched, task, inputs):
        jin = task.meta["job_inputs"]; ev = ev_text(inputs.get("evidence") or {})
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
        jid = task.meta["job_id"]; ev = ev_text(inputs.get("evidence") or {})
        obj = _extract_json(inputs.get("orient") or "") or {}
        delivs = [str(d) for d in (obj.get("deliverables") or [])][:6] or ["the complete analysis"]
        for d in delivs:
            ap = ("You are a focused sub-analyst with a CLEAN context. Using ONLY the evidence below, produce "
                  f"JUST this deliverable, concise + factual:\n\nDELIVERABLE: {d}\n\nEVIDENCE:\n{ev}")
            finding, _ = model.gen_fast(ap, maxn=320)
            o.memory.remember(jid, d, finding)
        return {"deliverables": delivs, "n": len(delivs)}, True, ValidationLevel.PROPERTY

    def synthesize(sched, task, inputs):
        jid = task.meta["job_id"]; jin = task.meta["job_inputs"]
        intent_obj = _extract_json(inputs.get("orient") or "") or {}
        findings = o.memory.recall(jid) or {}
        fb = "\n".join(f"- [{k}]: {v}" for k, v in findings.items())
        salient = ro._salient_tokens(jin["prompt"])
        tok_block = (f"\n\nVERBATIM TOKENS — the answer MUST contain each exactly: {salient}" if salient else "")
        sp = (f"REQUEST:\n{jin['prompt']}\n\nYOUR ORIENTATION PLAN:\n{inputs.get('orient') or ''}\n\n"
              f"FINDINGS FROM YOUR SUB-ANALYSES:\n{fb}{tok_block}\n\n"
              "Assemble the SINGLE final answer: conform EXACTLY to the requested output format "
              "(ONE JSON object with all requested keys) and include every verbatim token above.")
        ans, _ = model.gen_fast(sp, maxn=900)
        conf, gaps = ro._confidence(ans, intent_obj, salient)
        if verify and conf < 0.999 and gaps:
            rp = (sp + f"\n\nSELF-CHECK (verify+repair) — your draft did NOT fully satisfy the request. GAPS:\n"
                  + "\n".join(f"  - {g}" for g in gaps[:10])
                  + "\n\nProduce the FULL corrected answer that closes every gap.")
            ans2, _ = model.gen_fast(rp, maxn=950)
            c2, g2 = ro._confidence(ans2, intent_obj, salient)
            if c2 >= conf:
                ans, conf, gaps = ans2, c2, g2
        ans = ro._set_conf(ans, conf)
        o.memory.remember(jid, "_confidence", conf)
        return ans, True, ValidationLevel.SCHEMA

    def cyber_grade(result, task):
        ok, _ = grade_checks(result, task.meta["job_inputs"]["checks"]); return ok

    def commit_repair(sched, task, inputs):
        # FIX 2: clean-context repair re-enters SYNTHESIS with a canonical/completeness critique.
        jid = task.meta["job_id"]; jin = task.meta["job_inputs"]
        prior = inputs.get("_failed") or next((v for v in inputs.values() if isinstance(v, str)), "")
        intent_obj = _extract_json(o.memory.recall(jid, "_orient") or "") or {}
        salient = ro._salient_tokens(jin["prompt"])
        ans, conf = ro.resynth_canonical(model, jin, prior, intent_obj, salient)
        o.memory.remember(jid, "_confidence", conf); o.memory.remember(jid, "_repaired", True)
        return ans, True, ValidationLevel.PROVEN

    o.register_job_type("cyber", lambda j: True, mode)
    o.register_executor("det_evidence", det_evidence)
    o.register_executor("direct_act", direct_act)
    o.register_executor("orient", orient); o.register_executor("act", act); o.register_executor("synthesize", synthesize)
    o.register_executor("commit", lambda s, t, i: (next((v for v in i.values() if isinstance(v, str)), None), True, ValidationLevel.PROVEN))
    o.register_executor("commit_repair", commit_repair)
    o.register_validator("cyber_grade", cyber_grade)
    for k, owner in {"det_evidence": "primitive:det_primitives_cyber", "direct_act": "model_lane",
                     "orient": "main:OOD", "act": "agent:clean",
                     "synthesize": "main:synthesize", "commit": "commit_trap", "commit_repair": "agent:repair"}.items():
        o.assign_agent(k, owner)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--tasks", default="/tmp/cyber50.json")
    args = ap.parse_args()
    tasks = json.load(open(args.tasks))[:args.n]
    print(f"[load] gpt-oss-20b ...", flush=True)
    from gptoss_adapter import GptOssAdapter
    model = GptOssAdapter()
    o = build(model, mode="ooda")
    print(f"[run] {len(tasks)} cyber jobs through the kernel\n", flush=True)
    for t in tasks:
        p = f"{t.get('system','')}\n\n{t['prompt']}"
        r = o.submit(Job(t["id"], prompt=p, inputs={"prompt": p, "checks": t["checks"], "fixture": t.get("fixture")}))
        print(f"\n===== JOB {t['id']} (diff {t.get('difficulty','?')}) committed={r['committed']} =====", flush=True)
        print("  job_type:", r["job_type"], "| template:", r["template"],
              "| ledger_verified:", r["ledger_verified"], "| ledger_entries:", r["ledger_entries"])
        print("  TASK GRAPH (owner / state / verification):")
        for row in r["tasks"]:
            print(f"    {row['kind']:14s} owner={row['owner']:28s} {row['status']:10s} "
                  f"verify={row['verification']:9s} proof={(row['proof'] or '')[:10]}")
        cf = o.memory.recall(t["id"], "_confidence")
        rep = " (repaired)" if o.memory.recall(t["id"], "_repaired") else ""
        print(f"  self_conf={cf}  honest_conf={ro.reconcile_confidence(cf, r['committed'])}{rep}")
    print("\n===== DASHBOARD =====")
    print(json.dumps(o.dashboard(), indent=2, default=str))


if __name__ == "__main__":
    main()
