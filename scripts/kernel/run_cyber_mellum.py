#!/usr/bin/env python3
"""Run 50 cyber jobs on Mellum UNDER the Job Kernel.

Each cyber-50 task becomes a Job: the Orchestrator classifies it cyber -> a 3-node plan
(det_evidence ▸ analyze ▸ commit), assigns agents, and runs it on the task graph.
  - det_evidence  [primitive:det_primitives_cyber] : best-effort verified evidence pulled from the prompt
  - analyze       [model_lane:mellum]              : Mellum generates the structured-JSON analyst answer
  - commit        [commit_trap, validator=cyber_grade] : graded against the task's REAL harness checks
Every node is audited in the proof ledger; a master rollup reports the pass rate. (Single-turn, no fixture
tool access — so this is a floor for the fixture-heavy tasks; the value is the kernel orchestrating a real
model + det-primitives end-to-end with the real grader.)
"""
from __future__ import annotations
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.task_graph import PLAN_TEMPLATES, ValidationLevel
from kernel.orchestrator import Orchestrator, Job, PlanMemory
import det_primitives_cyber as cyber

# --- the cyber-model plan (model-backed analyze node) ---
PLAN_TEMPLATES["cyber_model"] = [
    {"id": "evidence", "kind": "det_evidence", "deps": []},
    {"id": "analyze",  "kind": "analyze",      "deps": ["evidence"]},
    {"id": "commit",   "kind": "commit",       "deps": ["analyze"],
     "validator": {"fn": "cyber_grade", "level": "ORACLE"}, "root": True},
]


# --- faithful-enough port of the harness runCheck for cyber checks (checks.mjs) ---
def _extract_json(answer: str):
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", answer or "", re.I)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    s, e = (answer or "").find("{"), (answer or "").rfind("}")
    if 0 <= s < e:
        try: return json.loads(answer[s:e + 1])
        except Exception: pass
    return None

def _contains(blob: str, term: str) -> bool:
    t = term.lower()
    if t in blob: return True
    sq = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    return sq(term) in sq(blob)

def grade_checks(answer: str, checks: list):
    blob = (answer or "").lower(); results = []; allok = True
    for c in checks:
        ty = c.get("type")
        if ty == "contains":
            okc = _contains(blob, c["value"])
        elif ty == "contains_all":
            okc = all(_contains(blob, v) for v in c["value"])
        elif ty == "regex":
            okc = re.search(c["pattern"], answer or "", re.I if "i" in c.get("flags", "") else 0) is not None
        elif ty == "cyber_structured_json":
            obj = _extract_json(answer)
            if obj is None:
                okc = False
            else:
                mi = c.get("min_items", {})
                okc = all(isinstance(obj.get(k), list) and len(obj.get(k, [])) >= n for k, n in mi.items())
                okc = okc and bool(str(obj.get("confidence", "")))
        else:
            okc = True
        results.append((c.get("label", ty), okc)); allok = allok and okc
    return allok, results


# --- lever 1: scheduler-owned FIXTURE READ ---
import glob, datetime
from collections import Counter
FIX_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "fixtures")
FIX_ROOT = os.path.abspath(os.path.join(os.getcwd(), "fixtures"))

def read_fixture(fixture: str, per_file=1400, max_files=6) -> dict:
    d = os.path.join(FIX_ROOT, fixture)
    files = {}
    if not os.path.isdir(d):
        return files
    for f in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.isfile(f) and os.path.getsize(f) < 300000 and len(files) < max_files:
            try: files[os.path.basename(f)] = open(f, errors="ignore").read()[:per_file]
            except Exception: pass
    return files

def _epochs(text: str, near: str = None):
    lines = [l for l in text.splitlines() if (near is None or near in l)]
    eps = []
    for l in lines:
        m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})", l)
        if m:
            try: eps.append(int(datetime.datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S").timestamp()))
            except Exception: pass
    return sorted(eps)

# --- levers 1+3: read the fixture + run det_primitives_cyber over it -> verified evidence ---
def det_evidence(sched, task, inputs):
    jin = task.meta["job_inputs"]
    fixture = jin.get("fixture")
    fdict = read_fixture(fixture) if fixture else {}
    ftext = "\n".join(fdict.values())
    allt = jin.get("prompt", "") + "\n" + ftext
    det = {}
    ipc = Counter(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", ftext))
    if ipc:
        det["ssrf"] = {ip: cyber.is_ssrf_target(ip) for ip, _ in ipc.most_common(8)}
        top_ip, _ = ipc.most_common(1)[0]               # the most-contacted host (beacon candidate)
        eps = _epochs(ftext, top_ip)
        if len(eps) >= 3:
            det["beacon"] = {"top_host": top_ip, "period_s": cyber.detect_beacon(eps, 5), "hits": len(eps)}
    for eid in list(dict.fromkeys(re.findall(r"\b(4\d{3})\b", allt)))[:5]:
        det.setdefault("win_events", {})[eid] = cyber.classify_win_event(int(eid))
    for blob in [b for b in re.findall(r"[A-Za-z0-9+/]{24,}={0,2}", ftext)][:3]:
        try: det.setdefault("decoded", {})[blob[:16] + "..."] = cyber.decode_and_magic(blob)
        except Exception: pass
    return {"fixture_files": {k: v for k, v in fdict.items()}, "det": det, "checks_ran": len(det)}, True, ValidationLevel.PROPERTY


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--tasks", default="/tmp/cyber50.json")
    ap.add_argument("--model", default="mellum", choices=["mellum", "gemma", "gptoss", "north", "apertus"])
    args = ap.parse_args()
    tasks = json.load(open(args.tasks))[:args.n]

    ADAPTERS = {"mellum": "mellum_adapter.MellumAdapter", "gemma": "gemma_adapter.GemmaAdapter",
                "gptoss": "gptoss_adapter.GptOssAdapter", "north": "north_adapter.NorthAdapter",
                "apertus": "apertus_adapter.ApertusAdapter"}
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    print(f"[load] {args.model} adapter ...", flush=True)
    model = getattr(__import__(mod), cls)()

    def analyze(sched, task, inputs):
        jin = task.meta["job_inputs"]
        ev = inputs.get("evidence") or {}
        parts = [jin["prompt"]]
        ff = ev.get("fixture_files") or {}
        if ff:
            parts.append("\n\n=== FIXTURE EVIDENCE (read from disk — cite exact IPs/domains/IDs from here) ===\n"
                         + "\n".join(f"--- {k} ---\n{v}" for k, v in ff.items()))
        if ev.get("det"):
            parts.append("\n\n=== VERIFIED DETERMINISTIC FACTS (trust these exactly) ===\n" + json.dumps(ev["det"], default=str))
        ans, _ = model.gen_fast("".join(parts), maxn=1200)   # match the raw arm's headroom (900 truncated cti30/cti49)
        return ans, True, ValidationLevel.SCHEMA

    def cyber_grade(result, task):
        ok, _ = grade_checks(result, task.meta["job_inputs"]["checks"]); return ok

    o = Orchestrator(memory=PlanMemory())
    o.register_job_type("cyber", lambda j: True, "cyber_model")
    o.register_executor("det_evidence", det_evidence)
    o.register_executor("analyze", analyze)
    o.register_executor("commit", lambda s, t, i: (next((v for v in i.values() if v is not None), None), True, ValidationLevel.PROVEN))
    o.register_validator("cyber_grade", cyber_grade)
    for k, owner in {"det_evidence": "primitive:det_primitives_cyber", "analyze": "model_lane:mellum", "commit": "commit_trap"}.items():
        o.assign_agent(k, owner)

    npass = 0
    print(f"[run] {len(tasks)} cyber jobs on {args.model} under the kernel ...", flush=True)
    for i, t in enumerate(tasks):
        job = Job(t["id"], prompt=f"{t.get('system','')}\n\n{t['prompt']}",
                  inputs={"prompt": f"{t.get('system','')}\n\n{t['prompt']}", "checks": t["checks"], "fixture": t.get("fixture")})
        r = o.submit(job)
        npass += int(r["committed"])
        print(f"  {t['id']:8s} diff{t.get('difficulty','?'):<2} {'PASS' if r['committed'] else 'fail '}  "
              f"(ledger {r['ledger_entries']} ✓{r['ledger_verified']})", flush=True)

    dash = o.dashboard()
    print(f"\n===== {len(tasks)}-cyber-on-{args.model}-under-kernel: {npass}/{len(tasks)} committed =====")
    print("  dashboard:", json.dumps(dash, default=str)[:260])


if __name__ == "__main__":
    main()
