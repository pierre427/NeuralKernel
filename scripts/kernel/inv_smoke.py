#!/usr/bin/env python3
"""Smoke the domain-agnostic investigation protocol on a handful of cyber tasks for a given model.
Reports per-task: routed kind, ACH leader, grade. Used to test whether a CAPABLE executor makes the
structured protocol pay off (weak models execute the stages poorly).

EMPIRICAL VERDICT (2026-06-22, cyber-50): the heavy ACH protocol this drives is NET-NEGATIVE vs single-shot —
gpt-oss 10/50 (this) vs 23/50 raw vs 29/50 thin (run_cyber_mellum), at ~3x the wall-clock; 0/5 on Mellum's
hardest. RETIRED as a default path. Kept as an experiment / for FRAME-gated, validation-justified escalation
only. The validated DEFAULT cyber kernel is the THIN path (run_cyber_mellum). See docs/neural-microkernel-design.md
"0b. Empirical validation" and the raw_baseline.py causal control."""
import sys, os, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernel.investigation as inv
from kernel.fixture_tools import FixtureSandbox, recon
from kernel.run_cyber_mellum import grade_checks, FIX_ROOT, det_evidence
from kernel.task_graph import Task

ADAPTERS = {"mellum": "mellum_adapter.MellumAdapter", "gptoss": "gptoss_adapter.GptOssAdapter",
            "gemma": "gemma_adapter.GemmaAdapter", "north": "north_adapter.NorthAdapter"}
CONTRACT = ("Return ONE JSON object: answer, confidence, known_facts, evidence, inferred_links, alternatives, "
            "uncertainties, next_steps (arrays for the list ones); put exact tokens/labels in values.")
# the cyber output contract — enforced deterministically by enforce_contract() from investigation state
CYBER_SPEC = {"keys": ["answer", "confidence", "known_facts", "evidence", "inferred_links", "alternatives",
                       "uncertainties", "next_steps"],
              "array_keys": ["known_facts", "evidence", "alternatives"]}


class CyberSource(inv.EvidenceSource):
    def __init__(self, fixture, prompt):
        self.sb = FixtureSandbox(os.path.join(FIX_ROOT, fixture)) if fixture else None
        self._recon = None
        if self.sb and self.sb.files:
            tk = Task("e", "d"); tk.meta["job_inputs"] = {"prompt": prompt, "fixture": fixture}
            self._recon = recon(self.sb, det_evidence(None, tk, {})[0].get("det"))

    def manifest(self):
        return self.sb.manifest() if (self.sb and self.sb.files) else "(no fixture)"

    def collect(self, query):
        if not self.sb or not self.sb.files:
            return ""
        parts = [self._recon] if self._recon else []
        for term in re.findall(r"\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z_][A-Za-z0-9_.]{3,}", query)[:3]:
            r = self.sb.grep(re.escape(term), max_hits=5)
            if "no matches" not in r:
                parts.append(r)
        return "\n".join(parts)[:900]


def _scen(t):
    return re.split(r"Return ONLY one JSON|Return one line|Return:|Return JSON|Answer with:", t["prompt"])[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptoss", choices=list(ADAPTERS))
    ap.add_argument("--ids", default="cti02,cti04,cti06,cti15,cti16")
    ap.add_argument("--tasks", default="/tmp/cyber50.json")
    args = ap.parse_args()
    mod, cls = ADAPTERS[args.model].rsplit(".", 1)
    model = getattr(__import__(mod), cls)()
    tasks = {t["id"]: t for t in json.load(open(args.tasks))}
    npass = 0; ids = args.ids.split(",")
    for tid in ids:
        t = tasks[tid]; src = CyberSource(t["fixture"], t["prompt"])
        rep, st = inv.investigate(model, src, _scen(t), contract_hint=CONTRACT, max_rounds=2, contract_spec=CYBER_SPEC)
        ok, per = grade_checks(rep, t["checks"]); npass += int(ok)
        print(f"  {tid:7s} d{t['difficulty']:<2} routed={st.frame.get('routed','diagnostic-full'):18s} "
              f"rounds={st.rounds} suff={st.sufficiency.get('sufficient')} conf={st.sufficiency.get('confidence')} "
              f"leader={st.leader.get('statement','')[:38]:38s} {'PASS' if ok else 'fail'} "
              f"failed={[l for l,o in per if not o]}", flush=True)
    print(f"\n===== {args.model} investigation-protocol: {npass}/{len(ids)} =====", flush=True)


if __name__ == "__main__":
    main()
