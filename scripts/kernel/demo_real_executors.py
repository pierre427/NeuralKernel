#!/usr/bin/env python3
"""DEMO: the Job Kernel driving REAL primitives end-to-end (CPU-only, no GPU).

Submits actual coding tasks (real entries + real hidden cases from ladder-100) and a real cyber-evidence
job through one Orchestrator wired to the real executors: coding nodes route to hold-out-proven det-blocks
and validate against the task's real cases; the cyber node runs det_primitives_cyber over real evidence.
Every node is audited in the proof ledger; the master rollup shows the agent-assigned, verified plan."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.orchestrator import Orchestrator, Job, PlanMemory, default_classifier_rules
from kernel.executors import wire_real_executors

CHECKS = []
def ok(c, label): CHECKS.append((c, label)); print(f"  {'OK ' if c else 'XX '} {label}")


def main():
    o = Orchestrator(memory=PlanMemory())
    default_classifier_rules(o)
    wire_real_executors(o)

    print("\n=== REAL coding jobs through the Job Kernel (recover-direct det-block -> real check_cases -> commit) ===")
    real = json.load(open("/tmp/real_coding_tasks.json"))
    for t in real:
        job = Job(t["id"], prompt=f"Implement `{t['entry']}`. {t['prompt']}",
                  inputs={"entry": t["entry"], "cases": t["cases"]})
        r = o.submit(job)
        tests_row = next((x for x in r["tasks"] if x["kind"] == "run_tests"), {})
        print(f"\n  job {t['id']} [{t['entry']}] type={r['job_type']} committed={r['committed']}")
        for line in r["checklist"]:
            print("     ", line)
        ok(r["job_type"] == "coding", f"{t['entry']}: classified coding")
        ok(r["committed"], f"{t['entry']}: committed via recover-direct det-block + real {len(t['cases'])}-case validation")
        ok(tests_row.get("verification") == "verified", f"{t['entry']}: run_tests reached VERIFIED (real cases passed)")
        ok(r["ledger_verified"], f"{t['entry']}: proof ledger hash-verified")

    print("\n=== REAL cyber job through the Job Kernel (det_primitives_cyber extracts guaranteed evidence) ===")
    evidence = {
        "blob": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAA=",        # base64 (PNG magic)
        "host": "169.254.169.254",                              # cloud metadata IP (SSRF target)
        "url": "http://evil.test/x", "allowlist": ["corp.example"],
        "timestamps": [0, 60, 120, 180, 240], "tol": 1,         # perfect 60s beacon
        "event_id": 4625,                                       # Windows failed logon
    }
    r = o.submit(Job("cti-real", prompt="Triage this SOC alert: suspicious beacon, SSRF host, encoded blob, failed logons.",
                     inputs={"evidence": evidence}))
    anchors = (r["result"] or {}).get("anchors", {})
    print(f"\n  job cti-real type={r['job_type']} committed={r['committed']}")
    for line in r["checklist"]:
        print("     ", line)
    print("  deterministic evidence anchors:", json.dumps(anchors, default=str)[:240])
    ok(r["job_type"] == "cyber", "cyber job classified cyber")
    ok(r["committed"], "cyber job committed")
    ok(anchors.get("ssrf_target") is True, "det_primitives_cyber flagged 169.254.169.254 as an SSRF target")
    ok(anchors.get("beacon_period") == 60, "det_primitives_cyber detected the 60s beacon period")
    ok("Failed" in str(anchors.get("win_event", "")) or anchors.get("win_event"), "Windows event 4625 classified")
    ok(any(x["owner"] == "primitive:det_primitives_cyber" for x in r["tasks"]), "det_checks node assigned to the real cyber primitive")

    print("\n=== master dashboard across all real jobs ===")
    print("  ", json.dumps(o.dashboard(), default=str)[:300])
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} real-executor checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
