#!/usr/bin/env python3
"""M1 model verification: the commit-trap-as-output-path on North over ladder coding tasks (North's domain).

Capability-contrast (NOT parity): gen_committed(min_level=ORACLE) commits ONLY ORACLE-verified-correct code
(running the task's real cases) and CATCHES/repairs the wrong ones — every emit is proof-carrying. The
always-accept control (min_level=SYNTAX) emits the answer unchanged, which equals production gen_fast
(the trap adds nothing when it commits). Loads North; run solo.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commit_trap import ValidationLevel
from kernel.commit_kernel import CommitKernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
def _ladder_path():
    # bundle-first (kernel/fixtures/), then the original repo path — works both standalone and in-tree.
    for p in (os.path.join(_HERE, "fixtures", "ladder_all_cases.json"),
              os.path.join(_REPO, "reports", "north-m0", "ladder_all_cases.json")):
        if os.path.exists(p):
            return p
    return os.path.join(_REPO, "reports", "north-m0", "ladder_all_cases.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--maxn", type=int, default=400)
    a = ap.parse_args()
    tasks = json.load(open(_ladder_path()))[:a.n]
    print(f"[load] North CommitKernel ...", flush=True)
    k = CommitKernel()
    committed = caught = repaired = 0
    for t in tasks:
        r = k.gen_committed(t, max_repairs=1, min_level=ValidationLevel.ORACLE, maxn=a.maxn)
        committed += int(r["committed"])
        caught += int(not r["committed"])
        repaired += int(r["committed"] and r["attempts"] > 0)
        print(f"  {t['id']:4s} {t['entry']:20s} d{t['difficulty']:<2} -> {r['action']:6s} {r['level']:8s} "
              f"attempts={r['attempts']} ledger_ok={r['ledger_verified']}", flush=True)
    print(f"\n===== M1 commit-kernel on North: {committed}/{len(tasks)} ORACLE-committed "
          f"({repaired} via feedback-repair), {caught} caught (NOT emitted) =====", flush=True)

    # always-accept control: at the SYNTAX bar the answer emits unchanged == production gen_fast
    t0 = tasks[0]
    ctrl = k.gen_committed(t0, max_repairs=0, min_level=ValidationLevel.SYNTAX, maxn=a.maxn)
    base, _ = k.gen_fast(t0["prompt"], maxn=a.maxn)
    print(f"[control] {t0['id']}: always-accept(SYNTAX) output == gen_fast: {ctrl['answer'] == base}", flush=True)


if __name__ == "__main__":
    main()
