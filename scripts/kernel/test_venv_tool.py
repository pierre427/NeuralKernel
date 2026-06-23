#!/usr/bin/env python3
"""CPU test for the venv_exec tool (no model; provisions a real hermetic venv, cached + reused across runs).
Proves: good/buggy candidates graded by cases; candidate runs OUT OF PROCESS (pid != parent); an infinite loop is
wall-clock-killed (not the kernel); a raising candidate is captured per-case; no-case runs can't validate; the
node passes only when the candidate passes all cases; venv_exec ships gated as op:venv_exec."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/ (so `kernel` imports as a package)
from kernel.venv_tool import run_candidate, _venv_ok, _clamp
from kernel.tools import register_builtins, TOOL_CATALOG
from kernel.task_graph import Scheduler

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


def main():
    # 1. good candidate, all cases pass, through the real hermetic venv (provisions on first call, then cached)
    g = run_candidate("def add(a, b):\n    return a + b", "add",
                      [{"args": [2, 3], "expected": 5}, {"args": [10, 20], "expected": 30}])
    ok(g.get("ok") and g.get("passed") == 2 and g.get("total") == 2, f"good candidate passes 2/2 in venv (exit={g.get('exit_code')})")
    if sys.platform == "darwin":
        ok(g.get("sandboxed_nonet") is True, "darwin: candidate ran under sandbox-exec (deny network*)")

    # 2. buggy candidate fails its cases and reports them
    b = run_candidate("def add(a, b):\n    return a * b", "add",
                      [{"args": [2, 3], "expected": 5}, {"args": [10, 20], "expected": 30}])
    ok((not b.get("ok")) and b.get("passed") == 0 and len(b.get("fails", [])) > 0, "buggy candidate fails 0/2, reports fails")

    # 3. OUT OF PROCESS: the candidate's pid differs from this process (in-process exec would match -> pass)
    iso = run_candidate("import os\ndef f(_):\n    return os.getpid()", "f", [{"args": [0], "expected": os.getpid()}])
    ok((not iso.get("ok")) and iso.get("passed") == 0, "candidate runs in a SEPARATE process (pid != parent)")

    # 4. timeout: an infinite loop is hard-killed by the wall clock, the kernel process survives
    t = run_candidate("def f(_):\n    while True:\n        pass", "f", [{"args": [0], "expected": 1}], timeout_s=1)
    ok((not t.get("ok")) and t.get("killed"), f"infinite-loop candidate -> wall-clock kill ({t.get('error')})")

    # 5. crashing candidate -> per-case error captured, no kernel crash
    c = run_candidate("def f(_):\n    raise ValueError('boom')", "f", [{"args": [0], "expected": 1}])
    ok((not c.get("ok")) and any("ValueError" in str(x.get("error", "")) for x in c.get("fails", [])),
       "raising candidate -> per-case error captured (kernel survives)")

    # 6. no cases -> cannot validate
    ok(not run_candidate("def f(x):\n    return x", "f", []).get("ok"), "no cases -> ok=False (cannot validate)")
    ok(not run_candidate("", "f", [{"args": [1], "expected": 1}]).get("ok"), "empty code -> ok=False (no entry)")

    # 7. _venv_ok: node passes ONLY when the candidate passed all cases
    ok(_venv_ok({"ok": True}, None) and not _venv_ok({"ok": False, "passed": 1}, None), "_venv_ok requires candidate ok=True")

    # 8. registration: venv_exec ships as a default tool, gated op:venv_exec
    wirings = register_builtins(Scheduler(), grant=True)
    w = next((x for x in wirings if x["kind"] == "venv_exec"), None)
    ok(w is not None and w["capability"] == "op:venv_exec" and "venv_exec" in TOOL_CATALOG,
       "venv_exec registered as a default tool (op:venv_exec)")

    # 9. WRITE CONTAINMENT (darwin sandbox): a candidate cannot persist outside its per-run scratch cwd
    if sys.platform == "darwin":
        evil = os.path.expanduser("~/.kernel_test_evil_probe")
        if os.path.exists(evil):
            os.remove(evil)
        w = run_candidate("def f(_):\n    open(%r, 'w').write('x')\n    return 'wrote'" % evil, "f",
                          [{"args": [0], "expected": "wrote"}])
        ok((not w.get("ok")) and (not os.path.exists(evil)), "candidate write OUTSIDE cwd is DENIED (no host file)")
        if os.path.exists(evil):
            os.remove(evil)

        # 10. VENV-POISONING CLOSED: a candidate cannot write into the shared venv's site-packages (cross-run)
        import glob as _g
        import kernel.venv_tool as _vt
        sp = _g.glob(os.path.join(_vt._VENV_DIR or "", "lib", "python*", "site-packages")) if _vt._VENV_DIR else []
        if sp:
            target = os.path.join(sp[0], "evil_poison.py")
            p = run_candidate("def f(_):\n    open(%r, 'w').write('X=1')\n    return 1" % target, "f",
                              [{"args": [0], "expected": 1}])
            ok((not p.get("ok")) and (not os.path.exists(target)),
               "candidate canNOT poison the shared venv site-packages (cross-run contamination closed)")

    # 11. BUG-SWEEP HIGH: a forged nonce-less output line (candidate os._exit's before the runner emits) is REJECTED
    FORGE = ("import os, sys, json\n"
             "sys.stdout.write(json.dumps({'outs': [{'v': 5}]}) + '\\n')\n"
             "sys.stdout.flush()\n"
             "os._exit(0)\n")
    fr = run_candidate(FORGE, "add", [{"args": [2, 3], "expected": 5}])
    ok(not fr.get("ok"), "forged nonce-less output -> REJECTED (parent accepts only the nonce-fenced runner line)")

    # 12. BUG-SWEEP HIGH: timeout/mem clamp (inf/nan/non-numeric/neg -> safe) so a model can't set timeout_s=1e12
    ok(_clamp(1e12, 0.1, 120, 10) == 120 and _clamp("abc", 0.1, 120, 10) == 10
       and _clamp(float("nan"), 0.1, 120, 10) == 10 and _clamp(-5, 0.1, 120, 10) == 0.1,
       "_clamp bounds timeout/mem (1e12->120, nan/non-numeric->default, neg->lo): wall-clock can't be disabled")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} venv_exec checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
