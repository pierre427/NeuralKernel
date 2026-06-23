#!/usr/bin/env python3
"""kernel/venv_tool.py — the `venv_exec` SYSCALL-tier tool: run model-written candidate code OUT OF PROCESS in a
hermetic Python venv, against its cases, under isolation. The model's write→test→validate substrate, and the
out-of-process executor that SEALS the hold-out gate's advisory-timeout caveat (an in-process SIGALRM can be
disarmed by the candidate; a subprocess wall-clock kill cannot).

Isolation (layered; out-of-process is solid, the rest is defense-in-depth — Docker is the hardening tier for a
genuinely hostile model). Per candidate RUN: a subprocess in its OWN process group (start_new_session) so a
timeout kills the whole tree (killpg), not just the leader; a wall-clock timeout (the real backstop); a CPU rlimit
(RLIMIT_CPU bites; RLIMIT_AS best-effort on darwin); and on darwin a sandbox-exec profile that denies network AND
denies file-WRITES outside the per-run cwd (so a candidate can't persist or POISON the shared venv's site-packages)
and denies reads of ~/.ssh|.aws|.gnupg. The candidate is exec'd in a FRESH namespace in the child; nothing reaches
back into the kernel (the parent only json-parses the child's last stdout line).

RESIDUALS (honest): file-READS elsewhere are still allowed but bounded by no-net (a read secret can only be handed
to the model, not exfiltrated remotely); a candidate that setsid's its OWN grandchild escapes killpg; the venv is
process-SHARED (not per-call hermetic); off-darwin / without sandbox-exec the net+write deny does NOT apply
(surfaced as result.sandboxed_nonet=False). `deps`/pip is NOT model-reachable (supply-chain). Docker
(harness/sandbox/docker.mjs: read-only, network=none, pids-limit) is the principled fix for these residuals.
NOT 'proven' — syscall tier; the hold-out gate confers trust.
"""
from __future__ import annotations
import os, sys, json, time, atexit, signal, shutil, tempfile, subprocess

try:
    import resource                                          # POSIX only (darwin/linux)
except ImportError:
    resource = None

try:                                                         # the TRUSTED comparator (runs in the parent only)
    from .holdout_gate import _struct_eq
except ImportError:
    from holdout_gate import _struct_eq


def _judge(out, truth):
    """Parent-side verdict for one child output `out` ({v|r|e}) vs the trusted `truth`. Untrusted code never runs
    here — `out` is data. A json value compares structurally; a non-json repr compares to repr(truth)."""
    if "e" in out:
        return False
    if "v" in out:
        return _struct_eq(out["v"], truth)
    return out.get("r") == repr(truth)                        # non-json result -> repr-equality


def _clamp(v, lo, hi, default):
    """Coerce a (possibly model-supplied / non-numeric / inf / nan) value into [lo, hi], else `default`."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):          # nan / inf
        return default
    return max(lo, min(x, hi))

_VENV_DIR = None                                             # cached hermetic venv path (provisioned on first use)

# Child-side RUNNER: run the (untrusted) candidate on each arg-list and emit its RAW OUTPUTS ONLY (no verdict in
# the child — the TRUSTED PARENT compares). The result line is FENCED by a per-run NONCE the parent injects via env
# and the runner POPS before exec'ing the candidate, then the runner os._exit(0)s (skipping atexit). So a candidate
# CANNOT forge an accepted result: an atexit/print/os._exit forge line lacks the nonce -> the parent rejects it,
# and os._exit prevents an atexit forge from running at all. (Residual: deep frame-introspection to steal the
# nonce from the parent runner's locals — closed only by the Docker tier; documented.)
_RUNNER = r'''import os, sys, json
def _main():
    spec = json.load(open(sys.argv[1]))
    ns = {}
    try:
        exec(spec["code"], ns)
    except BaseException as e:
        return {"exec_error": "%s: %s" % (type(e).__name__, e)}
    fn = ns.get(spec["entry"])
    if not callable(fn):
        return {"exec_error": "entry %r not defined/callable" % spec["entry"]}
    outs = []
    for args in spec["arglists"]:
        try:
            v = fn(*args)
        except BaseException as e:
            outs.append({"e": "%s: %s" % (type(e).__name__, e)}); continue
        try:
            json.dumps(v); outs.append({"v": v})
        except Exception:
            outs.append({"r": repr(v)[:400]})
    return {"outs": outs}
def _emit():
    _n = os.environ.pop("KERNEL_RUNNER_NONCE", "")
    _r = _main()
    sys.stdout.write(_n + json.dumps(_r) + "\n"); sys.stdout.flush()
    os._exit(0)
_emit()
'''

# Child-side CALL: run a (trusted, gate-passed) primitive on args, emit its RESULT (nonce-fenced, same discipline).
_CALL_RUNNER = r'''import os, sys, json
def _main():
    spec = json.load(open(sys.argv[1])); ns = {}
    try:
        exec(spec["code"], ns)
    except BaseException as e:
        return {"ok": False, "error": "exec: %s: %s" % (type(e).__name__, e)}
    fn = ns.get(spec["entry"])
    if not callable(fn):
        return {"ok": False, "error": "entry not callable"}
    try:
        r = fn(*spec.get("args", []))
    except BaseException as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    try:
        json.dumps(r); return {"ok": True, "result": r}
    except Exception:
        return {"ok": True, "result_repr": repr(r)[:500]}
def _emit():
    _n = os.environ.pop("KERNEL_RUNNER_NONCE", "")
    _r = _main()
    sys.stdout.write(_n + json.dumps(_r) + "\n"); sys.stdout.flush()
    os._exit(0)
_emit()
'''


def _cleanup():
    global _VENV_DIR
    if _VENV_DIR and os.path.isdir(_VENV_DIR):
        shutil.rmtree(_VENV_DIR, ignore_errors=True)


def _ensure_venv(deps=None):
    """Provision a hermetic venv once (cached); pip-install deps if requested (net ALLOWED here — provisioning is
    a trusted kernel step, distinct from the no-net candidate RUN). Returns the venv's python path."""
    global _VENV_DIR
    if not (_VENV_DIR and os.path.exists(os.path.join(_VENV_DIR, "bin", "python"))):
        d = tempfile.mkdtemp(prefix="kernel-venv-")
        subprocess.run([sys.executable, "-m", "venv", d], check=True, capture_output=True, timeout=180)
        _VENV_DIR = d
        atexit.register(_cleanup)
    py = os.path.join(_VENV_DIR, "bin", "python")
    if deps:
        subprocess.run([py, "-m", "pip", "install", "--quiet", *list(deps)], check=True, capture_output=True, timeout=420)
    return py


def _limits(mem_mb, cpu_s):
    if resource is None:
        return None
    def _set():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        except Exception:
            pass
        try:
            b = int(mem_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))     # best-effort (often unenforced on darwin)
        except Exception:
            pass
    return _set


def _killpg(proc):
    """Kill the whole process GROUP (not just the leader) so detached grandchildren a candidate spawned don't
    outlive the wall-clock kill. (A candidate that itself setsid's a grandchild still escapes — Docker is the
    real containment for that; this closes the common fork-and-detach case the review demonstrated.)"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _sandbox_profile(cwd):
    """macOS SBPL: no network; deny reads of common secret dirs; deny ALL writes except the per-run cwd + tmp —
    so a candidate can neither persist outside its scratch dir NOR poison the shared venv's site-packages
    (cross-run contamination). Last-match-wins, so the cwd allow overrides the blanket write-deny."""
    home = os.path.expanduser("~")
    cwd_real = os.path.realpath(cwd)
    # Allow writes ONLY under this run's cwd — NOT the whole tmp root, because the cached venv also lives under
    # tmp (mkdtemp); allowing all of tmp would re-open the venv-poisoning hole. /dev/null|urandom for hygiene.
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f'(deny file-read* (subpath "{home}/.ssh") (subpath "{home}/.aws") (subpath "{home}/.gnupg"))',
        "(deny file-write*)",
        f'(allow file-write* (subpath "{cwd_real}") (literal "/dev/null") (literal "/dev/urandom"))',
    ])


def _sandboxed_run(runner_src, spec_obj, *, timeout_s=10.0, mem_mb=512, cpu_s=None, allow_net=False,
                   python_exe=None, deps=None):
    """Run `runner_src` (a self-contained child script) over `spec_obj` (json) in the isolated subprocess — the
    ONE place the security-critical launch lives (sandbox profile + process-group kill + rlimits + no-bytecode).
    Returns the child's parsed json result, annotated with elapsed_ms/exit_code/sandboxed_nonet."""
    py = python_exe or _ensure_venv(deps)
    timeout_s = _clamp(timeout_s, 0.1, 120.0, 10.0)            # defense-in-depth clamp (reject inf/nan/neg/huge)
    mem_mb = int(_clamp(mem_mb, 16, 4096, 512))
    cpu_s = int(cpu_s or max(1, int(timeout_s) + 1))
    nonce = os.urandom(16).hex()                              # fences the runner's result line (anti-forge)
    with tempfile.TemporaryDirectory() as d:
        spec_path, runner_path = os.path.join(d, "spec.json"), os.path.join(d, "runner.py")
        try:
            json.dump(spec_obj, open(spec_path, "w"))
        except Exception as e:
            return {"ok": False, "error": f"spec not json-serializable: {type(e).__name__}: {e}"}
        open(runner_path, "w").write(runner_src)
        cmd = [py, runner_path, spec_path]
        sandboxed = False
        if not allow_net and sys.platform == "darwin" and shutil.which("sandbox-exec"):
            prof = os.path.join(d, "nonet.sb")
            open(prof, "w").write(_sandbox_profile(d))         # no-net + deny-write-outside-cwd (closes poisoning)
            cmd = ["sandbox-exec", "-f", prof] + cmd
            sandboxed = True
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", KERNEL_RUNNER_NONCE=nonce)
        t0 = time.time()
        try:
            # start_new_session => the child leads its own process GROUP, so a timeout can killpg the whole tree
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    preexec_fn=_limits(mem_mb, cpu_s), cwd=d, env=env, start_new_session=True)
        except Exception as e:
            return {"ok": False, "error": f"spawn failed: {type(e).__name__}: {e}"}
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _killpg(proc)                                      # kill the GROUP, not just the leader
            try:
                proc.communicate(timeout=5)                    # reap
            except Exception:
                pass
            return {"ok": False, "error": "timeout", "killed": True,
                    "elapsed_ms": round((time.time() - t0) * 1000, 1), "sandboxed_nonet": sandboxed}
        elapsed = round((time.time() - t0) * 1000, 1)
        # Accept ONLY the runner's NONCE-fenced line — a candidate-forged line (atexit/print/os._exit) lacks the
        # nonce and is rejected; no nonce line at all (candidate os._exit'd mid-run, or crashed) => fail closed.
        line = next((l for l in (out or "").splitlines() if l.startswith(nonce)), None)
        if line is None:
            res = {"ok": False, "error": "no nonce-fenced runner output (candidate crashed/forged?)",
                   "stderr": (err or "")[-300:]}
        else:
            try:
                res = json.loads(line[len(nonce):])
            except Exception:
                res = {"ok": False, "error": "unparseable runner output"}
        if proc.returncode not in (0, None) and "error" not in res and "outs" not in res and "exec_error" not in res:
            res = {"ok": False, "error": f"child exit {proc.returncode}", "stderr": (err or "")[-300:]}
        res["elapsed_ms"] = elapsed
        res["exit_code"] = proc.returncode
        res["sandboxed_nonet"] = sandboxed
        return res


def run_candidate(code, entry, cases, *, deps=None, timeout_s=10.0, mem_mb=512, cpu_s=None,
                  allow_net=False, python_exe=None):
    """Run `code` (must define `entry`) against `cases` in the sandbox; the child emits raw outputs and the
    TRUSTED PARENT compares them to each case's expected. ok=True iff it ran and passed ALL cases (>=1 case)."""
    if not code or not entry:
        return {"ok": False, "error": "code and entry are required"}
    cases = cases or []
    arglists = [c["args"] if "args" in c else ([c["input"]] if "input" in c else []) for c in cases]
    r = _sandboxed_run(_RUNNER, {"code": code, "entry": entry, "arglists": arglists}, timeout_s=timeout_s,
                       mem_mb=mem_mb, cpu_s=cpu_s, allow_net=allow_net, python_exe=python_exe, deps=deps)
    meta = {k: r.get(k) for k in ("elapsed_ms", "exit_code", "sandboxed_nonet", "killed") if r.get(k) is not None}
    if r.get("exec_error"):
        return {"ok": False, "error": r["exec_error"], "passed": 0, "total": len(cases), "fails": [], **meta}
    outs = r.get("outs")
    if outs is None:
        return {"ok": False, "error": r.get("error", "no outputs from child"), "passed": 0, "total": len(cases),
                "fails": [], **meta}
    if len(outs) != len(cases):                              # forged/short output -> reject (parent validates count)
        return {"ok": False, "error": f"child returned {len(outs)} of {len(cases)} outputs", "passed": 0,
                "total": len(cases), "fails": [], **meta}
    passed = 0; fails = []
    for i, (c, o) in enumerate(zip(cases, outs)):            # PARENT-side comparison — untrusted code can't reach it
        if "e" in o:
            fails.append({"case": i, "error": o["e"]})
        elif _judge(o, c.get("expected")):
            passed += 1
        else:
            fails.append({"case": i, "got": repr(o.get("v", o.get("r")))[:200], "expected": repr(c.get("expected"))[:200]})
    return {"ok": len(fails) == 0 and len(cases) > 0, "passed": passed, "total": len(cases), "fails": fails[:10], **meta}


def run_gate_in_venv(code, entry, generate_src, oracle_src, *, n=2000, seed=0, timeout_s=60.0, mem_mb=512,
                     python_exe=None):
    """HOLD-OUT GATE with the trust boundary in the right place: the TRUSTED PARENT exec's the gap's generate +
    oracle (gap code = the kernel's source of truth), builds n seeded novel instances and their truths, runs ONLY
    the untrusted candidate on those instances in the sandbox (one spawn, raw outputs), then judges in-parent.
    ok=True iff every instance matched (n>0). The candidate never sees the oracle nor reaches the comparator."""
    if not code or not entry:
        return {"ok": False, "error": "code and entry are required"}
    gns = {}
    try:
        exec(generate_src, gns); exec(oracle_src, gns)       # gap code is TRUSTED (kernel-supplied) -> runs in-parent
    except Exception as e:
        return {"ok": False, "error": f"gap generate/oracle exec: {type(e).__name__}: {e}", "passed": 0, "n": n}
    generate, oracle = gns.get("generate"), gns.get("oracle")
    if not (callable(generate) and callable(oracle)):
        return {"ok": False, "error": "gap must define generate(rng) and oracle(*inst)", "passed": 0, "n": n}
    import random as _random
    rng = _random.Random(seed)
    insts = []
    for _ in range(n):
        inst = generate(rng)
        insts.append(list(inst) if isinstance(inst, tuple) else [inst])
    r = _sandboxed_run(_RUNNER, {"code": code, "entry": entry, "arglists": insts},
                       timeout_s=timeout_s, mem_mb=mem_mb, python_exe=python_exe)
    meta = {k: r.get(k) for k in ("elapsed_ms", "sandboxed_nonet", "killed") if r.get(k) is not None}
    if r.get("exec_error"):
        return {"ok": False, "error": r["exec_error"], "passed": 0, "n": n, **meta}
    outs = r.get("outs")
    if outs is None:
        return {"ok": False, "error": r.get("error", "no outputs from child"), "passed": 0, "n": n, **meta}
    if len(outs) != n:                                       # forged/short output (can't fake n unknown answers)
        return {"ok": False, "error": f"child returned {len(outs)} of {n} outputs", "passed": 0, "n": n, **meta}
    passed = 0; first_fail = None
    for inst, o in zip(insts, outs):                         # PARENT-side compare vs the trusted oracle's truth
        if "e" in o:
            if first_fail is None: first_fail = {"inst": repr(inst)[:160], "error": o["e"]}
            continue
        try:
            truth = oracle(*inst)
        except Exception as e:
            if first_fail is None: first_fail = {"inst": repr(inst)[:160], "oracle_error": f"{type(e).__name__}: {e}"}
            continue
        if _judge(o, truth):
            passed += 1
        elif first_fail is None:
            first_fail = {"inst": repr(inst)[:160], "got": repr(o.get("v", o.get("r")))[:160], "truth": repr(truth)[:160]}
    return {"ok": passed == n and n > 0, "passed": passed, "n": n, "first_fail": first_fail, **meta}


def call_in_venv(code, entry, args, *, timeout_s=10.0, mem_mb=512, python_exe=None):
    """Call a (trusted, gate-passed) primitive on args in the sandbox; returns {ok, result} (or result_repr if the
    return value isn't json-able). Keeps even a registered primitive's dispatch out-of-process."""
    if not code or not entry:
        return {"ok": False, "error": "code and entry are required"}
    return _sandboxed_run(_CALL_RUNNER, {"code": code, "entry": entry, "args": list(args or [])},
                          timeout_s=timeout_s, mem_mb=mem_mb, python_exe=python_exe)


def _venv_ok(result, task) -> bool:
    """venv_exec node passes iff the candidate ran AND passed all its cases."""
    return isinstance(result, dict) and result.get("ok") is True


def venv_exec_handler(sched, inputs):
    inputs = inputs or {}
    # NOTE: `deps` (pip install) is deliberately NOT model-reachable — installing a model-chosen package runs
    # arbitrary setup code with network (supply-chain / typosquat). deps stays a kernel-only run_candidate param;
    # the kernel pre-provisions any needed deps. allow_net is likewise ignored here (the model can't lift no-net).
    # timeout_s/mem_mb come from the model -> pass raw; _sandboxed_run._clamp rejects inf/nan/neg/huge so a
    # model can't defeat the wall-clock (timeout_s=1e12) or the rlimit. allow_net/python_exe are NOT model-liftable.
    return run_candidate(inputs.get("code"), inputs.get("entry"), inputs.get("cases", []),
                         timeout_s=inputs.get("timeout_s", 10), mem_mb=inputs.get("mem_mb", 512),
                         allow_net=False, python_exe=None)


def venv_tool_spec():
    """The ToolSpec for venv_exec (registered as a default syscall-tier tool; venv is provisioned lazily on first
    call, so registration is cheap)."""
    try:
        from .tools import ToolSpec
    except ImportError:
        from tools import ToolSpec
    return ToolSpec(name="venv_exec", handler=venv_exec_handler, result_ok=_venv_ok,
                    description="Run model-written Python code OUT OF PROCESS in a hermetic, no-network, "
                                "time+resource-limited venv against test cases; returns per-case pass/fail. Use "
                                "to write, test, and validate candidate code safely. Passes iff all cases pass.",
                    signature="venv_exec(code:str, entry:str, cases:[{args|input, expected}], deps?, timeout_s=10, mem_mb=512, allow_net=False)")


__all__ = ["run_candidate", "run_gate_in_venv", "call_in_venv", "venv_exec_handler", "venv_tool_spec", "_venv_ok"]
