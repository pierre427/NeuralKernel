#!/usr/bin/env python3
"""kernel/holdout_gate.py — the REUSABLE hold-out admission gate (Step 1 of the self-extending det plane).

The standing trust rule (docs/shape-catalog.md): no deterministic primitive is trusted until it scores 100% on
THOUSANDS of NOVEL instances vs an INDEPENDENT ground truth. Today each docs/shape-appliance-bundle/hold_out_*.py
hand-codes its own seeded generator + independent oracle + __main__ loop — there is NO shared driver. This
generalizes that one methodology into a single driver so a self-extending pipeline can gate a freshly-written
primitive PROGRAMMATICALLY before it is ever trusted.

A gate is three INDEPENDENT pieces (independence is the whole point — a memorized lookup scores ~0% held-out):
  candidate(*inst) -> answer        the primitive under test
  generate(rng)   -> inst (tuple)   a SEEDED novel-instance generator (never the benchmark's cases)
  oracle(*inst)   -> truth          a DIFFERENTLY-CODED ground truth (brute force / stdlib / different formula)
                                    ...or verify(inst, got) -> bool for output-verifier primitives (self-checking
                                    answers where no closed oracle is cheap, e.g. "the returned SAT assignment
                                    actually satisfies every clause").

run_gate -> GateResult{passed (ok==n), ok, n, first_fail, elapsed}. The candidate runs under a per-instance
wall-clock timeout (a bad/hanging primitive fails, never hangs the gate).

PLACEBO control (design §5.4, "fail->pass alone is NOT proof"): gate_with_placebo runs a deliberately-WRONG
mutant through the SAME gate and requires it to FAIL — a gate a placebo also passes has no discriminating power
and is not gating anything. A self-extending pipeline must clear candidate-passes AND placebo-fails before
promotion.

stdlib only (signal/time/random) so it is import-safe anywhere; SIGALRM timeout is main-thread only (documented).
"""
from __future__ import annotations
import copy, signal, time, random
from dataclasses import dataclass
from typing import Callable, Optional, Any, Tuple

# CAVEATS (by design; the caller owns these):
#  * The per-instance timeout (SIGALRM) is SAME-PROCESS and ADVISORY — a candidate can disarm it and FS/net are
#    not blocked. It catches buggy/slow primitives, NOT adversarial ones. Real isolation is the venv-subprocess
#    tool (Step 2); gate self-written code THROUGH that, then run this gate on the result.
#  * Do NOT nest run_gate(), and a candidate must not itself use _timed — the itimer is global; a nested cancel
#    wipes the outer deadline. SIGALRM is main-thread only (off-thread it fails CLOSED: counted as a miss).
#  * compare/_struct_eq is VALUE equality (True==1, 1.0==1; nan never matches -> fails closed). For type-strict
#    primitives pass a custom `compare`. `verify`'s independence is the caller's responsibility.


class _Timeout(Exception):
    pass


def _timed(fn: Callable, args: tuple, t: float = 5.0):
    """Run fn(*args) under a wall-clock timeout (SIGALRM, main thread only). Raises _Timeout on overrun."""
    def _h(sig, frm):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


_SCALAR = (bool, int, float, str, bytes, type(None))

def _struct_eq(a: Any, b: Any) -> bool:
    """Structural equality tolerant of tuple/list interchange and dict ordering. SECURITY (CRITICAL): leaf equality
    is TYPE-GATED to exact JSON scalars so a hostile candidate output cannot win equality via a custom __eq__ or an
    int/str subclass (`class AlwaysEq: __eq__ = lambda s,o: True` would otherwise pass every instance and defeat the
    placebo). Containers must be EXACT list/tuple/dict (a dict subclass with an adversarial keys() is rejected); the
    TRUSTED side (b) drives dict-key iteration. Mismatched/non-scalar/subclass leaves fail CLOSED."""
    ta, tb = type(a), type(b)
    if ta in (list, tuple) and tb in (list, tuple):
        return len(a) == len(b) and all(_struct_eq(x, y) for x, y in zip(a, b))
    if ta is dict and tb is dict:
        return set(a.keys()) == set(b.keys()) and all(_struct_eq(a[k], b[k]) for k in b)
    if ta not in _SCALAR or tb not in _SCALAR:
        return False                                          # non-scalar / hostile-typed leaf -> not equal
    if ta is bool or tb is bool:                              # bool is an int subclass; require both-or-neither bool
        return a is b
    return a == b                                             # both exact JSON scalars: ordinary (numeric) equality


@dataclass
class GateResult:
    passed: bool
    ok: int
    n: int
    elapsed_s: float
    first_fail: Optional[Tuple[Any, Any, Any]] = None   # (instance, got, truth) of the first miss
    n_distinct: int = 0                                 # distinct generated instances (novelty signal)
    note: str = ""


def run_gate(candidate: Callable, generate: Callable, oracle: Optional[Callable] = None, *,
             n: int = 2000, seed: int = 0, compare: Callable = _struct_eq,
             verify: Optional[Callable] = None, timeout_s: float = 5.0,
             min_distinct: Optional[int] = None) -> GateResult:
    """Score `candidate` on `n` seeded novel instances against an INDEPENDENT truth. passed iff ok == n AND the
    generator produced enough DISTINCT instances (a degenerate generator that ignores `rng` lets a hard-coded
    candidate score n/n — so passed fails CLOSED unless n_distinct >= min_distinct; default catches a constant
    generator, n_distinct surfaced for stricter caller policy). Untrusted-candidate BaseExceptions (SystemExit,
    KeyboardInterrupt) are caught as a MISS, never allowed to crash the gate."""
    assert oracle is not None or verify is not None, "supply an independent oracle or a verify()"
    rng = random.Random(seed)
    ok = 0
    first_fail: Optional[Tuple[Any, Any, Any]] = None
    seen = set()
    t0 = time.time()
    for _ in range(n):
        inst = generate(rng)
        if not isinstance(inst, tuple):
            inst = (inst,)
        try:
            seen.add(repr(inst))                                 # novelty signal (repr is hashable + cheap)
        except Exception:
            pass
        try:
            got = _timed(candidate, copy.deepcopy(inst), timeout_s)   # candidate gets its OWN copy: an untrusted
            #   primitive cannot mutate the inputs the INDEPENDENT oracle sees (shared-mutable false-pass). A
            #   non-deepcopyable instance fails CLOSED here.
        except BaseException as e:                               # crash/timeout/SystemExit/KeyboardInterrupt = MISS
            if first_fail is None:
                first_fail = (inst, f"CANDIDATE RAISED {type(e).__name__}: {e}", None)
            continue
        try:
            if verify is not None:
                good, truth = bool(verify(inst, got)), None
            else:
                truth = oracle(*inst)
                good = bool(compare(got, truth))
        except BaseException as e:                               # a broken/hostile oracle/verify also fails closed
            if first_fail is None:
                first_fail = (inst, got, f"ORACLE/VERIFY RAISED {type(e).__name__}: {e}")
            continue
        if good:
            ok += 1
        elif first_fail is None:
            first_fail = (inst, got, truth)
    n_distinct = len(seen)
    # novelty floor: reject a degenerate generator. Default = catch a constant generator (n_distinct < 2 when
    # n>=2); a caller gating an UNTRUSTED generator should pass a stricter min_distinct.
    floor = min_distinct if min_distinct is not None else (2 if n >= 2 else 1)
    novel = (n_distinct >= min(n, floor))
    note = "" if novel else f"degenerate generator: only {n_distinct} distinct of {n} instances"
    # passed requires n>0 (no vacuous promotion), ok==n, AND a non-degenerate generator — all fail CLOSED.
    return GateResult(passed=(n > 0 and ok == n and novel), ok=ok, n=n, elapsed_s=round(time.time() - t0, 3),
                      first_fail=first_fail, n_distinct=n_distinct, note=note)


def gate_with_placebo(candidate: Callable, placebo: Callable, generate: Callable,
                      oracle: Optional[Callable] = None, **kw) -> Tuple[GateResult, GateResult, bool]:
    """Run the real candidate AND a deliberately-wrong placebo through the SAME gate. Returns
    (cand, placebo, gate_is_sound) where gate_is_sound == (cand.passed AND NOT placebo.passed) — the gate only
    means something if it accepts the real primitive and rejects the placebo (design §5.4)."""
    cand = run_gate(candidate, generate, oracle, **kw)
    plac = run_gate(placebo, generate, oracle, **kw)
    return cand, plac, (cand.passed and not plac.passed)


def prove(name: str, candidate: Callable, generate: Callable, oracle: Optional[Callable] = None, **kw) -> GateResult:
    """Programmatic admission assertion: run_gate and RAISE unless 100%. The promotion gate of the pipeline."""
    r = run_gate(candidate, generate, oracle, **kw)
    if not r.passed:
        raise AssertionError(f"{name}: hold-out gate FAILED {r.ok}/{r.n}  first_fail={r.first_fail}")
    return r


__all__ = ["GateResult", "run_gate", "gate_with_placebo", "prove", "_struct_eq"]
