#!/usr/bin/env python3
"""CPU regression for the reusable hold-out gate (no model). Proves: (1) an EXISTING trusted primitive (c_gcd)
re-proves at 100% through the generic driver vs an INDEPENDENT stdlib oracle; (2) a second primitive (is_prime,
wheel) passes vs a brute-force oracle; (3) the PLACEBO control discriminates (a wrong gcd FAILS the same gate);
(4) a hanging candidate is killed by the per-instance timeout instead of hanging the gate."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ (det_primitives)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # kernel/
from holdout_gate import run_gate, gate_with_placebo
from det_primitives import c_gcd                                                   # a real TRUSTED primitive

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


# --- seeded novel-instance generators + INDEPENDENT (differently-coded) oracles ---
def gen_gcd(rng):
    return (rng.randint(1, 10 ** 9), rng.randint(1, 10 ** 9))

def cand_is_prime(n):                       # 6k+-1 wheel
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def brute_is_prime(n):                       # INDEPENDENT: full trial division 2..n-1
    if n < 2: return False
    return all(n % d for d in range(2, n))

def gen_small_int(rng):
    return (rng.randint(0, 5000),)


def main():
    # 1. existing trusted primitive re-proves at 100% via the generic gate, vs an INDEPENDENT stdlib oracle
    g = run_gate(c_gcd, gen_gcd, math.gcd, n=3000, seed=7)
    ok(g.passed and g.ok == 3000, f"c_gcd 100% vs math.gcd ({g.ok}/{g.n}, {g.elapsed_s}s)  first_fail={g.first_fail}")

    # 2. a second primitive (wheel) gated against a brute-force oracle of a DIFFERENT algorithm
    p = run_gate(cand_is_prime, gen_small_int, brute_is_prime, n=3000, seed=11)
    ok(p.passed, f"is_prime(wheel) 100% vs brute trial-division ({p.ok}/{p.n}, {p.elapsed_s}s)  first_fail={p.first_fail}")

    # 3. PLACEBO control: a deliberately-wrong gcd (a+b) must FAIL the same gate (discrimination power)
    cand, plac, sound = gate_with_placebo(c_gcd, (lambda a, b: a + b), gen_gcd, math.gcd, n=1000, seed=3)
    ok(cand.passed, "placebo run: real c_gcd still passes")
    ok(not plac.passed, f"placebo (a+b) FAILS the gate ({plac.ok}/{plac.n}) — the gate discriminates")
    ok(sound, "gate_with_placebo sound: candidate passes AND placebo fails (fail->pass alone is NOT proof)")

    # 4. a hanging candidate is caught by the per-instance timeout, not allowed to hang the gate
    import time as _t
    slow = run_gate((lambda x: _t.sleep(10) or x), gen_small_int, (lambda x: x), n=2, seed=0, timeout_s=0.3)
    ok((not slow.passed) and slow.first_fail is not None, "hanging candidate -> timeout -> fail (gate never hangs)")

    # 5. HIGH-fix: a candidate that MUTATES a shared mutable instance cannot corrupt the oracle into agreeing
    def gen_list(rng):
        return ([rng.randint(0, 9) for _ in range(5)],)
    mut = run_gate((lambda xs: (xs.__setitem__(0, 7), 7)[1]), gen_list, (lambda xs: xs[0]), n=500, seed=1)
    ok(not mut.passed, f"mutating candidate FAILS (oracle sees a pristine copy, not the corrupted input) ({mut.ok}/{mut.n})")

    # 6. HIGH-fix: an empty run (n<=0) must NOT vacuously pass an untested primitive
    vac = run_gate((lambda x: 1 / 0), gen_small_int, (lambda x: x), n=0)
    ok((not vac.passed) and vac.ok == 0, "n=0 -> passed=False (no vacuous promotion)")

    # 7. BUG-SWEEP CRITICAL: a candidate returning a hostile-__eq__ object can NO LONGER pass (type-gated leaf)
    class AlwaysEq:
        def __eq__(self, o): return True
        def __hash__(self): return 0
    ae = run_gate((lambda x: AlwaysEq()), gen_small_int, (lambda x: x), n=200, seed=2)
    ok(not ae.passed, "hostile __eq__ candidate (AlwaysEq) is REJECTED (leaf equality is type-gated)")
    cand_ae, plac_ae, sound_ae = gate_with_placebo((lambda x: AlwaysEq()), (lambda x: x + 1), gen_small_int, (lambda x: x), n=200, seed=2)
    ok(not cand_ae.passed, "placebo path: the AlwaysEq candidate also fails the gate (placebo no longer defeated)")

    # 8. BUG-SWEEP HIGH: a candidate raising BaseException (SystemExit) is a MISS, NOT a gate crash
    def _sysexit(x): raise SystemExit("bye")
    se = run_gate(_sysexit, gen_small_int, (lambda x: x), n=5, seed=0)
    ok((not se.passed) and se.first_fail is not None, "candidate raising SystemExit -> miss, gate does not crash")

    # 9. BUG-SWEEP HIGH: a degenerate (constant) generator is caught -> fail closed
    deg = run_gate((lambda x: 16), (lambda rng: (4,)), (lambda x: x * x), n=2000, seed=0)
    ok((not deg.passed) and deg.n_distinct == 1, f"constant generator -> degenerate, fail-closed ({deg.note})")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} hold-out-gate checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
