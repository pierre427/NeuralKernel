#!/usr/bin/env python3
"""CPU-only test for kernel/lexical_proposer.governed_propose — the M5 scheduler-owned-k disposal over the lexical
proposer (the LAST continuation-k seam; project_scheduler_owned_k).

No model: governed_propose runs over the REAL routable registry (det_primitives + tool catalog) on CPU. cost is
injected via cost_source so the cost/budget brakes are deterministic. Proves the doctrine:
  1. CLEAR WINNER (c_gcd / "greatest common divisor") -> action 'route', k==1 (thin: a clear lexical winner needs
     no fan-out), candidates[0] is c_gcd.
  2. CLOSE RACE (count letters) -> the top-1/top-2 margin is tiny -> high INDEPENDENT score-uncertainty -> WIDEN
     k>1 (buy candidates to validate-the-branch, since lexical scores are UNCALIBRATED).
  3. GAP-DETECT (gibberish) -> nothing clears the floor -> action 'synthesize' (Step-3 trigger).
  4. COST veto (close race + cost=1.0 + KPolicy(k_min=0)) -> the DISCRETIONARY widening is braked away (k smaller
     than the idle close-race case); floor-cleared still implies a >=1 route.
  5. BUDGET caps k (close race + small budget -> k clamped below the disposed k).
  6. VALIDATE-THE-BRANCH: validate_proven REJECTS a lexical false-positive (count_leading_zeros, int-only, on
     string-counting cases) and ACCEPTS the right primitive (count_byte) — so the caller skips the false-positive a
     close race surfaced. (Never trust a lexical hunch like a proven gate.)
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_proposer_k.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.lexical_proposer import governed_propose, validate_proven, propose, ROUTE_THRESHOLD
from kernel.k_policy import KPolicy

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")

IDLE = lambda: 0.0     # injected idle cost (no host pressure) so cost never silently brakes the no-cost cases


def test_clear_winner():
    print("\n[1] CLEAR WINNER: c_gcd strongly matches one PROVEN primitive -> route, k==1 (thin)")
    res = governed_propose("c_gcd", "greatest common divisor of two integers",
                           policy=KPolicy(), cost_source=IDLE)
    ok(res["action"] == "route", f"action == 'route' (got {res['action']!r})")
    ok(res["k"] == 1, f"k == 1 (a clear winner needs no fan-out) (got {res['k']})")
    ok(len(res["candidates"]) == 1, f"exactly one candidate returned (got {len(res['candidates'])})")
    ok(res["candidates"][0]["name"] == "c_gcd",
       f"candidates[0] is c_gcd (got {res['candidates'][0]['name']!r})")
    ok(res["score_uncertainty"] < 0.5, f"score_uncertainty is LOW for a clear winner (got {res['score_uncertainty']})")
    ok(res["audit"]["used_signal"] == "thin_default",
       f"audit used_signal == 'thin_default' (got {res['audit']['used_signal']!r})")


def test_close_race():
    print("\n[2] CLOSE RACE: 'count letters' yields a tiny top-1/top-2 margin -> high uncertainty -> WIDEN k>1")
    # sanity: the real registry actually produces a close race here (top-2 within a hair).
    cands = propose("count letters", "count the letters in a string", top_k=4)
    margin = (cands[0]["score"] - cands[1]["score"]) / max(cands[0]["score"], 1e-9)
    ok(margin < 0.2, f"the real registry top-1/top-2 margin is small ({margin:.3f}) -> a genuine close race")

    res = governed_propose("count letters", "count the letters in a string",
                           policy=KPolicy(), cost_source=IDLE)
    ok(res["action"] == "route", f"action == 'route' (got {res['action']!r})")
    ok(res["k"] > 1, f"k > 1 (the close race WIDENED the disposal) (got {res['k']})")
    ok(len(res["candidates"]) == res["k"], f"candidate list length matches k (got {len(res['candidates'])} vs {res['k']})")
    ok(res["score_uncertainty"] > 0.5, f"score_uncertainty is HIGH for a close race (got {res['score_uncertainty']})")
    ok(res["audit"]["used_signal"] == "independent_disagreement",
       f"audit used_signal == 'independent_disagreement' (the margin is the trusted lever) "
       f"(got {res['audit']['used_signal']!r})")
    return res["k"]   # the idle close-race k, the baseline the cost-veto test brakes below


def test_gap_detect():
    print("\n[3] GAP-DETECT: a gibberish query clears nothing above the floor -> action 'synthesize'")
    res = governed_propose("zzqx_flumph", "nonexistent capability", policy=KPolicy(), cost_source=IDLE)
    ok(res["action"] == "synthesize", f"action == 'synthesize' (got {res['action']!r})")
    ok(res["k"] == 0, f"k == 0 (nothing to route) (got {res['k']})")
    ok(res["candidates"] == [], "no candidates on the synthesize path")
    ok(res["audit"]["used_signal"] == "gap_detect", f"audit used_signal == 'gap_detect' (got {res['audit']['used_signal']!r})")
    ok(res["audit"]["top_score"] < ROUTE_THRESHOLD,
       f"the best score was below the floor ({res['audit']['top_score']} < {ROUTE_THRESHOLD})")


def test_cost_veto(idle_close_k):
    print("\n[4] COST veto: close race + cost=1.0 + KPolicy(k_min=0) -> discretionary widening braked away")
    # k_min=0 lets cost collapse the DISCRETIONARY part to 0 (the degenerate syscall-head policy); but the floor
    # was cleared, so governed_propose still routes the single best (k>=1). The braking shows as k < the idle case.
    res = governed_propose("count letters", "count the letters in a string",
                           policy=KPolicy(k_min=0), cost_source=lambda: 1.0)
    ok(res["action"] == "route", f"action == 'route' (floor cleared, still route the best) (got {res['action']!r})")
    ok(res["k"] >= 1, f"k >= 1 (floor-cleared always routes at least the best) (got {res['k']})")
    ok(res["k"] < idle_close_k,
       f"k ({res['k']}) < the idle close-race k ({idle_close_k}) -> the discretionary widening was braked")
    ok(res["audit"]["disposed_k"] == 0, f"the policy disposed_k collapsed to 0 under cost (got {res['audit']['disposed_k']})")
    ok(res["audit"]["used_signal"] == "cost_braked",
       f"audit used_signal indicates cost braking (got {res['audit']['used_signal']!r})")


def test_budget_caps(idle_close_k):
    print("\n[5] BUDGET caps k: close race + a small budget -> k clamped below the disposed k")
    # idle cost so the POLICY wants to widen (disposed_k == idle_close_k), but a tiny budget can only afford fewer.
    res = governed_propose("count letters", "count the letters in a string",
                           policy=KPolicy(), cost_source=IDLE, budget=1.0)
    ok(res["action"] == "route", f"action == 'route' (got {res['action']!r})")
    ok(res["audit"]["disposed_k"] == idle_close_k,
       f"the policy still WANTED to widen (disposed_k == {idle_close_k}) (got {res['audit']['disposed_k']})")
    ok(res["k"] < idle_close_k, f"a small budget CLAMPED k ({res['k']}) below the disposed k ({idle_close_k})")
    ok(res["k"] >= 1, f"k >= 1 (floor cleared) (got {res['k']})")
    ok(res["audit"]["budget_note"] in ("budget-clamped", "budget-exhausted"),
       f"audit budget_note shows the clamp (got {res['audit']['budget_note']!r})")


def test_validate_the_branch():
    print("\n[6] VALIDATE-THE-BRANCH: validate_proven rejects a lexical false-positive, accepts the right primitive")
    # The close race for 'count letters' surfaces count_leading_zeros (int-only) alongside count_byte (THE counter).
    # A lexical score can't tell them apart; validate-the-branch (actually RUNNING the candidate on the span's
    # example cases) can — so the caller skips the false-positive and keeps the one that works.
    cands = [c["name"] for c in propose("count letters", "count the letters in a string", top_k=4)]
    ok("count_leading_zeros" in cands and "count_byte" in cands,
       f"the close race surfaced BOTH the false-positive and the right primitive (got {cands})")

    string_cases = [{"args": ["hello", "l"], "expected": 2}, {"args": ["banana", "a"], "expected": 3}]
    ok(validate_proven("count_leading_zeros", string_cases) is False,
       "validate_proven REJECTS count_leading_zeros on string-counting cases (the lexical false-positive)")
    ok(validate_proven("count_byte", string_cases) is True,
       "validate_proven ACCEPTS count_byte on those same cases (the right primitive)")
    ok(validate_proven("count_byte", []) is False,
       "validate_proven with no cases is False (an un-validatable reuse is not trusted)")


if __name__ == "__main__":
    test_clear_winner()
    idle_close_k = test_close_race()
    test_gap_detect()
    test_cost_veto(idle_close_k)
    test_budget_caps(idle_close_k)
    test_validate_the_branch()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} proposer-k checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
