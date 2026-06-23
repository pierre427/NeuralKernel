#!/usr/bin/env python3
"""CPU-only test for kernel/fact_loop.py — the M5-governed verified-fact re-prefill (Mode-A multiplier).

No model: a deterministic _FakeAdapter whose gen_fast(prompt) answer depends ONLY on whether the verified-facts
block is present in the prompt (so we can control base-pass vs base-fail and whether facts FIX the answer). Proves:
  - the load-bearing INERTNESS invariant (no facts -> prompt & generation byte-identical to plain gen_fast),
  - facts are rendered with label+value over the original prompt,
  - THIN by default (base validates -> no fact-fetch, no re-prefill),
  - the LIFT path (base fails, idle cost, no budget -> re-prefill fires and the fact makes validation pass),
  - cost can VETO the spend (high cost + a k_min=0 policy -> k=0, stay thin),
  - an exhausted budget can VETO the spend (budget=0 -> k=0, stay thin).
The real-model lift (token-delivery actually working) is proven separately by mode_a_factuse.py on North.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_fact_loop.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.fact_loop import (Fact, render_facts, augment_prompt, fact_reprefill,
                              FactLoopResult, governed_fact_loop)
from kernel.k_policy import KPolicy

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")

FACTS_HEADER = "Verified facts (computed by a trusted tool"   # the marker render_facts emits


class _FakeAdapter:
    """Deterministic, NO model. gen_fast returns (text, n) where the text depends ONLY on whether the
    verified-facts block is present:
      * facts present  -> `with_facts` answer (the "the fact fixed it" branch),
      * facts absent   -> `base` answer (the base attempt).
    n is len(text.split()) so callers can assert on a stable token count. A 'calls' log lets tests count gens.
    """
    def __init__(self, base="WRONG", with_facts="RIGHT"):
        self.base = base
        self.with_facts = with_facts
        self.calls = []

    def gen_fast(self, prompt, maxn=512):
        self.calls.append(prompt)
        text = self.with_facts if (FACTS_HEADER in prompt) else self.base
        return text, len(text.split())


def _fact_provider(_prompt):
    # a verified deterministic fact (no model call), mirroring count('strawberry','r') = 3
    return [Fact("count('strawberry','r')", 3)]


def test_inertness():
    print("\n[1] inertness: no facts -> prompt UNCHANGED and generation identical to plain gen_fast")
    p = "How many r in strawberry?"
    ok(augment_prompt(p, []) == p, "augment_prompt(p, []) returns the prompt UNCHANGED")
    ok(augment_prompt(p, None) == p, "augment_prompt(p, None) returns the prompt UNCHANGED")
    fake = _FakeAdapter(base="base-answer")
    direct = fake.gen_fast(p)
    rp = fact_reprefill(fake, p, [])
    ok(rp == direct, f"fact_reprefill(fake, p, []) == fake.gen_fast(p) (same text & n): {rp!r} == {direct!r}")
    ok(fake.calls[-1] == p, "the re-prefilled prompt passed to the adapter was byte-identical to p")


def test_facts_rendered():
    print("\n[2] facts rendered: label + value over the original prompt")
    p = "How many r in strawberry?"
    aug = augment_prompt(p, [Fact("count", "3")])
    ok("count" in aug, "augmented prompt contains the fact label 'count'")
    ok("3" in aug, "augmented prompt contains the fact value '3'")
    ok(p in aug, "augmented prompt still contains the original prompt")
    ok(render_facts([]) == "", "render_facts([]) is the empty string (inertness at the renderer)")


def test_thin_path():
    print("\n[3] THIN: base answer PASSES validate -> no fact-fetch, no re-prefill, one gen")
    fake = _FakeAdapter(base="RIGHT", with_facts="RIGHT")
    provider_calls = {"n": 0}
    def provider(pr):
        provider_calls["n"] += 1
        return _fact_provider(pr)
    res = governed_fact_loop(fake, "q", fact_provider=provider, validate=lambda a: a == "RIGHT",
                             policy=KPolicy(), cost_source=lambda: 0.0)
    ok(isinstance(res, FactLoopResult), "returns a FactLoopResult")
    ok(res.used_facts is False, "used_facts == False on the thin path")
    ok(res.total_gens == 1, f"total_gens == 1 (got {res.total_gens})")
    ok(res.reprefill_attempts == 0, f"reprefill_attempts == 0 (got {res.reprefill_attempts})")
    ok(res.k == 0, f"k == 0 on the thin path (got {res.k})")
    ok(res.audit.get("base_validated") is True, "audit base_validated is True")
    ok(provider_calls["n"] == 0, "fact_provider was NOT called on the thin path (no fact-fetch)")
    ok(len(fake.calls) == 1, "exactly one model gen on the thin path")


def test_lift_path():
    print("\n[4] LIFT: base FAILS, idle cost, no budget -> re-prefill fires and the fact makes validation pass")
    fake = _FakeAdapter(base="WRONG", with_facts="RIGHT")
    res = governed_fact_loop(fake, "q", fact_provider=_fact_provider, validate=lambda a: a == "RIGHT",
                             policy=KPolicy(), cost_source=lambda: 0.0, budget=None)
    ok(res.used_facts is True, "used_facts == True (we spent)")
    ok(res.reprefill_attempts == 1, f"reprefill_attempts == 1 (got {res.reprefill_attempts})")
    ok(res.total_gens == 2, f"total_gens == 2 (base + reprefill) (got {res.total_gens})")
    ok(res.k >= 1, f"k >= 1 (the policy/cost/budget granted the spend) (got {res.k})")
    ok(res.n_facts == 1, f"n_facts == 1 (got {res.n_facts})")
    ok(res.audit.get("base_validated") is False, "audit base_validated is False (base failed)")
    ok(res.audit.get("final_validated") is True, "audit final_validated is True (the fact fixed it)")
    ok(res.answer == "RIGHT", f"final answer is the fact-corrected one (got {res.answer!r})")
    ok(any(FACTS_HEADER in c for c in fake.calls), "the second gen actually saw the verified-facts block")


def test_cost_governed():
    print("\n[5] COST-governed: base fails but cost=1.0 (k_min=0 policy) -> k=0, stay thin, audit shows cost braking")
    fake = _FakeAdapter(base="WRONG", with_facts="RIGHT")
    # k_min=0 lets discretionary cost-braking collapse k to 0 (the degenerate syscall-head policy); default cost_brake.
    policy = KPolicy(k_min=0)
    res = governed_fact_loop(fake, "q", fact_provider=_fact_provider, validate=lambda a: a == "RIGHT",
                             policy=policy, cost_source=lambda: 1.0, budget=None)
    ok(res.k == 0, f"k == 0 under high cost (got {res.k})")
    ok(res.used_facts is False, "used_facts == False (cost vetoed the spend)")
    ok(res.total_gens == 1, f"total_gens == 1 (stayed thin) (got {res.total_gens})")
    ok(res.answer == "WRONG", "returns the (failed) base answer when it cannot afford to spend")
    ok(res.audit.get("used_signal") == "cost_braked",
       f"audit used_signal indicates cost braking (got {res.audit.get('used_signal')!r})")
    ok(not any(FACTS_HEADER in c for c in fake.calls), "no fact-reprefill gen happened")


def test_budget_governed():
    print("\n[6] BUDGET: base fails, idle cost, budget=0.0 -> k=0, stay thin, audit budget_note is exhausted/clamped")
    fake = _FakeAdapter(base="WRONG", with_facts="RIGHT")
    res = governed_fact_loop(fake, "q", fact_provider=_fact_provider, validate=lambda a: a == "RIGHT",
                             policy=KPolicy(), cost_source=lambda: 0.0, budget=0.0)
    ok(res.k == 0, f"k == 0 with an exhausted budget (got {res.k})")
    ok(res.used_facts is False, "used_facts == False (budget vetoed the spend)")
    ok(res.total_gens == 1, f"total_gens == 1 (stayed thin) (got {res.total_gens})")
    ok(res.audit.get("budget_note") in ("budget-exhausted", "budget-clamped"),
       f"audit budget_note in (budget-exhausted, budget-clamped) (got {res.audit.get('budget_note')!r})")
    ok(not any(FACTS_HEADER in c for c in fake.calls), "no fact-reprefill gen happened")


def test_review_hardening():
    print("\n[7] review hardening: empty-provider honesty + fail-safe cost/validate (no audit lies, no crashes)")
    # 7a: spend authorized but provider yields NO facts -> augment_prompt would be a verbatim re-run;
    #     must stay THIN with an honest audit (never claim used_facts when zero facts were delivered).
    fake = _FakeAdapter(base="WRONG", with_facts="RIGHT")
    res = governed_fact_loop(fake, "q", fact_provider=lambda p: [], validate=lambda a: a == "RIGHT",
                             policy=KPolicy(), cost_source=lambda: 0.0, budget=None)
    ok(res.used_facts is False, "empty provider -> used_facts False (no false capability-spend claim)")
    ok(res.n_facts == 0 and res.reprefill_attempts == 0 and res.total_gens == 1,
       f"empty provider -> stayed thin, no redundant gen (gens={res.total_gens})")
    ok("no verified facts" in res.audit.get("reason", ""), "audit reason honestly states no facts available")
    ok(len(fake.calls) == 1, "empty provider -> exactly one (base) gen, NOT a verbatim re-run")
    # 7b: a broken cost sensor must FAIL-SAFE to thin (cost->1.0), never crash, never force a spend.
    fake2 = _FakeAdapter(base="WRONG", with_facts="RIGHT")
    def boom(): raise RuntimeError("sensor down")
    res2 = governed_fact_loop(fake2, "q", fact_provider=_fact_provider, validate=lambda a: a == "RIGHT",
                              policy=KPolicy(k_min=0), cost_source=boom, budget=None)
    ok(res2.used_facts is False and res2.total_gens == 1, "broken cost sensor -> fail-safe THIN, no crash")
    # 7c: a validator that RAISES on the base answer must count as not-validated (-> spend path), never crash;
    #     the loop then recovers via the facts. Also checks the audit records the policy's disposed_k.
    fake3 = _FakeAdapter(base="WRONG", with_facts="RIGHT")
    def flaky_validate(a):
        if a == "WRONG":
            raise ValueError("cannot validate the base answer")
        return a == "RIGHT"
    res3 = governed_fact_loop(fake3, "q", fact_provider=_fact_provider, validate=flaky_validate,
                              policy=KPolicy(), cost_source=lambda: 0.0, budget=None)
    ok(res3.used_facts is True and res3.answer == "RIGHT",
       "validator raising on base -> treated as failed, loop recovers via facts (no crash)")
    ok(res3.audit.get("disposed_k") is not None, "audit records the policy's disposed_k (realized k=1 vs disposed)")


if __name__ == "__main__":
    test_inertness()
    test_facts_rendered()
    test_thin_path()
    test_lift_path()
    test_cost_governed()
    test_budget_governed()
    test_review_hardening()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} fact-loop checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
