#!/usr/bin/env python3
"""M5 capability-multiplier loop — GOVERNED verified-fact re-prefill (the Mode-A finding turned into a policy).

THE FINDING (kernel/mode_a_factuse.py, CONFIRMED on North): a verified fact delivered AS TOKENS lifts the model
from ~0 to ~1.0 on a task it cannot do unaided — because tokens are processed from layer 0, so early layers turn
the fact into an OPERAND. The same fact delivered as a mid-stack RESIDUAL edit (G3) did NOT become usable: it
arrived too late / in the wrong basis. So the kernel's capability-multiplier is NOT a residual injector; it is:

    compute a verified fact with a hold-out-proven DETERMINISTIC primitive
      -> inject it AS TOKENS (re-prefill the prompt with a "Verified facts" block)
      -> the model re-reads from layer 0 and USES it.

THE GOVERNANCE (M5, project_scheduler_owned_k + project_grounding_result's thin/heavy verdict): spending is not
free — a fact-fetch + a second full generation costs a model call and latency, and a heavy protocol HURTS an
already-capable model. So KPolicy must GOVERN whether/how-much to spend:

  * THIN BY DEFAULT — if the BASE attempt already passes validation, do NOT fetch facts and do NOT re-prefill
    (k=0, one gen). The verified fact is the capability MULTIPLIER, not a tax on every prompt.
  * SPEND ONLY WHEN EARNED — a base attempt that FAILS validation is an INDEPENDENT, ground-truth-measured
    uncertainty of 1.0 (validate-the-branch, the trustworthy lever — not raw model self-confidence, which
    miscalibrates). Only THEN do we ask KPolicy for k, with cost (kernel proprioception) and budget brakes.
  * COST / BUDGET CAN STILL SAY NO — under high host load (cost->1 brakes discretionary widening) or an exhausted
    trap budget, k collapses to 0: stay thin, return the (failed) base answer, and AUDIT why we did not spend.
    "fail->pass is not proof" applied to spending: don't pay for a second gen you can't afford or aren't allowed.

LOAD-BEARING INERTNESS INVARIANT: with no facts the loop is byte-identical to a plain `adapter.gen_fast`.
`augment_prompt(p, [])` returns `p` UNCHANGED, so `fact_reprefill(adapter, p, [])` == `adapter.gen_fast(p, maxn)`
(same text & n). The fact machinery can never silently perturb the base path — it only ever ADDS a verified block.

Model-agnostic: the only model surface used is `adapter.gen_fast(prompt, maxn) -> (text, n)`. No MLX, no model load
here — CPU-testable with a fake adapter (kernel/test_fact_loop.py). Wires KPolicy/CostModel exactly like
kernel/escalate_solve.py (cost_source / budget / charge_budget; independent uncertainty as the lever)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from kernel.k_policy import KPolicy, KSignals
from kernel.k_cost import CostModel, charge_budget


@dataclass
class Fact:
    """A verified fact computed by a trusted (hold-out-proven) deterministic primitive — NOT a model guess.

    label  a human/audit name for the computation, e.g. "count('strawberry','r')".
    value  the verified result to be delivered to the model as an operand token sequence, e.g. 3.
    """
    label: str
    value: Any


def render_facts(facts) -> str:
    """Render verified facts as a token block the model re-reads from layer 0 (the Mode-A delivery channel).

    No facts (empty / None) -> '' (so callers can prepend unconditionally without perturbing the base prompt).
    Otherwise a header that tells the model to TRUST these over its own (miscalibrated) counting/arithmetic,
    followed by one '- {label} = {value}' line per fact.
    """
    facts = facts or []
    if not facts:
        return ""
    lines = ["Verified facts (computed by a trusted tool — trust these over your own counting):"]
    for f in facts:
        lines.append(f"- {f.label} = {f.value}")
    return "\n".join(lines)


def augment_prompt(prompt: str, facts) -> str:
    """Prepend the verified-facts block to the prompt.

    INERTNESS INVARIANT (load-bearing): facts empty/None -> return `prompt` UNCHANGED (no whitespace, no header),
    so the fact path can never perturb the base attempt. Otherwise: render_facts(facts) + '\\n\\n' + prompt.
    """
    if not facts:
        return prompt
    return render_facts(facts) + "\n\n" + prompt


def fact_reprefill(adapter, prompt: str, facts, maxn: int = 512):
    """Deliver `facts` AS TOKENS, then generate — the Mode-A re-prefill (facts become operands from layer 0).

    Returns (text, n) from `adapter.gen_fast(augment_prompt(prompt, facts), maxn=maxn)`.

    INVARIANT: fact_reprefill(adapter, prompt, []) is IDENTICAL to adapter.gen_fast(prompt, maxn) — same text & n —
    because augment_prompt with no facts returns the prompt byte-for-byte unchanged.
    """
    return adapter.gen_fast(augment_prompt(prompt, facts), maxn=maxn)


@dataclass
class FactLoopResult:
    """The governed loop's outcome + a full audit trail (so the scheduler/proof-ledger can see WHY it spent or not)."""
    answer: str
    used_facts: bool          # did we re-prefill with verified facts? (False on every THIN path)
    n_facts: int              # how many facts were injected (0 unless we re-prefilled)
    base_attempts: int        # model gens on the base attempt (always 1)
    reprefill_attempts: int   # model gens on fact-reprefill (0 or 1)
    total_gens: int           # base_attempts + reprefill_attempts (1 when thin, 2 when we spend)
    k: int                    # the scheduler-disposed/budget-admitted continuation count for the spend decision
    audit: dict               # {used_signal, reason, cost, budget_note, base_validated, final_validated, disposed_k}


def _ok(validate, answer) -> bool:
    """Run the validate-the-branch lever, fail-SAFE: a validator that raises counts as NOT validated (never crash
    the loop on a flaky validator — the trust lever degrading must not take down the kernel)."""
    try:
        return bool(validate(answer))
    except Exception:
        return False


def governed_fact_loop(adapter, prompt, *, fact_provider, validate, policy,
                       sysinfo=None, cost_source=None, budget=None, importance=0.7, maxn=512) -> FactLoopResult:
    """Thin-by-default verified-fact re-prefill, GOVERNED by KPolicy + cost + budget.

    Args:
      adapter        anything with gen_fast(prompt, maxn) -> (text, n). No model is loaded here.
      prompt         the user/task prompt.
      fact_provider  fact_provider(prompt) -> list[Fact]. Verified DETERMINISTIC facts; cheap, makes NO model call.
                     Called ONLY when we have decided to spend (k>=1) — never on the thin path.
      validate       validate(answer) -> bool. The validate-the-branch lever, judged against verified facts /
                     ground truth (NOT model self-report). Used on the base attempt and, if we spend, the reprefill.
      policy         a KPolicy. The caller owns its shape (k_min/k_max/cost_brake). decide() disposes k.
      sysinfo        a sysinfo dict for CostModel().cost_from_sysinfo when no cost_source is given.
      cost_source    callable()->cost in [0,1] (injectable; e.g. tests/demos). Overrides sysinfo when present.
      budget         remaining trap budget (float) for charge_budget; None == untracked (admit the policy's k).
      importance     task importance -> scales the discretionary spend ceiling in KPolicy.
      maxn           max new tokens per generation.

    Algorithm:
      1. base: a0 = adapter.gen_fast(prompt, maxn).  (base_attempts = 1)
      2. if validate(a0): THIN — return a0 (used_facts=False, total_gens=1, k=0; audit base_validated=True). No
         fact-fetch, no re-prefill: the multiplier is reserved for prompts the base attempt cannot do.
      3. else (base FAILED -> independent uncertainty = 1.0):
           cost = cost_source() if given else CostModel().cost_from_sysinfo(sysinfo or {})['cost']
           dec  = policy.decide(KSignals(disagreement=1.0, cost=cost, importance=importance))   # validate-the-branch
           k    = charge_budget(dec, budget).admitted_k                                          # cost+budget brakes
           if k >= 1: fetch verified facts, RE-PREFILL them as tokens, return the new answer (used_facts=True,
                      reprefill_attempts=1, total_gens=2; audit final_validated=validate(a1)).
           else:      stay THIN — cost or an exhausted budget vetoed the spend; return the (failed) base answer
                      (used_facts=False, k=0, total_gens=1) and AUDIT why (used_signal, budget_note).
    """
    # 1. base attempt
    a0, _ = adapter.gen_fast(prompt, maxn=maxn)
    base_ok = _ok(validate, a0)

    # 2. THIN: the base attempt already passes -> never fetch facts, never re-prefill (the multiplier is reserved).
    if base_ok:
        audit = {
            "used_signal": "thin_default",
            "reason": "base attempt validated -> no fact-fetch, no re-prefill (thin)",
            "cost": None,
            "budget_note": None,
            "base_validated": True,
            "final_validated": True,
        }
        return FactLoopResult(answer=a0, used_facts=False, n_facts=0, base_attempts=1,
                              reprefill_attempts=0, total_gens=1, k=0, audit=audit)

    # 3. base FAILED -> an INDEPENDENT, ground-truth uncertainty of 1.0. Ask the policy whether to spend.
    if cost_source is not None:
        try:
            cost = float(cost_source())
        except Exception:
            cost = 1.0      # fail-safe: a broken cost sensor must brake toward THIN, never force a spend
    else:
        cost = CostModel().cost_from_sysinfo(sysinfo or {})["cost"]
    dec = policy.decide(KSignals(disagreement=1.0, cost=cost, importance=importance))
    charge = charge_budget(dec, budget)
    k = charge.admitted_k

    if k >= 1:
        # SPEND was authorized: compute verified facts (no model call).
        facts = fact_provider(prompt) or []
        if not facts:
            # The provider yielded nothing -> augment_prompt returns the prompt UNCHANGED, so a re-prefill would be
            # a verbatim re-run of the base attempt. Do NOT burn a redundant gen or claim a capability spend: stay
            # thin and audit honestly (the governance record must never overstate what happened).
            audit = {"used_signal": dec.used_signal, "reason": "spend authorized but no verified facts available -> stayed thin",
                     "cost": cost, "budget_note": charge.note, "base_validated": False, "final_validated": False,
                     "disposed_k": dec.k}
            return FactLoopResult(answer=a0, used_facts=False, n_facts=0, base_attempts=1,
                                  reprefill_attempts=0, total_gens=1, k=0, audit=audit)
        # re-prefill the verified facts as tokens (the Mode-A multiplier). The loop realizes exactly ONE re-prefill,
        # so the realized k is 1; the policy's disposed k is recorded separately (audit honesty, not aspiration).
        a1, _ = fact_reprefill(adapter, prompt, facts, maxn)
        audit = {
            "used_signal": dec.used_signal,
            "reason": dec.reason,
            "cost": cost,
            "budget_note": charge.note,
            "base_validated": False,
            "final_validated": _ok(validate, a1),
            "disposed_k": dec.k,
        }
        return FactLoopResult(answer=a1, used_facts=True, n_facts=len(facts), base_attempts=1,
                              reprefill_attempts=1, total_gens=2, k=1, audit=audit)

    # k == 0: cost-braking or an exhausted/clamped budget vetoed the spend -> stay thin, keep the base answer.
    audit = {
        "used_signal": dec.used_signal,
        "reason": dec.reason,
        "cost": cost,
        "budget_note": charge.note,
        "base_validated": False,
        "final_validated": False,
        "disposed_k": dec.k,
    }
    return FactLoopResult(answer=a0, used_facts=False, n_facts=0, base_attempts=1,
                          reprefill_attempts=0, total_gens=1, k=0, audit=audit)


__all__ = ["Fact", "render_facts", "augment_prompt", "fact_reprefill", "FactLoopResult", "governed_fact_loop"]
