#!/usr/bin/env python3
"""LIVE DISPATCH for the governed verified-fact loop — the "propose-which-fact / dispose-whether-to-spend" seam.

This wires the existing M5-governed fact-loop (kernel/fact_loop.py) into a single live dispatch path:

    prompt -> ROUTER (which verified fact does this prompt need?) -> governed_fact_loop (spend?) -> answer
              |                                                       |
              propose                                                 dispose (KPolicy + cost + budget)

The two halves of the seam, kept distinct on purpose (mirrors learned-proposes / kernel-disposes):
  * ROUTER — lexical, deterministic, model-free. First spec whose matches(prompt) fires owns the prompt. A spec
    binds (a) a capability token, (b) a matcher, (c) a verified-fact PROVIDER built from a hold-out-proven
    DETERMINISTIC primitive (det_primitives.py — NO model), and (d) a VALIDATE that judges a model answer against
    that same verified ground truth. The router only PROPOSES which fact applies; it never spends a model call.
  * DISPOSE — governed_fact_loop runs the proposed fact under KPolicy + cost + budget: THIN by default (the base
    attempt already passes -> no fact-fetch, no re-prefill), SPEND only when the base attempt fails validation
    (an independent ground-truth uncertainty of 1.0) AND cost/budget allow it. The fact is delivered AS TOKENS
    (the Mode-A re-prefill), so the model re-reads it from layer 0 and turns it into an operand.

CAPABILITY GATE — two layers, both fail-closed:
  * At the dispatch() call: `capabilities` (a granted set) gates the fact path. No router match, or the matched
    spec's cap not granted, => PASS THROUGH (plain gen_fast). capabilities=None == ungated (trust the caller).
  * At the scheduler: register_fact_loop_executor declares op's required cap so the Scheduler's admit() enforces
    it kernel-side (an un-granted cap is rejected before the executor ever runs — the scheduler is the gate).

LOAD-BEARING INERTNESS INVARIANT (inherited from fact_loop): when `routed is None` (no match OR cap-denied) the
dispatch path is BYTE-IDENTICAL to `adapter.gen_fast(prompt, maxn)[0]` — the fact machinery can never perturb a
prompt it does not own. A non-matching/chat prompt comes out exactly as the bare model would have produced it.

Model-agnostic: the only model surface is `adapter.gen_fast(prompt, maxn) -> (text, n)`. The det-primitive
computations ARE the verified facts (no model). CPU-testable with a fake adapter (kernel/test_fact_dispatch.py)."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable, Optional

from kernel.fact_loop import Fact, governed_fact_loop
from kernel.task_graph import ValidationLevel


# ===========================================================================
# Verified-fact deterministic primitives (NO model). count_byte / c_gcd come from the hold-out-proven
# det_primitives ISA; is_prime is a tiny trial-division primitive kept local so this module imports cleanly
# without pulling in any model-bearing file (north_syscall_e2e defines an equivalent for the head-gated path).
# ===========================================================================
from det_primitives import count_byte, c_gcd


def _is_prime(n: int) -> bool:
    """Deterministic primality by trial division (the verified fact for is_prime prompts). Pure, no model."""
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# ===========================================================================
# A routed fact specification: matcher + capability + (provider, validate) factories.
# ===========================================================================
@dataclass
class FactSpec:
    """One verified-fact capability the router can dispatch to.

    name          a short id, e.g. "letter_count" (also the routed-label dispatch returns and the audit key).
    cap           the capability token a caller/scheduler must grant to run this fact path, e.g. "fact:letter_count".
    matches       matches(prompt:str) -> bool : does this prompt need this fact? (lexical, model-free.)
    make_provider make_provider(prompt) -> (fact_provider(_p) -> list[Fact]) : a CLOSURE that, when the loop
                  decides to spend, computes the VERIFIED fact for THIS prompt with a det primitive (no model).
                  Returning [] (args not extractable) makes the loop stay thin honestly (no false spend).
    make_validate make_validate(prompt) -> (validate(answer:str) -> bool) : judges a model answer against the
                  verified ground truth for THIS prompt (substring/parse match — NOT model self-report).
    """
    name: str
    cap: str
    matches: Callable
    make_provider: Callable
    make_validate: Callable


class FactRouter:
    """Ordered registry of FactSpecs. route() returns the FIRST spec whose matches(prompt) fires, else None.

    Order is registration order (first registered wins on overlap), so register the more specific matchers first.
    """
    def __init__(self, specs=None):
        self.specs: list[FactSpec] = []
        for s in (specs or []):
            self.register(s)

    def register(self, spec: FactSpec):
        self.specs.append(spec)
        return self

    def route(self, prompt: str) -> Optional[FactSpec]:
        for spec in self.specs:
            try:
                if spec.matches(prompt):
                    return spec
            except Exception:
                # a flaky matcher must never crash routing — treat it as a non-match (fail toward pass-through).
                continue
        return None


# ===========================================================================
# Argument extraction (the north_syscall_e2e.extract_args spirit) — pull the operands a det primitive needs.
# Each returns None when the args are not unambiguously present, so the provider can stay thin honestly.
# ===========================================================================
_STOP = {"word", "letter", "many", "times", "does", "occur", "show", "shows", "that", "this", "with", "from",
         "into", "your", "exactly", "count", "appear", "appears", "number", "which", "what", "there", "tell",
         "could", "please", "have"}


def _extract_letter_count(prompt: str):
    """(word, letter) for a 'how many <letter> in <word>' prompt, or None. Mirrors north_syscall_e2e."""
    p = prompt
    ch = re.search(r"letter\s+['\"]?([A-Za-z])['\"]?|['\"]([A-Za-z])['\"]", p)
    c = (ch.group(1) or ch.group(2)) if ch else None
    m = re.search(r"word\s+['\"]?([A-Za-z]{3,})['\"]?", p) or re.search(r"['\"]([A-Za-z]{3,})['\"]", p)
    if m:
        w = m.group(1)
    else:  # fallback: the longest content token (skip stopwords) — the target word is the salient one
        toks = [t for t in re.findall(r"[A-Za-z]{4,}", p) if t.lower() not in _STOP]
        w = max(toks, key=len) if toks else None
    return (w, c) if (w and c) else None


def _extract_gcd(prompt: str):
    """(a, b) for a gcd prompt, anchored to the operands (not incidental years/indices), or None."""
    p = prompt
    for pat in (r"gcd\(\s*(\d+)\s*,\s*(\d+)\)", r"(\d+)\s*[/:]\s*(\d+)", r"(\d+)\s+and\s+(\d+)"):
        m = re.search(pat, p)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    ns = re.findall(r"\d+", p)
    return (int(ns[0]), int(ns[1])) if len(ns) == 2 else None     # refuse if ambiguous (>2 numbers)


def _extract_is_prime(prompt: str):
    """(n,) for an is-prime prompt, or None (refuse if ambiguous)."""
    p = prompt
    m = re.search(r"(?:is|whether)\s+(\d+)\b|\b(\d+)\s+(?:is\s+)?(?:a\s+)?prime", p, re.I)
    if m:
        return (int(m.group(1) or m.group(2)),)
    ns = re.findall(r"\d+", p)
    return (int(ns[0]),) if len(ns) == 1 else None


# --- answer matching: does the model's free-text answer contain the verified value? -----------------------
def _answer_has_int(answer: str, value: int) -> bool:
    """True iff the verified integer appears as a standalone token in the answer (word-boundary, not a substring
    of a larger number). The validate-the-branch judge for numeric facts — checked against ground truth, not vibes."""
    return re.search(rf"(?<!\d){re.escape(str(int(value)))}(?!\d)", answer or "") is not None


def _answer_has_bool(answer: str, value: bool) -> bool:
    """True iff the answer's yes/no polarity matches the verified primality (is/ is not prime)."""
    a = (answer or "").lower()
    neg = ("not prime" in a) or ("isn't prime" in a) or ("is not a prime" in a) or ("composite" in a) \
        or bool(re.search(r"\bno\b", a))
    pos = ("is prime" in a) or ("is a prime" in a) or bool(re.search(r"\byes\b", a))
    if value:
        return pos and not neg
    return neg and not pos


# ===========================================================================
# default_router — seed with three PROVEN det primitives (letter-count, gcd, is_prime).
# ===========================================================================
def _make_letter_count_provider(prompt: str):
    args = _extract_letter_count(prompt)

    def provider(_p):
        if not args:
            return []                                  # args not extractable -> no fact -> loop stays thin honestly
        w, c = args
        return [Fact(f"count_byte({w!r}, {c!r})", count_byte(w, c))]
    return provider


def _make_letter_count_validate(prompt: str):
    args = _extract_letter_count(prompt)
    truth = count_byte(*args) if args else None

    def validate(answer):
        return truth is not None and _answer_has_int(answer, truth)
    return validate


def _make_gcd_provider(prompt: str):
    args = _extract_gcd(prompt)

    def provider(_p):
        if not args:
            return []
        a, b = args
        return [Fact(f"c_gcd({a}, {b})", c_gcd(a, b))]
    return provider


def _make_gcd_validate(prompt: str):
    args = _extract_gcd(prompt)
    truth = c_gcd(*args) if args else None

    def validate(answer):
        return truth is not None and _answer_has_int(answer, truth)
    return validate


def _make_is_prime_provider(prompt: str):
    args = _extract_is_prime(prompt)

    def provider(_p):
        if not args:
            return []
        (n,) = args
        return [Fact(f"is_prime({n})", _is_prime(n))]
    return provider


def _make_is_prime_validate(prompt: str):
    args = _extract_is_prime(prompt)
    truth = _is_prime(*args) if args else None

    def validate(answer):
        return truth is not None and _answer_has_bool(answer, truth)
    return validate


# matchers — lexical, model-free. is_prime is registered BEFORE gcd would be a risk, and letter-count's matcher
# requires the counting intent so a bare gcd/prime prompt never falls into it.
def _m_letter_count(p: str) -> bool:
    pl = p.lower()
    return ("letter" in pl or "character" in pl) and \
        bool(re.search(r"how many|number of|count|times|occur|appear", pl))


def _m_gcd(p: str) -> bool:
    pl = p.lower()
    return ("gcd" in pl) or ("greatest common divisor" in pl) or ("greatest common factor" in pl)


def _m_is_prime(p: str) -> bool:
    return bool(re.search(r"\bprime\b", p, re.I)) and bool(re.search(r"\d", p))


def default_router() -> FactRouter:
    """A router seeded with three hold-out-proven deterministic primitives, each parsing its own args from the
    prompt, computing the VERIFIED value with the det primitive (NO model), and validating a model answer against
    that ground truth. Specs are ordered most-specific-first (letter_count, gcd, is_prime)."""
    return FactRouter([
        FactSpec("letter_count", "fact:letter_count", _m_letter_count,
                 _make_letter_count_provider, _make_letter_count_validate),
        FactSpec("gcd", "fact:gcd", _m_gcd, _make_gcd_provider, _make_gcd_validate),
        FactSpec("is_prime", "fact:is_prime", _m_is_prime, _make_is_prime_provider, _make_is_prime_validate),
    ])


# ===========================================================================
# dispatch — the live path. route -> (pass-through | cap-denied pass-through | governed fact-loop).
# ===========================================================================
def dispatch(adapter, prompt, *, router, policy, cost_source=None, budget=None, capabilities=None,
             importance=0.7, maxn=512) -> dict:
    """LIVE DISPATCH: route a prompt to its verified-fact primitive, run the governed fact-loop, or pass through.

    Returns a dict:
      {'answer', 'routed', 'result', 'total_gens'}
        answer      the final text handed to the user.
        routed      the spec name when the fact path ran; None on a plain pass-through; 'cap-denied:<name>' when a
                    spec matched but its capability was not granted (still a pass-through, but audited as denied).
        result      the FactLoopResult when the fact path ran, else None.
        total_gens  model generations spent (1 on any pass-through; res.total_gens — 1 thin / 2 spend — otherwise).

    Algorithm:
      1. spec = router.route(prompt).
      2. PASS THROUGH (byte-identical to gen_fast) when spec is None, OR (capabilities is not None and the spec's
         cap is not in capabilities). routed = None / 'cap-denied:<name>' accordingly.
      3. otherwise run governed_fact_loop with the spec's provider+validate and return its outcome.

    INERTNESS: whenever routed is None/cap-denied, answer == adapter.gen_fast(prompt, maxn)[0] exactly."""
    spec = router.route(prompt)

    # cap-denied pass-through: a fact applies but its capability was not granted -> do NOT run the fact path.
    if spec is not None and capabilities is not None and spec.cap not in capabilities:
        answer, _ = adapter.gen_fast(prompt, maxn=maxn)
        return {"answer": answer, "routed": "cap-denied:" + spec.name, "result": None, "total_gens": 1}

    # no fact applies -> plain pass-through (byte-identical to the bare model).
    if spec is None:
        answer, _ = adapter.gen_fast(prompt, maxn=maxn)
        return {"answer": answer, "routed": None, "result": None, "total_gens": 1}

    # a fact applies and is authorized -> dispose via the governed loop (thin-by-default; spend only when earned).
    res = governed_fact_loop(adapter, prompt,
                             fact_provider=spec.make_provider(prompt),
                             validate=spec.make_validate(prompt),
                             policy=policy, cost_source=cost_source, budget=budget,
                             importance=importance, maxn=maxn)
    return {"answer": res.answer, "routed": spec.name, "result": res, "total_gens": res.total_gens}


# ===========================================================================
# Scheduler executor — register the live dispatch as a kernel op (task_graph executor ABI, matched exactly).
# ===========================================================================
def _job_prompt(task) -> Optional[str]:
    """Read the task's prompt the way the other executors do (kernel/executors.py _jin, run_coding_kernel
    draft_model): task.meta['job_inputs']['prompt'], falling back to wired input artifacts / the contract."""
    jin = (task.meta or {}).get("job_inputs", {}) or {}
    if isinstance(jin.get("prompt"), str):
        return jin["prompt"]
    # fall back to an input artifact named 'prompt' (the convention for inputs wired as artifact refs)
    return None


def _fact_loop_verdict(out: dict) -> bool:
    """Was a dispatch result's answer trustworthy? The single source of truth for the executor's ok and the
    paired validator (so both agree). A pass-through (routed None / cap-denied; result None) carries no fact
    contract -> True (an honest JUDGMENT pass; the bare model's answer, unperturbed). A routed fact path is
    trusted iff its BASE already validated (thin) OR its re-prefilled verified facts validated (spend); a routed
    spend that STILL failed validation -> False (the node fails -> repair/abort as usual)."""
    res = out.get("result") if isinstance(out, dict) else None
    if res is None:
        return True
    return bool(res.audit.get("base_validated")) or (res.used_facts and bool(res.audit.get("final_validated")))


def register_fact_loop_executor(scheduler, adapter, router, policy, *, op="fact_loop",
                                cap=None, level=ValidationLevel.PROPERTY,
                                cost_source=None, budget=None, importance=0.7, maxn=512):
    """Register `op` as a scheduler EXECUTOR + its paired VALIDATOR (the live kernel dispatch), matching
    task_graph's ABI exactly.

    EXECUTOR — fn(sched, task, inputs) -> (result, ok, ValidationLevel), the executor-ABI shape:
      * reads the task's prompt (inputs['prompt'] when a dependency/artifact wired one in, else
        task.meta['job_inputs']['prompt'] — the run_coding_kernel/executors `_jin` convention),
      * calls dispatch(adapter, prompt, router=..., policy=..., capabilities=None) — admit() already gated the cap,
        so the executor does NOT double-gate,
      * result = the dispatch dict {answer, routed, result, total_gens} (so downstream nodes read result['answer']
        and the proof ledger records the routing/spend), ok/level = the self-computed verdict (_fact_loop_verdict).

    VALIDATOR — registered under key `<op>_validated` as fn(result, task) -> bool. NOTE the binding ABI detail:
    task_graph's Scheduler validates a node through its REGISTERED VALIDATOR (Scheduler._validate via
    task.validator), NOT through the executor's returned ok/level (that 2nd/3rd tuple element is the ABI shape but
    is discarded by _execute). So to make a node read VALIDATED/VERIFIED rather than pass-through JUDGMENT, the
    TASK must carry `validator={"fn": <op>_validated, "level": "<level>"}`. This validator recomputes the same
    verdict from the dispatch dict, so the executor's ok and the node's pass/fail always agree. A task that omits
    the validator still RUNS (judgment pass-through) — declaring it just upgrades the recorded verification.

    CAPABILITY GATE — the SCHEDULER's job, via the GENERIC fail-closed gate admit() runs for every task
    (Scheduler.admit: each cap in task.capabilities must be in scheduler.capabilities, else SchedulerReject). So a
    fact_loop TASK declares `capabilities=[cap]` (cap default 'fact:'+op) and the kernel grants it with
    `scheduler.capabilities |= {cap}`. An un-granted cap is rejected at admit() before the executor runs — the
    same gate exercised by test_capability_gate_fail_closed. (We deliberately do NOT register `op` in the
    syscall-op registry: that path MANDATES a validator AND a ceiling enforced at admit-time; here validation is a
    normal registered validator, leaving the op usable judgment-only when a caller wants a bare pass-through.)

    Returns {'cap': <required cap token>, 'validator': <validator key>, 'level': '<level name>'} so the caller can
    grant the cap and stamp the task's validator, e.g.
        reg = register_fact_loop_executor(s, adapter, router, policy)
        s.capabilities |= {reg['cap']}
        s.admit(Task('fl', 'fact_loop', capabilities=[reg['cap']], is_root=True,
                     validator={'fn': reg['validator'], 'level': reg['level']},
                     meta={'job_inputs': {'prompt': prompt}}))
    """
    cap = cap or ("fact:" + op)
    vkey = op + "_validated"

    def _executor(sched, task, inputs):
        prompt = None
        if isinstance(inputs, dict):                       # a dependency/artifact wired a prompt in
            v = inputs.get("prompt")
            if isinstance(v, str):
                prompt = v
        if prompt is None:
            prompt = _job_prompt(task)
        if prompt is None:
            return {"answer": None, "routed": None, "result": None, "total_gens": 0}, False, ValidationLevel.NONE

        out = dispatch(adapter, prompt, router=router, policy=policy, cost_source=cost_source, budget=budget,
                       capabilities=None,        # the scheduler's admit() already gated the cap; don't double-gate
                       importance=importance, maxn=maxn)
        verified = _fact_loop_verdict(out)
        return out, verified, (level if verified else ValidationLevel.NONE)

    def _validator(result, task):
        return _fact_loop_verdict(result)

    scheduler.register_executor(op, _executor)
    scheduler.register_validator(vkey, _validator)
    return {"cap": cap, "validator": vkey, "level": level.name}


__all__ = ["FactSpec", "FactRouter", "default_router", "dispatch", "register_fact_loop_executor"]
