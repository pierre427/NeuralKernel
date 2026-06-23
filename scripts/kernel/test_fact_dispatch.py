#!/usr/bin/env python3
"""CPU-only test for kernel/fact_dispatch.py — the LIVE DISPATCH path over the M5-governed fact-loop.

No model: a deterministic _FakeAdapter whose gen_fast(prompt) answer depends ONLY on whether the verified-facts
block is present (so we control base-wrong vs facts-fix-it) — same convention as test_fact_loop.py /
test_scheduler.py. Proves the dispatch seam end-to-end:
  1. ROUTING: default_router() routes a letter-count prompt -> letter_count, a gcd prompt -> gcd, chat -> None.
  2. PASS-THROUGH inertness: an unrelated prompt -> routed None and answer byte-identical to gen_fast (the fact
     path can never perturb a prompt it does not own).
  3. LIFT: a fact-needing prompt whose BASE answer is wrong but becomes right once the verified-facts block is
     present -> routed == spec, result.used_facts True, the answer validates.
  4. CAPABILITY gate: capabilities lacking the spec.cap -> cap-denied pass-through (no fact path); granting the
     cap -> the fact path runs.
  5. SCHEDULER EXECUTOR: the REAL kernel.task_graph.Scheduler (pure Python) admits + runs a fact_loop task through
     register_fact_loop_executor, and an un-granted capability is rejected by admit() (fail-closed).
The real-model lift (token delivery actually working on North) is proven separately by run_fact_loop_north.py.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_fact_dispatch.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.fact_dispatch import (FactSpec, FactRouter, default_router, dispatch,
                                  register_fact_loop_executor)
from kernel.fact_loop import Fact, FactLoopResult
from kernel.k_policy import KPolicy
from kernel.task_graph import Scheduler, Task, SchedulerReject, TaskState, ValidationLevel
from kernel.commit_trap import TrapBudget

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")

FACTS_HEADER = "Verified facts (computed by a trusted tool"   # the marker render_facts emits


class _FakeAdapter:
    """Deterministic, NO model. gen_fast(prompt) -> (text, n):
      * facts block ABSENT -> `base` (the wrong base attempt),
      * facts block PRESENT -> an answer that ECHOES the verified value(s) from the facts block, so it validates
        for ANY routed fact (letter-count int, gcd int, is_prime bool) without hard-coding a single answer.
    n is len(text.split()); a 'calls' log lets tests count gens and inspect what the adapter saw."""
    def __init__(self, base="the answer is 999"):
        self.base = base
        self.calls = []

    def gen_fast(self, prompt, maxn=512):
        self.calls.append(prompt)
        if FACTS_HEADER in prompt:
            text = _answer_from_facts_block(prompt)
        else:
            text = self.base
        return text, len(text.split())


def _answer_from_facts_block(prompt: str) -> str:
    """Build an answer that USES the verified facts block — echo each '- label = value' value so the answer
    validates regardless of which primitive routed (mirrors a model that re-read the fact from layer 0). The
    facts block is everything before the '\\n\\n' separator augment_prompt inserts before the original prompt."""
    vals = [m.group(1).strip() for m in re.finditer(r"=\s*([^\n]+)", prompt.split("\n\n")[0])]
    if not vals:
        return "I am not sure."
    parts = []
    for v in vals:
        if v in ("True", "False"):
            parts.append("Yes, it is prime." if v == "True" else "No, it is not prime (composite).")
        else:
            parts.append(f"The exact answer is {v}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
def test_routing():
    print("\n[1] routing: letter-count -> letter_count, gcd -> gcd, chat -> None")
    r = default_router()
    s_lc = r.route("Could you tell me how many times the letter r occurs in the word strawberry?")
    ok(s_lc is not None and s_lc.name == "letter_count", f"letter-count prompt routes to letter_count (got {getattr(s_lc,'name',None)!r})")
    s_gcd = r.route("What is the greatest common divisor of 1071 and 462?")
    ok(s_gcd is not None and s_gcd.name == "gcd", f"gcd prompt routes to gcd (got {getattr(s_gcd,'name',None)!r})")
    s_pr = r.route("Is 1000003 a prime number? Answer precisely.")
    ok(s_pr is not None and s_pr.name == "is_prime", f"prime prompt routes to is_prime (got {getattr(s_pr,'name',None)!r})")
    s_chat = r.route("Write a two-line haiku about autumn leaves.")
    ok(s_chat is None, f"unrelated/chat prompt routes to None (got {getattr(s_chat,'name',None)!r})")
    # the spec carries the capability token the gate keys on
    ok(s_lc.cap == "fact:letter_count", f"letter_count spec declares cap 'fact:letter_count' (got {s_lc.cap!r})")


def test_passthrough_inertness():
    print("\n[2] PASS-THROUGH inertness: a non-matching prompt is byte-identical to plain gen_fast")
    fake = _FakeAdapter(base="autumn winds drift / crisp leaves spiral to the ground")
    p = "Write a two-line haiku about autumn leaves."
    direct, _ = fake.gen_fast(p)                 # what the bare model produces
    n_before = len(fake.calls)
    out = dispatch(fake, p, router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0)
    ok(out["routed"] is None, f"routed is None on a non-matching prompt (got {out['routed']!r})")
    ok(out["result"] is None, "result is None on a pass-through (no fact-loop ran)")
    ok(out["answer"] == direct, f"answer byte-identical to gen_fast(prompt)[0] (got {out['answer']!r})")
    ok(out["total_gens"] == 1, f"total_gens == 1 on a pass-through (got {out['total_gens']})")
    ok(len(fake.calls) == n_before + 1, "the pass-through made exactly one model gen (the bare attempt)")
    ok(fake.calls[-1] == p, "the prompt passed to the adapter was byte-identical (no facts block prepended)")


def test_lift():
    print("\n[3] LIFT: base answer wrong, becomes right once the verified-facts block is present")
    fake = _FakeAdapter(base="It appears 2 times.")   # WRONG: strawberry has 3 r's
    p = "How many times does the letter r occur in the word strawberry?"
    out = dispatch(fake, p, router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0, budget=None)
    ok(out["routed"] == "letter_count", f"routed == 'letter_count' (got {out['routed']!r})")
    res = out["result"]
    ok(isinstance(res, FactLoopResult), "result is a FactLoopResult")
    ok(res.used_facts is True, "result.used_facts True (the loop spent on the verified-facts re-prefill)")
    ok(out["total_gens"] == 2, f"total_gens == 2 (base + re-prefill) (got {out['total_gens']})")
    ok(res.audit.get("base_validated") is False, "audit base_validated False (base answer '2' was wrong)")
    ok(res.audit.get("final_validated") is True, "audit final_validated True (the verified fact fixed it)")
    # the answer must contain the VERIFIED count (3), judged against ground truth — not the wrong base.
    ok(re.search(r"(?<!\d)3(?!\d)", out["answer"]) is not None, f"answer contains the verified count 3 (got {out['answer']!r})")
    ok(any(FACTS_HEADER in c for c in fake.calls), "the second gen actually saw the verified-facts block")
    # a gcd lift too (different primitive, same dispatch path)
    fake2 = _FakeAdapter(base="The gcd is 7.")        # WRONG: gcd(1071,462) == 21
    out2 = dispatch(fake2, "What is the greatest common divisor of 1071 and 462?",
                    router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0)
    ok(out2["routed"] == "gcd" and out2["result"].used_facts is True, "gcd prompt also lifts via the fact path")
    ok(re.search(r"(?<!\d)21(?!\d)", out2["answer"]) is not None, f"gcd answer contains the verified 21 (got {out2['answer']!r})")


def test_thin_passthrough_when_base_right():
    print("\n[3b] THIN: a routed prompt whose BASE already validates -> no re-prefill (multiplier reserved)")
    fake = _FakeAdapter(base="There are exactly 3 of them.")   # already correct for strawberry/r
    p = "How many times does the letter r occur in the word strawberry?"
    out = dispatch(fake, p, router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0)
    ok(out["routed"] == "letter_count", "still ROUTED (a fact applies) ...")
    ok(out["result"].used_facts is False and out["total_gens"] == 1, "... but THIN: base validated, no re-prefill")


def test_capability_gate():
    print("\n[4] CAPABILITY gate: missing cap -> cap-denied pass-through; granted cap -> the fact path runs")
    p = "How many times does the letter r occur in the word strawberry?"
    spec = default_router().route(p)
    # (a) empty granted set -> the matching spec's cap is denied -> pass-through (no fact path).
    fake = _FakeAdapter(base="It appears 2 times.")
    out_denied = dispatch(fake, p, router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0,
                          capabilities=set())
    ok(out_denied["routed"] == "cap-denied:letter_count", f"empty caps -> routed 'cap-denied:letter_count' (got {out_denied['routed']!r})")
    ok(out_denied["result"] is None and out_denied["total_gens"] == 1, "cap-denied -> no fact-loop, single gen")
    ok(out_denied["answer"] == "It appears 2 times.", "cap-denied -> the bare (wrong) base answer, fact path NOT run")
    ok(not any(FACTS_HEADER in c for c in fake.calls), "cap-denied -> no verified-facts block was ever generated")
    # (b) a set lacking THIS cap (but non-empty) is also denied.
    fake_b = _FakeAdapter(base="It appears 2 times.")
    out_other = dispatch(fake_b, p, router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0,
                         capabilities={"fact:gcd"})
    ok(out_other["routed"] == "cap-denied:letter_count", "a cap set lacking the spec's cap is also denied")
    # (c) granting the cap -> the fact path runs and lifts.
    fake2 = _FakeAdapter(base="It appears 2 times.")
    out_ok = dispatch(fake2, p, router=default_router(), policy=KPolicy(), cost_source=lambda: 0.0,
                      capabilities={spec.cap})
    ok(out_ok["routed"] == "letter_count", "granting the cap -> routed letter_count (fact path authorized)")
    ok(out_ok["result"] is not None and out_ok["result"].used_facts is True, "granted cap -> the fact path runs and spends")
    ok(re.search(r"(?<!\d)3(?!\d)", out_ok["answer"]) is not None, "granted-cap answer contains the verified count 3")


def test_scheduler_executor():
    print("\n[5] SCHEDULER EXECUTOR: real task_graph.Scheduler admits + runs op:fact_loop; un-granted cap rejected")
    p = "How many times does the letter r occur in the word strawberry?"

    # (a) un-granted cap is REJECTED by admit() (fail-closed) — declare the task's cap but DON'T grant it.
    s0 = Scheduler()                                   # empty granted capability set
    fake0 = _FakeAdapter(base="It appears 2 times.")
    reg = register_fact_loop_executor(s0, fake0, default_router(), KPolicy(), op="fact_loop",
                                      cost_source=lambda: 0.0)
    cap, vfn, vlevel = reg["cap"], reg["validator"], reg["level"]
    ok(cap == "fact:fact_loop", f"register returns the required cap token (got {cap!r})")
    ok(vfn == "fact_loop_validated", f"register returns the paired validator key (got {vfn!r})")
    rejected = False
    try:
        s0.admit(Task("fl", "fact_loop", capabilities=[cap], is_root=True,
                      validator={"fn": vfn, "level": vlevel}, meta={"job_inputs": {"prompt": p}}))
    except SchedulerReject:
        rejected = True
    ok(rejected, "admit() REJECTS a fact_loop task whose cap was not granted (fail-closed)")

    # (b) granting the cap -> the task admits, runs through the executor, and the verified fact LIFTS it.
    s = Scheduler(capabilities={cap})
    fake = _FakeAdapter(base="It appears 2 times.")    # WRONG base -> the loop must spend
    reg2 = register_fact_loop_executor(s, fake, default_router(), KPolicy(), op="fact_loop", cost_source=lambda: 0.0)
    t = s.admit(Task("fl", "fact_loop", capabilities=[cap], is_root=True,
                     validator={"fn": reg2["validator"], "level": reg2["level"]},
                     meta={"job_inputs": {"prompt": p}}))
    ok(t.status in (TaskState.READY, TaskState.BLOCKED), "granted-cap fact_loop task ADMITS")
    summary = s.run()
    ok(summary["committed"], "the fact_loop task ran and COMMITTED through the scheduler")
    out = s.result("fl")
    ok(isinstance(out, dict) and out.get("routed") == "letter_count", f"executor result routed letter_count (got {out and out.get('routed')!r})")
    ok(out.get("result") is not None and out["result"].used_facts is True, "the scheduled run spent on the verified fact")
    ok(re.search(r"(?<!\d)3(?!\d)", out.get("answer") or "") is not None, f"scheduled answer contains the verified count 3 (got {out.get('answer')!r})")
    ok(s.tasks["fl"].verification.value in ("validated", "verified"), f"node verification reflects a verified fact (got {s.tasks['fl'].verification.value})")
    ok(summary["ledger_verified"], "the proof ledger is hash-verified for the scheduled fact_loop run")

    # (c) a PLAIN pass-through (chat prompt) through the scheduler still commits (validator True for pass-through).
    s2 = Scheduler(capabilities={cap})
    fake2 = _FakeAdapter(base="autumn leaves fall / a quiet golden descent")
    reg3 = register_fact_loop_executor(s2, fake2, default_router(), KPolicy(), op="fact_loop", cost_source=lambda: 0.0)
    s2.admit(Task("chat", "fact_loop", capabilities=[cap], is_root=True,
                  validator={"fn": reg3["validator"], "level": reg3["level"]},
                  meta={"job_inputs": {"prompt": "Write a two-line haiku about autumn leaves."}}))
    sm2 = s2.run()
    ok(sm2["committed"], "a chat prompt pass-through also commits through the scheduler (judgment pass)")
    ok((s2.result("chat") or {}).get("routed") is None, "the chat run was a pass-through (routed None)")


if __name__ == "__main__":
    test_routing()
    test_passthrough_inertness()
    test_lift()
    test_thin_passthrough_when_base_right()
    test_capability_gate()
    test_scheduler_executor()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} fact-dispatch checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
