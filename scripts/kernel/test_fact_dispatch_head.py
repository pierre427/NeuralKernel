#!/usr/bin/env python3
"""CPU-only test for kernel/fact_dispatch_head.py — the LEARNED-head router at the fact-dispatch seam.

NO model, NO checkpoint: HeadFactRouter is constructed with an INJECTED decide_fn(prompt) -> (class_str, conf),
so we drive every (class, confidence) case deterministically and assert the class->FactSpec mapping + the
confidence-threshold gate + the lexical fallback, all without touching mlx or a safetensors file. The real-model
routing (head reads North's residual and beats the lexical matcher on paraphrases) is proven separately by
run_fact_dispatch_head_north.py.

Cases (mirrors the task spec):
  (a) confident mapped classes route to the SAME FactSpec default_router() builds:
      count_byte 0.9 -> letter_count ; c_gcd -> gcd ; is_prime -> is_prime.
  (b) 'none' -> None (no fact; pass-through with no fallback).
  (c) a real class but conf 0.3 (< threshold 0.5) -> None (or the lexical fallback if one is wired).
  (d) with lexical_fallback=default_router(), a head 'none' on a CLEARLY-lexical prompt falls back and STILL routes.
  (e) the returned object is a FactSpec usable by dispatch (has .name / .cap / .make_provider / .make_validate),
      and dispatch() actually drives it on a fake adapter (the head router is a true drop-in for the lexical one).

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_fact_dispatch_head.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.fact_dispatch import FactSpec, default_router, dispatch
from kernel.fact_dispatch_head import HeadFactRouter, HEAD_CLASS_TO_SPEC
from kernel.fact_loop import FactLoopResult
from kernel.k_policy import KPolicy

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")

FACTS_HEADER = "Verified facts (computed by a trusted tool"   # the marker render_facts emits (see test_fact_dispatch)


def _fixed_decide(cls, conf):
    """A decide_fn that returns the same (class, conf) for every prompt — drives a single mapping/threshold case."""
    return lambda _prompt: (cls, conf)


class _FakeAdapter:
    """Deterministic, NO model (same convention as test_fact_dispatch._FakeAdapter): the facts-block-absent answer
    is the wrong `base`; once the verified-facts block is present, echo each '= value' so the answer validates for
    ANY routed primitive. Used only by case (e) to prove the head router drives dispatch end to end."""
    def __init__(self, base="the answer is 999"):
        self.base = base
        self.calls = []

    def gen_fast(self, prompt, maxn=512):
        self.calls.append(prompt)
        if FACTS_HEADER in prompt:
            vals = [m.group(1).strip() for m in re.finditer(r"=\s*([^\n]+)", prompt.split("\n\n")[0])]
            parts = []
            for v in vals:
                if v in ("True", "False"):
                    parts.append("Yes, it is prime." if v == "True" else "No, it is not prime (composite).")
                else:
                    parts.append(f"The exact answer is {v}.")
            text = " ".join(parts) if parts else "I am not sure."
        else:
            text = self.base
        return text, len(text.split())


# ---------------------------------------------------------------------------
def test_mapped_classes_route_to_same_specs():
    print("\n[a] confident mapped classes -> the SAME FactSpecs default_router() builds")
    base = default_router()                                   # the spec pool the head proposes from
    by_name = {s.name: s for s in base.specs}
    cases = [("count_byte", "letter_count"), ("c_gcd", "gcd"), ("is_prime", "is_prime")]
    for cls, spec_name in cases:
        r = HeadFactRouter(decide_fn=_fixed_decide(cls, 0.9), spec_source=base)
        s = r.route("any prompt — the head decision is injected, text is irrelevant here")
        ok(s is not None and s.name == spec_name,
           f"class {cls!r} @0.9 -> spec {spec_name!r} (got {getattr(s, 'name', None)!r})")
        ok(s is by_name.get(spec_name),
           f"  ... and it is default_router()'s OWN {spec_name!r} spec (same provider/validate/cap reused)")
    # the class->spec map is exactly the documented one (no surprise classes)
    ok(HEAD_CLASS_TO_SPEC == {"count_byte": "letter_count", "c_gcd": "gcd", "is_prime": "is_prime"},
       f"HEAD_CLASS_TO_SPEC is the documented mapping (got {HEAD_CLASS_TO_SPEC!r})")


def test_none_class_routes_to_none():
    print("\n[b] 'none' (no trap) -> None with no fallback (clean pass-through)")
    r = HeadFactRouter(decide_fn=_fixed_decide("none", 0.99))
    ok(r.route("Write a two-line haiku about autumn leaves.") is None, "class 'none' @0.99 -> None")
    # an unmapped trap class the fact-loop ships no primitive for is ALSO an honest None
    r2 = HeadFactRouter(decide_fn=_fixed_decide("shortest_dist", 0.99))
    ok(r2.route("shortest path from node 0 to 5") is None,
       "class 'shortest_dist' @0.99 -> None (no kernel primitive -> honest boundary)")


def test_subthreshold_routes_to_none():
    print("\n[c] a real class but conf 0.3 (< threshold 0.5) -> None (the degenerate scheduler-owned-k gate)")
    r = HeadFactRouter(decide_fn=_fixed_decide("c_gcd", 0.3))     # no fallback
    ok(r.route("What is gcd(1071, 462)?") is None, "class 'c_gcd' @0.3 < 0.5 -> None (sub-threshold, no fallback)")
    # exactly-at-threshold fires (>= is inclusive); just-below does not
    ok(HeadFactRouter(decide_fn=_fixed_decide("c_gcd", 0.5)).route("gcd?") is not None,
       "conf == threshold (0.5) fires (>= is inclusive)")
    ok(HeadFactRouter(decide_fn=_fixed_decide("c_gcd", 0.49)).route("gcd?") is None,
       "conf just below threshold (0.49) -> None")
    # custom threshold honored
    ok(HeadFactRouter(decide_fn=_fixed_decide("c_gcd", 0.6), conf_threshold=0.8).route("gcd?") is None,
       "conf 0.6 with a raised threshold 0.8 -> None")


def test_subthreshold_falls_back_to_lexical():
    print("\n[c'] sub-threshold WITH a lexical fallback -> the deterministic matcher still routes")
    fb = default_router()
    r = HeadFactRouter(decide_fn=_fixed_decide("c_gcd", 0.3), lexical_fallback=fb)
    s = r.route("What is the greatest common divisor of 1071 and 462?")
    ok(s is not None and s.name == "gcd",
       f"sub-threshold head + lexical fallback on a gcd prompt -> gcd via fallback (got {getattr(s,'name',None)!r})")


def test_none_falls_back_on_lexical_prompt():
    print("\n[d] head 'none' on a CLEARLY-lexical prompt + lexical_fallback -> falls back and STILL routes")
    fb = default_router()
    r = HeadFactRouter(decide_fn=_fixed_decide("none", 0.95), lexical_fallback=fb)
    # the head says 'no trap' (wrong here), but the deterministic matcher rescues the obvious letter-count prompt.
    s = r.route("How many times does the letter r occur in the word strawberry?")
    ok(s is not None and s.name == "letter_count",
       f"head 'none' overridden by lexical fallback on a letter-count prompt -> letter_count (got {getattr(s,'name',None)!r})")
    # and a genuine chat prompt still ends up None even WITH the fallback (fallback also finds no fact)
    ok(r.route("Write a two-line haiku about autumn leaves.") is None,
       "head 'none' + fallback on a true chat prompt -> still None (fallback also routes nothing)")


def test_returned_spec_drives_dispatch():
    print("\n[e] the returned object is a FactSpec usable by dispatch() (drop-in for the lexical router)")
    base = default_router()
    r = HeadFactRouter(decide_fn=_fixed_decide("count_byte", 0.9), spec_source=base)
    s = r.route("How many times does the letter r occur in the word strawberry?")
    # structural: it is a FactSpec with the dispatch-required surface
    ok(isinstance(s, FactSpec), "routed object is a FactSpec instance")
    for attr in ("name", "cap", "make_provider", "make_validate"):
        ok(hasattr(s, attr), f"  spec has .{attr}")
    ok(s.cap == "fact:letter_count", f"spec carries the capability token (got {s.cap!r})")
    # behavioral: dispatch() drives the HeadFactRouter exactly as it would the lexical one, and LIFTS a wrong base.
    fake = _FakeAdapter(base="It appears 2 times.")           # WRONG: strawberry has 3 r's
    out = dispatch(fake, "How many times does the letter r occur in the word strawberry?",
                   router=r, policy=KPolicy(), cost_source=lambda: 0.0)
    ok(out["routed"] == "letter_count", f"dispatch(router=HeadFactRouter) routed letter_count (got {out['routed']!r})")
    ok(isinstance(out["result"], FactLoopResult) and out["result"].used_facts is True,
       "the governed fact-loop spent on the verified fact through the head router")
    ok(re.search(r"(?<!\d)3(?!\d)", out["answer"]) is not None,
       f"the lifted answer contains the verified count 3 (got {out['answer']!r})")
    # pass-through inertness still holds via the head router: a 'none' decision -> byte-identical bare gen.
    r_none = HeadFactRouter(decide_fn=_fixed_decide("none", 0.95))
    fake2 = _FakeAdapter(base="autumn winds drift / crisp leaves spiral down")
    p = "Write a two-line haiku about autumn leaves."
    direct, _ = fake2.gen_fast(p)
    out2 = dispatch(fake2, p, router=r_none, policy=KPolicy(), cost_source=lambda: 0.0)
    ok(out2["routed"] is None and out2["answer"] == direct,
       "head 'none' -> dispatch pass-through byte-identical to bare gen_fast (inertness preserved)")


if __name__ == "__main__":
    test_mapped_classes_route_to_same_specs()
    test_none_class_routes_to_none()
    test_subthreshold_routes_to_none()
    test_subthreshold_falls_back_to_lexical()
    test_none_falls_back_on_lexical_prompt()
    test_returned_spec_drives_dispatch()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} head-router checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
