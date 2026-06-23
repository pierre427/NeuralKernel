#!/usr/bin/env python3
"""kernel/decompose.py — MIXED-PROMPT DECOMPOSER (prose / code span router).

A large mixed prompt (prose + a code artifact + more prose) is SEGMENTED into ordered spans; each span is ROUTED by
the scheduler:
  • prose span  -> model generation (answer just this part, with the full prompt as context).
  • code span   -> (1) LEXICAL LOOKUP of an already-PROVEN primitive whose name matches the requested entry — the
                   fastpath: reused, zero codegen; else (2) GENERATE + sandbox-CHECK via the admit-gated op:venv_exec
                   (the candidate must define a parameterless _selftest() that returns True; it runs out-of-process)
                   with ONE repair on failure.
Results are STITCHED back into one answer, each span carrying its TRUST TIER.

Honest tiering (the crux): a bespoke class has no INDEPENDENT oracle, so it earns CHECKED ("ran + its self-test
passed in a sandbox"), never PROVEN ("hold-out-gated vs an independent oracle" — that path is self_extend.py, which
needs a generator+oracle the user prompt does not supply). The decomposer LABELS each span's tier; it never claims
proof it cannot back.
"""
from __future__ import annotations
import os, sys, re, json, time
from dataclasses import dataclass, field
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kernel.task_graph import Scheduler, Task
from kernel.tools import register_builtins
from kernel.lexical_proposer import propose, validate_proven, governed_propose

_FRAMING = re.compile(r"<\|[A-Z_]+\|>")


def _clean(t):
    return _FRAMING.sub("", t or "").strip()


def _json_block(t):
    t = _clean(t)
    m = re.search(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", t) or re.search(r"([\[{][\s\S]*[\]}])", t)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _extract_code(t):
    m = re.search(r"```(?:[a-z+]*)\s*\n([\s\S]*?)```", _clean(t), re.I)
    return (m.group(1) if m else _clean(t)).strip()


def _objects(arr_text):
    """Yield top-level {...} object substrings from a JSON-ish array body — brace-matched + string-aware, so one
    MALFORMED element can't swallow the rest (per-span recovery instead of all-or-nothing json.loads)."""
    objs, depth, start, q, esc = [], 0, None, None, False
    for i, ch in enumerate(arr_text):
        if q:
            esc = (ch == "\\" and not esc)
            if ch == q and not esc:
                q = None
            continue
        if ch in "\"'":
            q = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(arr_text[start:i + 1]); start = None
    return objs


def _salvage(obj):
    """Best-effort recovery of a malformed span object: pull the structural string fields, DROP the bad parts
    (e.g. a broken sample_cases array). A code span survives without cases -> routes via lexical reuse / generate."""
    def s(k):
        m = re.search(r'"' + k + r'"\s*:\s*"([^"]*)"', obj)
        return m.group(1) if m else ""
    kind = s("kind")
    return {"kind": kind, "entry": s("entry"), "language": s("language") or "python",
            "signature": s("signature"), "instruction": s("instruction"), "sample_cases": []} if kind else None


def _parse_spans(text):
    """Robust span parse: clean json.loads first; else extract + parse/salvage each object so a single malformed
    element (a frequent small-model glitch, e.g. args missing a bracket) doesn't collapse the whole decomposition."""
    text = _clean(text)
    arr = _json_block(text)
    if isinstance(arr, list):
        return arr
    m = re.search(r"\[([\s\S]*)\]", text)
    out = []
    for obj in _objects(m.group(1) if m else text):
        try:
            out.append(json.loads(obj))
        except Exception:
            sal = _salvage(obj)
            if sal:
                out.append(sal)
    return out


@dataclass
class Span:
    idx: int
    kind: str                # "prose" | "code"
    instruction: str
    language: str = "python"
    entry: str = ""          # function/class name for a code span
    signature: str = ""
    cases: list = field(default_factory=list)   # [{args|input, expected}] — used to validate a proven-primitive reuse


_DECOMP_PROMPT = (
    "Split the USER PROMPT into an ordered JSON array of PARTS for separate handling. Each part is either\n"
    '  {"kind":"prose","instruction":"<what to answer in prose>"}  or\n'
    '  {"kind":"code","language":"python","entry":"<function or class name>","signature":"<sig>",'
    '"instruction":"<what to build>","sample_cases":[{"args":[<arg>,...],"expected":<value>}]}\n'
    "For a plain FUNCTION, include 1-3 concrete sample_cases (args -> expected) so it can be validated/reused. "
    "For a CLASS or where examples are awkward, use \"sample_cases\":[]. Preserve order; merge adjacent prose. "
    "Output ONE JSON array and nothing else.\n\nUSER PROMPT:\n___P___\n\nJSON:")

_CODE_PROMPT = (
    "Write {lang} code for the BUILD below. Then add a parameterless function `_selftest()` that constructs/exercises "
    "it and returns True iff it works (use asserts inside). Output ONLY one ```{lang} ...``` block, nothing else.\n\n"
    "FULL TASK (context):\n{ctx}\n\nBUILD: {instr}\n")

_PROSE_PROMPT = (
    "Answer ONLY this part of the user's request, in prose. Do NOT write code.\n\n"
    "FULL REQUEST (context):\n{ctx}\n\nANSWER THIS PART: {instr}\n")


def decompose_prompt(model, prompt, maxn=1000):
    out, _ = model.gen_fast(_DECOMP_PROMPT.replace("___P___", prompt), maxn=maxn)
    arr = _parse_spans(out)
    spans = []
    if isinstance(arr, list):
        for i, s in enumerate(arr):
            if not isinstance(s, dict):
                continue
            kind = "code" if s.get("kind") == "code" else "prose"
            cases = s.get("sample_cases")
            spans.append(Span(i, kind, str(s.get("instruction", "")).strip(), str(s.get("language", "python")),
                              str(s.get("entry", "")).strip(), str(s.get("signature", "")).strip(),
                              cases if isinstance(cases, list) else []))
    return spans or [Span(0, "prose", prompt)]            # fallback: whole prompt = one prose span


class Decomposer:
    def __init__(self, model, *, k_policy=None, cost_source=None, route_budget=None):
        self.model = model
        self.sched = Scheduler()
        register_builtins(self.sched, grant=True)         # registers op:venv_exec (+ datetime/sysinfo/trace)
        self.events = []
        # M5 (project_scheduler_owned_k): OPTIONAL scheduler-owned-k disposal over the code-span fastpath. When
        # k_policy is None the routing is byte-identical to before (fixed propose(top_k=6) + validate loop). When
        # set, governed_propose disposes how many proven candidates to validate-the-branch (a clear winner -> 1; a
        # close race -> the top-k) AND gap-detects: if nothing clears the route floor, skip the futile reuse loop
        # and go straight to GENERATE (the self-extending plane's synthesize path), cost/budget-braked.
        self.k_policy = k_policy
        self.cost_source = cost_source
        self.route_budget = route_budget

    def _venv_check(self, code, entry="_selftest"):
        """Run the candidate's self-test OUT-OF-PROCESS through the admit-gated op:venv_exec. CHECKED tier."""
        t0 = time.time()
        task = Task(task_id=f"venv:{len(self.events)}", kind="venv_exec",
                    validator={"fn": "venv_exec_ok", "level": "SCHEMA"}, capabilities=["op:venv_exec"])
        self.sched.admit(task)
        res, ex_ok, _ = self.sched.executors["venv_exec"](self.sched, task,
                        {"code": code, "entry": entry, "cases": [{"args": [], "expected": True}], "timeout_s": 10})
        res = res or {}
        ok = bool(ex_ok and res.get("ok"))
        ev = {"kind": "venv_check", "ok": ok, "ms": int((time.time() - t0) * 1000),
              "err": res.get("error"), "fails": res.get("fails")}
        self.events.append(ev)
        return ev

    def _gen_code(self, span, ctx, extra=""):
        out, _ = self.model.gen_fast(
            _CODE_PROMPT.format(lang=span.language, ctx=ctx[:1000], instr=span.instruction + extra), maxn=1800)
        return _extract_code(out)

    def route(self, span, ctx):
        if span.kind == "prose":
            out, _ = self.model.gen_fast(_PROSE_PROMPT.format(ctx=ctx[:1000], instr=span.instruction), maxn=1000)
            return {"span": span, "tier": "prose", "content": _clean(out)}

        # FASTPATH: the Step-4 lexical proposer RANKS the routable registry; validate-the-branch DISPOSES.
        # M5: if a k_policy is set, governed_propose disposes the candidate pool (clear winner -> 1; close race ->
        # top-k) and GAP-DETECTS (nothing clears the floor -> straight to codegen). Else the original fixed top_k=6.
        if self.k_policy is not None:
            g = governed_propose(span.entry, span.instruction, policy=self.k_policy,
                                 cost_source=self.cost_source, budget=self.route_budget)
            self.events.append({"kind": "governed_propose", "entry": span.entry, "action": g["action"],
                                "k": g["k"], **{k: g["audit"].get(k) for k in ("used_signal", "score_uncertainty")}})
            cands = g["candidates"] if g["action"] == "route" else []   # synthesize -> empty -> falls to codegen
        else:
            cands = propose(span.entry, span.instruction, top_k=6)
        for c in cands:                                   # first PROVEN candidate that PASSES the span's cases wins
            if c["tier"] == "proven" and span.cases and validate_proven(c["name"], span.cases):
                return {"span": span, "tier": "proven-reuse", "primitive": c["name"], "score": c["score"],
                        "validated": True,
                        "content": (f"# routed to PROVEN primitive `{c['name']}`  {c['signature']}\n"
                                    f"# validated on {len(span.cases)} example case(s) — no codegen.")}
        top = cands[0] if cands else None                 # else high-confidence lexical reuse (no cases to validate)
        if top and not span.cases and top["score"] >= 0.85:
            return {"span": span, "tier": "reuse-lexical", "primitive": top["name"], "score": top["score"],
                    "validated": False,
                    "content": (f"# routed to {top['tier']} capability `{top['name']}`  {top['signature']}\n"
                                f"# lexical match {top['score']} (no examples to validate against).")}

        code = self._gen_code(span, ctx)                  # no trusted capability fits -> generate + sandbox-CHECK
        chk = self._venv_check(code)
        if chk["ok"]:
            return {"span": span, "tier": "checked", "content": code}
        fb = str(chk.get("err") or (chk.get("fails") or [{}])[0])[:300]    # one repair, with the failure as feedback
        code2 = self._gen_code(span, ctx, extra=f"\n\nThe previous attempt FAILED its self-test: {fb}. Fix it.")
        chk2 = self._venv_check(code2)
        return {"span": span, "tier": "checked" if chk2["ok"] else "checked-fail",
                "content": code2, "repaired": chk2["ok"], "detail": None if chk2["ok"] else (chk2.get("err") or chk2.get("fails"))}

    def solve(self, prompt):
        spans = decompose_prompt(self.model, prompt)
        results = [self.route(s, prompt) for s in spans]
        return {"spans": spans, "results": results, "answer": stitch(results)}


def stitch(results):
    parts = []
    for r in results:
        s = r["span"]
        if r["tier"] == "prose":
            parts.append(r["content"])
        elif r["tier"] == "proven-reuse":
            parts.append(f"```python\n{r['content']}\n```\n_[kernel: PROVEN primitive reused — `{r['primitive']}` "
                         f"(score {r.get('score')}, validated on examples), zero codegen]_")
        elif r["tier"] == "reuse-lexical":
            parts.append(f"```python\n{r['content']}\n```\n_[kernel: reused `{r['primitive']}` "
                         f"({r.get('score')} lexical match, unvalidated)]_")
        else:
            label = {"checked": "CHECKED ✓ sandbox self-test passed",
                     "checked-fail": "CHECKED ✗ self-test still failing"}[r["tier"]]
            rep = " (repaired once)" if r.get("repaired") else ""
            parts.append(f"```{s.language}\n{r['content']}\n```\n_[kernel: {label}{rep}]_")
    return "\n\n".join(parts)


__all__ = ["Span", "Decomposer", "decompose_prompt", "lexical_lookup", "stitch"]


if __name__ == "__main__":                                # demo: a real mixed prompt (prose + class + prose)
    from north_adapter import NorthAdapter
    PROMPT = (sys.argv[1] if len(sys.argv) > 1 else
              "I'm building a tiny analytics tool. First, briefly explain what a moving average is and when it helps. "
              "Then write a Python function `gcd(a, b)` that returns the greatest common divisor of two integers. "
              "Then write a Python class `RollingStats` that ingests numbers one at a time and can return the count, "
              "mean, min, max, and a simple moving average over the last k values. Finally, give one tip for choosing "
              "the window size k.")
    print("[load] north ..."); model = NorthAdapter()
    d = Decomposer(model)
    t0 = time.time(); res = d.solve(PROMPT)
    print(f"\n=== DECOMPOSITION ({len(res['spans'])} spans, {time.time()-t0:.1f}s) ===")
    for r in res["results"]:
        s = r["span"]
        print(f"  span {s.idx}: {s.kind:5} tier={r['tier']:13} {('entry='+s.entry) if s.entry else s.instruction[:50]}")
    print("\n=== STITCHED ANSWER ===\n")
    print(res["answer"])
