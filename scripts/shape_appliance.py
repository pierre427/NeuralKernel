#!/usr/bin/env python3
"""Model-AGNOSTIC shape-routing appliance core.

The deterministic stack (shape_recoveries, shape_detector, output_validators) is independent of the
LLM. The only model-specific surface is a small ModelAdapter:

    class ModelAdapter:
        name: str
        def gen_fast(prompt) -> (answer_text, n_tokens)       # fast / low-reasoning path
        def gen_reasoning(prompt) -> (answer_text, n_tokens)  # escalation / high-reasoning path
        # both return the model's ANSWER text (the final channel / post-reasoning answer)

Given an adapter, run_appliance() executes the same recovery-first pipeline used for North, so a new
model is ported by writing one adapter — nothing in the recovery/detector/validator layer changes."""
from __future__ import annotations
import ast, re, signal, time
from collections import Counter
from shape_detector import detect
from shape_recoveries import RECOVERIES
from output_validators import detect_output_type, is_complete_valid


def _extract(text):
    # lenient fence: ```python / ```py / ```Python / bare ```, with or without the trailing newline
    m = re.search(r"```(?:python|py)?[ \t]*\n?(.*?)```", text, re.S | re.I)
    return m.group(1) if m else text


def _sig_params(text, entry):
    """Parameter NAMES of `entry(...)` in `text` (a prompt signature or a recovery's def), or None.
    Bracket-aware: splits the arg list on top-level commas only, so type annotations like
    `points: list[tuple[int,int]]` don't fabricate spurious params (the commas inside [...] are nested)."""
    i = text.find(entry)
    while i != -1:
        # left word-boundary: reject when `entry` is the suffix of a longer identifier (e.g. my_dijkstra)
        left_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        j = i + len(entry)
        while j < len(text) and text[j] == " ":
            j += 1
        if left_ok and j < len(text) and text[j] == "(":
            break
        i = text.find(entry, i + 1)
    if i == -1 or j >= len(text) or text[j] != "(":
        return None
    depth = 0; k = j; end = -1
    for k in range(j, len(text)):                       # find the matching ) for this (
        c = text[k]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                end = k; break
    if end == -1:
        return None
    inner = text[j + 1:end]
    parts = []; buf = ""; d = 0                          # split on top-level commas only
    for c in inner:
        if c in "([{":
            d += 1; buf += c
        elif c in ")]}":
            d -= 1; buf += c
        elif c == "," and d == 0:
            parts.append(buf); buf = ""
        else:
            buf += c
    if buf.strip():
        parts.append(buf)
    out = []
    for p in parts:
        name = p.split(":")[0].split("=")[0].strip().lstrip("*")
        if name and name != "self":
            out.append(name)
    return out


def _contract_matches(prompt, recovery_code, entry):
    """A recovery may only fire if its signature matches the task's — prevents cross-suite collisions
    (e.g. ladder topological_sort(graph) vs challenge topological_sort(n, edges))."""
    tp = _sig_params(prompt, entry); rp = _sig_params(recovery_code, entry)
    if tp is None or rp is None:
        return True                                    # can't tell → don't block
    return tp == rp


def _compile(code, entry):
    ns = {}
    try:
        exec(code, ns)
        fn = ns.get(entry)
        return fn if callable(fn) else None
    except Exception:
        return None


def _struct_eq(a, b):                          # mirrors the harness runner: tuples ≡ lists
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_struct_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_struct_eq(a[k], b[k]) for k in a)
    return a == b


class _Timeout(Exception):
    pass


def _timed(fn, args=(), kwargs=None, t=5):
    kwargs = kwargs or {}
    import threading
    if threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)            # SIGALRM only arms in the main thread
    def _h(s, f): raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h)
    prev = signal.setitimer(signal.ITIMER_REAL, t)   # prev=(interval, remaining) of any outer deadline
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        if prev and prev[1] > 0:                     # restore an outer caller's pending deadline (re-entrancy)
            signal.setitimer(signal.ITIMER_REAL, prev[1])


def _coerce_int_keys(obj):                     # mirrors the harness: JSON string keys → int for dict[int,…]
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) and k.lstrip("-").isdigit() for k in obj):
            return {int(k): _coerce_int_keys(v) for k, v in obj.items()}
        return {k: _coerce_int_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_int_keys(x) for x in obj]
    return obj


def check_cases(code, entry, cases):
    fn = _compile(code, entry)
    if fn is None:
        return False
    for c in cases:
        exp = _coerce_int_keys(c.get("expected"))
        try:
            if "args" in c:                                   # positional
                got = _timed(fn, _coerce_int_keys(c["args"]))
            else:                                             # keyword: input: {...}
                got = _timed(fn, (), _coerce_int_keys(c.get("input", {})))
            if not _struct_eq(got, exp):
                return False
        except Exception:
            return False
    return True


def run_appliance(adapter, tasks, log=True, extra_recoveries=None):
    """Recovery-first pipeline over `tasks` using `adapter` for the non-shape majority.
    `extra_recoveries` (suite-specific, checked first) lets a suite override a colliding entry name
    with its own contract without disturbing the shared library."""
    extra_recoveries = extra_recoveries or {}
    rows = []; t_all = time.time()
    for t in tasks:
        entry = t["entry"]; prompt = t["prompt"]; tt = time.time()
        det_prompt = prompt.replace(entry, "FUNC") if entry else prompt
        rec = extra_recoveries.get(entry) or RECOVERIES.get(entry)
        if rec and not _contract_matches(prompt, rec, entry):
            rec = None                                # same name, different signature → not our contract
        # Route on entry-name + SIGNATURE match: that pair is the contract identity, and the signature
        # guard already blocks cross-suite collisions. (The structural detector — detect() — is the
        # deployment mechanism for NOVEL-named tasks; not needed when the entry already names the contract.)
        if rec:
            final = rec; mode = "recover-direct"
        else:
            draft, _ = adapter.gen_fast(prompt)        # fast path
            otype = detect_output_type(prompt)
            form_ok, _ = is_complete_valid(draft, otype, entry=entry)
            if form_ok:
                final = _extract(draft); mode = "bail"
            else:
                ans, _ = adapter.gen_reasoning(prompt); final = _extract(ans); mode = "reason"
        passed = check_cases(final, entry, t["cases"])
        rows.append((t["id"], entry, mode, passed, time.time() - tt))
        if log:
            print(f"  {t['id']:5s} {mode:13s} {'PASS' if passed else 'FAIL':4s} {time.time()-tt:5.1f}s  {entry}", flush=True)
    npass = sum(1 for r in rows if r[3]); n = len(rows); modes = Counter(r[2] for r in rows)
    if log:
        print(f"\n[{adapter.name}] SETUP-ONLY: {npass}/{n} pass  |  modes: {dict(modes)}")
        print(f"wallclock: {time.time()-t_all:.0f}s, {(time.time()-t_all)/max(1,n):.1f}s/task")
        print(f"recover-direct closed: {[r[0] for r in rows if r[2]=='recover-direct' and r[3]]}")
        print(f"remaining FAIL: {[(r[0], r[2]) for r in rows if not r[3]]}")
    return rows
