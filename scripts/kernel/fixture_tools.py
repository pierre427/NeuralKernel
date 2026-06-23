#!/usr/bin/env python3
"""Grounding layer: a deterministic, READ-ONLY investigation toolset over a cyber fixture directory, plus
a bounded ReAct loop that lets a model FIND facts instead of confabulating them.

The KERNEL owns the tools (model proposes a query as text, the sandbox executes it deterministically and
resumes the model with the result — the microkernel's "scheduler owns I/O + park/resume" pattern). The
sandbox reads the fixture files ONLY; it has NO access to the task's checks (no answer-key leakage — this
is the same data the full agentic harness exposes). det() runs hold-out-proven det_primitives_cyber fns.
"""
from __future__ import annotations
import os, re, glob, json
try:
    import det_primitives_cyber as cyber
except Exception:
    cyber = None


def _coerce(a: str):
    a = a.strip().strip('"').strip("'")
    try: return int(a)
    except ValueError: pass
    try: return float(a)
    except ValueError: pass
    return a


class FixtureSandbox:
    """Read-only investigation surface over one fixture dir. Bounded output; counts calls."""
    def __init__(self, fixture_dir: str, per_file_cap: int = 400000):
        self.dir = fixture_dir
        self.files = {}
        if fixture_dir and os.path.isdir(fixture_dir):
            for f in sorted(glob.glob(os.path.join(fixture_dir, "*"))):
                if os.path.isfile(f) and os.path.getsize(f) < per_file_cap:
                    try: self.files[os.path.basename(f)] = open(f, errors="ignore").read()
                    except Exception: pass
        self.calls = 0
        self.log = []

    def manifest(self) -> str:
        return ", ".join(f"{k}({len(v)}b)" for k, v in self.files.items()) or "(no fixture files)"

    def grep(self, pattern: str, max_hits: int = 25) -> str:
        self.calls += 1
        try: rx = re.compile(pattern, re.I)
        except re.error: return f"(invalid regex: {pattern!r})"
        out = []
        for name, text in self.files.items():
            for i, ln in enumerate(text.splitlines()):
                if rx.search(ln):
                    out.append(f"{name}:{i}: {ln.strip()[:240]}")
                    if len(out) >= max_hits:
                        return "\n".join(out) + f"\n(+more, showing first {max_hits})"
        return "\n".join(out) if out else "(no matches)"

    def read(self, name: str, offset: int = 0, limit: int = 1800) -> str:
        self.calls += 1
        key = next((k for k in self.files if name.lower() in k.lower()), None)
        if not key:
            return f"(no such file; available: {list(self.files)})"
        return self.files[key][int(offset):int(offset) + int(limit)] or "(empty slice)"

    def det(self, fn: str, *args) -> str:
        self.calls += 1
        f = getattr(cyber, fn, None) if cyber else None
        if not callable(f):
            avail = [n for n in dir(cyber) if not n.startswith("_") and callable(getattr(cyber, n))] if cyber else []
            return f"(no such primitive: {fn}; available: {avail})"
        try:
            return json.dumps(f(*[_coerce(a) for a in args]), default=str)
        except Exception as e:
            return f"(error calling {fn}: {e})"


_CMD = re.compile(r"^\s*(GREP|READ|DET|DONE)\b[:\-]?\s*(.*)$", re.I)


def _dispatch(sb: FixtureSandbox, cmd: str) -> str:
    m = _CMD.match(cmd)
    if not m:
        return "(unparsable command — use: GREP <regex> | READ <file> <offset> <limit> | DET <fn> <args> | DONE <answer>)"
    verb, rest = m.group(1).upper(), m.group(2).strip()
    if verb == "GREP":
        return sb.grep(rest.strip().strip('"').strip("'") or ".")
    if verb == "READ":
        parts = rest.split()
        name = parts[0] if parts else ""
        off = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
        lim = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1800
        return sb.read(name, off, lim)
    if verb == "DET":
        parts = rest.split()
        return sb.det(parts[0], *parts[1:]) if parts else "(DET needs a primitive name)"
    return "DONE"


def recon(sb: FixtureSandbox, det_facts: dict = None) -> str:
    """Deterministic kernel-driven reconnaissance: present what's actually in the evidence so a weak model
    can DRILL DOWN from real facts instead of cold-guessing search syntax. No model calls."""
    from collections import Counter
    parts = [f"FILES: {sb.manifest()}"]
    alltext = "\n".join(sb.files.values())
    ips = Counter(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", alltext))
    if ips:
        parts.append("IP frequency (host=hits, most-contacted first): " + ", ".join(f"{ip}={n}" for ip, n in ips.most_common(12)))
    doms = Counter(re.findall(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", alltext.lower()))
    doms = [(d, n) for d, n in doms.items() if not re.fullmatch(r"\d+(?:\.\d+)+", d)]
    if doms:
        parts.append("domains seen: " + ", ".join(f"{d}={n}" for d, n in sorted(doms, key=lambda x: -x[1])[:10]))
    eids = Counter(re.findall(r"\b4\d{3}\b", alltext))
    if eids:
        parts.append("4xxx event ids: " + ", ".join(f"{e}={n}" for e, n in eids.most_common(8)))
    for name, text in sb.files.items():
        parts.append(f"--- {name} (head) ---\n{text[:480]}")
    if det_facts:
        parts.append("VERIFIED DETERMINISTIC FACTS (trust exactly): " + json.dumps(det_facts, default=str)[:700])
    return "\n".join(parts)


INVESTIGATE_PREAMBLE = (
    "You are a forensic investigator with a READ-ONLY toolset over the evidence. You have INITIAL RECON "
    "below (file heads, IP/domain/event frequencies, verified facts). DRILL DOWN to nail the answer — do "
    "not guess and do not invent file contents. Issue ONE command at a time, choosing real tokens from the "
    "recon:\n"
    "  GREP <regex>            e.g. GREP 10.10.3.4   (all lines mentioning that host)\n"
    "  READ <file> <off> <lim> e.g. READ conn.log 0 2000   (see the full table past the head)\n"
    "  DET <fn> <args>         e.g. DET classify_win_event 4624 | DET is_ssrf_target 169.254.169.254\n"
    "  DONE <answer>           when grounded (include exact IPs/IDs/usernames), or 'insufficient evidence'\n"
    "Output ONLY the single next command line.")


def investigate(model, sandbox: FixtureSandbox, question: str, budget: int = 6, det_facts: dict = None) -> tuple:
    """Bounded ReAct loop, SEEDED with deterministic recon: the model drills down from real presented
    evidence. Returns (grounded_answer, transcript). Honest by construction — unfound facts -> 'insufficient
    evidence' rather than confabulation."""
    seed = recon(sandbox, det_facts)
    transcript = [f"INITIAL RECON:\n{seed}"]
    for _ in range(budget):
        prompt = (f"{INVESTIGATE_PREAMBLE}\n\nQUESTION: {question}\n\n" + "\n\n".join(transcript) + "\n\nNext command:")
        raw, _ = model.gen_fast(prompt, maxn=120)
        cmd = next((l for l in (raw or "").splitlines() if l.strip()), "DONE")
        m = _CMD.match(cmd)
        if m and m.group(1).upper() == "DONE":
            ans = m.group(2).strip()
            if ans and "insufficient" not in ans.lower():
                return ans, transcript[1:]
            break
        transcript.append(f"> {cmd.strip()}\n{_dispatch(sandbox, cmd)[:600]}")
    # force a grounded answer from everything gathered (recon + drill-downs)
    ans, _ = model.gen_fast(
        f"QUESTION: {question}\n\nALL EVIDENCE GATHERED (cite exact tokens from here only):\n" + "\n\n".join(transcript)
        + "\n\nAnswer using ONLY this evidence; include the exact IPs/IDs/usernames. If genuinely not "
          "present, say exactly 'insufficient evidence'.", maxn=300)
    return ans, transcript[1:]
