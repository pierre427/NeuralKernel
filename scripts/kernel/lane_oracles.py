#!/usr/bin/env python3
"""Lane verification-oracles compiled into runtime validator predicates (reclaim trace-backlog #6).

The fable/mythos trace curriculum distils each behavioral LANE down to one machine-checkable
`verificationOracle` (see reports/trace-curriculum/full-lessons/lessons.jsonl, 23k lessons / 9 lanes).
Training-only, those oracles never gate a live commit. This module turns the CRISP ones into
`(result, task) -> bool` predicates with the exact signature the Scheduler's validator registry expects
(task_graph.Scheduler.register_validator), so a plan node can reference one by name and the kernel will
enforce it — turning the curriculum into always-on guardrails.

Implemented (the crisp, result-checkable subset + the §5 trust invariant):
  evidence_cited       (verification_honesty / general_agent_behavior)
                       -- a completion claim must cite concrete evidence, OR honestly say it's unverified.
  format_contract_ok   (output_discipline)
                       -- the result must validate against the task's declared output_contract.
  no_secret_or_raw_cot (data_pipeline_ops safety + design §5 "block commits that emit secrets / raw CoT")
                       -- the committed artifact must not leak secrets/keys or raw chain-of-thought.

Deferred (not a single-result check): repair_loop (a repair-EXAMPLE shape, a training-data validator),
tool_call_cadence / coordination_planning (action-selection, not a result), frontend/ml/ domain evidence.

Pure-Python: json + re only. No kernel import (so it can register onto any Scheduler/Orchestrator).
"""
from __future__ import annotations
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
def _as_text_and_dict(result: Any) -> tuple[str, dict | None]:
    if isinstance(result, dict):
        return json.dumps(result, default=str), result
    if isinstance(result, str):
        try:
            d = json.loads(result)
        except (ValueError, TypeError):
            d = None
        return result, (d if isinstance(d, dict) else None)
    return json.dumps(result, default=str), None


def _contract(task: Any) -> dict:
    if task is None:
        return {}
    c = getattr(task, "output_contract", None)
    if c is None and isinstance(task, dict):
        c = task.get("output_contract")
    return c or {}


# ---------------------------------------------------------------------------
# verification_honesty: "final claim must cite a concrete command, artifact, screenshot, or explicit
# unverified status." Honest abstention PASSES (design: honest 'insufficient evidence' > confident-wrong).
# ---------------------------------------------------------------------------
_UNVERIFIED = ("insufficient evidence", "unverified", "not verified", "could not verify",
               "cannot verify", "no evidence", "unable to verify")
_EVIDENCE_KEYS = ("evidence", "known_facts", "citations", "proof", "proof_ref", "inferred_links")
# a count must be ANCHORED to a result keyword, so a bare date "1/2" or ratio "9/10 severity" is NOT evidence
_COUNT_RE = re.compile(r"\b\d+\s*/\s*\d+\s+(?:pass|passed|passing|fail|failed|cases?|tests?|checks?|ok)\b", re.I)
_RANCOUNT_RE = re.compile(r"\b\d+\s+(?:passed|failed|cases?|tests?|checks?)\b", re.I)
_FENCE_RE = re.compile(r"```")                                   # a cited fenced code/command block
_SHELL_RE = re.compile(r"(?m)^\s*\$\s+\S")                       # a shell-prompt command line
_INLINE_CMD_RE = re.compile(r"`[^`]*\s[^`]*`")                   # a multi-token inline span (a command, not a bare `9/10` ratio)
_ARTIFACT_RE = re.compile(r"\b[\w./-]+\.(?:py|json|jsonl|txt|md|sh|log|csv|js|ts|tsx|go|rs|c|cpp|h|"
                          r"yaml|yml|toml|cfg|ini|env|pem|pcap|bin|html|xml|sql)\b")


def evidence_cited(result: Any, task: Any = None) -> bool:
    """True if the result cites concrete evidence OR honestly declares itself unverified. Heuristic: a
    denylist of confident-without-evidence shapes (it cannot confirm a cited file actually exists), but a
    bare date/ratio/"proof"/single-word backtick no longer counts — those were trivial honesty-gate bypasses."""
    text, d = _as_text_and_dict(result)
    low = text.lower()
    if any(m in low for m in _UNVERIFIED):                       # honest abstention is a valid outcome
        return True
    if isinstance(d, dict) and any(d.get(k) for k in _EVIDENCE_KEYS):
        return True
    if _COUNT_RE.search(text) or _RANCOUNT_RE.search(text):      # "12/12 passed", "3 tests passed"
        return True
    if _FENCE_RE.search(text) or _SHELL_RE.search(text) or _INLINE_CMD_RE.search(text):
        return True                                              # a cited command / fenced snippet
    if _ARTIFACT_RE.search(text):                                # a cited artifact path
        return True
    return False                                                 # a bare, evidence-free confident claim


# ---------------------------------------------------------------------------
# output_discipline: "output must validate against the requested format contract."
# ---------------------------------------------------------------------------
def format_contract_ok(result: Any, task: Any = None) -> bool:
    c = _contract(task)
    if not isinstance(c, dict) or not c:
        return True                                              # no enforceable contract -> nothing to check
    typ = str(c.get("type") or "").lower()
    req = c.get("required_fields") or c.get("required_keys") or []
    if not isinstance(req, (list, tuple)):
        req = [req]
    inc = c.get("must_include") or []
    if not isinstance(inc, (list, tuple)):
        inc = [inc]
    if req:                                                      # enforce required keys regardless of type
        _, d = _as_text_and_dict(result)
        if d is None:
            return False
        return all(isinstance(k, str) and k in d for k in req)   # a non-str required key is malformed -> fail closed
    if inc:                                                      # enforce required substrings regardless of type
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return all(str(s) in text for s in inc)
    if typ in ("json", "object", "structured"):
        _, d = _as_text_and_dict(result)
        return d is not None                                     # must at least parse as a structured object
    if typ in ("text", "string"):
        return True                                              # text with no required substrings -> nothing specific
    return False                                                 # a non-empty contract we cannot interpret -> fail CLOSED


# ---------------------------------------------------------------------------
# data_pipeline_ops safety + design §5: a commit must not leak secrets/keys or raw chain-of-thought.
# ---------------------------------------------------------------------------
# Distinctive value-shaped secret tokens (low false-positive). Best-effort — NOT a real secret scanner.
_SECRET_RES = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                        # AWS access key id
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"),            # any PEM private key (RSA/EC/PGP/ENCRYPTED/...)
    re.compile(r"\bsk[-_](?:live_|test_)?[A-Za-z0-9]{20,}"),                    # openai / stripe-style secret key
    re.compile(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),     # github classic / fine-grained tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),                             # slack token
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}"),                          # bearer token
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:@/]+:[^\s@/]+@"),                   # credentials embedded in a URL
]
# keyword := "value" — matches BOTH a raw string (password = "x") AND the json-serialized form
# ("password": "x") because the keyword may itself be quoted. The value is captured so placeholders are exempt.
_INLINE_SECRET_RE = re.compile(
    r"""(?ix) ['"]? \b (?: aws_secret_access_key | api[_-]?key | secret | passwd | password
        | access[_-]?token | auth[_-]?token | client[_-]?secret | token ) \b ['"]? \s* [:=] \s* ['"]([^'"]{6,})['"]""")
# a dict KEY whose name denotes a credential (checked against its string value, placeholder-exempt)
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:^|[._-])(?:aws_secret_access_key|api[_-]?key|secret|passwd|password|access[_-]?token"
    r"|auth[_-]?token|client[_-]?secret|private[_-]?key|token)$")
_PLACEHOLDER_RE = re.compile(r"(?i)(redact|placeholder|example|dummy|sample|your[_-]|<[^>]+>|\*{3,}|x{6,}|\.\.\.|changeme|^none$|^null$)")
_COT_MARKERS = ("<think>", "</think>", "<|channel|>analysis", "chain-of-thought:", "raw reasoning:")


def _walk_kv(obj, key=None):
    """Yield (nearest-dict-key, leaf-value) for every leaf in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_kv(v, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_kv(v, key)
    else:
        yield key, obj


def no_secret_or_raw_cot(result: Any, task: Any = None) -> bool:
    """True if the artifact is CLEAN. Best-effort denylist (NOT a substitute for a real secret scanner):
    distinctive token shapes; an inline `keyword: "value"` in raw OR json-serialized form; and a sensitive
    dict KEY carrying a non-placeholder string value; plus raw chain-of-thought markers. Placeholders
    (REDACTED / <your-key> / **** / example) are exempt to avoid blocking docs and templates."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    low = text.lower()
    if any(m in low for m in _COT_MARKERS):
        return False
    if any(p.search(text) for p in _SECRET_RES):
        return False
    for m in _INLINE_SECRET_RE.finditer(text):                  # keyword := "value" (raw or json form)
        if not _PLACEHOLDER_RE.search(m.group(1)):
            return False
    struct = result if isinstance(result, (dict, list)) else None
    if struct is None and isinstance(result, str):
        try:
            j = json.loads(result)
            struct = j if isinstance(j, (dict, list)) else None
        except (ValueError, TypeError):
            struct = None
    if struct is not None:                                      # a credential-named key with a real string value
        for k, v in _walk_kv(struct):
            if k and isinstance(v, str) and len(v) >= 8 and _SENSITIVE_KEY_RE.search(k) and not _PLACEHOLDER_RE.search(v):
                return False
    return True


# ---------------------------------------------------------------------------
# Registry — lane -> predicate, and a helper to install them as Scheduler/Orchestrator validators.
# ---------------------------------------------------------------------------
LANE_ORACLES = {
    "verification_honesty": evidence_cited,
    "output_discipline": format_contract_ok,
    "data_pipeline_ops": no_secret_or_raw_cot,
}

# the names a plan node references in {"fn": ..., "level": ...}
ORACLE_VALIDATORS = {
    "evidence_cited": evidence_cited,
    "format_contract_ok": format_contract_ok,
    "no_secret_or_raw_cot": no_secret_or_raw_cot,
}


def register_lane_oracles(target) -> list[str]:
    """Register every lane-oracle predicate as a validator on a Scheduler or Orchestrator (anything with
    register_validator(name, fn)). Purely additive: a registered validator is only used by a node that
    references it by name, so this never changes existing plans. Returns the names registered."""
    for name, fn in ORACLE_VALIDATORS.items():
        target.register_validator(name, fn)
    return list(ORACLE_VALIDATORS)


__all__ = ["evidence_cited", "format_contract_ok", "no_secret_or_raw_cot",
           "LANE_ORACLES", "ORACLE_VALIDATORS", "register_lane_oracles"]
