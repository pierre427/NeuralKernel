#!/usr/bin/env python3
"""Output-contract validators — the bailout's safety gate, across output TYPES (not just Python).

The early bailout (return as soon as a valid answer appears, skipping reasoning) is only SOUND if the
validator actually confirms the answer. So we validate by the task's expected output contract:
  • python  — extracts the code block, parses (AST), defines the entry function
  • json    — parses; checks required keys / array-ness when the prompt states them
  • xml     — parses (well-formed) via ElementTree
  • text    — "must include X" / regex contracts
Stronger contracts → safer bailout. Shape tasks additionally validate by differential vs the trusted
recovery (output_validators handles the FORM; the shape layer handles SEMANTICS)."""
from __future__ import annotations
import ast, json, re
import xml.etree.ElementTree as ET


def detect_output_type(prompt):
    p = (prompt or "").lower()
    if re.search(r"\bxml\b|well-?formed xml|<\w+>.*</\w+>", p):
        return "xml"
    if re.search(r"\bjson\b|json object|valid json", p):
        return "json"
    if "```python" in (prompt or "") or re.search(r"\bfunction\b|def \w|return only the function", p):
        return "python"
    return "text"


def _extract(output, lang=""):
    m = re.search(r"```(?:%s)?\s*\n(.*?)```" % lang, output, re.S)
    return m.group(1).strip() if m else output.strip()


def validate_python(output, entry=None):
    code = _extract(output, "python")
    if "def " not in code:
        return False, "no function definition"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    if entry and re.search(r"def\s+" + re.escape(entry) + r"\b", code) is None:
        return False, f"missing entry function {entry}"
    return True, "ok"


def validate_json(output, required_keys=None):
    text = _extract(output, "json")
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"invalid json: {e}"
    if required_keys:
        if not isinstance(obj, dict):
            return False, "json is not an object"
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return False, f"missing keys {missing}"
    return True, "ok"


def validate_xml(output):
    text = _extract(output, "xml")
    m = re.search(r"<[\s\S]*>", text)
    if m:
        text = m.group(0)
    try:
        ET.fromstring(text)
    except Exception as e:
        return False, f"malformed xml: {e}"
    return True, "ok"


def validate_includes(output, must_include):
    missing = [s for s in (must_include or []) if s not in output]
    return (not missing), ("ok" if not missing else f"missing required: {missing}")


def is_complete_valid(output, otype=None, prompt=None, entry=None, required_keys=None, must_include=None):
    """Form-level validity for the bailout. Returns (ok, reason). otype inferred from prompt if None."""
    otype = otype or detect_output_type(prompt or output)
    if otype == "python":
        return validate_python(output, entry)
    if otype == "json":
        return validate_json(output, required_keys)
    if otype == "xml":
        return validate_xml(output)
    return validate_includes(output, must_include)


if __name__ == "__main__":
    cases = [
        ("python ok", "```python\ndef f(x):\n    return x+1\n```", dict(otype="python", entry="f"), True),
        ("python syntax", "```python\ndef f(x):\n    return x+\n```", dict(otype="python"), False),
        ("python missing entry", "```python\ndef g(x):\n    return x\n```", dict(otype="python", entry="f"), False),
        ("json ok", '```json\n{"a": 1, "b": [2,3]}\n```', dict(otype="json", required_keys=["a", "b"]), True),
        ("json bad", '{"a": 1,}', dict(otype="json"), False),
        ("json missing key", '{"a": 1}', dict(otype="json", required_keys=["a", "b"]), False),
        ("xml ok", "<root><item id='1'>x</item></root>", dict(otype="xml"), True),
        ("xml malformed", "<root><item></root>", dict(otype="xml"), False),
        ("text includes", "the answer is FORTY-TWO and done", dict(otype="text", must_include=["FORTY-TWO"]), True),
        ("text missing", "the answer is unknown", dict(otype="text", must_include=["FORTY-TWO"]), False),
    ]
    ok = 0
    for name, out, kw, exp in cases:
        got, reason = is_complete_valid(out, **kw)
        flag = "OK" if got == exp else "FAIL"
        ok += got == exp
        print(f"  {flag:4s} {name:22s} -> {got} ({reason})")
    # detection
    assert detect_output_type("Return a JSON object with keys") == "json"
    assert detect_output_type("emit well-formed XML") == "xml"
    assert detect_output_type("Write a Python function `f`") == "python"
    print(f"\noutput_validators self-test: {ok}/{len(cases)} contract checks + type detection OK")
