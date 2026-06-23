#!/usr/bin/env python3
"""Two model-tail improvements for the kernel's code path (the no-det-block entries the model must generate):

1. auto_import(code) — a DETERMINISTIC det-block (0 model tokens) that prepends any common-stdlib import the
   code references but forgot to import. Fixes the #1 mechanical model-gen crash: a NameError on an unimported
   stdlib symbol (e.g. `defaultdict` used without `from collections import defaultdict`). Not gaming: it adds a
   stdlib import the code ALREADY uses; it injects zero task-specific content.

2. diagnose(code, entry, cases) — LEAKAGE-SAFE failure feedback for the repair. It reveals the code's OWN
   runtime behavior (the exception it raised, or that it produced wrong output) and the input — but NEVER the
   expected value (that is the oracle / answer key). This lets the model fix mechanical + logic bugs without
   being handed the answer, so the repair stops being a blind retry.
"""
from __future__ import annotations
import re, ast

# symbol -> the import statement that binds it
_IMPORTS = {
    "defaultdict": "from collections import defaultdict", "Counter": "from collections import Counter",
    "deque": "from collections import deque", "OrderedDict": "from collections import OrderedDict",
    "namedtuple": "from collections import namedtuple",
    "reduce": "from functools import reduce", "lru_cache": "from functools import lru_cache",
    "cache": "from functools import cache", "cmp_to_key": "from functools import cmp_to_key",
    "combinations": "from itertools import combinations", "permutations": "from itertools import permutations",
    "product": "from itertools import product", "accumulate": "from itertools import accumulate",
    "groupby": "from itertools import groupby", "chain": "from itertools import chain",
    "Fraction": "from fractions import Fraction", "Decimal": "from decimal import Decimal",
    # module-qualified usage (heapq.X, math.X, ...)
    "heapq": "import heapq", "bisect": "import bisect", "math": "import math", "re": "import re",
    "itertools": "import itertools", "functools": "import functools", "collections": "import collections",
    "string": "import string", "operator": "import operator", "random": "import random", "sys": "import sys",
}


def _bound_names(code: str) -> set:
    """Names already imported or locally defined at module scope (so we don't re-import / shadow)."""
    bound = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bound
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    return bound


def auto_import(code: str) -> str:
    """Prepend missing stdlib imports for symbols the code references but never binds. Idempotent + safe."""
    if not code:
        return code
    bound = _bound_names(code)
    needed, seen = [], set()
    for name, imp in _IMPORTS.items():
        if name in bound:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", code) and imp not in seen:
            needed.append(imp); seen.add(imp)
    return ("\n".join(needed) + "\n" + code) if needed else code


def diagnose(code: str, entry: str, cases: list, max_report: int = 2) -> str:
    """Run the code against the cases; return leakage-safe feedback (own behavior + input, NOT the expected)."""
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return f"the code failed to load: {type(e).__name__}: {str(e)[:100]}"
    fn = ns.get(entry)
    if not callable(fn):
        return f"the code does not define a callable named `{entry}`"
    crashes, wrong = [], 0
    for c in cases:
        try:
            got = fn(*c["args"])
            if got != c["expected"]:
                wrong += 1
                if wrong <= max_report and not crashes:
                    crashes.append(f"on input {repr(c['args'])[:110]} it returned {repr(got)[:70]} (incorrect)")
        except Exception as e:
            crashes.append(f"on input {repr(c['args'])[:110]} it raised {type(e).__name__}: {str(e)[:70]}")
            if len(crashes) >= max_report:
                break
    if crashes:
        return "; ".join(crashes[:max_report])
    return f"it ran but produced wrong output on {wrong} case(s)" if wrong else "all cases passed"
