#!/usr/bin/env python3
"""The deterministic INSTRUCTION SET for the LLM-as-JIT — primitives modeled on the most
battle-tested deterministic primitive sets in computing: the i86 ISA (and its extensions), the
C standard library, and the Python standard library. (P. Lamy.)

Rationale: the operations LLMs do worst fuzzily — counting characters, substring search, bit
manipulation, exact arithmetic, sorting, parsing — are exactly the ones hardware gives *dedicated
instructions* for (POPCNT, SSE4.2 string ops, BMI) and stdlibs give *proven functions* for
(strlen/strstr/qsort, str.count/sorted/Counter). They must be EXACT, so they are the right
deterministic primitives to hand the model. The model composes glue + control flow from these
(its strength) instead of reinventing them (its weakness — the capability gaps).

Three tiers, lowest to highest. Each primitive names the hardware instruction / C function /
Python it mirrors. All are pure, deterministic, side-effect-free.
"""
from __future__ import annotations
import math, json, bisect as _bisect, zlib
from collections import Counter as _Counter

# ============================================================ TIER 0 — i86 ISA (bit/string/exact)
def popcount(x: int) -> int:
    """POPCNT — count set bits."""
    return bin(x & ((1 << max(x.bit_length(), 1)) - 1)).count("1") if x >= 0 else bin(x & 0xFFFFFFFFFFFFFFFF).count("1")

def count_trailing_zeros(x: int) -> int:
    """TZCNT (BMI1) — count trailing zero bits."""
    return (x & -x).bit_length() - 1 if x else 0

def count_leading_zeros(x: int, width: int = 64) -> int:
    """LZCNT (BMI1) — count leading zero bits in a `width`-bit word."""
    return width - x.bit_length() if x else width

def bit_scan_forward(x: int) -> int:
    """BSF — index of lowest set bit (-1 if none)."""
    return (x & -x).bit_length() - 1 if x else -1

def parity(x: int) -> int:
    """PF — 1 if an odd number of set bits, else 0."""
    return popcount(x) & 1

def rotl(x: int, n: int, width: int = 32) -> int:
    """ROL — rotate-left a `width`-bit word."""
    n %= width; m = (1 << width) - 1; x &= m
    return ((x << n) | (x >> (width - n))) & m

def rotr(x: int, n: int, width: int = 32) -> int:
    """ROR — rotate-right a `width`-bit word."""
    return rotl(x, width - (n % width), width)

def byte_swap(x: int, width: int = 32) -> int:
    """BSWAP — reverse byte order of a `width`-bit word."""
    b = x.to_bytes(width // 8, "big"); return int.from_bytes(b[::-1], "big")

def crc32(data: bytes) -> int:
    """CRC32 (SSE4.2) — cyclic redundancy checksum."""
    return zlib.crc32(data) & 0xFFFFFFFF

def count_byte(data, needle) -> int:
    """REPNE SCAS / memchr-count — count occurrences of `needle` in `data`. THE counter that fixes
    'how many r's in strawberry' (== 3) where fuzzy counting fails."""
    return list(data).count(needle)

def string_index(haystack: str, needle: str) -> int:
    """PCMPESTRI (SSE4.2) / strstr — index of first occurrence of `needle`, or -1."""
    return haystack.find(needle)

# ============================================================ TIER 1 — C standard library
def c_strlen(s: str) -> int: return len(s)                      # string.h
def c_strchr(s: str, c: str) -> int: return s.find(c)           # string.h (index or -1)
def c_strcmp(a: str, b: str) -> int: return (a > b) - (a < b)   # string.h (-1/0/1)
def c_isdigit(c: str) -> bool: return len(c) == 1 and c.isdigit()   # ctype.h
def c_isalpha(c: str) -> bool: return len(c) == 1 and c.isalpha()   # ctype.h
def c_isalnum(c: str) -> bool: return len(c) == 1 and c.isalnum()   # ctype.h
def c_isspace(c: str) -> bool: return len(c) == 1 and c.isspace()   # ctype.h
def c_toupper(c: str) -> str: return c.upper()                  # ctype.h
def c_tolower(c: str) -> str: return c.lower()                  # ctype.h
def c_atoi(s: str) -> int:                                      # stdlib.h
    s = s.strip(); m = "-" if s[:1] == "-" else ""; d = "".join(ch for ch in s.lstrip("+-") if ch.isdigit())
    return int(m + d) if d else 0
def c_qsort(xs, key=None) -> list: return sorted(xs, key=key)   # stdlib.h
def c_bsearch(sorted_xs, x) -> int:                             # stdlib.h (index or -1)
    i = _bisect.bisect_left(sorted_xs, x)
    return i if i < len(sorted_xs) and sorted_xs[i] == x else -1
def c_abs(x): return abs(x)                                     # stdlib.h
def c_pow(b, e): return math.pow(b, e)                          # math.h
def c_sqrt(x): return math.isqrt(x) if isinstance(x, int) and x >= 0 else math.sqrt(x)  # math.h
def c_floor(x): return math.floor(x)                            # math.h
def c_gcd(a, b): return math.gcd(a, b)                          # C++/numerics

# ============================================================ TIER 2 — Python standard library
def py_counter(xs) -> dict: return dict(_Counter(xs))           # collections.Counter
def py_most_common(xs):                                         # Counter.most_common
    return [k for k, _ in _Counter(xs).most_common()]
def py_factorial(n: int) -> int: return math.factorial(n)       # math.factorial
def py_comb(n: int, k: int) -> int: return math.comb(n, k)      # math.comb
def py_json_parse(s: str): return json.loads(s)                 # json.loads
def py_json_dump(o) -> str: return json.dumps(o)                # json.dumps
def py_bisect_insort(xs: list, x) -> None: _bisect.insort(xs, x)  # bisect.insort

# ---- regex composition primitives (so the model writes only the outer loop; closes l80) ----
def match_char_class(spec: str, c: str) -> bool:
    """Regex char-class body match: 'abc', 'a-z', '^abc' (negated)."""
    neg = spec.startswith("^"); body = spec[1:] if neg else spec
    hit = False; i = 0
    while i < len(body):
        if i + 2 < len(body) and body[i + 1] == "-":
            if body[i] <= c <= body[i + 2]: hit = True
            i += 3
        else:
            if body[i] == c: hit = True
            i += 1
    return (not hit) if neg else hit

def match_quantified(token_fn, quant: str, text: str, pos: int) -> list:
    """The backtracking primitive: positions reachable after matching `token_fn` (a char->bool) with
    `quant` ('' / '*' / '+' / '?') starting at `pos`. Longest-first for greedy backtracking. With
    this + match_char_class, a regex matcher is just an outer token loop."""
    if quant == "":
        return [pos + 1] if pos < len(text) and token_fn(text[pos]) else []
    if quant == "?":
        return ([pos + 1] if pos < len(text) and token_fn(text[pos]) else []) + [pos]
    reach = []; p = pos
    while p < len(text) and token_fn(text[p]):
        p += 1; reach.append(p)
    return reach[::-1] + ([pos] if quant == "*" else [])


# the reference "manual" handed to the model — its deterministic instruction set
MANUAL = """Trusted deterministic PRIMITIVES already imported and in scope (USE these; do NOT
reimplement them). Modeled on the i86 ISA, C stdlib, and Python stdlib.
  # bit/exact (i86 ISA): popcount, count_trailing_zeros, count_leading_zeros, bit_scan_forward,
  #   parity, rotl, rotr, byte_swap, crc32
  count_byte(data, needle) -> int        # count occurrences (REPNE SCAS) — e.g. count_byte('strawberry','r')==3
  string_index(haystack, needle) -> int  # substring index or -1 (strstr / SSE4.2)
  # C stdlib: c_strlen, c_strchr, c_strcmp, c_isdigit, c_isalpha, c_isalnum, c_isspace,
  #   c_toupper, c_tolower, c_atoi, c_qsort(xs,key=None), c_bsearch(sorted,x), c_abs,
  #   c_pow, c_sqrt, c_floor, c_gcd
  # Python stdlib: py_counter, py_most_common, py_factorial, py_comb, py_json_parse, py_json_dump, py_bisect_insort
  match_char_class(spec, c) -> bool      # regex class body: 'abc','a-z','^abc'
  match_quantified(token_fn, quant, text, pos) -> list  # backtracking: positions after a quantified token
"""

# the source text to prepend before exec'ing a model's composition (so the primitives are in scope)
def primitive_src() -> str:
    import inspect, sys
    mod = sys.modules[__name__]
    parts = ["import math, json, bisect as _bisect, zlib", "from collections import Counter as _Counter"]
    for name, obj in vars(mod).items():
        if callable(obj) and getattr(obj, "__module__", None) == __name__ and name not in ("primitive_src",):
            try:
                parts.append(inspect.getsource(obj))
            except Exception:
                pass
    return "\n".join(parts)


if __name__ == "__main__":
    # self-test the primitives + the strawberry case
    assert count_byte("strawberry", "r") == 3 and count_byte("bookkeeper", "e") == 3
    assert popcount(0b1011) == 3 and count_trailing_zeros(0b1000) == 3
    assert c_atoi("  -42abc") == -42 and c_bsearch([1, 3, 5, 7], 5) == 2
    assert match_char_class("a-z", "m") and not match_char_class("^abc", "a")
    assert match_quantified(lambda c: c == "a", "*", "aaab", 0) == [3, 2, 1, 0]
    print("det_primitives self-test OK — strawberry count_byte('strawberry','r') =", count_byte("strawberry", "r"))
    print("primitive_src() length:", len(primitive_src()), "chars")
