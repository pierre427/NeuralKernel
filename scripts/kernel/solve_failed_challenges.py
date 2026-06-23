#!/usr/bin/env python3
"""Ask the SAME model (North) to legitimately solve the coding tasks it failed (w47 to_snake_case, c156
resp_parser) — WITHOUT cheating / memorizing the visible cases. Uses the self-extending pipeline: a TRUSTED,
independently-coded oracle + a novel-instance generator (the source of truth I supply) gate North's candidate on
THOUSANDS of fresh inputs. A solution that hardcodes the 3-5 visible cases fails the hold-out gate by construction;
only a genuinely general algorithm passes. North gets the repair loop (feedback = a held-out instance it got
wrong, never the visible answers), so it iterates toward the general solution.

Run: python kernel/solve_failed_challenges.py [--task w47|c156|both] [--attempts 6] [--n 2000]
"""
import sys, os, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.self_extend import Gap, SelfExtender
from kernel.task_graph import Scheduler
from kernel.code_fixups import auto_import


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text or "", re.I)
    if m:
        return m.group(1)
    i = (text or "").find("def ")
    return text[i:] if i >= 0 else (text or "")


# ── w47: to_snake_case ── trusted oracle = canonical 2-regex camel->snake (handles acronyms: HTMLParser->html_parser)
W47_ORACLE = r'''import re
def oracle(s):
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()
'''
W47_GEN = r'''import random
_W = ['get','set','element','by','id','parse','html','xml','http','request','to','from','value','node','list',
      'map','key','data','user','name','count','max','min','sum','index','buffer','stream','config','url','json']
def generate(rng):
    parts = [rng.choice(_W) for _ in range(rng.randint(1, 4))]
    out = ''
    for i, w in enumerate(parts):
        r = rng.random()
        if r < 0.25: out += w.upper()                 # acronym, e.g. HTML / ID  (the hard case)
        elif i == 0 and r < 0.6: out += w             # camelCase first word lowercase
        else: out += w.capitalize()                   # PascalCase / subsequent words
    return (out or 'x',)
'''

# ── c156: resp_parser ── trusted oracle = independent recursive RESP parser; generator = a trusted RESP encoder
C156_ORACLE = r'''def oracle(data):
    pos = [0]
    def line():
        i = data.index('\r\n', pos[0]); s = data[pos[0]:i]; pos[0] = i + 2; return s
    def parse():
        t = data[pos[0]]; pos[0] += 1; ln = line()
        if t == '+': return ln
        if t == '-': return ln
        if t == ':': return int(ln)
        if t == '$':
            n = int(ln)
            if n == -1: return None
            s = data[pos[0]:pos[0] + n]; pos[0] += n + 2; return s
        if t == '*':
            n = int(ln)
            if n == -1: return None
            return [parse() for _ in range(n)]
        raise ValueError('bad RESP %r' % t)
    return parse()
'''
C156_GEN = r'''import random
def _enc(rng, depth):
    opts = ['str', 'int', 'bulk', 'null'] + (['arr'] if depth < 2 else [])
    t = rng.choice(opts)
    if t == 'str':  return '+' + ''.join(rng.choice('OKabcPONGxyz') for _ in range(rng.randint(1, 6))) + '\r\n'
    if t == 'int':  return ':' + str(rng.randint(-1000, 1000)) + '\r\n'
    if t == 'bulk':
        s = ''.join(rng.choice('helloworld0123') for _ in range(rng.randint(0, 10)))
        return '$%d\r\n%s\r\n' % (len(s), s)
    if t == 'null': return '$-1\r\n'
    n = rng.randint(0, 3)                              # array
    return '*%d\r\n%s' % (n, ''.join(_enc(rng, depth + 1) for _ in range(n)))
def generate(rng):
    return (_enc(rng, 0),)
'''

TASKS = {
    "w47": dict(entry="to_snake_case",
                prompt=("Write a Python function `to_snake_case(s: str) -> str` that converts camelCase or "
                        "PascalCase to snake_case. 'getElementById' -> 'get_element_by_id'. Acronyms collapse: "
                        "'HTMLParser' -> 'html_parser'. Return ONLY the function definition in a ```python block."),
                sample_cases=[{"args": ["getElementById"], "expected": "get_element_by_id"},
                              {"args": ["HTMLParser"], "expected": "html_parser"},
                              {"args": ["simple"], "expected": "simple"}],
                generate_src=W47_GEN, oracle_src=W47_ORACLE),
    "c156": dict(entry="resp_parser",
                 prompt=("Write a Python function `resp_parser(data: str) -> object` that parses Redis RESP: "
                         "'+' simple strings, '-' errors (return the text), ':' integers, '$' bulk strings "
                         "($-1 -> None), '*' arrays (*-1 -> None). Return the parsed Python value. Return ONLY "
                         "the function definition in a ```python block."),
                 sample_cases=[{"args": ["+OK\r\n"], "expected": "OK"}, {"args": [":42\r\n"], "expected": 42},
                               {"args": ["$5\r\nhello\r\n"], "expected": "hello"},
                               {"args": ["*3\r\n:1\r\n:2\r\n:3\r\n"], "expected": [1, 2, 3]},
                               {"args": ["$-1\r\n"], "expected": None}],
                 generate_src=C156_GEN, oracle_src=C156_ORACLE),
}


def _verify_oracle(spec):
    """Sanity: the TRUSTED oracle must itself pass the visible cases + the generator must produce varied inputs."""
    ns = {}
    exec(spec["oracle_src"], ns); exec(spec["generate_src"], ns)
    oracle, generate = ns["oracle"], ns["generate"]
    for c in spec["sample_cases"]:
        got = oracle(*c["args"])
        assert got == c["expected"], f"ORACLE WRONG on {c['args']}: got {got!r} != {c['expected']!r}"
    import random
    rng = random.Random(1)
    seen = {repr(generate(rng)) for _ in range(200)}
    assert len(seen) > 50, f"generator not diverse: {len(seen)} distinct"
    return len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["w47", "c156", "both"])
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    todo = ["w47", "c156"] if args.task == "both" else [args.task]

    # verify oracles BEFORE loading the model (fail fast if my reference is wrong)
    for tid in todo:
        d = _verify_oracle(TASKS[tid])
        print(f"[oracle-check] {tid}: trusted oracle passes all visible cases; generator {d}+ distinct/200", flush=True)

    from north_adapter import NorthAdapter
    print("[load] north ...", flush=True)
    model = NorthAdapter()

    def synth(gap, feedback):
        p = gap.description
        if feedback:
            if feedback.get("fails"):                         # a VISIBLE example failed
                f = feedback["fails"][0]
                args = gap.sample_cases[f["case"]]["args"] if f.get("case") is not None else "?"
                hint = (f"Your function is WRONG on this example: input {args!r} -> your output {f.get('got')!r}, "
                        f"but the correct answer is {f.get('expected')!r}. Fix it, and handle EVERY case "
                        f"consistently — a top-level value must decode exactly like a nested one (same rules, no "
                        f"leftover delimiters).")
            else:                                             # a HELD-OUT (novel, unseen) instance failed
                hint = (f"Your attempt was INCORRECT on a held-out input you were NOT shown: {feedback.get('first_fail')}. "
                        f"The examples are insufficient — write a GENERAL algorithm correct for ALL inputs of this kind.")
            p = gap.description + "\n\n" + hint + " Return ONLY the function."
        out, _ = model.gen_fast(p, maxn=1600)
        return auto_import(_extract_code(out))

    for tid in todo:
        spec = TASKS[tid]
        gap = Gap(entry=spec["entry"], description=spec["prompt"], sample_cases=spec["sample_cases"],
                  generate_src=spec["generate_src"], oracle_src=spec["oracle_src"], n=args.n, seed=0)
        print(f"\n===== solving {tid} ({spec['entry']}) — North + hold-out gate ({args.n} novel/seed) =====", flush=True)
        ext = SelfExtender(Scheduler(), synthesize=synth, max_attempts=args.attempts)
        tr = ext.extend(gap)
        if tr.get("registered"):
            ga = next(a for a in tr["attempts"] if a["result"] == "registered")
            print(f"  ✅ SOLVED on attempt {ga['attempt']}/{args.attempts}: passed hold-out gate {ga['gate']['passed']}/{ga['gate']['n']} "
                  f"+ 2nd-seed re-gate {ga.get('regate', {}).get('passed')} (GENERAL, not memorized)", flush=True)
            print("  --- North's general solution ---\n" + "\n".join("    " + l for l in tr["code"].strip().splitlines()), flush=True)
        else:
            print(f"  ❌ NOT solved in {args.attempts} attempts (reason: {tr.get('reason')})", flush=True)
            for a in tr["attempts"]:
                stg = a.get("gate") or a.get("sample")
                print(f"     attempt {a['attempt']}: {a['result']}  {stg}", flush=True)


if __name__ == "__main__":
    main()
