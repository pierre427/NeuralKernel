#!/usr/bin/env python3
"""North A/B DEMONSTRATION for kernel/fact_loop.py — the M5-governed verified-fact re-prefill (Mode-A multiplier).

THE CLAIM, on a real model: a verified fact computed by a DETERMINISTIC primitive and delivered AS TOKENS
(re-prefill) lifts the base model on a task it does unreliably — and KPolicy GOVERNS the spend so the multiplier
is THIN by default (paid only when the base attempt fails validation) and is VETOED under host load.

THE TASK — a grounded "count-then-add" slice where the base model is unreliable but the primitive is exact:
  prompt = `In the text "{word}", count how many times the letter '{c}' appears, then add {k} to that count.
            Reply with only the resulting digit (the count plus {k}, mod 10).`
  verified fact = word.count(c)   (DETERMINISTIC — no model)
  ground truth  = (word.count(c) + k) % 10
A MIX is built: half EASY (short word, the letter appears 0-1 times -> base usually right) + half HARD (long word
with the letter repeated 3-6 times -> base usually miscounts). k varies across 0..9. The deterministic count is the
SAME for every arm; only HOW it is used (never / always / governed) differs.

FOUR ARMS over the SAME N tasks (report ACCURACY vs ground truth + AVG model gens):
  1. base          adapter.gen_fast(prompt) only                                  -> 1 gen/task (the FLOOR).
  2. always_ground fact_reprefill(adapter, prompt, facts) ALWAYS                    -> 1 gen/task (the CEILING).
  3. governed_idle governed_fact_loop(policy=KPolicy(k_min=0,k_max=2), cost=0.0)    -> thin on base-pass, spend on
                   base-fail: acc should approach the ceiling at avg_gens strictly in (1,2) (paid only on failures).
  4. governed_busy same but cost=1.0 (host saturated)                              -> cost VETOES the spend: stays
                   THIN (reprefills~0, acc~base, gens~1).

VERDICT (written to logs/fact_loop_north.json): governed_idle acc >> base AND within ~0.15 of always_ground
(multiplier works); governed_idle avg_gens strictly in (1,2) (thin — paid only on base failures); governed_busy
reprefills==0 and acc==base (cost vetoes under load).

  Real North A/B:  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/run_fact_loop_north.py
  CPU smoke (fake): /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/run_fact_loop_north.py --fake --n 40
"""
from __future__ import annotations
import sys, os, re, json, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.fact_loop import Fact, fact_reprefill, governed_fact_loop, augment_prompt
from kernel.k_policy import KPolicy

MAXN = 8                                                   # ~8 new tokens — enough for first_digit extraction
FACTS_HEADER = "Verified facts (computed by a trusted tool"  # the marker render_facts emits (fake keys off it)
EASY_LEN = (4, 6)        # short words: the target letter appears 0-1 times -> base usually right
HARD_LEN = (14, 20)      # long words: the target letter repeated 3-6 times -> base usually miscounts


def first_digit(text: str) -> str:
    """First \\d character in the generated text (the answer is a single resulting digit)."""
    m = re.search(r"\d", text or "")
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------------------------------------------
# Task generation: a MIX of EASY (base usually right) and HARD (base usually miscounts) count-then-add instances.
# ---------------------------------------------------------------------------------------------------------------
_FILLER = "bdfghjklmnpqstvwxyz"   # consonants that are NOT a typical target letter, for non-target padding


def _make_word(rng, length, c, reps):
    """Build a length-`length` lowercase word containing the letter `c` EXACTLY `reps` times.

    `reps` non-overlapping `c` slots are placed at random positions; every other slot is a filler letter that is
    never `c`. So word.count(c) == reps by construction (the deterministic ground-truth count is known exactly)."""
    reps = max(0, min(reps, length))
    positions = set(rng.sample(range(length), reps))
    chars = []
    for i in range(length):
        if i in positions:
            chars.append(c)
        else:
            chars.append(rng.choice([f for f in _FILLER if f != c]))
    return "".join(chars)


def make_tasks(n, seed=0):
    """N tasks: first half EASY, second half HARD; k cycled across 0..9. Each task carries its own ground truth."""
    rng = random.Random(seed)
    tasks = []
    n_easy = n // 2
    for i in range(n):
        easy = i < n_easy
        c = rng.choice("abcdefghijklmnopqrstuvwxyz")
        if easy:
            length = rng.randint(*EASY_LEN)
            reps = rng.randint(0, 1)                       # 0-1 occurrences -> trivial count
        else:
            length = rng.randint(*HARD_LEN)
            reps = rng.randint(3, 6)                       # 3-6 occurrences -> base tends to miscount
        reps = min(reps, length)
        word = _make_word(rng, length, c, reps)
        k = i % 10                                         # vary k across 0..9
        assert word.count(c) == reps
        tasks.append({"word": word, "c": c, "k": k, "easy": easy,
                      "count": reps, "truth": (reps + k) % 10})
    return tasks


def build_prompt(word, c, k):
    return (f'In the text "{word}", count how many times the letter \'{c}\' appears, then add {k} to that count. '
            f'Reply with only the resulting digit (the count plus {k}, mod 10).')


def closures_for(task):
    """Per-task fact_provider + validate closures (verified DETERMINISTIC fact; validate vs ground truth)."""
    word, c, k = task["word"], task["c"], task["k"]
    truth = str((word.count(c) + k) % 10)
    fact_provider = (lambda _p, word=word, c=c: [Fact(f"number of '{c}' in the text", word.count(c))])
    validate = (lambda ans, truth=truth: first_digit(ans) == truth)
    return fact_provider, validate


# ---------------------------------------------------------------------------------------------------------------
# FAKE adapter (NO model) — a deterministic stand-in so the full 4-arm harness runs on CPU as a STRUCTURAL smoke.
# ---------------------------------------------------------------------------------------------------------------
class _FakeAdapter:
    """Deterministic, NO model. Two behaviours, exactly as the demo needs:

    (a) BASE prompt (no verified-facts block): return a WRONG digit for HARD tasks and the RIGHT digit for EASY
        ones -> partial base accuracy (EASY pass, HARD fail), so the arms have something to lift.
    (b) GROUNDED prompt (contains the "Verified facts" block): parse the injected count and the `add {k}` operand
        out of the prompt and return (count + k) % 10 -> grounding lifts accuracy to the ceiling.

    It reconstructs ground truth from the PROMPT TEXT alone (the word in quotes, the target letter, k) — it never
    sees the task dict — so it is a faithful structural stand-in for a model reading the same prompt. n = MAXN.
    """
    name = "fake-count-then-add"

    def __init__(self):
        self.calls = []

    @staticmethod
    def _parse_prompt(prompt):
        """Pull (word, c, k) from the task sentence (present in both base and grounded prompts)."""
        mw = re.search(r'In the text "([^"]*)"', prompt)
        mc = re.search(r"the letter '(.)'", prompt)
        mk = re.search(r"add (\d+) to that count", prompt)
        word = mw.group(1) if mw else ""
        c = mc.group(1) if mc else ""
        k = int(mk.group(1)) if mk else 0
        return word, c, k

    def gen_fast(self, prompt, maxn=MAXN):
        self.calls.append(prompt)
        word, c, k = self._parse_prompt(prompt)
        truth = (word.count(c) + k) % 10

        if FACTS_HEADER in prompt:
            # (b) GROUNDED: trust the injected verified count (parse "... = <n>") instead of self-counting.
            mfact = re.search(r"=\s*(\d+)", prompt.split(FACTS_HEADER, 1)[1])
            count = int(mfact.group(1)) if mfact else word.count(c)
            return str((count + k) % 10), maxn

        # (a) BASE: EASY -> correct (the base model can do short/low-count); HARD -> a deterministic WRONG digit.
        easy = len(word) <= EASY_LEN[1] and word.count(c) <= 1
        if easy:
            return str(truth), maxn
        wrong = (truth + 1) % 10                            # off-by-one: a plausible miscount, always != truth
        return str(wrong), maxn


# ---------------------------------------------------------------------------------------------------------------
# The four arms.
# ---------------------------------------------------------------------------------------------------------------
def arm_base(adapter, tasks):
    hits = gens = 0
    for t in tasks:
        _, validate = closures_for(t)
        text, _ = adapter.gen_fast(build_prompt(t["word"], t["c"], t["k"]), maxn=MAXN)
        hits += int(validate(text)); gens += 1
    return {"accuracy": round(hits / len(tasks), 3), "avg_gens": round(gens / len(tasks), 3),
            "reprefills_fired": len(tasks)}            # always_ground/base reprefill-count is informational only


def arm_always_ground(adapter, tasks):
    hits = gens = 0
    for t in tasks:
        fact_provider, validate = closures_for(t)
        prompt = build_prompt(t["word"], t["c"], t["k"])
        text, _ = fact_reprefill(adapter, prompt, fact_provider(prompt), maxn=MAXN)
        hits += int(validate(text)); gens += 1
    return {"accuracy": round(hits / len(tasks), 3), "avg_gens": round(gens / len(tasks), 3),
            "reprefills_fired": len(tasks)}


def arm_governed(adapter, tasks, *, cost):
    """governed_fact_loop with KPolicy(k_min=0,k_max=2). cost=0.0 -> idle (spend on base-fail); cost=1.0 -> busy
    (cost brakes discretionary widening to k=0 -> stay thin). budget=None (untracked)."""
    policy = KPolicy(k_min=0, k_max=2)
    hits = gens = fired = 0
    for t in tasks:
        fact_provider, validate = closures_for(t)
        prompt = build_prompt(t["word"], t["c"], t["k"])
        res = governed_fact_loop(adapter, prompt, fact_provider=fact_provider, validate=validate,
                                 policy=policy, cost_source=(lambda c=cost: c), budget=None, importance=0.7,
                                 maxn=MAXN)
        # accuracy is measured vs GROUND TRUTH, independent of the loop's own validate verdict.
        hits += int(validate(res.answer))
        gens += res.total_gens
        fired += int(res.used_facts)
    return {"accuracy": round(hits / len(tasks), 3), "avg_gens": round(gens / len(tasks), 3),
            "reprefills_fired": fired}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of tasks (same set across all arms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true",
                    help="use the deterministic FAKE adapter (NO model) for a CPU structural smoke")
    args = ap.parse_args()

    tasks = make_tasks(args.n, seed=args.seed)
    n_easy = sum(1 for t in tasks if t["easy"])
    print(f"[tasks] n={len(tasks)}  easy={n_easy}  hard={len(tasks) - n_easy}  "
          f"(adapter={'FAKE (no model)' if args.fake else 'North'})", flush=True)

    if args.fake:
        adapter = _FakeAdapter()
    else:
        from north_adapter import NorthAdapter
        print("[load] north ...", flush=True)
        adapter = NorthAdapter()

    arms = {}
    print("[run] base ...", flush=True)
    arms["base"] = arm_base(adapter, tasks)
    print("[run] always_ground ...", flush=True)
    arms["always_ground"] = arm_always_ground(adapter, tasks)
    print("[run] governed_idle (cost=0.0) ...", flush=True)
    arms["governed_idle"] = arm_governed(adapter, tasks, cost=0.0)
    print("[run] governed_busy (cost=1.0) ...", flush=True)
    arms["governed_busy"] = arm_governed(adapter, tasks, cost=1.0)

    # ---- table ----
    print("\n===== fact-loop North A/B (count-then-add; verified count delivered AS TOKENS) =====", flush=True)
    print(f"  {'arm':16s} {'accuracy':>9} {'avg_gens':>9} {'reprefills_fired':>17}", flush=True)
    for name in ("base", "always_ground", "governed_idle", "governed_busy"):
        a = arms[name]
        print(f"  {name:16s} {a['accuracy']:9} {a['avg_gens']:9} {a['reprefills_fired']:17d}", flush=True)

    # ---- verdict ----
    base_acc = arms["base"]["accuracy"]
    ceil_acc = arms["always_ground"]["accuracy"]
    gi = arms["governed_idle"]
    gb = arms["governed_busy"]

    multiplier_works = (gi["accuracy"] > base_acc + 0.10) and (gi["accuracy"] >= ceil_acc - 0.15)
    thin_paid_on_fail = 1.0 < gi["avg_gens"] < 2.0
    cost_vetoes = (gb["reprefills_fired"] == 0) and (abs(gb["accuracy"] - base_acc) <= 0.02)
    passed = bool(multiplier_works and thin_paid_on_fail and cost_vetoes)

    verdict = {
        "adapter": "fake" if args.fake else "north",
        "n": len(tasks), "seed": args.seed, "n_easy": n_easy, "n_hard": len(tasks) - n_easy,
        "maxn": MAXN, "arms": arms,
        "checks": {
            "multiplier_works": {
                "passed": bool(multiplier_works),
                "detail": (f"governed_idle acc {gi['accuracy']} vs base {base_acc} (need > base+0.10) and "
                           f"within 0.15 of always_ground {ceil_acc} (need >= {round(ceil_acc - 0.15, 3)})")},
            "thin_paid_only_on_base_failures": {
                "passed": bool(thin_paid_on_fail),
                "detail": f"governed_idle avg_gens {gi['avg_gens']} strictly in (1, 2)"},
            "cost_vetoes_under_load": {
                "passed": bool(cost_vetoes),
                "detail": (f"governed_busy reprefills_fired {gb['reprefills_fired']} (need 0) and acc {gb['accuracy']} "
                           f"~= base {base_acc} (|diff| <= 0.02)")},
        },
        "VERDICT": ("PASS: verified-fact re-prefill is a GOVERNED capability multiplier on North — governed_idle "
                    "approaches the always_ground ceiling at <2 gens/task (paid only on base failures), and host "
                    "load (cost=1.0) vetoes the spend (stays thin, acc~base)."
                    if passed else
                    "FAIL: see checks (multiplier_works / thin_paid_only_on_base_failures / cost_vetoes_under_load)."),
    }

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    outp = os.path.join(logdir, "fact_loop_north.json")
    json.dump(verdict, open(outp, "w"), indent=2, default=str)

    print("\n===== VERDICT =====", flush=True)
    for cname, c in verdict["checks"].items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {cname}: {c['detail']}", flush=True)
    print(f"  VERDICT: {verdict['VERDICT']}", flush=True)
    print(f"  (-> {outp})", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
