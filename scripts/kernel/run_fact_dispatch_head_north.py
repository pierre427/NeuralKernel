#!/usr/bin/env python3
"""LEARNED-head routing vs lexical routing on real North — the head proposes which verified fact from the weights.

Proves the upgrade kernel/fact_dispatch_head.py makes to the dispatch ROUTER: the trained syscall head reads
North's L24 residual at the prompt-end decision point and proposes a primitive class, which maps onto the SAME
FactSpec the lexical default_router() builds. The decisive contrast is the KEYWORD-FREE PARAPHRASE set: prompts
that NEED count_byte / c_gcd / is_prime but DON'T say "how many letter" / "gcd" / "prime", so the lexical matcher
misses while the head — trained on oblique phrasings — should still route. Plain chat prompts must route to none
(no false trap) under both routers.

For each prompt we print: HEAD route vs LEXICAL route vs EXPECTED. We compute head_routing_accuracy and
lexical_routing_accuracy over the set and write logs/fact_dispatch_head_north.json with a VERDICT:
  PASS iff head_acc >= lexical_acc AND the head routes >= 1 paraphrase the lexical matcher missed.

If /tmp/north_syscall_head.safetensors does not exist yet, prints a clear message and exits 2 (does NOT crash).

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/run_fact_dispatch_head_north.py
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEAD_PATH = "/tmp/north_syscall_head.safetensors"

# (label, prompt, expected_spec_name_or_None, is_paraphrase)
# expected is the FactSpec name the verified-fact loop should run (None == no trap / pass-through).
# is_paraphrase marks the keyword-free prompts where the LEXICAL matcher is expected to MISS but the head shouldn't.
PROMPTS = [
    # --- keyword-free PARAPHRASES (lexical matcher misses; head should route) ---
    ("para-count", 'tally each appearance of the third letter of the alphabet in "saccharic"', "letter_count", True),
    ("para-gcd", "what is the largest number dividing both 1071 and 462 evenly", "gcd", True),
    ("para-prime", "does 1000003 have any divisors other than 1 and itself", "is_prime", True),
    # --- keyword-explicit controls (both routers should route) ---
    ("kw-count", "How many times does the letter r occur in the word strawberry?", "letter_count", False),
    ("kw-gcd", "What is the greatest common divisor of 1071 and 462?", "gcd", False),
    ("kw-prime", "Is 1000003 a prime number? Answer precisely.", "is_prime", False),
    # --- plain chat (both routers -> none / pass-through) ---
    ("chat-1", "Write a two-line haiku about autumn leaves.", None, False),
    ("chat-2", "Give me three friendly tips for staying calm before a talk.", None, False),
]


def main():
    # fail-soft if the head checkpoint isn't trained yet (the user trains it concurrently).
    if not os.path.exists(HEAD_PATH):
        print(f"head checkpoint not found: {HEAD_PATH} — train north_syscall_head.py --layer 24 first", flush=True)
        sys.exit(2)

    from mlx_lm.utils import load
    from north_syscall_head import MODEL
    from kernel.fact_dispatch import default_router
    from kernel.fact_dispatch_head import HeadFactRouter

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()

    lexical = default_router()
    # the head proposes from the SAME spec pool (share the lexical router instance) and falls back to it when the
    # head proposes no fact — so head routing can only ADD to, never subtract from, the lexical baseline.
    head_router = HeadFactRouter(model=model, tok=tok, head_path=HEAD_PATH,
                                 lexical_fallback=lexical, spec_source=lexical)
    # a PURE head router (no fallback) too, to expose what the head alone proposes (diagnostic, not scored).
    head_only = HeadFactRouter(model=model, tok=tok, head_path=HEAD_PATH, spec_source=lexical)

    print("\n===== LEARNED-head routing vs lexical routing on North =====", flush=True)
    print(f"  {'label':12} {'expected':13} {'head':13} {'head(no-fb)':13} {'lexical':13}  paraphrase", flush=True)
    rows = []
    head_correct = lex_correct = 0
    paraphrases_head_won = []   # paraphrases the head routed correctly that lexical missed
    for label, prompt, expected, is_para in PROMPTS:
        h = head_router.route(prompt)
        ho = head_only.route(prompt)
        lx = lexical.route(prompt)
        h_name = getattr(h, "name", None)
        ho_name = getattr(ho, "name", None)
        lx_name = getattr(lx, "name", None)
        h_ok = (h_name == expected)
        lx_ok = (lx_name == expected)
        head_correct += int(h_ok)
        lex_correct += int(lx_ok)
        # the win condition: a paraphrase the head routes to the expected fact but lexical does NOT route at all.
        head_won_para = bool(is_para and h_name == expected and expected is not None and lx_name != expected)
        if head_won_para:
            paraphrases_head_won.append(label)
        rows.append({"label": label, "prompt": prompt, "expected": expected,
                     "head": h_name, "head_no_fallback": ho_name, "lexical": lx_name,
                     "is_paraphrase": is_para, "head_ok": h_ok, "lexical_ok": lx_ok,
                     "head_won_paraphrase": head_won_para})
        print(f"  {label:12} {str(expected):13} {str(h_name):13} {str(ho_name):13} {str(lx_name):13}  "
              f"{'yes' if is_para else '-'}"
              f"{'   <- head won' if head_won_para else ''}", flush=True)

    n = len(PROMPTS)
    head_acc = head_correct / n
    lex_acc = lex_correct / n
    head_ge_lexical = head_acc >= lex_acc
    routes_missed_paraphrase = len(paraphrases_head_won) >= 1
    verdict_pass = head_ge_lexical and routes_missed_paraphrase

    out = {
        "n_prompts": n,
        "head_routing_accuracy": round(head_acc, 4),
        "lexical_routing_accuracy": round(lex_acc, 4),
        "head_ge_lexical": head_ge_lexical,
        "paraphrases_head_routed_that_lexical_missed": paraphrases_head_won,
        "head_routes_ge1_missed_paraphrase": routes_missed_paraphrase,
        "rows": rows,
        "VERDICT": ("PASS: the learned head routes the fact dispatch at least as accurately as the lexical "
                    "matcher AND recovers >=1 keyword-free paraphrase the lexical matcher missed — the trained "
                    "in-weights decision proposes the right verified fact where surface keywords fail."
                    if verdict_pass else
                    "PARTIAL: head did not both match-or-beat lexical accuracy and recover a missed paraphrase — "
                    "see head_ge_lexical / paraphrases_head_routed_that_lexical_missed and the rows."),
    }

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "fact_dispatch_head_north.json")
    json.dump(out, open(logpath, "w"), indent=2, default=str)

    print("\n===== verdict =====", flush=True)
    print(f"  head_routing_accuracy    : {head_acc*100:.1f}%  ({head_correct}/{n})", flush=True)
    print(f"  lexical_routing_accuracy : {lex_acc*100:.1f}%  ({lex_correct}/{n})", flush=True)
    print(f"  head >= lexical          : {head_ge_lexical}", flush=True)
    print(f"  paraphrases head won     : {paraphrases_head_won}", flush=True)
    print(f"VERDICT: {out['VERDICT']}", flush=True)
    print(f"(-> {os.path.relpath(logpath)})", flush=True)


if __name__ == "__main__":
    main()
