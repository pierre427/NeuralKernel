#!/usr/bin/env python3
"""Shape detector — routes a task to a primitive by STRUCTURE (description + signature), NOT by name.

The whole point (P. Lamy): a novel task of a known shape must route correctly even if its function is
named differently. So detection keys on the problem's *content* — the vocabulary of the shape and the
signature type-structure — and abstains when nothing scores confidently. Each shape carries its
hold-out-proven primitive and its validator, so a confident match yields a verifiable recovery."""
from __future__ import annotations
import re

# Each shape: structural KEYWORDS (in the description) + SIGNATURE hints (type structure) + the
# hold-out-proven primitive + the validator strategy. Detection is name-agnostic.
SHAPES = [
    {"name": "sat_dpll", "primitive": "dpll", "validator": "verify_sat",
     "kw": [r"\bsat\b", r"\bcnf\b", r"claus", r"satisf", r"\bdpll\b", r"\bliteral", r"unit propagation"],
     "sig": [r"list\[list\[int\]\]"]},
    {"name": "grapheme", "primitive": "unicode_grapheme_count", "validator": "differential",
     "kw": [r"grapheme", r"user-perceived", r"combining (char|mark)", r"\bzwj\b", r"regional indicator",
            r"variation selector", r"cluster"], "sig": [r"\(s:\s*str\)\s*->\s*int"]},
    {"name": "symbolic_diff", "primitive": "symbolic_differentiate", "validator": "numeric_diff",
     "kw": [r"differentiat", r"derivative", r"symbolic", r"polynomial.*(respect|degree)"], "sig": []},
    {"name": "regex_nfa", "primitive": "regex_to_nfa_match", "validator": "differential",
     "kw": [r"\bnfa\b", r"thompson", r"finite automaton", r"regex.*(match|engine|nfa)", r"automaton"],
     "sig": []},
    {"name": "wavelet", "primitive": "WaveletTree", "validator": "differential",
     "kw": [r"wavelet", r"\brank\b.*(quer|array)", r"\bquantile\b", r"k-?th smallest"], "sig": []},
    {"name": "nonogram", "primitive": "nonogram_line", "validator": "brute_force",
     "kw": [r"nonogram", r"\bclue", r"block lengths"], "sig": []},
    {"name": "link_cut", "primitive": "LinkCutTree", "validator": "brute_force",
     "kw": [r"link.?cut", r"dynamic (tree|forest|connectivity)", r"\blink\b.*\bcut\b"], "sig": []},
    {"name": "centroid", "primitive": "centroid_decompose", "validator": "invariant",
     "kw": [r"centroid", r"decompos.*tree", r"tree.*decompos"], "sig": []},
    {"name": "lambda", "primitive": "lambda_eval", "validator": "known_forms",
     "kw": [r"lambda calculus", r"beta.?reduc", r"church (numeral|encoding)", r"normal form"], "sig": []},
    # --- high-reuse advanced shapes (det_primitives_advanced), structurally detected ---
    {"name": "max_flow", "primitive": "Dinic", "validator": "verify_flow",
     "kw": [r"max(imum)? flow", r"min.?cut", r"flow network", r"dinic", r"edmonds.?karp"], "sig": []},
    {"name": "scc", "primitive": "scc_tarjan", "validator": "verify_scc",
     "kw": [r"strongly connected", r"\bscc\b", r"tarjan", r"kosaraju"], "sig": []},
    {"name": "convex_hull", "primitive": "convex_hull", "validator": "verify_hull",
     "kw": [r"convex hull", r"monotone chain", r"graham scan", r"jarvis"], "sig": []},
    {"name": "lis", "primitive": "lis_length", "validator": "brute_force",
     "kw": [r"longest increasing subsequence", r"\blis\b", r"patience sort"], "sig": []},
    {"name": "xor_basis", "primitive": "gf2_xor_basis", "validator": "verify_xor",
     "kw": [r"xor basis", r"linear basis", r"max(imum)? xor", r"gaussian elimination.*xor", r"gf\(?2\)?"], "sig": []},
    {"name": "sieve", "primitive": "sieve_primes", "validator": "verify_primes",
     "kw": [r"sieve", r"eratosthenes", r"all primes (up to|below)", r"prime.*(up to|less than)"], "sig": []},
    {"name": "modpow", "primitive": "modpow", "validator": "verify_modpow",
     "kw": [r"modular exponent", r"\bmod(ular)? pow", r"square.and.multiply", r"\(base.*exp.*mod"], "sig": []},
    {"name": "stable_sum", "primitive": "stable_sum", "validator": "differential",
     "kw": [r"kahan", r"compensated sum", r"stable sum", r"floating.?point.*(precision|sum)",
            r"numerically stable"], "sig": []},
    {"name": "optimal_bst", "primitive": "optimal_bst_intervals", "validator": "brute_force",
     "kw": [r"optimal binary search tree", r"optimal bst", r"knuth"], "sig": []},
    {"name": "range_kth", "primitive": "persistent_segment_tree_kth", "validator": "differential",
     "kw": [r"persistent segment tree", r"k-?th smallest", r"range.*k-?th", r"merge sort tree"], "sig": []},
    {"name": "aho_corasick", "primitive": "aho_corasick_dp_count", "validator": "brute_force",
     "kw": [r"aho.?corasick", r"forbidden pattern", r"do not contain", r"avoid.*substring"], "sig": []},
    {"name": "cellular_automaton", "primitive": "cellular_automaton_rule", "validator": "differential",
     "kw": [r"cellular automaton", r"wolfram rule", r"elementary.*automaton", r"rule.?number"], "sig": []},
    {"name": "hindley_milner", "primitive": "type_inference_hindley_milner", "validator": "known_forms",
     "kw": [r"hindley.?milner", r"type infer", r"most general type", r"infer.*type", r"algorithm w"], "sig": []},
    {"name": "program_synthesis", "primitive": "program_synthesis_from_io", "validator": "brute_force",
     "kw": [r"program synthesis", r"synthesize.*(expression|program)", r"input.?output examples",
            r"simplest.*expression"], "sig": []},
    {"name": "banker_round", "primitive": "correct_round_half_even", "validator": "differential",
     "kw": [r"round.?half.?to.?even", r"banker", r"round half to even", r"ieee 754.*round"], "sig": []},
    {"name": "hungarian", "primitive": "hungarian_algorithm", "validator": "brute_force",
     "kw": [r"hungarian", r"kuhn.?munkres", r"assignment problem", r"minimum cost assignment"], "sig": []},
    {"name": "poly_mult", "primitive": "fft_ntt_poly_multiply", "validator": "differential",
     "kw": [r"\bntt\b", r"number theoretic transform", r"multiply.*polynomial", r"polynomial.*multipl"], "sig": []},
    # --- ladder-100 core algorithmic/parsing shapes ---
    {"name": "longest_palindrome", "primitive": "longest_palindrome_substring", "validator": "brute_force",
     "kw": [r"longest palindrom", r"palindromic substring", r"expand around center"],
     "sig": [r"\(s:\s*str\)\s*->\s*str"]},
    {"name": "permutations", "primitive": "permutations", "validator": "brute_force",
     "kw": [r"all permutations", r"permutations of", r"lexicograph"],
     "sig": [r"->\s*list\[list\[int\]\]"]},
    {"name": "bfs_shortest_path", "primitive": "bfs_shortest_path", "validator": "verify_path",
     "kw": [r"shortest path", r"number of edges", r"unweighted graph", r"\bbfs\b"],
     "sig": [r"start:\s*int,\s*end:\s*int\)\s*->\s*int"]},
    {"name": "directed_cycle", "primitive": "has_cycle", "validator": "brute_force",
     "kw": [r"contains a cycle", r"directed graph.*cycle", r"cycle detection", r"\bhas_cycle\b"],
     "sig": [r"\)\s*->\s*bool"]},
    {"name": "bipartite", "primitive": "is_bipartite", "validator": "brute_force",
     "kw": [r"bipartite", r"2-?colorable", r"two-?colou?r"],
     "sig": [r"\)\s*->\s*bool"]},
    {"name": "topo_sort", "primitive": "topological_sort", "validator": "verify_topo",
     "kw": [r"topological order", r"topological sort", r"\bdag\b", r"directed acyclic"],
     "sig": [r"->\s*list\[int\]"]},
    {"name": "dijkstra", "primitive": "dijkstra", "validator": "verify_path",
     "kw": [r"shortest distance", r"weighted directed graph", r"dijkstra", r"neighbor.*weight"],
     "sig": [r"list\[list\]\]"]},
    {"name": "csv_line", "primitive": "parse_csv_line", "validator": "differential",
     "kw": [r"parse.*csv", r"csv line", r"quoted field", r"escaped quote"],
     "sig": [r"\(line:\s*str\)\s*->\s*list\[str\]"]},
    {"name": "eval_expr", "primitive": "eval_expression", "validator": "differential",
     "kw": [r"evaluat.*expression", r"operator precedence", r"mathematical expression"],
     "sig": [r"\(expr:\s*str\)\s*->\s*float"]},
    {"name": "json_parse", "primitive": "parse_json", "validator": "differential",
     "kw": [r"json parser", r"parse.*json", r"do not use json module"],
     "sig": [r"\(s:\s*str\)\s*->\s*object"]},
    {"name": "sql_tokenize", "primitive": "tokenize_sql", "validator": "differential",
     "kw": [r"tokeniz.*sql", r"sql query.*token", r"token types"],
     "sig": [r"\(query:\s*str\)\s*->\s*list\[dict"]},
    {"name": "simple_regex", "primitive": "match_simple_regex", "validator": "differential",
     "kw": [r"simple regex", r"regex matcher", r"without using the re module", r"character classes"],
     "sig": [r"\(pattern:\s*str,\s*text:\s*str\)\s*->\s*bool"]},
    {"name": "relative_date", "primitive": "parse_relative_date", "validator": "differential",
     "kw": [r"relative date", r"days (ago|later)", r"weeks (ago|later)", r"months (ago|later)"],
     "sig": [r"\(base:\s*str,\s*relative:\s*str\)\s*->\s*str"]},
    {"name": "query_engine", "primitive": "query_engine", "validator": "differential",
     "kw": [r"query engine", r"sql-?like query", r"\$gt", r"\$lt", r"order_by", r"select.*where"],
     "sig": [r"\(data:\s*list\[dict\]"]},
]


def detect(prompt, signature=None, min_score=2):
    """Return (shape_dict, score) for the best structural match, or (None, score) if below threshold.
    Name-agnostic: scores description keywords (1 each) + signature hints (2 each, strong)."""
    text = (prompt or "").lower()
    sig = (signature or prompt or "").lower()
    best, best_score = None, 0
    for sh in SHAPES:
        score = sum(1 for p in sh["kw"] if re.search(p, text))
        score += sum(2 for p in sh.get("sig", []) if re.search(p, sig))
        if score > best_score:
            best, best_score = sh, score
    return (best, best_score) if best_score >= min_score else (None, best_score)


if __name__ == "__main__":
    import json, subprocess, os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # dump expert-50 prompts via node (id + prompt) — detect WITHOUT using the entry name
    try:
        out = subprocess.check_output(
            ["node", "-e",
             "import('./suites/expert-50/tasks.mjs').then(m=>console.log(JSON.stringify("
             "m.tasks.map(t=>({id:t.id,entry:t.eval&&t.eval.entry||t.entry,prompt:t.prompt})))))"],
            cwd=root, text=True)
        tasks = json.loads(out)
    except Exception as e:
        print("could not load suite:", e); tasks = []

    # known shape ground truth for the residuals we built primitives for
    KNOWN = {"x07": "centroid", "x12": "grapheme", "x33": "symbolic_diff", "x34": "nonogram",
             "x35": "lambda", "x36": "regex_nfa", "x38": "sat_dpll", "x42": "link_cut", "x45": "wavelet",
             "x06": "xor_basis", "x02": "convex_hull", "x21": "lis"}
    hits = miss = false_pos = 0
    print("Detecting shape from DESCRIPTION (entry name NOT used):\n")
    for t in tasks:
        # strip the entry name from the prompt so detection is purely structural
        prompt = (t.get("prompt") or "")
        if t.get("entry"):
            prompt = prompt.replace(t["entry"], "FUNC")
        sh, score = detect(prompt)
        name = sh["name"] if sh else None
        exp = KNOWN.get(t["id"])
        if exp:
            ok = name == exp
            hits += ok; miss += (not ok)
            print(f"  {t['id']:5s} expect={exp:13s} detected={str(name):13s} score={score} {'OK' if ok else 'MISS'}")
        elif name:
            false_pos += 1  # detected a shape for a task we didn't pre-map (may be correct or spurious)
    total = hits + miss
    print(f"\nknown-shape recall: {hits}/{total}; other tasks routed (review for FP/extra coverage): {false_pos}")
