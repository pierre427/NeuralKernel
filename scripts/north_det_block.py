#!/usr/bin/env python3
"""Deterministic logic block ON the LLM stack — a constraint op the router fires to in a bypass.

P. Lamy's point: the deterministic step doesn't have to be fuzzy generation OR an external CPU
syscall. The model's forward pass is already exact GPU computation; only the final sample is
"fuzzy". So a deterministic logic block is just a NON-LEARNED op inserted into the same on-device
pipeline — a co-processor/functional-unit the router dispatches to, not a remote trap.

This proves the mechanism on North: a deterministic CONSTRAINT block runs on North's logits
on-device (MLX), in the generation loop, enforcing an EXACT property by masking — bypassing
unconstrained fuzzy sampling. Demonstrated with a real spec constraint: "no imports" (mask the
import/from tokens) and "must be a fenced code block". The block is deterministic, on-stack, and
verifiable — the same shape a JSON/grammar/regex constraint or a validator-gate would take.
"""
from __future__ import annotations
import os
import argparse, re, time
import mlx.core as mx
from mlx_lm.utils import load

MODEL = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")
SYSTEM = "You are a coding assistant. Provide precise, correct implementations."


def build_forbidden_mask(tok, vocab, words):
    """A deterministic block's static state: token ids whose DECODED form contains a forbidden
    word (import/from ...) — masked to -inf on-device during the constrained region."""
    bad = []
    for tid in range(vocab):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if any(re.search(rf"\b{w}\b", s) for w in words):
            bad.append(tid)
    return mx.array(bad, dtype=mx.int32)


def gen(model, tok, prompt, end_text, max_tok, constrain=False, forbidden=None):
    ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                  add_generation_prompt=True, reasoning=False)
    cache = model.make_cache()
    out = []
    cur = list(ids)
    neg = mx.array(-1e9, dtype=mx.float32)
    for _ in range(max_tok):
        logits = model(mx.array([cur], dtype=mx.int32), cache=cache)[0, -1, :]
        if constrain and forbidden is not None:
            # DETERMINISTIC LOGIC BLOCK, on-device: mask forbidden tokens before argmax (a bypass
            # of pure fuzzy sampling — exact, no CPU round-trip, runs in the MLX graph).
            logits = mx.where(
                mx.zeros_like(logits).at[forbidden].add(1.0) > 0, neg, logits
            )
        nxt = int(mx.argmax(logits).item())
        if nxt == end_text:
            break
        out.append(nxt)
        cur = [nxt]
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Write a Python function `dedup(xs: list) -> list` that removes duplicates preserving order. Return ONLY the function in a ```python code block.")
    ap.add_argument("--max-tok", type=int, default=400)
    args = ap.parse_args()
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    end_text = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    vocab = model.args.vocab_size if hasattr(model, "args") else len(tok)

    print("[build] deterministic constraint block: forbid {import, from} tokens", flush=True)
    forbidden = build_forbidden_mask(tok, vocab, ["import", "from"])
    print(f"  masked {forbidden.size} token ids", flush=True)

    for label, constrain in [("FUZZY (unconstrained)", False), ("DETERMINISTIC-BLOCK (constrained)", True)]:
        t0 = time.time()
        ans = gen(model, tok, args.prompt, end_text, args.max_tok, constrain=constrain, forbidden=forbidden)
        code = re.search(r"```python\n(.*?)```", ans, re.S)
        body = code.group(1) if code else ans
        has_import = bool(re.search(r"\b(import|from)\b", body))
        print(f"\n[{label}] ({time.time()-t0:.1f}s)  has_import={has_import}  has_codeblock={bool(code)}", flush=True)
        print("  " + body.strip().replace("\n", "\n  ")[:300], flush=True)


if __name__ == "__main__":
    main()
