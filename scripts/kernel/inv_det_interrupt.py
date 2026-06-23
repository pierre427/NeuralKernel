#!/usr/bin/env python3
"""Phase-4 M4 PARITY + CAPABILITY-CONTRAST: a deterministic in-decode INTERRUPT on North.

NOT a pure parity proof (an armed constraint changes the output BY DESIGN). Two checks:
  IDENTITY control: gen_constrained(forbidden=None) == production gen_fast, token-for-token (==0) — the interrupt
                    is overhead-free / bit-identical when inert.
  CAPABILITY contrast: with a {import, from} mask, gen_constrained ENFORCES no-import on a prompt that the
                    unconstrained decode answers WITH an import — the kernel forces an exact property on-device.

Reuses north_det_block (build_forbidden_mask + the mx.where logit mask). Loads North; run solo."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.utils import load
from north_gateway_invariants import MODEL, PROMPT
from north_det_block import build_forbidden_mask
from north_adapter import NorthAdapter
from kernel.trapped_adapter import TrappedNorthAdapter

PROMPT_IMPORT = ("Write a Python function `mean(xs)` that returns the arithmetic mean of a list of numbers using "
                 "numpy. Return ONLY the function in a ```python code block.")
MAXN = 200


def _attach(cls, model, tok, eos):
    a = cls.__new__(cls); a.model, a.tok, a.eos = model, tok, eos
    a.plane = None
    return a


def _has_import(text):
    m = re.search(r"```python\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    return bool(re.search(r"\b(import|from)\b", body))


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    vocab = model.args.vocab_size
    forbidden = build_forbidden_mask(tok, vocab, ["import", "from"])
    base = _attach(NorthAdapter, model, tok, eos)
    ta = _attach(TrappedNorthAdapter, model, tok, eos)

    # IDENTITY control: inert interrupt == production gen_fast (token-for-token)
    out_fast, n_fast = base.gen_fast(PROMPT, maxn=MAXN)
    out_inert, n_inert = ta.gen_constrained(PROMPT, forbidden=None, maxn=MAXN)
    identity = (out_inert == out_fast and n_inert == n_fast)

    # CAPABILITY contrast: enforce no-import where the unconstrained decode imports
    out_unc, _ = base.gen_fast(PROMPT_IMPORT, maxn=MAXN)
    out_con, _ = ta.gen_constrained(PROMPT_IMPORT, forbidden=forbidden, maxn=MAXN)
    unc_import = _has_import(out_unc)
    con_import = _has_import(out_con)

    # PASS requires: inert==production (identity), the constraint enforced (no import), AND the contrast is
    # non-vacuous (the unconstrained decode actually DID import — else the "enforcement" proves nothing).
    R = {"masked_ids": int(forbidden.size), "identity_eq_gen_fast": identity,
         "unconstrained_has_import": unc_import, "constrained_has_import": con_import,
         "RESULT": "PASS" if (identity and unc_import and not con_import) else "FAIL"}
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_det_interrupt.json"), "w"), indent=2, default=str)
    print(f"  identity(inert)==gen_fast: {identity}  | unconstrained has_import={unc_import} -> "
          f"constrained has_import={con_import}  ({forbidden.size} ids masked)", flush=True)
    print(json.dumps(R, default=str), flush=True)
    sys.exit(0 if R["RESULT"] == "PASS" else 1)


if __name__ == "__main__":
    main()
