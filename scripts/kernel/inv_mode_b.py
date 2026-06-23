#!/usr/bin/env python3
"""Mode-B B0 PARITY: the result-conditioned injector is DISABLED-IDENTITY (==0) on real North, and the injection
point is LIVE (a nonzero write changes the output) — so the ==0 is a real zero, not a dead hook.

Strict == (design §5.1, no tolerance):
  (1) prefill: run_stack_modeb(injector up==0, armed, NONZERO result vec) == model() -> max|diff| == 0
  (2) decode:  generate via run_stack_modeb (up==0) == production gen_fast, TOKEN-FOR-TOKEN
  (3) liveness: a NONZERO-up injector at the same site CHANGES the output (max|diff| > 0) -> the hook is real
This is the substrate + safety invariant for B2 (the trained steering, which must then clear a hold-out oracle +
placebo control). Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/inv_mode_b.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
from north_gateway_invariants import MODEL, PROMPT, maxabs
from north_adapter import NorthAdapter
from kernel.mode_b import ModeBInjector, run_stack_modeb

MAXN = 64
RESULT_DIM = 16


def _gen_modeb(model, tok, eos, inj, site, rv, prompt, maxn):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, reasoning=False)
    cache = make_prompt_cache(model)
    def step(seq):
        logits = run_stack_modeb(model, mx.array([seq], dtype=mx.int32), inj, site, rv, cache)
        return int(mx.argmax(logits[0, -1, :]).item())
    nxt = step(list(ids)); out = []
    for _ in range(maxn):
        if nxt == eos:
            break
        out.append(nxt); nxt = step([nxt])
    return tok.decode(out), len(out)


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    d = int(model.args.hidden_size); nL = int(model.args.num_hidden_layers); site = nL // 2
    a = NorthAdapter.__new__(NorthAdapter); a.model, a.tok, a.eos = model, tok, eos

    inputs = mx.array(np.array([tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                                        add_generation_prompt=True, reasoning=False)], dtype=np.int32))
    rv = mx.ones((RESULT_DIM,), dtype=mx.float32)                       # a NONZERO result vector

    inj = ModeBInjector(d, result_dim=RESULT_DIM); mx.eval(inj.parameters())   # up==0 (untrained)
    base = model(inputs); mx.eval(base)
    p_prefill = maxabs(run_stack_modeb(model, inputs, inj, site, rv), base)

    # liveness: a nonzero-up injector at the SAME site must change the output (the hook is real, not dead)
    live = ModeBInjector(d, result_dim=RESULT_DIM)
    live.up.weight = mx.random.normal(live.up.weight.shape) * 0.02
    mx.eval(live.parameters())
    p_live = maxabs(run_stack_modeb(model, inputs, live, site, rv), base)

    base_txt, n = a.gen_fast(PROMPT, maxn=MAXN)
    inj_txt, n2 = _gen_modeb(model, tok, eos, inj, site, rv, PROMPT, MAXN)
    decode_eq = (inj_txt == base_txt and n2 == n)

    checks = {
        "prefill_identity (==0)": (p_prefill == 0.0),
        "decode_token_parity (==0)": decode_eq,
        "injection_point_is_live (nonzero up changes output)": (p_live > 0.0),
    }
    R = {"site": site, "hidden_size": d, "result_dim": RESULT_DIM, "n_base": n,
         "p_prefill": p_prefill, "p_live": p_live, "checks": checks,
         "RESULT": "PASS" if all(checks.values()) else "FAIL"}
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_mode_b.json"), "w"), indent=2, default=str)
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'XX '} {k}: {v}", flush=True)
    print(json.dumps({k: R[k] for k in ("site", "p_prefill", "p_live", "n_base", "RESULT")}, default=str), flush=True)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
