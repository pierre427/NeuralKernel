#!/usr/bin/env python3
"""M5 rung-2 PARITY: the MoE-entropy tap is INERT (==0) on real North, and yields a valid [0,1] uncertainty signal.

The OBSERVE half of scheduler-owned-k must be observationally null: installing the gate-softmax tap must not move
a single output bit. We prove, strict == (design §5.1, no tolerance):
  (1) gen_with_entropy (tap LIVE) == production gen_fast, TOKEN-FOR-TOKEN  -> the tap is inert;
  (2) after remove(), gen_fast == the same oracle                          -> install/restore is exact;
  (3) the tap covers all 48 MoE layers (layer-0 dense MLP skipped), every per-step signal reads 48 layers;
  (4) the recorded signal is finite, in [0,1], one per produced token, and VARIES (a real signal, not a constant).

Run solo: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/inv_moe_entropy.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.utils import load
from north_gateway_invariants import MODEL, PROMPT
from north_adapter import NorthAdapter
from kernel.moe_entropy import MoEEntropyTap, gen_with_entropy

MAXN = 96
EXPECT_MOE_LAYERS = 48           # North: num_hidden_layers=49, first_k_dense_replace=1 -> 48 MoE blocks


def _attach(model, tok, eos):
    a = NorthAdapter.__new__(NorthAdapter)
    a.model, a.tok, a.eos = model, tok, eos
    return a


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    a = _attach(model, tok, eos)

    out_base, n = a.gen_fast(PROMPT, maxn=MAXN)                  # untapped oracle
    print(f"[oracle] gen_fast n={n}", flush=True)

    tap = MoEEntropyTap(model).install()
    n_tap = tap.n_tapped
    out_t, n_t, sigs = gen_with_entropy(a, tap, PROMPT, maxn=MAXN)
    tap.remove()
    out_after, n_after = a.gen_fast(PROMPT, maxn=MAXN)           # removal must restore the oracle exactly

    aggs = [s["agg"] for s in sigs]
    eq_tap = (out_t == out_base and n_t == n)
    eq_rm = (out_after == out_base and n_after == n)
    layers_ok = (n_tap == EXPECT_MOE_LAYERS) and all(s["n_layers"] == EXPECT_MOE_LAYERS for s in sigs)
    finite_in01 = (len(aggs) > 0
                   and all(math.isfinite(x) for x in aggs)
                   and all(-1e-9 <= s["min"] and s["max"] <= 1.0 + 1e-6 for s in sigs))
    one_per_token = (len(sigs) >= n >= 1)                        # >=1 routing signal per produced token
    varies = (len(aggs) > 1 and (max(aggs) - min(aggs) > 1e-6))

    R = {
        "n_base": n, "n_tapped_layers": n_tap, "n_signals": len(sigs),
        "agg_min": (min(aggs) if aggs else None), "agg_max": (max(aggs) if aggs else None),
        "agg_mean": (sum(aggs) / len(aggs) if aggs else None),
        "checks": {
            "inert_tokens_eq (==0)": eq_tap,
            "remove_restores_oracle": eq_rm,
            "all_48_moe_layers_tapped": layers_ok,
            "signal_finite_in_[0,1]": finite_in01,
            "one_signal_per_token": one_per_token,
            "signal_varies_real": varies,
        },
    }
    allok = all(R["checks"].values())
    R["RESULT"] = "PASS" if allok else "FAIL"

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_moe_entropy.json"), "w"), indent=2, default=str)
    for k, v in R["checks"].items():
        print(f"  {'OK ' if v else 'XX '} {k}: {v}", flush=True)
    print(json.dumps({k: R[k] for k in ("n_base", "n_tapped_layers", "n_signals", "agg_min", "agg_max", "agg_mean", "RESULT")}, default=str), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
