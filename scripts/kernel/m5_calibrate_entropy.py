#!/usr/bin/env python3
"""M5 rung-3: CALIBRATION GATE for the MoE gate-softmax entropy signal (the OBSERVE half of scheduler-owned-k).

The calibration trap (project_scheduler_owned_k): entropy == "uncertainty" ONLY if it is CALIBRATED — i.e. only
if it actually predicts hardness/correctness. Our own evidence says model self-confidence MIScalibrates, so
KPolicy keeps KSignals.uncertainty_calibrated=False until THIS measurement earns it. "fail->pass is not proof"
applied to routing: we do not let the policy widen k on a signal we have not shown is real.

METHOD. Run a LABELED coding set (deterministic ground truth via check_cases) through North's RAW model lane with
the inert entropy tap LIVE. Per task record pass/fail + a battery of entropy AGGREGATES (mean/peak across tokens;
first-token; specific MoE layers L8/L24/L48; volatility). Then test whether any aggregate predicts FAILURE:
  * AUC = P(entropy(fail) > entropy(pass)) — 0.5 == no signal; >=0.65 (a modest bar) == usable; <0.5 == inverted.
  * point-biserial r vs passed.
A POSITIVE CONTROL (task difficulty -> fail) validates the pipeline + that the label set carries signal at all.

VERDICT -> docs/logs: if some aggregate clears the bar with the right sign, mark THAT feature calibrated and emit
its (entropy->uncertainty) mapping for KPolicy; else record MISCALIBRATED -> KPolicy must fall back to an
INDEPENDENT signal (seed-disagreement / validate-the-branch), which rung-3b (--seeds) measures.

Self-verifies the tap is still INERT (gen_with_entropy==gen_fast on task[0]) before trusting any number.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/m5_calibrate_entropy.py \
        [--tasks /tmp/expert50.json] [--n 50] [--maxn 1200] [--seeds 0]
"""
from __future__ import annotations
import os
import sys, os, json, re, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mlx_lm.utils import load
from north_adapter import NorthAdapter
from kernel.moe_entropy import MoEEntropyTap, gen_with_entropy
from shape_appliance import check_cases

MODEL = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text or "", re.I)
    if m:
        return m.group(1)
    i = (text or "").find("def ")
    return text[i:] if i >= 0 else (text or "")


def _layer_at(sig, li):
    for j, v in sig["per_layer"]:
        if j == li:
            return v
    return None


def task_features(sigs):
    """Per-task entropy aggregates over the per-token signal trace `sigs`."""
    means = [s["mean"] for s in sigs]
    maxs = [s["max"] for s in sigs]
    L8 = [v for v in (_layer_at(s, 8) for s in sigs) if v is not None]
    L24 = [v for v in (_layer_at(s, 24) for s in sigs) if v is not None]
    L48 = [v for v in (_layer_at(s, 48) for s in sigs) if v is not None]
    def m(a): return float(np.mean(a)) if a else float("nan")
    return {
        "seq_mean_of_mean": m(means),               # the broad 0.85 band
        "seq_peak_of_mean": (max(means) if means else float("nan")),
        "seq_mean_of_max": m(maxs),                 # avg of the most-uncertain layer
        "first_tok_mean": (means[0] if means else float("nan")),
        "seq_std_of_mean": (float(np.std(means)) if means else float("nan")),
        "L8_mean": m(L8), "L24_mean": m(L24), "L48_mean": m(L48),
    }


FEATURES = ["seq_mean_of_mean", "seq_peak_of_mean", "seq_mean_of_max", "first_tok_mean",
            "seq_std_of_mean", "L8_mean", "L24_mean", "L48_mean"]


def auc_fail(scores, fail):
    """AUC for predicting FAILURE from a score (higher score -> more likely fail). 0.5 == chance."""
    s = np.asarray(scores, float); f = np.asarray(fail, int)
    ok = ~np.isnan(s)
    s, f = s[ok], f[ok]
    pos, neg = s[f == 1], s[f == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum(int(p > n) + 0.5 * int(p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pbr(scores, passed):
    s = np.asarray(scores, float); p = np.asarray(passed, float)
    ok = ~np.isnan(s)
    if ok.sum() < 3 or np.std(s[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(s[ok], p[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/tmp/expert50.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--maxn", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=0, help="rung-3b: also measure seed-disagreement (k samples@0.8)")
    args = ap.parse_args()

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    a = NorthAdapter.__new__(NorthAdapter); a.model, a.tok, a.eos = model, tok, eos
    tasks = json.load(open(args.tasks))[:args.n]

    tap = MoEEntropyTap(model).install()

    # --- self-check: the (optimized) tap is still INERT before we trust any calibration number ---
    t0 = tasks[0]
    c_oracle, n_oracle = a.gen_fast(t0["prompt"], maxn=args.maxn)
    c_tap, n_tap, _ = gen_with_entropy(a, tap, t0["prompt"], maxn=args.maxn)
    inert = (c_tap == c_oracle and n_tap == n_oracle)
    print(f"[inert] gen_with_entropy==gen_fast on {t0['id']}: {inert}", flush=True)
    if not inert:
        print("XX tap NOT inert — aborting (calibration numbers would be untrustworthy)", flush=True)
        sys.exit(2)

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    rows_path = os.path.join(logdir, "m5_calib_rows.jsonl")
    rows = []
    with open(rows_path, "w") as fh:
        for t in tasks:
            text, ntok, sigs = gen_with_entropy(a, tap, t["prompt"], maxn=args.maxn)
            code = _extract_code(text)
            passed = bool(code) and bool(check_cases(code, t["entry"], t.get("cases", [])))
            feats = task_features(sigs)
            row = {"id": t["id"], "difficulty": t.get("difficulty"), "passed": passed, "ntok": ntok, **feats}
            if args.seeds > 0:
                outs = []
                for sd in range(args.seeds):
                    st, _ = a.gen_sample(t["prompt"], temp=0.8, seed=sd, maxn=args.maxn)
                    sc = _extract_code(st)
                    outs.append(bool(sc) and bool(check_cases(sc, t["entry"], t.get("cases", []))))
                row["seed_pass_rate"] = sum(outs) / len(outs)
                row["seed_disagreement"] = 1.0 - abs(2 * (sum(outs) / len(outs)) - 1.0)   # 1 == maximally split
            rows.append(row)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  {t['id']:5s} d{str(t.get('difficulty')):>2} {'PASS' if passed else 'fail'} "
                  f"ntok={ntok:4d} mean={feats['seq_mean_of_mean']:.3f} peak={feats['seq_peak_of_mean']:.3f} "
                  f"L24={feats['L24_mean']:.3f}", flush=True)

    tap.remove()

    passed = [r["passed"] for r in rows]
    fail = [0 if p else 1 for p in passed]
    npass = sum(passed)
    print(f"\n[labels] {npass}/{len(rows)} passed (raw North model lane)", flush=True)

    analysis = {}
    for feat in FEATURES:
        sc = [r[feat] for r in rows]
        analysis[feat] = {"auc_fail": round(auc_fail(sc, fail), 3), "pbr_passed": round(pbr(sc, passed), 3)}
    # positive control: difficulty -> fail (must show signal if the label set + pipeline are sound)
    diffs = [r["difficulty"] if r["difficulty"] is not None else float("nan") for r in rows]
    analysis["_control_difficulty"] = {"auc_fail": round(auc_fail(diffs, fail), 3), "pbr_passed": round(pbr(diffs, passed), 3)}
    if args.seeds > 0:
        sd = [r.get("seed_disagreement", float("nan")) for r in rows]
        analysis["seed_disagreement"] = {"auc_fail": round(auc_fail(sd, fail), 3), "pbr_passed": round(pbr(sd, passed), 3)}

    # verdict: strongest ENTROPY feature in EITHER direction (|AUC-0.5|), reporting the sign. A first pass that
    # scored only the naive "high->fail" direction mislabeled a strong INVERTED signal (high entropy -> SUCCESS)
    # as miscalibrated; m5_reanalyze_calib.py does the both-direction + length-confound analysis (verdict2.json).
    def _strength(f):
        a = analysis[f]["auc_fail"]
        return 0.0 if math.isnan(a) else abs(a - 0.5)
    best = max(FEATURES, key=_strength)
    best_auc = analysis[best]["auc_fail"]
    best_dir = "high->fail" if best_auc > 0.5 else "high->pass(INVERTED)"
    signal = _strength(best) >= 0.15            # AUC >= 0.65 in either direction
    # NB: a present signal is NOT auto-trusted — an inverted / single-suite / confounded signal must clear a
    # confound + action control first, so KPolicy's uncertainty_calibrated stays False for raw entropy regardless.
    verdict = {
        "n": len(rows), "n_pass": npass, "maxn": args.maxn, "seeds": args.seeds,
        "analysis": analysis,
        "best_entropy_feature": best, "best_entropy_auc_fail": best_auc, "best_direction": best_dir,
        "best_auc_either_dir": round(max(best_auc, 1 - best_auc), 3),
        "control_difficulty_auc": analysis["_control_difficulty"]["auc_fail"],
        "VERDICT": (f"SIGNAL PRESENT ({best} {best_dir}) but NOT wired -> KPolicy lever = independent uncertainty "
                    f"(seed-disagreement / validate-the-branch); see m5_reanalyze_calib.py for confound control"
                    if signal else "NO ENTROPY SIGNAL -> use independent uncertainty (disagreement / validate-the-branch)"),
    }
    json.dump(verdict, open(os.path.join(logdir, "m5_calib_verdict.json"), "w"), indent=2, default=str)
    print("\n===== M5 rung-3 calibration =====", flush=True)
    for k, v in analysis.items():
        print(f"  {k:24s} AUC(fail)={v['auc_fail']}  r(passed)={v['pbr_passed']}", flush=True)
    print(f"\nbest entropy feature: {best} AUC={best_auc} | difficulty control AUC={verdict['control_difficulty_auc']}", flush=True)
    print(f"VERDICT: {verdict['VERDICT']}", flush=True)
    print(f"(rows -> {rows_path}; verdict -> logs/m5_calib_verdict.json)", flush=True)


if __name__ == "__main__":
    main()
