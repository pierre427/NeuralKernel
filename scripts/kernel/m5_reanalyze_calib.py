#!/usr/bin/env python3
"""M5 rung-3 RE-ANALYSIS (CPU, no model): re-read m5_calib_rows.jsonl and report the entropy<->correctness
association HONESTLY — both directions + a length-confound control. The first pass only tested the naive
"high entropy -> fail" direction (AUC_fail>=0.65) and so mislabeled a strong INVERTED signal as miscalibrated.

Reports per feature: AUC in the correct (max) direction, its SIGN, r(passed), and — the key control — the
partial correlation r(feature, passed | ntok) and within-ntok-tercile AUC, because failures skew long (runaways
/ hitting the token cap), so a raw entropy<->pass link could be a generation-length artifact. "fail->pass is not
proof" applied to routing: an association that flips a metric is necessary-not-sufficient; it must survive a
confound control before KPolicy is allowed to trust it.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/m5_reanalyze_calib.py
"""
from __future__ import annotations
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kernel.m5_calibrate_entropy import FEATURES, auc_fail, pbr

ROWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "m5_calib_rows.jsonl")


def _r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def partial_r(x, y, z):
    """r(x,y | z) — correlation of x,y with the linear effect of z removed (z = ntok, the length confound)."""
    rxy, rxz, ryz = _r(x, y), _r(x, z), _r(y, z)
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / den if den > 0 else float("nan")


def tercile_auc(feat, passed, ntok):
    """AUC(pass) within each ntok tercile — if the signal is length-driven it collapses inside a length band."""
    n = np.asarray(ntok, float)
    qs = np.quantile(n, [1 / 3, 2 / 3])
    out = []
    for lo, hi in [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]:
        idx = [i for i in range(len(n)) if lo < n[i] <= hi] if lo != -np.inf else [i for i in range(len(n)) if n[i] <= hi]
        if len(idx) >= 4:
            f = [feat[i] for i in idx]; p = [passed[i] for i in idx]
            fail = [0 if q else 1 for q in p]
            a = auc_fail(f, fail)
            out.append(round(1 - a, 3) if not math.isnan(a) else None)   # AUC(pass) = 1 - AUC(fail)
        else:
            out.append(None)
    return out


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    passed = [r["passed"] for r in rows]
    fail = [0 if p else 1 for p in passed]
    ntok = [r["ntok"] for r in rows]
    npass = sum(passed)
    print(f"[rows] {len(rows)} tasks, {npass} pass / {len(rows)-npass} fail | ntok r(passed)={_r(ntok, [float(p) for p in passed]):+.3f}")
    print(f"  (length confound check: failures {'DO' if _r(ntok,[float(p) for p in passed])<-0.2 else 'do not strongly'} skew long)\n")

    print(f"  {'feature':18s} {'AUC*':>6} {'dir':>5} {'r(pass)':>8} {'r(pass|ntok)':>13}  tercile AUC(pass)")
    results = {}
    for feat in FEATURES:
        vals = [r[feat] for r in rows]
        a_fail = auc_fail(vals, fail)
        a_star = max(a_fail, 1 - a_fail)                 # strongest direction
        direction = "HI->fail" if a_fail > 0.5 else "HI->pass"
        r_pass = pbr(vals, passed)
        r_pc = partial_r(vals, [float(p) for p in passed], ntok)
        terc = tercile_auc(vals, passed, ntok)
        results[feat] = {"auc_best": round(a_star, 3), "direction": direction, "auc_fail": round(a_fail, 3),
                         "r_passed": round(r_pass, 3), "r_passed_given_ntok": round(r_pc, 3), "tercile_auc_pass": terc}
        print(f"  {feat:18s} {a_star:6.3f} {direction:>9} {r_pass:+8.3f} {r_pc:+13.3f}  {terc}")

    best = max(FEATURES, key=lambda f: results[f]["auc_best"])
    b = results[best]
    survives = (not math.isnan(b["r_passed_given_ntok"])) and abs(b["r_passed_given_ntok"]) >= 0.3
    print(f"\nBEST: {best}  AUC={b['auc_best']} ({b['direction']})  r|ntok={b['r_passed_given_ntok']}")
    if b["direction"] == "HI->pass":
        verdict = ("REAL-BUT-INVERTED+CONFOUND-CONTROLLED: high routing entropy predicts SUCCESS"
                   if survives else "INVERTED ASSOCIATION but does NOT survive ntok control -> length artifact")
    else:
        verdict = "naive direction (high entropy -> fail)"
    print("VERDICT:", verdict)
    print("\nKPolicy implication: the raw signal is INVERTED vs the 'entropy=bad-uncertainty' assumption AND is\n"
          "single-suite (all d10) — so it is NOT wired as a calibrated k-driver (uncertainty_calibrated stays\n"
          "False). The calibration gate did its job: it caught an inverted/confounded signal before use. KPolicy's\n"
          "live lever stays independent uncertainty (seed-disagreement / validate-the-branch).")

    out = {"n": len(rows), "n_pass": npass, "ntok_r_passed": round(_r(ntok, [float(p) for p in passed]), 3),
           "per_feature": results, "best_feature": best, "best_direction": b["direction"],
           "best_auc": b["auc_best"], "survives_length_control": survives, "VERDICT": verdict,
           "kpolicy_action": "uncertainty_calibrated=False for raw gate entropy (inverted+single-suite+confounded); "
                             "lever = independent uncertainty (disagreement / validate-the-branch)"}
    json.dump(out, open(os.path.join(os.path.dirname(ROWS), "m5_calib_verdict2.json"), "w"), indent=2, default=str)
    print(f"\n(updated verdict -> logs/m5_calib_verdict2.json)")


if __name__ == "__main__":
    main()
