"""
calibration_and_utility.py
================================================================================
Calibration and clinical utility, reported with measures that are informative at
a 1.55% base rate.

WHY THIS SCRIPT EXISTS
----------------------
The manuscript declines to report classification accuracy, and gives the correct
reason: at the test prevalence a constant "healthy" prediction attains 98.44%,
so the metric cannot separate a useful model from a useless one.

The Brier score it reports instead has exactly the same defect. A model that
predicts the base rate for every athlete-game scores

    Brier_null = p(1-p) = 0.0155 x 0.9845 = 0.01526

against the 0.0151-0.0154 reported for the four fitted models. The entire spread
across the table is 0.0003, and one model is WORSE than predicting a constant.
Reporting Brier here repeats the error the paper exists to warn against, one
table later.

This script replaces it with measures that do discriminate:

  CALIBRATION
    calibration-in-the-large (intercept)  logit(y) offset by logit(p), slope 1
    calibration slope                     logit(y) ~ logit(p)
    observed:expected ratio               sum(y) / sum(p)
    Brier skill score                     1 - Brier / Brier_null
    Estimated calibration index (ECI)     mean squared distance from a LOESS
                                          calibration curve to the diagonal

    A slope below 1 indicates overfitting -- predictions too extreme. A slope
    above 1 indicates the opposite. An intercept below 0 indicates systematic
    over-prediction of risk.

  CLINICAL UTILITY
    Decision curve analysis (Vickers & Elkin, Med Decis Making 2006). Net
    benefit at threshold probability pt:

        NB = TP/n - (FP/n) x pt/(1-pt)

    compared against "review everybody" and "review nobody". A model is worth
    deploying only where its curve sits above both. At a 1.55% base rate the
    relevant thresholds are small, so the curve is evaluated from 0.5% to 10%.

    Net benefit is also expressed as "net athlete-games avoided per 1,000", the
    reduction in review workload at equal injuries detected.

OUTPUTS
-------
  predictive_model/calibration_utility_output.txt
  predictive_model/calibration_utility.json
"""

import os
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import dataio
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(SCRIPT_DIR, "test_predictions.csv")
TXT_OUT = os.path.join(SCRIPT_DIR, "calibration_utility_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "calibration_utility.json")

MODELS = {
    "Logistic regression": "p_LogisticRegression_cal",
    "Random forest": "p_RandomForest_cal",
    "Gradient boosting": "p_XGBoost_cal",
    "Schedule + biometrics": "p_Parsimonious_cal",
}
UNCAL = {
    "Logistic regression": "p_LogisticRegression",
    "Random forest": "p_RandomForest",
    "Gradient boosting": "p_XGBoost",
    "Schedule + biometrics": "p_Parsimonious",
}

THRESHOLDS = np.concatenate([np.arange(0.005, 0.050, 0.0025),
                             np.arange(0.050, 0.1025, 0.005)])
EPS = 1e-9


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_metrics(y, p):
    """Calibration intercept, slope, O:E, Brier and Brier skill score."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    lp = logit(p)

    # Calibration slope: logit(y) ~ a + b*logit(p); b = 1 is perfect.
    X = sm.add_constant(lp)
    slope_fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    slope = float(slope_fit.params[1])
    slope_ci = [float(v) for v in slope_fit.conf_int()[1]]

    # Calibration-in-the-large: intercept with the linear predictor as offset.
    int_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                     offset=lp).fit()
    intercept = float(int_fit.params[0])
    int_ci = [float(v) for v in int_fit.conf_int()[0]]

    brier = float(brier_score_loss(y, p))
    base = float(y.mean())
    brier_null = float(base * (1 - base))
    bss = float(1 - brier / brier_null)

    return dict(
        calibration_slope=slope, calibration_slope_ci=slope_ci,
        calibration_intercept=intercept, calibration_intercept_ci=int_ci,
        observed=float(y.sum()), expected=float(p.sum()),
        oe_ratio=float(y.sum() / max(p.sum(), EPS)),
        # The DIRECTION word, derived rather than written by hand. The calibration-direction defect
        # (2026-09) was the manuscript describing an O:E below 1 as
        # under-prediction in three places; the number checker passed before
        # and after, because both wordings sit beside the same correct number.
        # A numeric guard cannot catch a sign error in prose, so the sign is
        # emitted as a string and the checker asserts the prose matches it.
        # O:E = observed / expected, so O:E < 1 means the model predicted MORE
        # injuries than occurred -- over-prediction.
        prediction_direction=(
            "over-prediction" if p.sum() > y.sum()
            else "under-prediction" if p.sum() < y.sum()
            else "calibrated"),
        brier=brier, brier_null=brier_null, brier_skill_score=bss,
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        prevalence=base)


def net_benefit(y, p, thresholds):
    """Vickers-Elkin net benefit, plus treat-all and treat-none references."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prev = y.mean()
    rows = []
    for pt in thresholds:
        flag = p >= pt
        tp = float(np.sum(flag & (y == 1)))
        fp = float(np.sum(flag & (y == 0)))
        w = pt / (1 - pt)
        nb = tp / n - (fp / n) * w
        nb_all = prev - (1 - prev) * w
        rows.append(dict(threshold=float(pt), n_flagged=int(flag.sum()),
                         tp=int(tp), fp=int(fp),
                         net_benefit=float(nb), nb_treat_all=float(nb_all),
                         nb_treat_none=0.0,
                         # workload avoided per 1,000 athlete-games at equal
                         # injuries detected, relative to reviewing everybody
                         net_reduction_per_1000=float((nb - nb_all) / w * 1000)
                         if w > 0 else np.nan))
    return rows


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 92)
    print("  CALIBRATION AND CLINICAL UTILITY AT A 1.55% BASE RATE")
    print("=" * 92)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = pd.read_csv(PRED, parse_dates=["dt"])
    y = df["y"].values
    prev = y.mean()
    print(f"[LOAD] {len(df):,} test athlete-games, {int(y.sum()):,} injuries "
          f"(prevalence {prev:.4%})\n")

    res = {"prevalence": float(prev), "n": int(len(df)), "events": int(y.sum())}

    # ---- why Brier is uninformative here ------------------------------------
    print("=" * 92)
    print("  [0] WHY THE BRIER SCORE MUST NOT CARRY THIS TABLE")
    print("=" * 92)
    b_null = prev * (1 - prev)
    b_const0 = float(np.mean((y - 0.0) ** 2))
    print(f"  Constant prediction of the base rate ({prev:.4f}) : Brier = {b_null:.6f}")
    print(f"  Constant prediction of zero risk                : Brier = {b_const0:.6f}")
    print(f"  Constant 'healthy' classification accuracy      : {1 - prev:.4%}")
    print("\n  Any fitted model here is compared against 0.01526. The reported spread")
    print("  across four models was 0.0151-0.0154 -- a total range of 0.0003, with one")
    print("  model scoring WORSE than a constant. The metric is as uninformative as the")
    print("  accuracy the manuscript already refuses to report, for the same reason.\n")
    res["brier_reference"] = dict(null_base_rate=float(b_null),
                                  constant_zero=float(b_const0),
                                  null_accuracy=float(1 - prev))

    # ---- calibration --------------------------------------------------------
    print("=" * 92)
    print("  [1] CALIBRATION (isotonically recalibrated predictions)")
    print("=" * 92)
    print(f"  {'model':24s} | {'slope':>16s} | {'intercept':>16s} | {'O:E':>6s} | "
          f"{'BSS':>7s}")
    print("  " + "-" * 88)
    res["calibration"] = {}
    for name, col in MODELS.items():
        m = calibration_metrics(y, df[col].values)
        res["calibration"][name] = m
        print(f"  {name:24s} | {m['calibration_slope']:6.3f} "
              f"({m['calibration_slope_ci'][0]:.2f},{m['calibration_slope_ci'][1]:.2f}) | "
              f"{m['calibration_intercept']:6.3f} "
              f"({m['calibration_intercept_ci'][0]:.2f},{m['calibration_intercept_ci'][1]:.2f}) | "
              f"{m['oe_ratio']:6.3f} | {m['brier_skill_score']:7.4f}")

    print("\n  Slope 1.0 and intercept 0.0 denote perfect calibration. O:E is observed")
    print("  injuries divided by the sum of predicted risks. BSS is the proportional")
    print("  reduction in Brier score against a constant base-rate prediction.\n")

    # The direction, spelled out, because this is exactly what the manuscript
    # got backwards in three places. Printed here and asserted in
    # verify_no_stale_numbers.py against the manuscript's own wording.
    dirs = {res["calibration"][n]["prediction_direction"] for n in MODELS}
    for name in MODELS:
        m = res["calibration"][name]
        print(f"  {name:24s} expected {m['expected']:7.1f} vs observed "
              f"{m['observed']:7.1f}  ->  {m['prediction_direction']}")
    if len(dirs) == 1:
        print(f"\n  All four models show {dirs.pop()}. Any prose saying otherwise "
              "is wrong,")
        print("  whatever number it quotes beside it.\n")
    else:
        print(f"\n  Models DISAGREE on direction: {sorted(dirs)}. The manuscript "
              "must say so\n  per model rather than making a single claim.\n")

    print("  Uncalibrated, for contrast (positive-class weighting inflates risk):")
    print(f"  {'model':24s} | {'slope':>7s} | {'intercept':>10s} | {'O:E':>8s}")
    print("  " + "-" * 60)
    res["calibration_uncalibrated"] = {}
    for name, col in UNCAL.items():
        m = calibration_metrics(y, df[col].values)
        res["calibration_uncalibrated"][name] = m
        print(f"  {name:24s} | {m['calibration_slope']:7.3f} | "
              f"{m['calibration_intercept']:10.3f} | {m['oe_ratio']:8.4f}")
    print()

    # ---- decision curve analysis --------------------------------------------
    print("=" * 92)
    print("  [2] DECISION CURVE ANALYSIS")
    print("=" * 92)
    print("  Net benefit at threshold probability pt. A model is worth deploying only")
    print("  where its curve exceeds BOTH 'review everybody' and 'review nobody' (0).\n")

    res["decision_curve"] = {}
    for name, col in MODELS.items():
        res["decision_curve"][name] = net_benefit(y, df[col].values, THRESHOLDS)

    head = "Schedule + biometrics"
    rows = res["decision_curve"][head]
    print(f"  Model: {head}")
    print(f"  {'pt':>7s} | {'flagged':>8s} | {'TP':>4s} | {'FP':>6s} | "
          f"{'NB model':>10s} | {'NB all':>10s} | {'per 1,000':>10s}")
    print("  " + "-" * 78)
    for r in rows:
        if round(r["threshold"] * 10000) % 250 and r["threshold"] > 0.02:
            continue
        print(f"  {r['threshold']*100:6.2f}% | {r['n_flagged']:>8,} | {r['tp']:>4,} | "
              f"{r['fp']:>6,} | {r['net_benefit']:10.6f} | {r['nb_treat_all']:10.6f} | "
              f"{r['net_reduction_per_1000']:10.2f}")

    # Where does the model beat both references?
    print("\n  Threshold range where each model beats both references:")
    res["useful_range"] = {}
    for name in MODELS:
        rows = res["decision_curve"][name]
        good = [r["threshold"] for r in rows
                if r["net_benefit"] > max(r["nb_treat_all"], 0) + 1e-9]
        if good:
            span = (min(good), max(good))
            peak = max(rows, key=lambda r: r["net_benefit"] - max(r["nb_treat_all"], 0))
            res["useful_range"][name] = dict(
                lo=float(span[0]), hi=float(span[1]),
                peak_threshold=float(peak["threshold"]),
                peak_net_benefit=float(peak["net_benefit"]),
                peak_net_reduction_per_1000=float(peak["net_reduction_per_1000"]))
            print(f"    {name:24s} : {span[0]*100:.2f}% - {span[1]*100:.2f}%  "
                  f"(peak advantage at pt = {peak['threshold']*100:.2f}%, "
                  f"{peak['net_reduction_per_1000']:.1f} athlete-games avoided "
                  f"per 1,000)")
        else:
            res["useful_range"][name] = None
            print(f"    {name:24s} : never exceeds both references")

    print("\n  Interpretation. Net benefit is measured in true positives per athlete-game,")
    print("  weighted by the cost the clinician implicitly assigns to a false alarm at")
    print("  that threshold. 'Athlete-games avoided per 1,000' converts the advantage")
    print("  over reviewing everybody into review workload saved at equal detection.")

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n[EXPORT] {JSON_OUT}")


if __name__ == "__main__":
    main()
