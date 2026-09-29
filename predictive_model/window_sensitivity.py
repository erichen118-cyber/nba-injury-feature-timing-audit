"""
window_sensitivity.py
================================================================================
Does the result depend on the injury-to-game matching window?

WHY THIS SCRIPT EXISTS
----------------------
The outcome is an injury RECORDED within 1-3 calendar days of a game the
athlete played.
That window is a construction choice, and it is the first thing a reviewer will
challenge, because the study's central claim concerns the timing of absence:

    "The model detects athletes who have recently stopped playing and are about
     to be entered on an injury report."

If the rest gradient were an artifact of when reports are FILED rather than when
athletes are HURT, tightening the window should move it. The upstream merge in
Data/compile_data_corrected.py fixes the maximum at calendar gap 3, so the
window can be narrowed but not widened without the raw injury database:

    window 1-1   injury recorded the calendar day after the game
    window 1-2   recorded within two calendar days     <- prespecified arm
    window 1-3   the primary specification

Same-day reports are excluded by construction and always were: a date-only
report cannot be ordered against an evening tip-off, and only 13 of 5,422
source records have an appearance on the transaction date.

Narrowing costs events, so intervals widen. What matters is whether the point
estimates move.

WHAT IS REPORTED
----------------
For each window, under the identical temporal split and the identical fixed
hyperparameters:

  * the headline 17-covariate schedule-and-biometrics model
  * the lagged workload block (the principal negative finding). Its size is
    printed, never written here: it was "110-covariate" until the rebound-percentage removal dropped the
    three rebound-percentage bases on 2026-09-22, and a count in a docstring is
    a copy that goes stale silently.
  * the restricted-cohort collapse (the study's decisive experiment)
  * the empirical rest gradient

OUTPUTS
-------
  predictive_model/window_sensitivity_output.txt
  predictive_model/window_sensitivity.json
"""

import os
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

import dataio

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(SCRIPT_DIR, "window_sensitivity_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "window_sensitivity.json")

# Maximum CALENDAR gap, not floored elapsed hours (2026-09-18). The
# two sensitivity arms are the same two builds as before under their true
# names: the old "_w0" was calendar gap 1 only and the old "_w1" was calendar
# gaps 1-2. The primary is calendar 1-3, which is what the historical "0-2"
# window always actually was.
WINDOW_SUFFIXES = {1: "_w1", 2: "_w2", 3: ""}

# Minimum calendar gap. Fixed at 1 by construction, not a tunable: a
# date-only report cannot be ordered against an evening tip-off, so a
# same-day match cannot be said to follow the game.
MIN_GAP = 1
WINDOW_KEYS = [f"window_{MIN_GAP}_{w}" for w in sorted(WINDOW_SUFFIXES)]

from features import BASE_VARS, BIO, LAG_SUFFIXES, PARS

# Block sizes, derived. The side-by-side table below is printed OUTSIDE the
# per-window loop, so it cannot borrow that loop's `workload_bio`; without
# these it had the literal "(110)" typed in, which the rebound-percentage removal made wrong by three
# bases on 2026-09-22.
N_PARS = len(PARS)
N_WORKLOAD_BIO = len(BASE_VARS) * len(LAG_SUFFIXES) + len(BIO)

TRAIN_SEASONS = list(range(2015, 2021))
TEST_SEASONS = [2022, 2023]
RNG = np.random.default_rng(42)

REST_BINS = [0, 1, 2, 4, 6, 100]
REST_LABELS = ["1 (b2b)", "2", "3-4", "5-6", "7+"]


def boot_ci(y, p, fn, groups, n=500):
    yv, pv, gv = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(gv)
    rows_of = {g: np.flatnonzero(gv == g) for g in uniq}
    vals = np.empty(n)
    for i in range(n):
        drawn = RNG.choice(uniq, len(uniq), replace=True)
        s = np.concatenate([rows_of[g] for g in drawn])
        vals[i] = fn(yv[s], pv[s]) if yv[s].sum() else np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def fit_eval(tr, te, cols):
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    pw = (tr.y == 0).sum() / max(1, (tr.y == 1).sum())
    m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                          reg_lambda=2.0, scale_pos_weight=pw, eval_metric="aucpr",
                          n_jobs=-1, random_state=42)
    m.fit(imp.transform(tr[cols]), tr.y)
    p = m.predict_proba(imp.transform(te[cols]))[:, 1]
    prev = te.y.mean()
    pr = average_precision_score(te.y, p)
    lo, hi = boot_ci(te.y.values, p, average_precision_score, te.personId.values)
    return dict(roc_auc=float(roc_auc_score(te.y, p)), pr_lift=float(pr / prev),
                lift_ci=[float(lo / prev), float(hi / prev)],
                prevalence=float(prev), n=int(len(te)), pos=int(te.y.sum()))


def load(suffix):
    return dataio.load(suffix)


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 96)
    print("  SENSITIVITY TO THE INJURY-TO-GAME MATCHING WINDOW")
    print("=" * 96)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    res = {}
    for w, suffix in WINDOW_SUFFIXES.items():
        try:
            df = load(suffix)
        except (FileNotFoundError, dataio.StaleDatasetError) as e:
            # RuntimeError subclass added 2026-09-18: a superseded parquet
            # for ONE window must skip that arm, exactly as a missing file
            # does, not kill the stage before the primary 1-3 arm runs.
            print(f"[SKIP] window 1-{w}: {e}")
            continue
        # BASE_VARS, not a hand-copied literal. This list was a second copy of
        # the builder's until 2026-09-22, so the rebound-percentage drop of the three
        # rebound-percentage bases would have had to be made here too -- and a
        # miss would have left this stage silently selecting covariates the
        # dataset no longer carries, or (worse, on a stale parquet) covariates
        # every other stage had dropped.
        lagged = [c for c in df.columns
                  if any(c.startswith(f"{v}_") for v in BASE_VARS)]
        workload_bio = lagged + BIO

        tr = df[df.season.isin(TRAIN_SEASONS)]
        te = df[df.season.isin(TEST_SEASONS)]
        te_norm = te[te.rest_days <= 2]
        tr_norm = tr[tr.rest_days <= 2]

        print("=" * 96)
        print(f"  WINDOW 1-{w} CALENDAR DAYS"
              + ("   <- primary specification" if w == 3 else "")
              + ("   <- prespecified sensitivity arm" if w == 2 else ""))
        print("=" * 96)
        print(f"  cohort {len(df):,} exposures, {int(df.y.sum()):,} events "
              f"({df.y.mean():.4%})")
        print(f"  test   {len(te):,} exposures, {int(te.y.sum()):,} events "
              f"({te.y.mean():.4%})\n")

        r = {}
        r["schedule_biometrics"] = fit_eval(tr, te, PARS)
        r["lagged_workload"] = fit_eval(tr, te, workload_bio)
        r["restricted_normal_schedule"] = fit_eval(tr_norm, te_norm, PARS)
        r["rest_days_alone"] = fit_eval(tr, te, ["rest_days"])

        print(f"  {'analysis':38s} | {'ROC-AUC':>8s} | {'PR lift':>8s} | {'95% CI':>14s}")
        print("  " + "-" * 80)
        # The workload block's size is FORMATTED IN, not typed: it was hardcoded
        # as "(110)" and would have gone stale the moment the rebound-percentage removal dropped three
        # bases on 2026-09-22, printing a covariate count three bases too high
        # next to numbers fitted on the real one.
        for k, nm in (("schedule_biometrics", f"Schedule + biometrics ({len(PARS)})"),
                      ("lagged_workload",
                       f"Lagged workload + biometrics ({len(workload_bio)})"),
                      ("rest_days_alone", "Rest days alone (1)"),
                      ("restricted_normal_schedule",
                       "RESTRICTED to normal schedule (17)")):
            v = r[k]
            print(f"  {nm:38s} | {v['roc_auc']:8.4f} | {v['pr_lift']:7.2f}x | "
                  f"[{v['lift_ci'][0]:.2f}, {v['lift_ci'][1]:.2f}]")

        # empirical rest gradient
        te2 = te[te.rest_days.notna()].copy()
        te2["bin"] = pd.cut(te2.rest_days, REST_BINS, labels=REST_LABELS)
        g = te2.groupby("bin", observed=True).agg(n=("y", "size"), pos=("y", "sum"))
        g["rate"] = g.pos / g.n
        print(f"\n  Empirical rest gradient (test seasons):")
        print("    " + "  ".join(f"{lab}: {g.loc[lab,'rate']*100:.3f}%"
                                 for lab in g.index))
        r["rest_gradient"] = {str(k): dict(n=int(v.n), pos=int(v.pos),
                                           rate=float(v.rate))
                              for k, v in g.iterrows()}
        opener = te[te.season_opener == 1]
        r["season_opener"] = dict(n=int(len(opener)), pos=int(opener.y.sum()),
                                  rate=float(opener.y.mean()) if len(opener) else np.nan)
        print()
        res[f"window_{MIN_GAP}_{w}"] = r

    # ---- side-by-side --------------------------------------------------------
    print("=" * 96)
    print("  SIDE BY SIDE")
    print("=" * 96)
    keys = [k for k in WINDOW_KEYS if k in res]
    print(f"  {'analysis':38s} | "
          + " | ".join(f"{k.replace('window_', '').replace('_', '-'):>16s}"
                       for k in keys))
    print("  " + "-" * (40 + 19 * len(keys)))
    for k, nm in (("schedule_biometrics", f"Schedule + biometrics ({N_PARS})"),
                  ("lagged_workload",
                   f"Lagged workload + biometrics ({N_WORKLOAD_BIO})"),
                  ("rest_days_alone", "Rest days alone (1)"),
                  ("restricted_normal_schedule", "RESTRICTED to normal schedule")):
        cells = []
        for kk in keys:
            v = res[kk][k]
            cells.append(f"{v['roc_auc']:.3f} / {v['pr_lift']:.2f}x")
        print(f"  {nm:38s} | " + " | ".join(f"{c:>16s}" for c in cells))
    print("\n  Each cell: ROC-AUC / PR-AUC lift over that window's own test base rate.")
    print("  Narrowing the window discards events and widens intervals; the question")
    print("  is whether the point estimates move.")

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n[EXPORT] {JSON_OUT}")


if __name__ == "__main__":
    main()
