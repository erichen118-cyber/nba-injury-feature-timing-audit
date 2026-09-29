"""
absence_confound_analysis.py
================================================================================
Is the model forecasting NEW injuries, or detecting athletes who are ALREADY
compromised?

WHY THIS SCRIPT EXISTS
----------------------
The parsimonious model (schedule load + biometrics, 12 features) reaches
ROC-AUC 0.726 / PR-AUC lift 3.58x on a temporally held-out test set. Inspecting
its SHAP map shows the dominant feature is `rest_days` -- and the direction is
the opposite of the workload hypothesis:

    rest_days   n        injuries   rate
    1 (b2b)     14,932       98     0.656%
    2           18,567      212     1.142%
    3            5,893      183     3.105%
    4            1,936      133     6.870%
    5+           5,804      127     2.188%

MORE rest precedes MORE injury. Playing a back-to-back is associated with a
THIRD the injury rate of a normally-rested game. Likewise, more games in the
last 7 days predicts LOWER risk (0.667% at 4 games vs 1.892% at 1 game).

This is not "rest causes injury". It is a selection/availability effect: a long
gap since an athlete's last game means they were ABSENT, and athletes are absent
mostly because something is wrong with them. The model, given `rest_days`, learns
"this athlete recently missed games, so an injury report is imminent."

Crucially the gradient PERSISTS among athletes with no injury recorded in the
prior 365 days (0.442% at 1 rest day -> 4.000% at 4). That does not exonerate the
feature: our injury labels are known to be incomplete (coverage rises 4.9x across
the panel), so "no recorded injury" does not mean "no absence". `rest_days` is
therefore best read as a proxy for UNRECORDED absence.

THE TEST
--------
If the model forecasts new injuries, it should retain skill on athletes who are
on a normal schedule and hence not returning from any gap. If it is an absence
detector, restricting to those athletes should collapse its performance.

  Cohort A (all test games)          -- includes returns from absence
  Cohort B (rest_days <= 2 only)     -- normal schedule; no return from a gap

and, orthogonally, four feature sets:

  rest_days alone (1f)
  availability only (4f)   : rest_days, b2b, games_last_7d, games_last_14d
  parsimonious (12f)       : availability + exposure + biometrics
  minus availability (8f)  : parsimonious with the 4 availability features removed

OUTPUTS
-------
  predictive_model/absence_confound_output.txt
  predictive_model/absence_confound.json
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

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import dataio
TXT_OUT = os.path.join(SCRIPT_DIR, "absence_confound_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "absence_confound.json")

from features import AVAIL, BIO, EXPOSURE, PARS

TRAIN_SEASONS = list(range(2015, 2021))
TEST_SEASONS = [2022, 2023]
RNG = np.random.default_rng(42)


def boot_ci(y, p, fn, groups, n=500):
    """Cluster bootstrap: resample ATHLETES, not rows. Rows nest within athletes
    (~115 games each), so a row-level resample understates every interval."""
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
    roc = roc_auc_score(te.y, p)
    pr = average_precision_score(te.y, p)
    lo, hi = boot_ci(te.y.values, p, average_precision_score, te.personId.values)
    return dict(roc_auc=roc, pr_auc=pr, pr_lift=pr / prev,
                lift_ci=[lo / prev, hi / prev], prevalence=float(prev),
                n=len(te), pos=int(te.y.sum()))


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 92)
    print("  ABSENCE CONFOUND ANALYSIS -- forecasting new injury, or detecting old injury?")
    print("=" * 92)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = dataio.load()
    tr = df[df.season.isin(TRAIN_SEASONS)]
    te = df[df.season.isin(TEST_SEASONS)]
    res = {"descriptive": {}, "models": {}}

    # ---- descriptive: the direction of the rest effect ----------------------
    # The rest curve is NOT monotone, and the earlier revision reported only its
    # peak ("0.656% at a back-to-back rising to 6.870% at four days' rest"),
    # stopping before the reversal. The full curve is printed here. Two changes
    # make it readable: rest_days is now differenced within season, so season
    # openers no longer enter the long-rest bins carrying the offseason; and the
    # 5+ bin is split, because it was hiding the turn.
    # From dataio, so train_and_evaluate.py's concussion gradient bins the same
    # way; the manuscript prints the two side by side (§4.6 against §4.10a).
    RB, LAB = dataio.REST_BINS, dataio.REST_LABELS

    def curve(sub, tag, store=None):
        g = (sub.groupby(pd.cut(sub.rest_days, RB), observed=True)
                .agg(n=("y", "size"), pos=("y", "sum")))
        print(f"  {'rest days':>16s} | {'n':>7s} | {'injuries':>8s} | {'rate':>7s}")
        print("  " + "-" * 50)
        for l, (_, r) in zip(LAB, g.iterrows()):
            print(f"  {l:>16s} | {r.n:>7,} | {int(r.pos):>8,} | {r.pos/r.n*100:>6.3f}%")
            if store is not None:
                store[l] = dict(n=int(r.n), pos=int(r.pos), rate=float(r.pos / r.n))
        op = sub[sub.season_opener == 1]
        if len(op):
            print(f"  {'season opener':>16s} | {len(op):>7,} | {int(op.y.sum()):>8,} | "
                  f"{op.y.mean()*100:>6.3f}%   (excluded from the curve above)")
            if store is not None:
                store["season_opener"] = dict(n=int(len(op)), pos=int(op.y.sum()),
                                              rate=float(op.y.mean()))

    print("[1] Empirical injury rate by pre-game rest (TEST seasons)")
    res["descriptive"]["rest_days"] = {}
    curve(te, "all", res["descriptive"]["rest_days"])
    # 2026-09-26 review: this block used to assert a fixed curve shape
    # ("climbs through four days") and an absence mechanism (held out and eased
    # back / cleared to return). The shape was contradicted by the rates printed
    # directly above it, and no source field records why an athlete missed a
    # game. The rates are now reported as observed, with no mechanism attached.
    print("\n  Descriptive rates only. The source records no reason for an athlete's")
    print("  gap (symptoms, clearance, coach's decision), so this curve does not")
    print("  identify a mechanism. The season-opener row is the athlete's first")
    print("  appearance and sits outside the curve; the opener NEGATIVE CONTROL")
    print("  (team's first game) is in negative_control_analyses.py.\n")

    # Panel-wide versions of the same curve, added 2026-08-17.
    #
    # The test-season curve is what accompanies the model evaluation, but it is
    # thin in the tails: once rest_days counts calendar days, the back-to-back
    # cell holds only 10 events in 6,811 exposures. That is the same order of
    # fragility that forced the season-opener claim to be withdrawn, so the
    # descriptive gradient should not be *carried* by it. A rest gradient is a
    # property of the exposure, not of the held-out split -- there is no leakage
    # in describing every season -- so the well-powered panels are computed here
    # and are what the manuscript should quote. The test-season curve stays, as
    # the version that matches the evaluation cohort.
    print("[1b] Same curve, FULL PANEL (2015-16 on) -- well-powered")
    res["descriptive"]["rest_days_full_panel"] = {}
    curve(df, "full", res["descriptive"]["rest_days_full_panel"])

    print("\n[1c] Same curve, HIGH-ASCERTAINMENT PANEL (2017-18 on)")
    res["descriptive"]["rest_days_high_ascertainment"] = {}
    curve(df[df.season >= 2017], "highasc",
          res["descriptive"]["rest_days_high_ascertainment"])
    print()

    print("[2] Same curve among athletes with NO injury recorded in the prior 365 days")
    clean = te[te.injuries_last_365d == 0]
    res["descriptive"]["rest_days_no_recorded_injury"] = {}
    curve(clean, "clean", res["descriptive"]["rest_days_no_recorded_injury"])
    print("\n  Because label coverage is incomplete, 'no recorded injury' does not mean")
    print("  'no absence'; compare these rates with [1] directly.\n")

    # ---- the decisive experiment -------------------------------------------
    sets = {
        f"rest_days alone (1f)": ["rest_days"],
        f"availability only ({len(AVAIL)}f)": AVAIL,
        f"parsimonious ({len(PARS)}f)": PARS,
        f"minus availability ({len(EXPOSURE + BIO)}f)": EXPOSURE + BIO,
    }
    cohorts = {
        "A. all test games": (tr, te),
        "B. normal schedule (rest<=2)": (tr[tr.rest_days <= 2], te[te.rest_days <= 2]),
    }
    KEY_PARS = f"parsimonious ({len(PARS)}f)"
    KEY_REST = "rest_days alone (1f)"

    print("=" * 92)
    print("  [3] FEATURE SETS, ALL GAMES VS NORMAL SCHEDULE")
    print("=" * 92)
    for cname, (trc, tec) in cohorts.items():
        print(f"\n  {cname}   (n={len(tec):,}, injuries={int(tec.y.sum())}, prevalence={tec.y.mean():.4%})")
        print(f"    {'feature set':26s} | {'ROC-AUC':>8s} | {'PR lift':>8s} | {'95% CI':>16s}")
        print("    " + "-" * 70)
        res["models"][cname] = {}
        for sname, cols in sets.items():
            r = fit_eval(trc, tec, cols)
            res["models"][cname][sname] = r
            print(f"    {sname:26s} | {r['roc_auc']:8.4f} | {r['pr_lift']:7.2f}x | "
                  f"[{r['lift_ci'][0]:.2f}, {r['lift_ci'][1]:.2f}]")

    a = res["models"]["A. all test games"][KEY_PARS]
    b = res["models"]["B. normal schedule (rest<=2)"][KEY_PARS]
    ra = res["models"]["A. all test games"][KEY_REST]

    print("\n" + "=" * 92)
    print("  CONCLUSION")
    print("=" * 92)
    # This block used to assert an availability ("absence detector")
    # mechanism. The manuscript withdrew it -- the index-game contrast cannot
    # identify an injury-onset or collider mechanism -- so only measured
    # quantities are printed here; interpretation belongs to the manuscript.
    print(f"  rest_days alone:            ROC {ra['roc_auc']:.4f} / lift {ra['pr_lift']:.2f}x")
    print(f"  {len(PARS)}-feature model, all games: ROC {a['roc_auc']:.4f} / lift {a['pr_lift']:.2f}x "
          f"(95% CI [{a['lift_ci'][0]:.2f}, {a['lift_ci'][1]:.2f}])")
    print(f"  same model, rest_days <= 2: ROC {b['roc_auc']:.4f} / lift {b['pr_lift']:.2f}x "
          f"(95% CI [{b['lift_ci'][0]:.2f}, {b['lift_ci'][1]:.2f}])")
    print()
    print("  These contrasts are descriptive. They do not identify why discrimination")
    print("  differs between cohorts, and must not be read as an absence or availability")
    print("  mechanism (manuscript S4.2).")

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n[EXPORT] {JSON_OUT}")


if __name__ == "__main__":
    main()
