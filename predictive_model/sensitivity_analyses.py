"""
sensitivity_analyses.py
================================================================================
Three analyses that test the study's central claim rather than decorate it.

A. HIGH-ASCERTAINMENT COHORT
   The strongest objection to this study is that incomplete injury recording
   manufactures the availability effect: if early-season injuries are missing
   from the record, then "absent but no recorded injury" athletes pollute the
   healthy class, and `rest_days` looks predictive for a purely clerical reason.
   Ascertainment rises 4.9-fold across the panel (0.38% -> 1.85%).

   The direct rebuttal is to discard the poorly-ascertained seasons and refit.
   If the findings hold on 2017-18 onward -- where recording is materially
   better -- the clerical explanation is much harder to sustain. This analysis
   was promised in build_predictive_dataset.py and never run.

B. POSITIONAL STRATIFICATION OF THE REST GRADIENT
   A biomechanical account predicts that the rest gradient tracks positional
   load, so centers (heavier, more contact, more rebounding) should differ from
   guards. A detected interaction would support that; a null one does not
   establish equal effects across positions and is not evidence for
   any particular mechanism (the availability mechanism was withdrawn). Tested formally by a rest x position
   interaction.

C. WORKLOAD x REST INTERACTION
   The natural reviewer question: does pre-game workload matter once you
   condition on schedule? Perhaps workload only bites for athletes who are
   ALSO poorly rested. Tested as an explicit interaction, and again inside the
   normal-schedule cohort where the availability signal is removed.

All models cluster standard errors on the athlete.

OUTPUT
------
  predictive_model/sensitivity_analyses_output.txt
  predictive_model/sensitivity_analyses.json
"""

import os
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.proportion import proportion_confint

import dataio

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read via dataio.load(), which resolves parquet or gzipped CSV. Hard-coding one
# extension is what silently broke this script when the builder switched to parquet.
TXT_OUT = os.path.join(SCRIPT_DIR, "sensitivity_analyses_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "sensitivity_analyses.json")

from features import BIO, HISTORY, PARS, SCHEDULE

FULL_TRAIN = list(range(2015, 2021))
HIGH_TRAIN = list(range(2017, 2021))   # 2017-18 onward: ascertainment >= ~1%
VAL = 2021
TEST = [2022, 2023]

RNG = np.random.default_rng(42)


def boot_lift_ci(y, p, groups, n=400):
    """Cluster bootstrap over athletes for PR-AUC lift."""
    yv, pv, gv = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(gv)
    rows_of = {g: np.flatnonzero(gv == g) for g in uniq}
    prev = yv.mean()
    vals = np.empty(n)
    for i in range(n):
        drawn = RNG.choice(uniq, len(uniq), replace=True)
        s = np.concatenate([rows_of[g] for g in drawn])
        vals[i] = (average_precision_score(yv[s], pv[s]) / yv[s].mean()
                   if yv[s].sum() else np.nan)
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


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
    lift = average_precision_score(te.y, p) / prev
    lo, hi = boot_lift_ci(te.y.values, p, te.personId.values)
    return dict(roc_auc=float(roc_auc_score(te.y, p)), pr_lift=float(lift),
                lift_ci=[lo, hi], n=int(len(te)), pos=int(te.y.sum()),
                prevalence=float(prev))


def rest_bin(s):
    return pd.cut(s, [0, 2, 4, 100], labels=["1-2 (normal)", "3-4 (short gap)", "5+ (long gap)"])


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 92)
    print("  SENSITIVITY ANALYSES -- ascertainment, position, workload x rest")
    print("=" * 92)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = dataio.load()
    res = {}

    # =====================================================================
    print("=" * 92)
    print("  [A] HIGH-ASCERTAINMENT COHORT -- does the finding survive dropping")
    print("      the poorly-recorded early seasons?")
    print("=" * 92)

    print("\n  Ascertainment by season (recorded injury rate):")
    g = df.groupby("season").agg(rows=("y", "size"), pos=("y", "sum"))
    g["rate_pct"] = (g.pos / g.rows * 100).round(3)
    for s, r in g.iterrows():
        mark = "  <- dropped in high-ascertainment cohort" if s < 2017 else ""
        print(f"    {s}-{str(s+1)[-2:]}  {int(r.rows):>6,} rows  {int(r.pos):>4} inj  "
              f"{r.rate_pct:>6.3f}%{mark}")

    te = df[df.season.isin(TEST)]
    res["A_ascertainment"] = {}

    for label, tr_seasons in (("Full panel (2015-16 on)", FULL_TRAIN),
                              ("High-ascertainment (2017-18 on)", HIGH_TRAIN)):
        tr = df[df.season.isin(tr_seasons)]
        print(f"\n  --- {label} ---")
        print(f"      train: {len(tr):,} rows, {int(tr.y.sum()):,} injuries "
              f"({tr.y.mean():.4%}), seasons {min(tr_seasons)}-{max(tr_seasons)}")

        out = {"train_rows": int(len(tr)), "train_pos": int(tr.y.sum())}
        for sname, cols in (("full", [c for c in df.columns
                                      if c not in ["personId", "Player", "dt",
                                                   "season", "y", "tissue"]]),
                            ("schedule+bio", PARS),
                            ("workload only", [c for c in df.columns
                                               if c not in ["personId", "Player", "dt",
                                                            "season", "y", "tissue"]
                                               + HISTORY + SCHEDULE])):
            r = fit_eval(tr, te, cols)
            out[sname] = r
            print(f"      {sname:16s} ({len(cols):>3d}f)  ROC {r['roc_auc']:.4f}   "
                  f"lift {r['pr_lift']:.2f}x  [{r['lift_ci'][0]:.2f}, {r['lift_ci'][1]:.2f}]")

        # the decisive experiment, refitted
        trn = tr[tr.rest_days <= 2]
        ten = te[te.rest_days <= 2]
        r = fit_eval(trn, ten, PARS)
        out["restricted_normal_schedule"] = r
        print(f"      {'restricted (rest<=2)':16s}        ROC {r['roc_auc']:.4f}   "
              f"lift {r['pr_lift']:.2f}x  [{r['lift_ci'][0]:.2f}, {r['lift_ci'][1]:.2f}]"
              f"   n={r['n']:,}, {r['pos']} injuries")
        res["A_ascertainment"][label] = out

    print("\n  INTERPRETATION: if the availability finding were an artifact of missing")
    print("  early-season injury records, discarding those seasons should weaken it.")

    # =====================================================================
    print("\n" + "=" * 92)
    print("  [B] IS THE REST GRADIENT THE SAME FOR EVERY POSITION?")
    print("=" * 92)
    print("""
  A biomechanical account predicts the gradient tracks positional load, so
  centers should differ from guards. A null interaction does not establish equal
  effects across positions, and it is not evidence for any particular mechanism.
""")

    t = te.copy()
    pos_cols = ["pos_C", "pos_F", "pos_G"]
    t["position"] = np.select([t[c] == 1 for c in pos_cols], ["C", "F", "G"],
                              default="Unstarted")
    t = t[t.position.isin(["C", "F", "G"]) & t.rest_days.notna()].copy()
    t["rb"] = rest_bin(t.rest_days)

    print(f"  {'position':10s} | {'rest band':16s} | {'n':>6s} | {'inj':>4s} | "
          f"{'rate':>7s} | {'95% CI':>16s}")
    print("  " + "-" * 76)
    res["B_position"] = {"rates": {}}
    for pos in ["C", "F", "G"]:
        sub = t[t.position == pos]
        for band in ["1-2 (normal)", "3-4 (short gap)", "5+ (long gap)"]:
            b = sub[sub.rb == band]
            if not len(b):
                continue
            k, n = int(b.y.sum()), len(b)
            lo, hi = proportion_confint(k, n, method="wilson")
            print(f"  {pos:10s} | {band:16s} | {n:>6,} | {k:>4} | {k/n*100:>6.3f}% | "
                  f"[{lo*100:5.2f}%, {hi*100:5.2f}%]")
            res["B_position"]["rates"][f"{pos}|{band}"] = dict(
                n=n, pos=k, rate=float(k / n), ci=[float(lo), float(hi)])
        # the gradient itself: short gap vs normal
        nb = sub[sub.rb == "1-2 (normal)"]
        sg = sub[sub.rb == "3-4 (short gap)"]
        rr = (sg.y.mean() / nb.y.mean()) if nb.y.mean() else np.nan
        print(f"  {'':10s} | {'-> risk ratio (short gap / normal)':50s} {rr:5.2f}x")
        res["B_position"][f"{pos}_risk_ratio"] = float(rr)
        print()

    # formal interaction test
    t["short_gap"] = (t.rb == "3-4 (short gap)").astype(int)
    m0 = smf.logit("y ~ short_gap + C(position)", data=t).fit(disp=0)
    m1 = smf.logit("y ~ short_gap * C(position)", data=t).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    from scipy.stats import chi2
    p_int = float(chi2.sf(lr, m1.df_model - m0.df_model))
    print(f"  Likelihood-ratio test, rest x position interaction:")
    print(f"    LR chi2({int(m1.df_model - m0.df_model)}) = {lr:.3f},  p = {p_int:.4f}")
    res["B_position"]["interaction_lr"] = dict(lr=float(lr), p=p_int)
    if p_int > 0.05:
        # A non-significant test does not establish equal effects, so
        # this branch no longer calls the gradient position-invariant or uses
        # it to choose between mechanisms.
        print("    -> No rest-by-position interaction detected. This does not establish")
        print("       equal effects across positions; no equivalence margin was tested.")
    else:
        print("    -> Significant interaction: the gradient DOES vary by position.")
        print("       This complicates the availability account and must be reported.")

    # =====================================================================
    print("\n" + "=" * 92)
    print("  [C] DOES PRE-GAME WORKLOAD MATTER, CONDITIONAL ON SCHEDULE?")
    print("=" * 92)

    d = df[df.season.isin(TEST)].copy()
    d = d.dropna(subset=["fieldGoalsAttempted_r5", "rest_days", "AGE", "BMI"])
    for c in ["fieldGoalsAttempted_r5", "numMinutes_r5", "rest_days", "AGE", "BMI"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()

    print(f"  Cohort: {len(d):,} athlete-games, {int(d.y.sum())} injuries, "
          f"{d.personId.nunique():,} athletes")
    print("  Logistic models, standard errors clustered on athlete.\n")

    res["C_interaction"] = {}
    for wl in ["fieldGoalsAttempted_r5_z", "numMinutes_r5_z"]:
        f_main = f"y ~ {wl} + rest_days_z + AGE_z + BMI_z"
        f_int = f"y ~ {wl} * rest_days_z + AGE_z + BMI_z"
        mm = smf.logit(f_main, data=d).fit(disp=0, cov_type="cluster",
                                           cov_kwds={"groups": d.personId})
        mi = smf.logit(f_int, data=d).fit(disp=0, cov_type="cluster",
                                          cov_kwds={"groups": d.personId})
        term = f"{wl}:rest_days_z"
        orr = float(np.exp(mi.params[term]))
        ci = mi.conf_int().loc[term]
        p = float(mi.pvalues[term])
        print(f"  --- workload = {wl.replace('_z','')}")
        print(f"    main effect,  workload      OR/SD {np.exp(mm.params[wl]):.3f}  "
              f"p = {mm.pvalues[wl]:.2e}")
        print(f"    main effect,  rest          OR/SD {np.exp(mm.params['rest_days_z']):.3f}  "
              f"p = {mm.pvalues['rest_days_z']:.2e}")
        print(f"    INTERACTION   workload*rest OR    {orr:.3f}  "
              f"[{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]  p = {p:.4f}")
        res["C_interaction"][wl] = dict(
            workload_or=float(np.exp(mm.params[wl])),
            rest_or=float(np.exp(mm.params["rest_days_z"])),
            interaction_or=orr, interaction_ci=[float(np.exp(ci[0])), float(np.exp(ci[1]))],
            interaction_p=p)
        print(f"    -> {'no' if p > 0.05 else 'SIGNIFICANT'} evidence that workload's effect "
              f"depends on how rested the athlete is.\n")

    # Does workload predict anything at all once availability is gone?
    print("  Discrimination from workload ALONE, inside the normal-schedule cohort")
    print("  (availability signal removed by construction):")
    tr = df[df.season.isin(FULL_TRAIN)]
    wl_cols = [c for c in df.columns
               if c not in ["personId", "Player", "dt", "season", "y", "tissue"]
               + HISTORY + SCHEDULE + BIO]
    r_all = fit_eval(tr, te, wl_cols)
    r_norm = fit_eval(tr[tr.rest_days <= 2], te[te.rest_days <= 2], wl_cols)
    print(f"    all test games        ({len(wl_cols)}f)  ROC {r_all['roc_auc']:.4f}  "
          f"lift {r_all['pr_lift']:.2f}x  [{r_all['lift_ci'][0]:.2f}, {r_all['lift_ci'][1]:.2f}]")
    print(f"    normal schedule only  ({len(wl_cols)}f)  ROC {r_norm['roc_auc']:.4f}  "
          f"lift {r_norm['pr_lift']:.2f}x  [{r_norm['lift_ci'][0]:.2f}, {r_norm['lift_ci'][1]:.2f}]")
    res["C_interaction"]["workload_only_all"] = r_all
    res["C_interaction"]["workload_only_normal_schedule"] = r_norm

    # =====================================================================
    print("\n" + "=" * 92)
    print("  [D] DO THE ABLATION CONCLUSIONS DEPEND ON THE ALGORITHM?")
    print("=" * 92)
    print("""
  The feature-group ablations are reported under gradient boosting, while the
  headline model -- selected on validation -- is a random forest. That invites
  the objection that the ablations are carried by a model we declined to report.

  The objection misplaces the bias. Selection bias arises from CHOOSING a model
  by its test performance; it does not contaminate an algorithm. In an ablation
  the algorithm is a fixed instrument applied identically to every covariate
  subset, and the quantity of interest is the DIFFERENCE between subsets.

  But that is an argument, and the question is empirical. So: refit every
  ablation under all three algorithms and see whether the conclusion moves.
""")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    tr = df[df.season.isin(FULL_TRAIN)]
    te = df[df.season.isin(TEST)]
    ytr, yte = tr.y.values, te.y.values
    prev = yte.mean()
    pw = (ytr == 0).sum() / (ytr == 1).sum()
    feature_cols = [c for c in df.columns
                    if c not in ["personId", "Player", "dt", "season", "y", "tissue"]]
    workload = [c for c in feature_cols if c not in HISTORY + SCHEDULE + BIO]

    def build(kind):
        if kind == "LogisticRegression":
            return Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler()),
                             ("c", LogisticRegression(max_iter=2000, C=0.1,
                                                      class_weight="balanced"))])
        if kind == "RandomForest":
            return Pipeline([("i", SimpleImputer(strategy="median")),
                             ("c", RandomForestClassifier(
                                 n_estimators=400, max_depth=10, min_samples_leaf=20,
                                 class_weight="balanced_subsample", n_jobs=-1,
                                 random_state=42))])
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("c", xgb.XGBClassifier(
                             n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                             reg_lambda=2.0, scale_pos_weight=pw, eval_metric="aucpr",
                             n_jobs=-1, random_state=42))])

    sets = {
        "full": feature_cols,
        "no_history": [c for c in feature_cols if c not in HISTORY],
        "schedule+bio": SCHEDULE + BIO,
        "workload+bio": workload + BIO,
        "history+bio": HISTORY + BIO,
    }
    algos = ["XGBoost", "RandomForest", "LogisticRegression"]

    print(f"  {'covariate set':16s} {'n':>4s} | " +
          " | ".join(f"{a:^17s}" for a in algos))
    print("  " + "-" * 76)
    res["D_algorithm"] = {}
    for sname, cols in sets.items():
        cells = []
        for a in algos:
            m = build(a)
            m.fit(tr[cols], ytr)
            p = m.predict_proba(te[cols])[:, 1]
            lift = average_precision_score(yte, p) / prev
            roc = roc_auc_score(yte, p)
            cells.append(f"{roc:.3f} / {lift:.2f}x")
            res["D_algorithm"].setdefault(a, {})[sname] = dict(
                roc_auc=float(roc), pr_lift=float(lift), n_features=len(cols))
        print(f"  {sname:16s} {len(cols):>4d} | " + " | ".join(f"{c:^17s}" for c in cells))
    print("\n  (each cell: ROC-AUC / PR-AUC lift)\n")

    print("  Does 'workload adds nothing beyond schedule' hold under each algorithm?")
    for a in algos:
        d = res["D_algorithm"][a]
        s, w = d["schedule+bio"]["pr_lift"], d["workload+bio"]["pr_lift"]
        print(f"    {a:20s} schedule {s:.2f}x  vs  workload {w:.2f}x  -> "
              f"{'HOLDS' if s > 1.5 * w else 'ATTENUATED'}")

    # A hardcoded paragraph here once named random forest "the selected
    # model" and quoted lifts from a superseded run. The selected model is read
    # from metrics.json, and no mechanism is asserted.
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), encoding="utf-8") as fh:
        selected = json.load(fh)["best_model"]
    print(f"\n  Selected model (metrics.json best_model): {selected}")
    if selected in res["D_algorithm"]:
        d = res["D_algorithm"][selected]
        print(f"    schedule+bio {d['schedule+bio']['pr_lift']:.2f}x  vs  "
              f"workload+bio {d['workload+bio']['pr_lift']:.2f}x  "
              f"vs  full {d['full']['pr_lift']:.2f}x")
    print("  Point estimates only, without intervals: read the per-algorithm contrasts")
    print("  above as a sensitivity check, not as evidence for any mechanism.")

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n[EXPORT] {JSON_OUT}")


if __name__ == "__main__":
    main()
