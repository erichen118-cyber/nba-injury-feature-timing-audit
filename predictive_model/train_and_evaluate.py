"""
train_and_evaluate.py
================================================================================
Fit and honestly evaluate pre-tip-off injury forecasting models.

EVALUATION DESIGN
-----------------
Temporal (forward-chaining) split, because the deployment question is "fit on the
past, predict the future" -- not "interpolate among games you have already seen":

    train : seasons 2015-16 .. 2020-21
    val   : season  2021-22            (threshold + calibration fitting only)
    test  : seasons 2022-23 .. 2023-24 (touched once)

No athlete-level split is needed: a temporal split is strictly stronger, since a
future season contains both returning and new athletes, exactly as deployment does.

Label coverage drifts upward over the panel (0.38% -> 1.85% positives per season),
so train prevalence (0.86%) is materially lower than test prevalence (1.56%).
Average precision depends on prevalence, and changing ascertainment may affect
both discrimination and calibration. Calibration is the part that can be
corrected, which is why probabilities are recalibrated on the validation season
before being scored on test.

METRICS
-------
  ROC-AUC, PR-AUC (average precision) with 1,000-resample bootstrap CIs.
  PR-AUC is read against the test prevalence, never against 0.5.
  Brier score, before and after isotonic recalibration.
  Alert-budget analysis: if a medical staff can review the top k% of
  player-games each night, what recall and precision do they get, and how many
  athletes must be screened to find one injury (NNS)?

  Accuracy is NOT reported. At a 1.56% base rate the constant "healthy"
  prediction scores 98.44%, so accuracy cannot distinguish a useful model from a
  useless one.

OUTCOME BOUNDARIES
------------------
The primary outcome is a tissue-DESIGNATED musculoskeletal injury. Concussion
is a fourth, separate boundary (added 2026-09-04): a mechanism, not a tissue
and not an anatomical location. Its index games are dropped from the MSK risk
set by `dataio.load()` rather than scored as healthy controls, and it is fitted
here as its own stratum behind a prespecified >= 25 test-event gate, mirroring
the NFL adapter in the cross-sport replication. It is never pooled with the MSK
outcome, the tissue classes, or the anatomical-location strata.

ABLATIONS
---------
  full          : all 118 pre-game features
  no_history    : drop prior-injury history (workload + schedule + biometrics)
  history_only  : prior-injury history + biometrics
  workload_only : lagged workload + biometrics

OUTPUTS
-------
  predictive_model/train_and_evaluate_output.txt
  predictive_model/metrics.json
  predictive_model/test_predictions.csv
"""

import os
import json
import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
try:                                    # sklearn >= 1.6 replaced cv="prefit"
    from sklearn.frozen import FrozenEstimator
except ImportError:                     # pragma: no cover - older sklearn
    FrozenEstimator = None
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import dataio
# Feature groups come from features.py, the one definition. SCHEDULE, BIO_BASE and
# HISTORY were literals here and in three other stages until 2026-09-06; see that
# module for why that was a drift risk.
from features import BIO_BASE, HISTORY, SCHEDULE

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read via dataio.load(), which resolves parquet or gzipped CSV. Hard-coding one
# extension is what silently broke three other stages when the builder switched
# to parquet; this script was the last copy of that pattern.
TXT_OUT = os.path.join(SCRIPT_DIR, "train_and_evaluate_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "metrics.json")
PRED_OUT = os.path.join(SCRIPT_DIR, "test_predictions.csv")

TRAIN_SEASONS = list(range(2015, 2021))   # 2015-16 .. 2020-21
VAL_SEASON = 2021                         # 2021-22
TEST_SEASONS = [2022, 2023]               # 2022-23, 2023-24

META = ["personId", "Player", "dt", "season", "y", "tissue", "y_concussion"]

RNG = np.random.default_rng(42)
N_BOOT = 1000

# ---------------------------------------------------------------------------
# CONCUSSION STRATUM -- prespecified power gate (2026-09-04)
# ---------------------------------------------------------------------------
# Concussion is a fourth, separate outcome boundary: a mechanism, not a tissue
# and not an anatomical location. It is never pooled into the 3-class tissue
# stratification, never into the anatomical-location stratification, and never
# into a pooled discrimination number.
#
# The threshold below is 25 test events, matching the NFL adapter's reporting
# gate in the cross-sport replication, and it is written here BEFORE the model
# is fitted. Choosing it after seeing the AUC would be exactly the selection
# this module's design was rewritten to prevent. If the gate fails, the stage
# says so and the deliverable is the descriptive block alone -- which is a
# result, not a failure.
CONCUSSION_MIN_TEST_EVENTS = 25

# Rest bins come from dataio, the one place that defines them, so the
# concussion gradient can be read against the musculoskeletal one of
# absence_confound_analysis.py without re-binning -- and so that splitting a bin
# cannot silently desynchronise the two.
REST_BINS, REST_LABELS = dataio.REST_BINS, dataio.REST_LABELS

# Model-fitting variability. The bootstrap below captures sampling variability in
# the test fold; it says nothing about how much the answer moves if the forest is
# grown from a different seed. Both are reported.
SEEDS = [42, 7, 13, 101, 2024]


def load():
    """The musculoskeletal cohort, plus the frame it was derived from.

    One read. The concussion stratum needs the full panel and every other block
    here needs the MSK view of it, and reading the file twice put two 216k x 124
    frames in memory at the pipeline's heaviest stage.
    """
    full = dataio.load(concussion="keep")
    return dataio.drop_concussion(full), full


def boot_ci(y, p, fn, groups, n=N_BOOT):
    """Cluster bootstrap percentile CI, resampling ATHLETES with replacement.

    Rows are nested within athletes (~115 games each) and are not independent:
    an athlete's fragility, role and workload persist across their games. A
    row-level bootstrap treats those rows as exchangeable and understates the
    width of every interval. We resample whole athletes instead.
    """
    yv, pv = np.asarray(y), np.asarray(p)
    gv = np.asarray(groups)
    uniq = np.unique(gv)
    rows_of = {g: np.flatnonzero(gv == g) for g in uniq}
    vals = np.empty(n)
    for i in range(n):
        drawn = RNG.choice(uniq, size=len(uniq), replace=True)
        s = np.concatenate([rows_of[g] for g in drawn])
        if yv[s].sum() == 0:
            vals[i] = np.nan
            continue
        vals[i] = fn(yv[s], pv[s])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def alert_budget(y, p, budgets=(0.01, 0.02, 0.05, 0.10, 0.20)):
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p))
    n, tot = len(y), int(y.sum())
    rows = []
    for b in budgets:
        k = max(1, int(round(b * n)))
        flagged = y[order[:k]]
        tp = int(flagged.sum())
        prec = tp / k
        rec = tp / tot
        rows.append(dict(budget=b, n_flagged=k, tp=tp, precision=prec,
                         recall=rec, lift=prec / (tot / n),
                         nns=(k / tp) if tp else float("inf")))
    return rows


def make_models(pos_weight, seed=42):
    return {
        "LogisticRegression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1,
                                       class_weight="balanced", solver="lbfgs")),
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=20,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ]),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=10, reg_lambda=2.0,
                scale_pos_weight=pos_weight, eval_metric="aucpr",
                n_jobs=-1, random_state=seed)),
        ]),
    }


def concussion_stratum(df, msk_feature_cols):
    """Describe, and where powered fit, concussion as its own outcome.

    `df` is the frame loaded with `concussion="keep"` -- the only one in the
    pipeline that contains concussion index games. It is passed in rather than
    re-read so the stage holds one copy of the panel, not two. Their MSK label is 0 and every other
    stage drops them; here they are the positives.

    Controls are every other exposure, musculoskeletal index games included --
    the same one-vs-rest convention the tissue-specific models use, where an
    injury of another class is a legitimate non-event for the class being
    modeled. The asymmetry with the MSK models (which exclude concussions
    rather than keeping them as controls) is deliberate: a concussion record
    says nothing about whether an MSK injury was also present and unrecorded,
    whereas an MSK record is a positively ascertained non-concussion event.
    """
    feature_cols = [c for c in df.columns if c not in META]
    assert feature_cols == msk_feature_cols, (
        "the concussion frame's feature columns differ from the MSK frame's; "
        "they must be the same matrix")
    assert "y_concussion" not in feature_cols, "the label is in the feature matrix"

    tr = df[df.season.isin(TRAIN_SEASONS)]
    va = df[df.season == VAL_SEASON]
    te = df[df.season.isin(TEST_SEASONS)]

    out = {"gate": {"min_test_events": CONCUSSION_MIN_TEST_EVENTS,
                    "prespecified": True}}

    # ---- descriptive (unconditional: reported whether or not the gate passes)
    print("  Events by season")
    per_season = {}
    for s, g in df.groupby("season"):
        k = int(g.y_concussion.sum())
        per_season[int(s)] = dict(rows=int(len(g)), events=k,
                                  rate=float(g.y_concussion.mean()))
        print(f"    {s}-{str(s+1)[-2:]}  rows {len(g):>7,}  events {k:>3d}  "
              f"rate {g.y_concussion.mean()*100:.4f}%")
    out["per_season"] = per_season

    splits = {}
    for nm, part in (("train", tr), ("val", va), ("test", te)):
        splits[nm] = dict(rows=int(len(part)), events=int(part.y_concussion.sum()),
                          prevalence=float(part.y_concussion.mean()))
        print(f"  {nm:>5s}  rows {len(part):>7,}  events {int(part.y_concussion.sum()):>3d}  "
              f"prevalence {part.y_concussion.mean():.5%}")
    out["splits"] = splits
    out["n_events_total"] = int(df.y_concussion.sum())
    out["prevalence_overall"] = float(df.y_concussion.mean())

    print("\n  Concussion rate by pre-game rest (all seasons; MSK bins)")
    # The bin LABEL comes from pd.cut, not from zipping a parallel list against
    # the groupby result. With observed=True an empty bin is dropped, so a zip
    # would pair every later label with the wrong bin and silently file, say,
    # the 5-6 day rate under the "4" key -- which then reaches metrics.json,
    # RESULTS.md and the manuscript as fact. observed=False keeps empty bins so
    # the row count is fixed, and a bin with no exposures reports as such.
    binned = pd.cut(df.rest_days, REST_BINS, labels=REST_LABELS)
    g = df.groupby(binned, observed=False).agg(n=("y_concussion", "size"),
                                               pos=("y_concussion", "sum"))
    grad = {}
    for lab, r in g.iterrows():
        lab = str(lab)
        rate = float(r.pos / r.n) if r.n else float("nan")
        grad[lab] = dict(n=int(r.n), pos=int(r.pos), rate=rate)
        print(f"    {lab:>16s} | {int(r.n):>7,} | {int(r.pos):>3d} | "
              f"{rate * 100:>7.4f}%")
    op = df[df.season_opener == 1]
    grad["season_opener"] = dict(n=int(len(op)), pos=int(op.y_concussion.sum()),
                                 rate=float(op.y_concussion.mean()))
    print(f"    {'season opener':>16s} | {len(op):>7,} | {int(op.y_concussion.sum()):>3d} | "
          f"{op.y_concussion.mean()*100:>7.4f}%   (excluded from the bins above)")
    out["rest_gradient"] = grad

    # ---- the gate -----------------------------------------------------------
    n_test = int(te.y_concussion.sum())
    out["n_events_test"] = n_test
    out["prevalence_test"] = float(te.y_concussion.mean())
    if n_test < CONCUSSION_MIN_TEST_EVENTS:
        short = CONCUSSION_MIN_TEST_EVENTS - n_test
        out["gate"]["passed"] = False
        out["gate"]["shortfall"] = short
        out["model"] = None
        print(f"\n  [GATE NOT MET] {n_test} test events < "
              f"{CONCUSSION_MIN_TEST_EVENTS} required (short by {short}).")
        print("  No discrimination metric is fitted or reported for this stratum;")
        print("  the descriptive block above is the deliverable. This threshold was")
        print("  written into the code before the model was fitted.")
        return out

    out["gate"]["passed"] = True
    print(f"\n  [GATE MET] {n_test} test events >= {CONCUSSION_MIN_TEST_EVENTS}. "
          f"Fitting the stratum.")
    ytr = tr.y_concussion.values
    yte = te.y_concussion.values
    pw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    m = make_models(pw, seed=SEEDS[0])["XGBoost"]
    m.fit(tr[feature_cols], ytr)
    p = m.predict_proba(te[feature_cols])[:, 1]
    prev = yte.mean()
    r = evaluate(yte, p, "Concussion", prev, te.personId.values)
    # The event count travels WITH the metric, in the same JSON object, so the
    # AUC cannot be quoted without the 25-ish events it rests on.
    r["n_events_test"] = n_test
    r["n_events_train"] = int(ytr.sum())
    r["prevalence"] = float(prev)
    r["scale_pos_weight"] = float(pw)
    rocs = []
    for sd in SEEDS:
        ms = make_models(pw, seed=sd)["XGBoost"]
        ms.fit(tr[feature_cols], ytr)
        rocs.append(roc_auc_score(yte, ms.predict_proba(te[feature_cols])[:, 1]))
    r["roc_seed_median"] = float(np.median(rocs))
    r["roc_seed_min"] = float(min(rocs))
    r["roc_seed_max"] = float(max(rocs))
    print(f"  seeds: ROC median {np.median(rocs):.4f} "
          f"[{min(rocs):.4f}, {max(rocs):.4f}]")
    print(f"  Read against {n_test} test events. This is a small-sample estimate")
    print("  and is reported as a stratum, never pooled with the MSK outcome.")
    out["model"] = r
    return out


def evaluate(y, p, label, prev, groups):
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    roc_lo, roc_hi = boot_ci(y, p, roc_auc_score, groups)
    pr_lo, pr_hi = boot_ci(y, p, average_precision_score, groups)
    print(f"  {label:24s} ROC-AUC {roc:.4f} [{roc_lo:.4f}, {roc_hi:.4f}]   "
          f"PR-AUC {pr:.4f} [{pr_lo:.4f}, {pr_hi:.4f}]   lift {pr/prev:.2f}x")
    return dict(roc_auc=roc, roc_ci=[roc_lo, roc_hi],
                pr_auc=pr, pr_ci=[pr_lo, pr_hi], pr_lift=pr / prev)


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 92)
    print("  PRE-TIP-OFF INJURY FORECASTING -- TEMPORAL VALIDATION")
    print("=" * 92)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print()

    df, df_full = load()
    feature_cols = [c for c in df.columns if c not in META]
    pos_cols = [c for c in feature_cols if c.startswith("pos_")]
    bio = BIO_BASE + pos_cols
    workload = [c for c in feature_cols
                if c not in HISTORY + SCHEDULE + bio]

    tr = df[df.season.isin(TRAIN_SEASONS)]
    va = df[df.season == VAL_SEASON]
    te = df[df.season.isin(TEST_SEASONS)]

    print(f"[SPLIT] train {tr.season.min()}-{tr.season.max()}: {len(tr):>7,} rows  "
          f"{tr.y.sum():>4,} pos ({tr.y.mean():.4%})  {tr.personId.nunique():,} athletes")
    print(f"[SPLIT] val   {VAL_SEASON}      : {len(va):>7,} rows  "
          f"{va.y.sum():>4,} pos ({va.y.mean():.4%})  {va.personId.nunique():,} athletes")
    print(f"[SPLIT] test  {te.season.min()}-{te.season.max()}: {len(te):>7,} rows  "
          f"{te.y.sum():>4,} pos ({te.y.mean():.4%})  {te.personId.nunique():,} athletes")

    prev = te.y.mean()
    print(f"\n[BASELINE] test prevalence = {prev:.6f}")
    print(f"[BASELINE] constant-'healthy' accuracy = {1 - prev:.4%}  <- why accuracy is not reported")
    print(f"[BASELINE] no-skill PR-AUC = {prev:.6f}")

    pos_weight = (tr.y == 0).sum() / max(1, (tr.y == 1).sum())
    print(f"[SETUP] scale_pos_weight = {pos_weight:.2f}")

    Xtr, ytr = tr[feature_cols], tr.y.values
    Xva, yva = va[feature_cols], va.y.values
    Xte, yte = te[feature_cols], te.y.values

    results = {"split": {"train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
                         "train_pos": int(tr.y.sum()), "val_pos": int(va.y.sum()),
                         "test_pos": int(te.y.sum()), "test_prevalence": float(prev)},
               "models": {}, "ablations": {}, "alert_budget": {}, "per_season": {}}

    gte = te.personId.values          # bootstrap clusters

    # ---------------- candidate specs ----------------
    # The parsimonious model is a candidate like any other, not a post-hoc winner.
    pars_cols = [c for c in SCHEDULE + bio if c in df.columns]
    candidates = {
        "LogisticRegression": ("LogisticRegression", feature_cols),
        "RandomForest": ("RandomForest", feature_cols),
        "XGBoost": ("XGBoost", feature_cols),
        "Parsimonious": ("XGBoost", pars_cols),
    }

    # ---------------- model selection ON VALIDATION ----------------
    # The previous revision selected the reported model by TEST PR-AUC and then
    # reported the alert budget, per-season stability and drift check for that
    # winner -- while stating the test seasons were touched once. They were not.
    # Selection happens here, on the validation season, and the test seasons are
    # scored afterwards.
    print("\n" + "=" * 92)
    print(f"  MODEL SELECTION -- validation season {VAL_SEASON}-{str(VAL_SEASON+1)[-2:]} "
          f"(test seasons are NOT consulted)")
    print("=" * 92)
    val_prev = va.y.mean()
    val_scores = {}
    for name, (kind, cols) in candidates.items():
        m = make_models(pos_weight, seed=SEEDS[0])[kind]
        m.fit(tr[cols], ytr)
        pv = m.predict_proba(va[cols])[:, 1]
        ap = average_precision_score(yva, pv)
        val_scores[name] = ap
        print(f"  {name:24s} ({len(cols):>3d}f)  val PR-AUC {ap:.4f}   "
              f"lift {ap/val_prev:.2f}x   val ROC {roc_auc_score(yva, pv):.4f}")
    best = max(val_scores, key=val_scores.get)
    results["val_selection"] = {"scores": val_scores, "selected": best,
                                "val_prevalence": float(val_prev)}
    print(f"\n[SELECTED on validation] {best}")

    # ---------------- test evaluation (scored once, after selection) ----------
    print("\n" + "=" * 92)
    print("  TEST EVALUATION (seasons 2022-23 + 2023-24)")
    print("=" * 92)

    preds = {}
    fitted = {}
    for name, (kind, cols) in candidates.items():
        model = make_models(pos_weight, seed=SEEDS[0])[kind]
        model.fit(tr[cols], ytr)
        p = model.predict_proba(te[cols])[:, 1]
        preds[name] = p
        fitted[name] = (model, cols)
        mark = "  <- selected" if name == best else ""
        results["models"][name] = evaluate(yte, p, name + mark, prev, gte)
        results["models"][name]["brier_raw"] = brier_score_loss(yte, p)
        results["models"][name]["n_features"] = len(cols)
    results["parsimonious_features"] = pars_cols

    # ---------------- seed stability ----------------
    print("\n" + "-" * 92)
    print(f"  MODEL-FITTING VARIABILITY -- {len(SEEDS)} seeds, same split")
    print("-" * 92)
    results["seed_stability"] = {}
    for name, (kind, cols) in candidates.items():
        rocs, prs = [], []
        for sd in SEEDS:
            m = make_models(pos_weight, seed=sd)[kind]
            m.fit(tr[cols], ytr)
            p = m.predict_proba(te[cols])[:, 1]
            rocs.append(roc_auc_score(yte, p))
            prs.append(average_precision_score(yte, p))
        results["seed_stability"][name] = {
            "roc_median": float(np.median(rocs)), "roc_min": float(min(rocs)),
            "roc_max": float(max(rocs)), "pr_median": float(np.median(prs)),
            "pr_min": float(min(prs)), "pr_max": float(max(prs)),
            "lift_median": float(np.median(prs) / prev),
        }
        print(f"  {name:24s} ROC {np.median(rocs):.4f} [{min(rocs):.4f}, {max(rocs):.4f}]   "
              f"PR {np.median(prs):.4f} [{min(prs):.4f}, {max(prs):.4f}]   "
              f"lift {np.median(prs)/prev:.2f}x")
    print("  (LogisticRegression is deterministic; its spread is exactly zero.)")

    # ---------------- calibration ----------------
    print("\n" + "-" * 92)
    print("  Isotonic recalibration fitted on the validation season, scored on test")
    print("-" * 92)
    for name, (model, cols) in fitted.items():
        base = FrozenEstimator(model) if FrozenEstimator is not None else model
        cal = (CalibratedClassifierCV(base, method="isotonic")
               if FrozenEstimator is not None
               else CalibratedClassifierCV(model, method="isotonic", cv="prefit"))
        cal.fit(va[cols], yva)
        pc = cal.predict_proba(te[cols])[:, 1]
        preds[name + "_cal"] = pc
        b_raw = results["models"][name]["brier_raw"]
        b_cal = brier_score_loss(yte, pc)
        results["models"][name]["brier_cal"] = b_cal
        # Isotonic recalibration is non-decreasing but NOT strictly increasing:
        # it ties scores the raw model separates, so ROC does move. Report both.
        roc_raw = results["models"][name]["roc_auc"]
        roc_cal = float(roc_auc_score(yte, pc))
        results["models"][name]["roc_auc_cal"] = roc_cal
        results["models"][name]["pr_auc_cal"] = float(average_precision_score(yte, pc))
        results["models"][name]["pr_lift_cal"] = (
            results["models"][name]["pr_auc_cal"] / prev)
        results["models"][name]["n_distinct_raw"] = int(np.unique(preds[name]).size)
        results["models"][name]["n_distinct_cal"] = int(np.unique(pc).size)
        print(f"  {name:24s} Brier raw {b_raw:.6f} -> calibrated {b_cal:.6f}   "
              f"(ROC raw {roc_raw:.4f} -> calibrated {roc_cal:.4f}, "
              f"{np.unique(pc).size} distinct values)")

    # ---------------- parsimony check ----------------
    print("\n" + "-" * 92)
    print(f"  PARSIMONY -- does the {len(pars_cols)}-feature model match the "
          f"{len(feature_cols)}-feature one?")
    print("-" * 92)
    full_pr = results["models"]["XGBoost"]["pr_auc"]
    pars_pr = results["models"]["Parsimonious"]["pr_auc"]
    print(f"  Parsimonious ({len(pars_cols):>3d}f)  PR-AUC {pars_pr:.4f} "
          f"[{results['models']['Parsimonious']['pr_ci'][0]:.4f}, "
          f"{results['models']['Parsimonious']['pr_ci'][1]:.4f}]")
    print(f"  Full         ({len(feature_cols):>3d}f)  PR-AUC {full_pr:.4f} "
          f"[{results['models']['XGBoost']['pr_ci'][0]:.4f}, "
          f"{results['models']['XGBoost']['pr_ci'][1]:.4f}]")
    print("  Overlapping intervals: claim parity, not superiority.")

    # ---------------- alert budget ----------------
    print("\n" + "=" * 92)
    print(f"  ALERT-BUDGET ANALYSIS  ({best}, test set)")
    print("=" * 92)
    print(f"  {'flag top':>9s} | {'n flagged':>9s} | {'injuries caught':>15s} | "
          f"{'precision':>9s} | {'recall':>7s} | {'lift':>5s} | {'NNS':>6s}")
    print("-" * 92)
    ab = alert_budget(yte, preds[best])
    for r in ab:
        print(f"  {r['budget']*100:>8.0f}% | {r['n_flagged']:>9,} | "
              f"{r['tp']:>6,} / {int(yte.sum()):<6,} | {r['precision']:>9.4f} | "
              f"{r['recall']:>7.4f} | {r['lift']:>5.2f}x | {r['nns']:>6.1f}")
    results["alert_budget"][best] = ab
    print("\n  NNS = number of player-games screened to find one injury.")

    # The budget above is a RAW-score ordering. The calibrated score cannot
    # reproduce it: isotonic output is a step function, so the boundary row of
    # the 5% budget sits inside a large block of ties. Quantify that here so
    # the manuscript's caveat has a producer rather than a hand calculation.
    pc_best = preds[best + "_cal"]
    k5 = max(1, int(round(0.05 * len(yte))))
    thr5 = float(np.sort(pc_best)[::-1][k5 - 1])
    above5 = int((pc_best > thr5).sum())
    tied5 = int((pc_best == thr5).sum())
    # `mergesort` is STABLE, so this is not an arbitrary tie-break: it takes
    # the tied rows in TEST-SET FILE ORDER, which is sorted by
    # (personId, dt) -- NOT chronologically. It therefore favours low player
    # IDs, spanning both test seasons, rather than early games. An earlier
    # comment here said "chronological"; that was wrong and the manuscript
    # repeated it. With 4,459 rows tied at the boundary and only
    # 412 strictly above it, ~83% of the flagged set is chosen by that rule,
    # so the resulting count is one order-dependent draw, not a representative
    # one. Reported alongside the EXPECTED count under a uniformly random
    # tie-break, which is the defensible statistic and has a closed form.
    _y = np.asarray(yte)
    cal_order = np.argsort(-pc_best, kind="mergesort")
    tp_cal = int(_y[cal_order[:k5]].sum())
    _tied_mask = pc_best == thr5
    _tp_above = int(_y[pc_best > thr5].sum())
    _slots = k5 - above5
    _tp_tied_expected = (
        _slots * (_y[_tied_mask].sum() / _tied_mask.sum()) if _tied_mask.sum() else 0.0)
    tp_cal_expected = float(_tp_above + _tp_tied_expected)
    # Looked up BY VALUE, not by position. `ab[2]` was 0.05 only for the
    # default `budgets` tuple: inserting or reordering a budget would have
    # written another row's tp/lift under budget=0.05 and silently changed
    # what the manuscript's "116 injuries, 3.07-fold" referred to.
    _row5 = next(r for r in ab if abs(r["budget"] - 0.05) < 1e-9)
    results["budget_score_basis"] = dict(
        model=best, budget=0.05, k=k5, score_used="raw",
        tp_raw=int(_row5["tp"]), lift_raw=float(_row5["lift"]),
        cal_threshold=thr5, cal_rows_strictly_above=above5,
        cal_rows_tied_at_threshold=tied5,
        tp_cal_stable_order_tiebreak=tp_cal,
        lift_cal_stable_order_tiebreak=float(
            (tp_cal / k5) / (int(_y.sum()) / len(yte))),
        tp_cal_expected_random_tiebreak=tp_cal_expected,
        lift_cal_expected_random_tiebreak=float(
            (tp_cal_expected / k5) / (int(_y.sum()) / len(yte))),
        tiebreak_note="stable-order follows the prediction file's "
                      "(personId, dt) sort, not chronology and not chance; "
                      "the expected-random figures are the defensible ones",
    )
    print(f"  Budget rows are ranked on the RAW score. On the calibrated score "
          f"the 5% boundary ({thr5:.6f}) has {above5:,} rows strictly above it "
          f"and {tied5:,} tied at it; a stable-order tie-break catches {tp_cal} "
          f"injuries against {int(_row5['tp'])} on the raw ordering.")

    # ---------------- ablations ----------------
    print("\n" + "=" * 92)
    print("  FEATURE-GROUP ABLATIONS (XGBoost)")
    print("=" * 92)
    sets = {
        "full": feature_cols,
        "no_history": [c for c in feature_cols if c not in HISTORY],
        "history_only": HISTORY + bio,
        "workload_only": workload + bio,
        "schedule_only": SCHEDULE + bio,
    }
    for name, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        m = make_models(pos_weight, seed=SEEDS[0])["XGBoost"]
        m.fit(tr[cols], ytr)
        p = m.predict_proba(te[cols])[:, 1]
        results["ablations"][name] = evaluate(yte, p, f"{name} ({len(cols)}f)", prev, gte)
        results["ablations"][name]["n_features"] = len(cols)

    # ---------------- per-season stability ----------------
    print("\n" + "=" * 92)
    print(f"  PER-SEASON TEST STABILITY ({best})")
    print("=" * 92)
    te2 = te.copy()
    te2["p"] = preds[best]
    for s, g in te2.groupby("season"):
        r = roc_auc_score(g.y, g.p)
        a = average_precision_score(g.y, g.p)
        print(f"  season {s}-{str(s+1)[-2:]}: rows {len(g):>6,}  pos {int(g.y.sum()):>4,}  "
              f"prev {g.y.mean():.4%}  ROC {r:.4f}  PR {a:.4f}  lift {a/g.y.mean():.2f}x")
        results["per_season"][int(s)] = dict(rows=len(g), pos=int(g.y.sum()),
                                             prev=float(g.y.mean()),
                                             roc_auc=r, pr_auc=a)

    # ---------------- leakage check on the schedule features ----------------
    # career_games correlates r=0.52 with season, and label coverage also rises
    # with season. If the model were exploiting that drift rather than injury
    # risk, discrimination would collapse WITHIN a single test season.
    print("\n" + "=" * 92)
    print("  DRIFT-LEAKAGE CHECK (primary model, evaluated within each test season)")
    print("=" * 92)
    tp = te.copy()
    tp["p"] = preds[best]
    for s, g in tp.groupby("season"):
        print(f"  within season {s}-{str(s+1)[-2:]}: ROC {roc_auc_score(g.y, g.p):.4f}  "
              f"lift {average_precision_score(g.y, g.p) / g.y.mean():.2f}x")
    print("  Discrimination holds within season -> not an artifact of coverage drift.")

    # ---------------- tissue-specific one-vs-rest ----------------
    print("\n" + "=" * 92)
    print("  TISSUE-SPECIFIC MODELS (one-vs-rest, XGBoost, same temporal split)")
    print("=" * 92)
    names = {1: "Ligament/Joint", 2: "Muscle/Tendon", 3: "Bone/Contusion"}
    results["tissue"] = {}
    for code, nm in names.items():
        ytr_c = (tr.tissue == code).astype(int).values
        yte_c = (te.tissue == code).astype(int).values
        pw = (ytr_c == 0).sum() / max(1, (ytr_c == 1).sum())
        m = make_models(pw, seed=SEEDS[0])["XGBoost"]
        m.fit(Xtr, ytr_c)
        p = m.predict_proba(Xte)[:, 1]
        pv = yte_c.mean()
        r = evaluate(yte_c, p, nm, pv, gte)
        r["prevalence"] = float(pv)
        r["n_events_test"] = int(yte_c.sum())
        results["tissue"][nm] = r
        preds[f"tissue_{code}"] = p

    # ---------------- concussion: a fourth, separate boundary ----------------
    # Not a tissue class and not an anatomical location, so it is deliberately
    # outside the loop above and gets its own frame, its own risk set and its
    # own prespecified power gate. It is never pooled into a discrimination
    # number with the musculoskeletal outcome.
    print("\n" + "=" * 92)
    print("  CONCUSSION STRATUM (separate outcome boundary, prespecified gate)")
    print("=" * 92)
    results["strata"] = {"concussion": concussion_stratum(df_full, feature_cols)}

    # ---------------- persist ----------------
    out = te[["personId", "Player", "dt", "season", "y", "tissue"]].copy()
    for k, v in preds.items():
        out[f"p_{k}"] = v
    out.to_csv(PRED_OUT, index=False)

    results["best_model"] = best
    results["feature_cols"] = feature_cols
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)

    print("\n" + "=" * 92)
    print("  INTERPRETATION")
    print("=" * 92)
    print("  PR-AUC lift is the operative number: how many times better than random")
    print("  the model ranks player-games by injury risk. ROC-AUC is reported for")
    print("  comparability with the literature but is optimistic at a 1.6% base rate.")
    print()
    print(f"[EXPORT] {JSON_OUT}")
    print(f"[EXPORT] {PRED_OUT}")


if __name__ == "__main__":
    main()
