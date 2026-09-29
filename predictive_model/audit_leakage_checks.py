"""audit_leakage_checks.py
================================================================================
ml-research-audit experiments for the NBA injury forecasting pipeline.

Two checks the existing pipeline does not perform:

  A. LABEL-PERMUTATION NEGATIVE CONTROL (checklist item 6).
     negative_control_analyses.py contains domain negative controls (exogenous
     vs endogenous gaps) but no shuffled-label control. If any leakage survives
     in the split/preprocessing machinery, a model fit to permuted labels will
     score above chance. Expectation: ROC-AUC ~= 0.50, lift ~= 1.0.

     SCOPE -- DO NOT OVERCLAIM THIS. It detects PROCEDURAL leakage: preprocessing
     fitted across the split, rows duplicated between train and test, a grouping
     variable that lets an athlete teach the model its own future. It does NOT
     detect FEATURE-CONSTRUCTION leakage -- the class of defect this project
     actually suffered. A covariate computed from the outcome game stops
     corresponding to its own row's label once labels are permuted, so the model
     cannot exploit it and the score collapses to chance either way. This control
     would have returned PASS on the original leaked models. Feature timing is
     audit item 1 and is established by reading build_predictive_dataset.py, not
     by this script.

     The permutation is GLOBAL -- labels are shuffled across the whole panel
     before the temporal split -- so the permuted test fold does not carry the
     real test prevalence: the panel-wide rate is below the test-season rate, so
     the permuted fold holds fewer positives at a lower prevalence than the real
     757 at 1.55%. AUC is invariant to prevalence and lift is referenced to the
     permuted fold's own base rate, so the verdict stands; but any null standard
     error computed for this control must use the PERMUTED counts, not the real
     ones.

     The PASS band is +/-`PASS_BAND_SE` Hanley-McNeil standard errors on the
     permuted split. The multiplier was chosen by eye, not derived; it is
     documented so it is not mistaken for a calculated threshold.

     **The permuted counts, the SE and the band are NOT restated here.** They
     move with the cohort -- the concussion exclusion and the playerteamId
     repair both shifted them -- and a hard-coded copy in this docstring was
     stale within a day of being written, claiming 503 positives and
     SE 0.01294 when the JSON said 526 and 0.012656. Read them from
     `audit_leakage_checks.json` -> `permutation`, which is written by this
     script on every run. See lines below on the mirror check for the same rule
     and the same reason.

  B. POSITION COVARIATE SENSITIVITY (checklist item 1).
     `position` WAS the career-modal startingPosition over the whole panel, so
     pos_C/pos_F/pos_G/pos_NeverStarted used games after the index game. Since
     2026-09-18 it is a prior-only tally and the covariate is
     admissible, so this arm is no longer a leakage disclosure. It is kept as
     an ordinary ablation: the 4 position dummies are 4 of 123 covariates and 4
     of the 17 in the parsimonious model, and refitting without them measures
     how much the reported numbers rest on them.

Mirrors train_and_evaluate.py exactly: same split, same hyperparameters, same
seed. Bootstrap intervals are skipped -- point estimates are what is at issue.

Validation that the mirror is faithful: the "with pos_*" arms of experiment B
must reproduce the published random-forest values exactly --
`metrics.json` -> models/RandomForest (123 covariates) and
`sensitivity_analyses.json` -> D_algorithm/RandomForest/schedule+bio (17).
Those two agreements are CHECKED AT RUN TIME below and written to the JSON as
`mirror_check`; the script prints a loud warning if either drifts. Do not
restate the values here -- a hard-coded pair in this docstring went stale at the
2026-08-17 rest_days rebuild and still claimed 0.7334 / 3.13x afterwards, which
is the exact failure `RESULTS.md` exists to prevent.

OUTPUTS
-------
  predictive_model/audit_leakage_checks_output.txt
  predictive_model/audit_leakage_checks.json
"""
import os
import sys
import json
import datetime
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

import dataio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(SCRIPT_DIR, "audit_leakage_checks_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "audit_leakage_checks.json")
METRICS_JSON = os.path.join(SCRIPT_DIR, "metrics.json")
SENS_JSON = os.path.join(SCRIPT_DIR, "sensitivity_analyses.json")

# PASS band for the permuted-label control, in Hanley-McNeil standard errors at
# AUC = 0.5 computed on the PERMUTED fold's own counts. The band was historically
# the literal interval 0.47-0.53, chosen by eye; expressing it in SE keeps it
# valid if the permuted positive count moves. 2.32 SE reproduces 0.47-0.53 at the
# counts this panel actually yields.
PASS_BAND_SE = 2.32



META = ["personId", "Player", "dt", "season", "y", "tissue", "y_concussion"]
# This script exists to mirror train_and_evaluate.py's specification, and warns in
# its own output when it stops mirroring it. Until 2026-09-06 it enforced that by
# carrying a hand-kept copy of these lists -- i.e. the mirror was maintained by the
# same mechanism whose failure it was built to detect. They now come from
# features.py, so the two cannot silently disagree about which covariates exist.
from features import BIO_BASE, SCHEDULE
TRAIN_SEASONS = list(range(2015, 2021))
VAL_SEASON = 2021
TEST_SEASONS = [2022, 2023]
SEED = 42


def rf(seed=SEED):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=10,
                                       min_samples_leaf=20,
                                       class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=seed))])


def xgbm(pos_weight, seed=SEED):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=10, reg_lambda=2.0,
                                  scale_pos_weight=pos_weight, eval_metric="aucpr",
                                  n_jobs=-1, random_state=seed))])


def score(model, tr, te, cols, ycol="y"):
    model.fit(tr[cols], tr[ycol].values)
    p = model.predict_proba(te[cols])[:, 1]
    prev = te[ycol].mean()
    return (roc_auc_score(te[ycol].values, p),
            average_precision_score(te[ycol].values, p) / prev)


def hanley_mcneil_se(n_pos, n_neg, auc=0.5):
    """Standard error of an AUC under the Hanley-McNeil formula.

    At auc = 0.5 this is the null SE, which is what the PASS band is built on.
    It must be computed on the PERMUTED fold's counts, not the real split's:
    the global permutation moves positives out of the test seasons, so the real
    757 / 47,927 understates the SE by about a fifth.
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(var)


def published(path, keys):
    """Read a published value out of a pipeline JSON, or None if absent."""
    try:
        with open(path, encoding="utf-8") as fh:
            node = json.load(fh)
    except (OSError, ValueError):
        return None
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def main():
    """Run the audit and RETURN AN EXIT CODE.

    Until 2026-09-18 this returned None and `__main__` did not `sys.exit()`, so
    the stage exited 0 whatever it found: a permutation result out of band, or a
    mirror check disagreeing with `train_and_evaluate.py`, printed a warning and
    passed. `run_pipeline.py` also suppresses captured output on exit 0, so
    nobody saw the warning either. Found in review; verified 2026-09-18
    that every flag currently passes, so making the stage able to fail does not
    break the pipeline today.

    The verdict is computed INSIDE the function `run_stage` calls, so it reaches
    the tracked log. It used to be computed after `tee.file.close()` and printed
    only to `tee.terminal`, which meant a failing run left
    `audit_leakage_checks_output.txt` ending at "[EXPORT] ..." exactly like a
    passing one. That was fixed once and reintroduced one layer out on the same
    day; `run_stage` removes the opportunity.
    """
    return dataio.run_stage(_audit, TXT_OUT)


def _audit():
    results = _run()
    failures = []
    bad_mirror = sorted(k for k, m in results["mirror_check"].items()
                        if not m["agrees"])
    if bad_mirror:
        failures.append(
            "this script no longer mirrors train_and_evaluate.py, or the "
            f"published JSON is stale, for: {bad_mirror}")
    bad_perm = sorted(k for k, m in results["permutation"]["models"].items()
                      if not m["pass"])
    if bad_perm:
        failures.append(
            "permuted-label discrimination is outside the prespecified "
            f"band for: {bad_perm}")
    print()
    if not failures:
        print("  PASS -- permutation in band and the mirror still holds.")
        return None
    return 1, "FAIL: " + "; ".join(failures)

def _run():
    df = dataio.load()
    feature_cols = [c for c in df.columns if c not in META]
    pos_cols = [c for c in feature_cols if c.startswith("pos_")]
    bio = BIO_BASE + pos_cols
    parsimonious = SCHEDULE + bio

    tr = df[df.season.isin(TRAIN_SEASONS)]
    te = df[df.season.isin(TEST_SEASONS)]
    pw = (tr.y == 0).sum() / max(1, (tr.y == 1).sum())

    results = {
        "run": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "seed": SEED,
        "n_covariates": len(feature_cols),
        "split": {"train_rows": len(tr), "test_rows": len(te)},
    }

    print("=" * 78)
    print("  A. LABEL-PERMUTATION NEGATIVE CONTROL")
    print("=" * 78)
    print(f"  {len(feature_cols)} covariates, same temporal split, seed {SEED}.")
    print("  Labels permuted across the whole panel, destroying any real signal.")
    print("  A leak-free pipeline must collapse to ROC-AUC ~0.50, lift ~1.0.\n")

    rng = np.random.default_rng(SEED)
    d2 = df.copy()
    d2["y_perm"] = rng.permutation(d2["y"].values)
    tr2 = d2[d2.season.isin(TRAIN_SEASONS)]
    te2 = d2[d2.season.isin(TEST_SEASONS)]
    pw2 = (tr2.y_perm == 0).sum() / max(1, (tr2.y_perm == 1).sum())

    n_pos = int(te2.y_perm.sum())
    n_neg = int(len(te2) - n_pos)
    se_perm = hanley_mcneil_se(n_pos, n_neg)
    lo, hi = 0.5 - PASS_BAND_SE * se_perm, 0.5 + PASS_BAND_SE * se_perm
    print(f"  Permuted test fold: {n_pos:,} positives / {n_neg:,} negatives "
          f"({n_pos / len(te2):.2%}).")
    print(f"  Null SE (Hanley-McNeil, permuted counts) {se_perm:.5f}; "
          f"PASS band +/-{PASS_BAND_SE} SE = {lo:.4f}-{hi:.4f}.\n")

    results["permutation"] = {
        "n_pos_permuted": n_pos, "n_neg_permuted": n_neg,
        "prevalence_permuted": float(n_pos / len(te2)),
        "null_se": se_perm, "pass_band_se": PASS_BAND_SE,
        "pass_band": [lo, hi], "models": {},
    }
    for name, m in (("RandomForest", rf()), ("XGBoost", xgbm(pw2))):
        auc, lift = score(m, tr2, te2, feature_cols, ycol="y_perm")
        n_se = (auc - 0.5) / se_perm
        ok = lo <= auc <= hi
        flag = "PASS" if ok else "*** INVESTIGATE ***"
        print(f"  {name:16s} permuted-label ROC-AUC {auc:.4f}   lift {lift:.2f}x   "
              f"{abs(n_se):.2f} SE {'above' if n_se >= 0 else 'below'} chance   {flag}")
        results["permutation"]["models"][name] = {
            "roc_auc": float(auc), "lift": float(lift),
            "n_se_from_chance": float(n_se), "pass": bool(ok),
        }

    print()
    print("=" * 78)
    print("  B. POSITION COVARIATE SENSITIVITY")
    print("=" * 78)
    print(f"  pos_* covariates: {pos_cols}")
    # This printed "Career-modal position uses games AFTER the index game
    # (17.25% of rows differ...)" until 2026-09-18. That stopped being true at
    # the career-modal position fix, when position became a prior-only tally, but the sentence kept
    # printing into the TRACKED log, where a reader would conclude the covariate
    # still leaks -- while this module's own docstring said the opposite and
    # RESULTS.md carried the corrected version. Found in review.
    # The two covariate counts are FORMATTED IN, not typed. They read "4 of 123"
    # and "4 of the 17" until 2026-09-22, when the rebound-percentage removal dropped 18 covariates --
    # and this sentence prints into the TRACKED log, which is exactly how the
    # superseded position-leakage claim survived here before.
    print("  Position is a prior-only tally since 2026-09-18, so this arm is an")
    print("  ordinary ablation, NOT a leakage disclosure: the 4 pos_* dummies")
    print(f"  are 4 of {len(feature_cols)} covariates and 4 of the "
          f"{len(parsimonious)} parsimonious ones.")
    print("  Refitting without them measures how much rests on them.\n")

    # Where each arm's "with pos_*" value must reproduce a published number.
    MIRRORS = {
        "full": (METRICS_JSON, ["models", "RandomForest", "roc_auc"],
                 "metrics.json models/RandomForest/roc_auc"),
        "parsimonious": (SENS_JSON,
                         ["D_algorithm", "RandomForest", "schedule+bio", "roc_auc"],
                         "sensitivity_analyses.json D_algorithm/RandomForest/schedule+bio"),
    }
    results["position_sensitivity"] = {}
    results["mirror_check"] = {}
    for key, label, cols in (("full", f"full ({len(feature_cols)})", feature_cols),
                             ("parsimonious",
                              f"parsimonious ({len(parsimonious)})", parsimonious)):
        keep = [c for c in cols if c not in pos_cols]
        a_with, l_with = score(rf(), tr, te, cols)
        a_wo, l_wo = score(rf(), tr, te, keep)
        print(f"  RandomForest {label:20s}")
        print(f"    with pos_*    ({len(cols):>3} cov)  ROC-AUC {a_with:.4f}  lift {l_with:.2f}x")
        print(f"    without pos_* ({len(keep):>3} cov)  ROC-AUC {a_wo:.4f}  lift {l_wo:.2f}x")
        print(f"    delta                       {a_wo - a_with:+.4f}       {l_wo - l_with:+.2f}x\n")
        results["position_sensitivity"][key] = {
            "n_cov_with": len(cols), "n_cov_without": len(keep),
            "roc_auc_with": float(a_with), "lift_with": float(l_with),
            "roc_auc_without": float(a_wo), "lift_without": float(l_wo),
            "delta_roc_auc": float(a_wo - a_with), "delta_lift": float(l_wo - l_with),
        }
        path, keys, where = MIRRORS[key]
        ref = published(path, keys)
        agrees = ref is not None and abs(float(ref) - a_with) < 5e-5
        results["mirror_check"][key] = {
            "source": where, "published": None if ref is None else float(ref),
            "recomputed": float(a_with), "agrees": bool(agrees),
        }

    # ---- mirror check -------------------------------------------------------
    # This script re-implements train_and_evaluate.py's fitting path. If that
    # copy ever drifts, every number above is evidence about the wrong pipeline.
    # The check is RUN rather than asserted in a comment: a hard-coded pair of
    # expected values in the module docstring went stale at the 2026-08-17
    # rest_days rebuild and kept claiming agreement afterwards.
    print("=" * 78)
    print("  MIRROR CHECK -- does this script still reproduce the published fits?")
    print("=" * 78)
    for key, m in results["mirror_check"].items():
        if m["published"] is None:
            print(f"  {key:14s} *** published value not readable at {m['source']}")
        elif m["agrees"]:
            print(f"  {key:14s} recomputed {m['recomputed']:.4f} == published "
                  f"{m['published']:.4f}  ({m['source']})")
        else:
            print(f"  {key:14s} *** DRIFT: recomputed {m['recomputed']:.4f} vs "
                  f"published {m['published']:.4f} at {m['source']}")
    if not all(m["agrees"] for m in results["mirror_check"].values()):
        print()
        print("  *** This script no longer mirrors train_and_evaluate.py, OR the")
        print("  *** published JSON is stale. Resolve before citing anything above.")
    print()

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"[EXPORT] {JSON_OUT}")
    return results


if __name__ == "__main__":
    sys.exit(main())
