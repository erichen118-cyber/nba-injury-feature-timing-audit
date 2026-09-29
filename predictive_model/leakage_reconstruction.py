"""
leakage_reconstruction.py
================================================================================
A 2x2 experiment that separates the two axes on which the historical leaked
result (ROC-AUC 0.7891) differs from the corrected one (ROC-AUC 0.6743).

WHY THIS STAGE EXISTS
---------------------
The study's settled claim states that correcting leakage "reduced discrimination
from ROC-AUC 0.789 to 0.674". That comparison is confounded. The original model,
retained in the authors' working repository,
used `train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)` -- a
RANDOM STRATIFIED split of player-games. The corrected analysis uses a temporal
split. So 0.7891 and 0.6743 differ on at least three axes, not the two the claim
names:

    (a) same-game features   -- the leaked model used index-game box scores
    (b) label-filtered history -- a builder-ordering defect
    (c) split design         -- random stratified vs temporal

For a paper whose contribution is an audit apparatus, an unattributable headline
comparison is the exact defect it criticizes in others. This stage measures (a)
and (c) separately, holding everything else fixed. It does NOT estimate (b) --
see the checklist below.

THE DESIGN (pre-decided 2026-09-20; not searched)
-------------------------------------------------
Four cells, one analytic cohort, one estimator, one seed:

                         | temporal split      | random stratified split
    ---------------------+---------------------+-------------------------
    corrected features   | cell 1 (= headline) | cell 3
    + same-game columns  | cell 2              | cell 4

  * Same-game columns are ADDED to the corrected feature set, never substituted
    for it. Substituting would measure a feature-set swap, not feature timing.
  * The random-split arm is a HISTORICAL STRESS TEST, explicitly labeled as
    such. A cell landing near 0.7891 does NOT establish reconstruction fidelity:
    many pipelines produce similar AUCs on this cohort.
  * The estimator, hyperparameters and seed are the headline model's, imported
    from `train_and_evaluate.make_models` so they cannot drift apart.

THE DISCREPANCY CHECKLIST -- written BEFORE the first run
---------------------------------------------------------
Any residual gap between cell 4 and the historical 0.7891 may come from the
axes below. They are enumerated here, in the commit that precedes the first
execution, so that the reported gap is read against a fixed list rather than
against whichever explanation the numbers later suggest.

  1. DATA AND COHORT VERSION. The historical model ran on every master row with
     a non-null `Is_Injured` (no Regular-Season filter, no `numMinutes > 0`
     filter, no >= 5 prior-in-panel-games filter, concussion index games
     retained). This stage runs on the corrected analytic cohort throughout, in
     all four cells. Cohort is held FIXED here and therefore is NOT measured.
  2. OUTCOME DEFINITION. The historical label was `Is_Injured`. The present
     label `y` is a tissue-DESIGNATED musculoskeletal injury in calendar days
     1-3 after the index game, with concussion index games removed. Different
     outcomes. Held fixed at the present definition in all four cells.
  3. PLAYING-HISTORY CONSTRUCTION (defect (b)). The historical builder
     computed playing history on a label-filtered panel. This stage uses the
     corrected three-stage builder in all four cells. Reconstructing the defect
     faithfully is out of proportion to what it would buy, so (b) receives NO
     estimated effect and remains a documented audit finding reported without an
     AUC attribution.
  4. FEATURE SPECIFICATION. The historical whitelist was 86 features of which
     ~89 master columns were index-game measurements (see SAME_GAME below); it
     also carried `numMinutes_Acute/_Chronic/_Spike` and the `fieldGoalsAttempted`
     and `reboundsTotal` equivalents. Those rolling windows are EXCLUDED from
     SAME_GAME: the corrected panel already carries shifted equivalents
     (`_r3`, `_r5`, `_acr`), so including the master's would confound a
     feature-set change with a timing change. SAME_GAME is index-game
     measurements only.
  5. MODEL SPECIFICATION. The historical model was a RandomForest
     (n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5,
     class_weight="balanced"). This stage uses the present headline estimator in
     all four cells, because the question is about feature timing and split
     design, not about estimator choice. Estimator is therefore a KNOWN,
     UNMEASURED axis of the residual gap.
  6. SPLIT MEMBERSHIP. The historical random split was drawn over its own
     (larger, differently filtered) row set, so its exact test rows cannot be
     recovered. Cells 3 and 4 redraw it over the present cohort with the same
     call signature and seed.
  7. EXECUTION AND SCORING. The historical AUC came from a single fit with no
     interval. Cells here report cluster-bootstrap intervals and a paired
     difference, so a small nominal gap may be within noise that the historical
     number never quantified.
  8. TRAINING-FOLD COMPOSITION -- ADDED 2026-09-20, AFTER THE FIRST RUN, IN
     REVIEW. The two split designs do not train on the same rows, and this
     was not disclosed in the original seven items. The temporal arm trains on
     2015-16..2020-21 only (142,572 rows) and never sees the 2021-22 validation
     season, because that season is reserved for model selection upstream. The
     random arm draws 75% of the WHOLE cohort (163,089 rows as of 2026-09-28), so it trains on
     14% more data, including the held-out validation season, and its test fold
     has a different prevalence (1.091% against 1.547%). The split-design
     contrast below therefore bundles training-set size and composition with
     split design proper, and is NOT a clean estimate of split design alone.
     It is left this way deliberately: cells 3 and 4 exist to reproduce the
     historical call signature, which drew over everything, and changing them
     to match the temporal arm's training rows would stop them being that.
     Whether the confound runs WITH or AGAINST the observed difference depends
     on the sign of that difference, so the stage computes and prints it rather
     than asserting it here. (Until 2026-09-28 this paragraph said the random
     arm "still scores LOWER", which the regular-season rerun made false: it now scores
     higher, in the same direction as its larger training set.)

NO SEED, HYPERPARAMETER OR SETTING MAY BE SEARCHED IN ORDER TO MAKE A CELL
APPROACH 0.7891. An unexplained gap is reported as unexplained.

OUTPUTS
-------
  predictive_model/leakage_reconstruction.json
  predictive_model/leakage_reconstruction_output.txt
"""

import datetime
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import dataio
from train_and_evaluate import (
    META, SEEDS, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASON, make_models,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MASTER = os.path.join(REPO_ROOT, "Data", "master_model_ready_data_corrected.csv")
TXT_OUT = os.path.join(SCRIPT_DIR, "leakage_reconstruction_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "leakage_reconstruction.json")

# The headline model is READ from metrics.json, never hardcoded. The validation
# margin between the leading candidates is 0.0018 PR-AUC and the selection has
# flipped repeatedly, so a literal here would silently compare the wrong model
# against itself after a rerun: the anchor below would still pass at 1e-6
# because both sides would read the same entry, while cell 1 had quietly stopped
# being the headline. Resolved at import so the failure is a startup error.
def _headline_model():
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    name = m.get("best_model") or m["val_selection"]["selected"]
    if name not in make_models(1.0, seed=0):
        raise SystemExit(
            f"metrics.json selected {name!r}, which make_models() does not "
            "build. The reconstruction cannot anchor cell 1 to a model it "
            "cannot construct -- add it to make_models or rerun the pipeline.")
    return name


HEADLINE_MODEL = _headline_model()

# The historical random split, reproduced by call signature rather than by
# recovering its row membership (checklist item 6).
RANDOM_SPLIT_KWARGS = dict(test_size=0.25, random_state=42)

N_BOOT = 1000
BOOT_SEED = 42

# Index-game measurements from the historical 86-feature whitelist. These are
# exactly the master columns that describe what happened IN the game whose
# injury is being predicted -- the feature-timing defect, isolated. Biometrics,
# starting position, prior-injury counts and the _Acute/_Chronic/_Spike rolling
# windows are deliberately excluded (checklist item 4).
SAME_GAME = [
    "numMinutes", "points", "assists", "reboundsTotal", "reboundsOffensive",
    "reboundsDefensive", "fieldGoalsMade", "fieldGoalsAttempted",
    "threePointersMade", "threePointersAttempted", "freeThrowsMade",
    "freeThrowsAttempted", "steals", "blocks", "blocksAgainst", "turnovers",
    "foulsPersonal", "foulsAgainst", "plusMinusPoints", "possessions",
    "fieldGoalsPercentage", "threePointersPercentage", "freeThrowsPercentage",
    "effectiveFieldGoalPercentage", "trueShootingPercentage", "usagePercentage",
    "estimatedUsagePercentage", "estimatedOffensiveRating", "offensiveRating",
    "spWorkOffensiveRating", "estimatedDefensiveRating", "defensiveRating",
    "spWorkDefensiveRating", "estimatedNetRating", "netRating",
    "spWorkNetRating", "assistPercentage", "assistToTurnoverRatio",
    "assistRatio", "offensiveReboundPercentage", "defensiveReboundPercentage",
    "reboundPercentage", "teamTurnoverPercentage",
    "estimatedTurnoverPercentage", "estimatedPace", "pace", "pacePer40",
    "spWorkPace", "playerImpactEstimate", "pointsOffTurnovers",
    "pointsSecondChance", "pointsFastBreak", "pointsInPaint",
    "opponentPointsOffTurnovers", "opponentPointsSecondChance",
    "opponentPointsFastBreak", "opponentPointsInPaint",
    "percentFieldGoalAttempts2Point", "percentFieldGoalAttempts3Point",
    "percentPoints2Point", "percentPoints2PointMidRange", "percentPoints3Point",
    "percentPointsFastBreak", "percentPointsFreeThrow",
    "percentPointsOffTurnovers", "percentPointsInPaint",
    "percentAssisted2PointMade", "percentUnassisted2PointMade",
    "percentAssisted3PointMade", "percentUnassisted3PointMade",
    "percentAssistedFieldGoalsMade", "percentUnassistedFieldGoalsMade",
    "percentTeamFieldGoalsMade", "percentTeamFieldGoalsAttempted",
    "percentTeamThreePointersMade", "percentTeamThreePointersAttempted",
    "percentTeamFreeThrowsMade", "percentTeamFreeThrowsAttempted",
    "percentTeamOffensiveRebounds", "percentTeamDefensiveRebounds",
    "percentTeamRebounds", "percentTeamAssists", "percentTeamTurnovers",
    "percentTeamSteals", "percentTeamBlocks", "percentTeamBlocksAgainst",
    "percentTeamFoulsPersonal", "percentTeamFoulsDrawn", "percentTeamPoints",
]


def attach_same_game(df):
    """Left-join index-game box scores onto the analytic panel, 1:1 or fail.

    The join key is (personId, normalized game date). The built panel does not
    carry `gameId`, and the builder derives its `dt` from `gameDateTimeEst`, so
    the date is the only shared key. The master is filtered to the SAME
    population the builder starts from -- Regular Season, played minutes -- so
    the key is unique on both sides; both uniqueness and full coverage are
    asserted rather than assumed, because a silent many-to-one would change the
    row count and break criterion 4 without any visible symptom.
    """
    print(f"[JOIN] reading {len(SAME_GAME)} index-game columns from the master")
    need = ["personId", "gameDateTimeEst", "gameType", "numMinutes"] + [
        c for c in SAME_GAME if c not in ("numMinutes",)]
    m = pd.read_csv(MASTER, usecols=need, low_memory=False)

    # Exactly the builder's population filters, in the builder's order.
    m["dt"] = pd.to_datetime(m["gameDateTimeEst"], errors="coerce")
    m = m.dropna(subset=["dt", "personId"])
    m = m[dataio.is_regular_season(m["gameType"])]
    m = m[(m["numMinutes"] > 0) & m["numMinutes"].notna()]
    m["key_day"] = m["dt"].dt.normalize()
    m = m[["personId", "key_day"] + SAME_GAME]
    print(f"[JOIN] master rows after the builder's population filters: {len(m):,}")

    dup = int(m.duplicated(subset=["personId", "key_day"]).sum())
    if dup:
        raise AssertionError(
            f"(personId, game day) is not unique in the filtered master: "
            f"{dup:,} duplicate keys. The join would change the row count.")

    left = df.copy()
    left["key_day"] = left["dt"].dt.normalize()
    dup_l = int(left.duplicated(subset=["personId", "key_day"]).sum())
    if dup_l:
        raise AssertionError(
            f"(personId, game day) is not unique in the analytic panel: "
            f"{dup_l:,} duplicate keys.")

    n_before = len(left)
    out = left.merge(m, on=["personId", "key_day"], how="left", validate="1:1")
    if len(out) != n_before:
        raise AssertionError(
            f"the join changed the row count: {n_before:,} -> {len(out):,}")

    unmatched = int(out[SAME_GAME].isna().all(axis=1).sum())
    print(f"[JOIN] unmatched analytic rows (no master row at that key): "
          f"{unmatched:,} of {len(out):,}")
    if unmatched:
        raise AssertionError(
            f"{unmatched:,} analytic rows found no master counterpart. The "
            "reconstruction would compare cells on different information.")

    # The historical script coerced to numeric and mapped infinities to NaN
    # before imputing. Same treatment here, so the same-game block is handled
    # the way the defect handled it.
    out[SAME_GAME] = out[SAME_GAME].apply(pd.to_numeric, errors="coerce")
    out[SAME_GAME] = out[SAME_GAME].replace([np.inf, -np.inf], np.nan)

    all_nan = [c for c in SAME_GAME if out[c].isna().all()]
    if all_nan:
        print(f"[JOIN] WARNING: {len(all_nan)} same-game columns are entirely "
              f"missing and will be dropped by the imputer: {all_nan}")
    return out.drop(columns=["key_day"]), all_nan


def boot_ci(y, p, fn, groups, rng, n=N_BOOT):
    """Cluster bootstrap percentile CI, resampling ATHLETES with replacement.

    The generator is passed IN and is consumed, not re-seeded per call. That
    matters for more than tidiness: `train_and_evaluate.boot_ci` draws ROC and
    then PR from one continuing generator, so re-seeding here made this stage's
    PR interval for the headline model disagree with metrics.json in the fourth
    decimal (0.0408 against 0.0410) while the ROC interval matched exactly. Two
    intervals for one number on one test fold is indistinguishable, in a ledger,
    from a real discrepancy. One generator per cell, seeded at BOOT_SEED and
    consumed ROC-then-PR, reproduces metrics.json bit for bit -- which the
    anchor below now asserts on all four values rather than on ROC alone.
    """
    yv, pv, gv = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(gv)
    rows_of = {g: np.flatnonzero(gv == g) for g in uniq}
    vals = np.empty(n)
    for i in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        s = np.concatenate([rows_of[g] for g in drawn])
        if yv[s].sum() == 0:
            vals[i] = np.nan
            continue
        vals[i] = fn(yv[s], pv[s])
    return [float(np.nanpercentile(vals, 2.5)),
            float(np.nanpercentile(vals, 97.5))]


def fit_score(df, cols, tr_idx, te_idx, label):
    """Fit the headline estimator on `tr_idx`, score on `te_idx`.

    Every cell carries a cluster-bootstrap interval, not a bare point estimate.
    All four AUCs are headed for manuscript prose, and a point estimate quoted
    without its interval is how a 0.01 difference gets read as an effect.
    """
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    ytr, yte = tr["y"].values, te["y"].values
    Xtr, Xte = tr[cols], te[cols]
    # Criterion 4 is about cohort drift, so the row counts are asserted against
    # the INDEX THE CELL WAS HANDED rather than recorded from a shared variable.
    # A future change that filters rows -- a dropna, a complete-case rule, a
    # merge that loses keys -- fires here instead of being averaged into a
    # summary that cannot disagree with itself.
    if len(Xtr) != len(tr_idx) or len(Xte) != len(te_idx):
        raise AssertionError(
            f"{label}: the feature matrix lost rows -- fit {len(Xtr)} of "
            f"{len(tr_idx)}, score {len(Xte)} of {len(te_idx)}")
    pos_weight = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    model = make_models(pos_weight, seed=SEEDS[0])[HEADLINE_MODEL]
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    roc = float(roc_auc_score(yte, p))
    pr = float(average_precision_score(yte, p))
    prev = float(yte.mean())
    groups = te["personId"].values
    rng = np.random.default_rng(BOOT_SEED)
    roc_ci = boot_ci(yte, p, roc_auc_score, groups, rng)
    pr_ci = boot_ci(yte, p, average_precision_score, groups, rng)
    print(f"  {label:44s} ROC-AUC {roc:.4f} [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]"
          f"   PR-AUC {pr:.4f} [{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]   "
          f"lift {pr / prev:.2f}x   ({len(cols)}f)")
    return dict(roc_auc=roc, roc_ci=roc_ci, pr_auc=pr, pr_ci=pr_ci,
                pr_lift=pr / prev,
                n_features=len(cols), n_train=int(len(tr)),
                n_test=int(len(te)), n_test_events=int(yte.sum()),
                test_prevalence=prev), p, np.asarray(te.index)


def paired_delta_ci(y, p_base, p_plus, groups, n=N_BOOT):
    """Cluster bootstrap of AUC(plus) - AUC(base) on ONE shared test fold.

    Paired: both predictions are resampled on the same drawn athletes, so the
    interval is about the difference rather than about two independent AUCs.
    Athletes are the resampling unit for the reason `train_and_evaluate.boot_ci`
    gives -- a player's ~115 games are not exchangeable rows.
    """
    rng = np.random.default_rng(BOOT_SEED)
    y = np.asarray(y)
    gv = np.asarray(groups)  # one fresh generator: this is a separate statistic
    uniq = np.unique(gv)
    rows_of = {g: np.flatnonzero(gv == g) for g in uniq}
    vals = np.empty(n)
    for i in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        s = np.concatenate([rows_of[g] for g in drawn])
        if y[s].sum() == 0 or y[s].sum() == len(s):
            vals[i] = np.nan
            continue
        vals[i] = roc_auc_score(y[s], p_plus[s]) - roc_auc_score(y[s], p_base[s])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return dict(point=float(roc_auc_score(y, p_plus) - roc_auc_score(y, p_base)),
                ci=[float(lo), float(hi)], n_boot=n,
                n_clusters=int(len(uniq)))


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 92)
    print("  LEAKAGE RECONSTRUCTION -- 2x2, feature timing x split design")
    print("=" * 92)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("  The random-split arms are a HISTORICAL STRESS TEST, not a")
    print("  validation of the 2026-06 model. See the module docstring.")
    print()

    base_df = dataio.load()
    base_cols = [c for c in base_df.columns if c not in META]
    df, all_nan = attach_same_game(base_df)
    plus_cols = base_cols + [c for c in SAME_GAME if c not in all_nan]

    n = len(df)
    print(f"\n[COHORT] one analytic cohort for all four cells: {n:,} rows, "
          f"{int(df.y.sum()):,} events, {df.personId.nunique():,} athletes")
    print(f"[FEATURES] corrected {len(base_cols)}  |  corrected + same-game "
          f"{len(plus_cols)}")

    # ---- split membership, computed ONCE per design so cells within a design
    # ---- are guaranteed identical (criterion 4).
    idx = np.arange(n)
    temporal_tr = idx[df.season.isin(TRAIN_SEASONS).values]
    temporal_te = idx[df.season.isin(TEST_SEASONS).values]
    rand_tr, rand_te = train_test_split(
        idx, stratify=df.y.values, **RANDOM_SPLIT_KWARGS)
    rand_tr, rand_te = np.sort(rand_tr), np.sort(rand_te)

    designs = {
        "temporal": dict(train=temporal_tr, test=temporal_te,
                         description="train 2015-16..2020-21, test 2022-23..2023-24"),
        "random_stratified": dict(train=rand_tr, test=rand_te,
                                  description="train_test_split(test_size=0.25, "
                                       "stratify=y, random_state=42)"),
    }
    for k, d in designs.items():
        print(f"[SPLIT] {k:18s} train {len(d['train']):>7,}  test "
              f"{len(d['test']):>7,}  test events {int(df.y.values[d['test']].sum()):>4,}")

    print("\n" + "=" * 92)
    print("  THE FOUR CELLS")
    print("=" * 92)
    cells, preds, members = {}, {}, {}
    for design, d in designs.items():
        for arm, cols in (("corrected", base_cols),
                          ("corrected_plus_same_game", plus_cols)):
            name = f"{design}__{arm}"
            r, p, te_index = fit_score(df, cols, d["train"], d["test"], name)
            r["design"] = design
            r["arm"] = arm
            r["n_rows_analytic"] = int(r["n_train"] + r["n_test"])
            cells[name] = r
            preds[name] = p
            members[name] = te_index

    # ---- criterion 4, asserted on what each cell ACTUALLY saw ---------------
    # An earlier version of this block assigned the same local `n` to all four
    # cells and then compared them, and compared `n_train`/`n_test` values
    # derived from one shared index pair per design. Neither comparison could
    # fail -- a guard measuring itself, which is the failure mode this project
    # keeps hitting. The counts below come from the fitted and scored matrices
    # (fit_score raises if either lost a row), and membership is compared as the
    # actual test INDEX element by element, so a cell that dropped or reordered
    # rows fires here.
    # WHAT THIS CAN AND CANNOT ASSERT, stated because the first version of the
    # criterion was written as "all four cells report an identical analytic row
    # count" and that is FALSE -- which only became visible once the guard was
    # capable of failing. The temporal cells cover 191,259 rows (142,572 train +
    # 48,687 test) because the 2021-22 season is reserved upstream for model
    # selection and neither temporal fold touches it; the random cells cover all
    # 216,104. The two designs therefore cannot have equal coverage without
    # either giving the temporal arm its calibration season (it must not have
    # it) or withholding a season from the random arm (which would stop it
    # reproducing the historical call). So the invariant that actually matters
    # is asserted instead, in three parts.
    coverage = {}
    for design in designs:
        names = [k for k in cells if cells[k]["design"] == design]

        # (1) within a design, the arms must be the same rows -- this is the
        #     cohort-drift check criterion 4 was really after.
        ref_name = names[0]
        for k in names[1:]:
            if not np.array_equal(members[ref_name], members[k]):
                raise AssertionError(
                    f"test-set membership differs within design '{design}': "
                    f"{ref_name} and {k} scored different rows "
                    f"({len(members[ref_name])} vs {len(members[k])}; "
                    f"{len(np.setdiff1d(members[ref_name], members[k])):,} "
                    "only in the first)")
            if cells[ref_name]["n_rows_analytic"] != cells[k]["n_rows_analytic"]:
                raise AssertionError(
                    f"row count differs within design '{design}': "
                    f"{ {j: cells[j]['n_rows_analytic'] for j in names} }")

        # (2) no cell may reach outside the single analytic cohort, and none may
        #     cover more of it than exists.
        cov = cells[ref_name]["n_rows_analytic"]
        if not 0 < cov <= n:
            raise AssertionError(
                f"design '{design}' covers {cov:,} rows, outside the "
                f"{n:,}-row cohort")
        if members[ref_name].max() >= n or members[ref_name].min() < 0:
            raise AssertionError(
                f"design '{design}' scored a row outside the cohort index")
        coverage[design] = cov

    # (3) the shortfall must be exactly the reserved calibration season -- if a
    #     cell ever loses rows for any OTHER reason, this is what catches it.
    n_val = int((df.season == VAL_SEASON).sum())
    if coverage["temporal"] + n_val != n:
        raise AssertionError(
            f"temporal cells cover {coverage['temporal']:,} of {n:,} rows; the "
            f"shortfall is {n - coverage['temporal']:,}, which is not the "
            f"{n_val:,}-row reserved calibration season. Something else "
            "dropped rows.")
    if coverage["random_stratified"] != n:
        raise AssertionError(
            f"random-split cells cover {coverage['random_stratified']:,} of "
            f"{n:,} rows; the random split should partition the whole cohort")

    print(f"\n[CRITERION 4] within each split design the two arms scored "
          f"identical rows, element for element.")
    print(f"  temporal cells cover {coverage['temporal']:,} of {n:,} rows -- "
          f"the {n_val:,}-row 2021-22 calibration season is reserved upstream "
          f"and is in neither temporal fold.")
    print(f"  random-split cells cover all {coverage['random_stratified']:,}. "
          f"The two designs therefore do NOT have equal coverage; see "
          f"checklist item 8.")

    # ---- cell 1 must BE the headline model ----------------------------------
    # All four values, not the point estimate alone. The intervals are the part
    # that silently diverged: with the bootstrap generator re-seeded per call
    # this cell reported a PR interval of 0.0279-0.0410 against metrics.json's
    # 0.0279-0.0408 for the same model on the same fold, and RESULTS.md carried
    # both. Checking only roc_auc could not see it.
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["models"][HEADLINE_MODEL]
    c1 = cells["temporal__corrected"]
    anchor = {}
    worst = 0.0
    for field in ("roc_auc", "pr_auc"):
        anchor[field] = {"metrics_json": ref[field], "cell1": c1[field],
                         "drift": abs(c1[field] - ref[field])}
        worst = max(worst, anchor[field]["drift"])
    for field in ("roc_ci", "pr_ci"):
        d = [abs(a - b) for a, b in zip(c1[field], ref[field])]
        anchor[field] = {"metrics_json": ref[field], "cell1": c1[field],
                         "drift": max(d)}
        worst = max(worst, max(d))
    print(f"[ANCHOR] cell 1 vs metrics.json {HEADLINE_MODEL}: "
          f"ROC {c1['roc_auc']:.6f}/{ref['roc_auc']:.6f}  "
          f"PR {c1['pr_auc']:.6f}/{ref['pr_auc']:.6f}  "
          f"worst drift over point estimates AND intervals {worst:.2e}")
    if worst > 1e-6:
        raise AssertionError(
            "cell 1 does not reproduce the headline model to 1e-6 on every "
            f"reported value (worst drift {worst:.3e}): {anchor}. The cells are "
            "not anchored to the reported analysis and nothing below can be "
            "read against it.")

    # ---- paired differences --------------------------------------------------
    print("\n" + "-" * 92)
    print("  PAIRED DIFFERENCE -- adding same-game columns, within one split")
    print("-" * 92)
    deltas = {}
    for design, d in designs.items():
        te = d["test"]
        deltas[design] = paired_delta_ci(
            df.y.values[te], preds[f"{design}__corrected"],
            preds[f"{design}__corrected_plus_same_game"],
            df.personId.values[te])
        r = deltas[design]
        print(f"  {design:18s} dROC {r['point']:+.4f}  "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  "
              f"({r['n_clusters']:,} athletes, {r['n_boot']} resamples)")

    # The split-design contrast is NOT paired and cannot be bootstrapped as a
    # difference: the two cells are scored on different test folds, drawn from
    # different athletes and different seasons, so there is no shared resampling
    # unit. It is reported as a difference of two point estimates whose own
    # intervals overlap heavily, and the manuscript must say so rather than
    # quoting -0.0101 as if it were an estimated effect with a standard error.
    split_effect = (cells["random_stratified__corrected"]["roc_auc"]
                    - cells["temporal__corrected"]["roc_auc"])
    ci_t = cells["temporal__corrected"]["roc_ci"]
    ci_r = cells["random_stratified__corrected"]["roc_ci"]
    print(f"\n  split design alone (random - temporal, corrected features): "
          f"{split_effect:+.4f}")
    print(f"    UNPAIRED -- different test folds, no shared resampling unit.")
    confound_same = (split_effect > 0) == (len(rand_tr) > len(temporal_tr))
    print(f"    ALSO CONFOUNDED -- the arms train on {len(temporal_tr):,} and "
          f"{len(rand_tr):,} rows (checklist item 8). The confound runs in the")
    print("    SAME direction as this difference, so it could produce it."
          if confound_same else
          "    OPPOSITE direction to this difference, so it cannot produce it.")
    print(f"    temporal {cells['temporal__corrected']['roc_auc']:.4f} "
          f"[{ci_t[0]:.4f}, {ci_t[1]:.4f}]  vs  random "
          f"{cells['random_stratified__corrected']['roc_auc']:.4f} "
          f"[{ci_r[0]:.4f}, {ci_r[1]:.4f}]")

    hist = 0.7891
    gap = hist - cells["random_stratified__corrected_plus_same_game"]["roc_auc"]
    print(f"  historical 0.7891 minus cell 4: {gap:+.4f} -- UNEXPLAINED, and "
          f"reported as such.")
    print("  Known unmeasured axes: estimator, cohort, outcome definition, "
          "history construction.")
    print("  See the discrepancy checklist in the module docstring. No setting")
    print("  was searched to close this gap.")

    out = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "design": {
            "purpose": "separate feature timing from split design in the "
                       "0.7891 -> corrected-headline comparison",
            "estimator": HEADLINE_MODEL,
            "seed": SEEDS[0],
            "random_split_kwargs": RANDOM_SPLIT_KWARGS,
            "same_game_columns": [c for c in SAME_GAME if c not in all_nan],
            "same_game_columns_dropped_all_missing": all_nan,
            "n_rows_analytic": n,
            "n_events_analytic": int(df.y.sum()),
            "n_athletes": int(df.personId.nunique()),
            "random_split_arms_are": "historical stress test, NOT a "
                                     "validation of the superseded model",
        },
        "cells": cells,
        "paired_delta_same_game": deltas,
        "split_design_effect_corrected_features": {
            "difference": float(split_effect),
            "paired": False,
            "why_not_paired": "the two cells are scored on different test "
                              "folds (different athletes, different seasons), "
                              "so there is no shared resampling unit and no "
                              "paired interval exists; read it against the two "
                              "cells' own overlapping intervals",
            "temporal_roc_ci": ci_t,
            "random_stratified_roc_ci": ci_r,
            # Computed, not fixed (2026-09-28). This used to assert the
            # confound "runs AGAINST the reported direction" and quote a
            # training count; after the regular-season rerun the difference turned
            # positive and the fixed sentence became false.
            "also_confounded_by_training_fold": (
                f"checklist item 8: the temporal arm trains on "
                f"{len(temporal_tr):,} rows (2015-16..2020-21, excluding the "
                f"reserved 2021-22 validation season) and the random arm on "
                f"{len(rand_tr):,} drawn from the whole cohort, so this "
                f"difference bundles training-set size and composition with "
                f"split design. More training data would be expected to raise "
                f"the random arm's score, which is the "
                + ("SAME direction as the observed difference, so the confound "
                   "could produce it"
                   if confound_same
                   else "OPPOSITE direction to the observed difference, so the "
                        "confound cannot produce it")
                + "; no clean estimate of split design alone is available "
                  "from these cells"),
            "train_rows_temporal": int(len(temporal_tr)),
            "train_rows_random": int(len(rand_tr)),
        },
        "historical_leaked_roc_auc": hist,
        "unexplained_gap_vs_cell4": float(gap),
        "unmeasured_axes": [
            "estimator (historical: RandomForest 300x12; here: "
            f"{HEADLINE_MODEL})",
            "cohort and data version (historical: unfiltered master rows)",
            "outcome definition (historical: Is_Injured)",
            "playing-history construction (the label-filtered history defect; deliberately not "
            "reconstructed)",
        ],
        "headline_anchor": {"model": HEADLINE_MODEL, "fields": anchor,
                            "worst_drift": worst, "tolerance": 1e-6},
    }
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {JSON_OUT}")


if __name__ == "__main__":
    sys.exit(main())
