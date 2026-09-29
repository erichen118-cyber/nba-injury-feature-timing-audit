"""
gee_lagged_association.py
================================================================================
Re-estimate the prespecified association screening (manuscript S4.2) with
STRICTLY PRE-GAME covariates.

WHY
---
The original screening fitted a population-averaged GEE to workload measured
DURING the index game -- the same game whose injury is the outcome. Acute
field-goal attempts came out strongly associated with every tissue class
(OR 1.21-1.43 per SD), and that association was then read, throughout the
applied literature, as evidence that workload anticipates injury.

It does not. A covariate measured during the game in which an athlete is hurt
reflects that game. The forecasting analysis already showed the lagged versions
of these covariates carry almost no predictive signal, but the association table
itself was never re-estimated -- it remained the last place in the paper where
concurrent measurement stood unchallenged, which is precisely the error the rest
of the paper is about. This closes that gap.

DESIGN -- TWO ARMS, BOTH FITTED HERE (revised 2026-09-21, phase 2)
------------------------------------------------------------------------
Until 2026-09-20 the "concurrent" column of this table was six HARDCODED
constants copied from the prior specification's published output. They could
not move when the pipeline moved, and the manuscript described them as a refit
on this cohort. They are gone. Both arms are now fitted here, on one explicit
cohort, differing ONLY in the timing of the workload window:

    lagged arm      <var>_r5   five-game mean ending at game t-1
    concurrent arm  <var>_c5   five-game mean ending at game t  (includes t)

Pairing five against five isolates TIMING. The historical specification paired
a five-game concurrent window (`fieldGoalsAttempted_Acute`) against a bare
index-game value (`reboundPercentage`), which confounds timing with
aggregation -- so its two covariates were not measuring the same contrast as
each other, let alone as the lagged arm. `reboundPercentage` has since been
dropped from this stage entirely for a separate reason: the rebound-percentage removal found its source
values invalid for two seasons. See WORKLOAD_BASE below.

`<var>_c5` reads game t and is therefore a leakage hazard: `dataio.load()`
drops it unless `concurrent="include"` is passed, which only this stage does,
and `audit_feature_timing.py` asserts it FAILS the same-row perturbation test
(Test B). It correctly passes the future-truncation test -- a window ending at
t reads nothing after t, so the defect is same-game dependence, not future
dependence.
  Population-averaged GEE, binomial family, logit link.
  Exchangeable working correlation, Huber-White robust variance.
  Clustered on athlete-season.
  Odds ratios per SD of the covariate (several covariates are proportions for
  which a one-unit change lies outside the observed range).

  Historical covariate       ->  the two arms fitted here
  ------------------------       -----------------------------
  fieldGoalsAttempted_Acute  ->  fieldGoalsAttempted_c5 / _r5
  reboundPercentage          ->  DROPPED 2026-09-22: invalid source
                                 values in 2015-16 and 2016-17
  AGE, BMI                   ->  unchanged (time-invariant within season)
  startingPosition           ->  position (career modal; the old one was a
                                 game-t starter flag, see build script)
  backToBack_Flag            ->  b2b (lagged by construction)

OUTPUT
------
  predictive_model/gee_lagged_association_output.txt
  predictive_model/gee_lagged_association.json
"""

import os
import json
import datetime
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import dataio

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read via dataio.load(), which resolves parquet or gzipped CSV. Hard-coding one
# extension is what silently broke this script when the builder switched to parquet.
TXT_OUT = os.path.join(SCRIPT_DIR, "gee_lagged_association_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "gee_lagged_association.json")

# The workload covariate(s), in both timings. `_r5` ends at t-1, `_c5` ends at
# t. Everything else in the model is identical between the arms.
#
# 2026-09-22 -- `reboundPercentage` REMOVED, so this is now a single
# covariate and the paired contrast carries 3 cells (one per tissue class)
# rather than 6.
#
# The column is invalid for 2015-16 and 2016-17: the vendor emitted the
# availability-denominated rebound share as identically 0.00 on ~64% of active
# appearances in those seasons that recorded a nonzero rebound count. Adding
# `C(season)` to the lagged arm collapsed its odds ratios to 1.053 (0.954-1.161),
# 1.048 (0.959-1.145) and 1.163 (1.006-1.345) -- two of three intervals then
# spanning 1 -- while field-goal attempts barely moved (1.337 -> 1.324,
# 1.460 -> 1.445, 1.507 -> 1.493).
#
# It is dropped OUTRIGHT rather than restricted to 2017-18+ or absorbed into a
# season term, and the reason is worth keeping: restricting would fit the
# table's two arms on two different cohorts, which is exactly the
# non-comparability the association-arm redesign existed to remove, and a season term would park
# invalid data in a nuisance parameter where nothing would ever look at it
# again. Field-goal attempts is the covariate every surviving claim rests on,
# and it is robust to season adjustment; the rebound arm was never load-bearing.
WORKLOAD_BASE = ["fieldGoalsAttempted"]

ARMS = {
    "lagged": dict(
        suffix="_r5",
        definition="five-game mean ending at game t-1 (games t-5..t-1); "
                   "strictly pre-game",
    ),
    "concurrent": dict(
        suffix="_c5",
        definition="five-game mean ending at game t (games t-4..t); INCLUDES "
                   "the index game and is not pre-game",
    ),
}

# Covariates standardised (z-scored) so coefficients read as OR per SD. The
# z-scoring is done on the SHARED cohort, once, so the two arms' ORs are on
# comparable scales.
SHARED_Z = ["AGE", "BMI"]

# Covariates reported in the S4.2 table that are IDENTICAL in both arms. They
# are tabulated so the manuscript's age association keeps a named producer;
# their two columns differ only through the adjustment, and the row carries
# retimed=False to say so.
COMPARE_SHARED = ["AGE"]


def arm_covars(suffix):
    return [f"{v}{suffix}" for v in WORKLOAD_BASE]


def arm_formula(suffix):
    return ("target ~ " + " + ".join(arm_covars(suffix))
            + " + AGE + BMI + C(position) + b2b")


OUTCOMES = {1: "Ligament/Joint", 2: "Muscle/Tendon", 3: "Bone/Contusion"}

# Paired-contrast bootstrap. 0 disables it (useful when iterating on the rest
# of the stage, since it dominates the runtime). Overridable from the
# environment so a debugging run need not edit the file.
N_BOOT = int(os.environ.get("GEE_N_BOOT", "500"))
BOOT_SEED = 20260921


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _fit_with_convergence_check(model, label):
    """Fit and return (fit, converged, n_iter), checking BOTH available signals.

    `GEEResults.converged` is real: `GEE.fit` sets
    `results.converged = (del_params < ctol)`. An earlier version of this
    function claimed the attribute did not exist -- that claim came from
    inspecting `dir(GEEResults)`, which misses it because it is set on the
    INSTANCE inside `fit()`, not declared on the class. Verified here
    2026-09-21 against statsmodels 0.14.6: `hasattr(fit, "converged")` is
    True, and a deliberately truncated fit returns `converged == False`.
    So the original `getattr(fit, "converged", True)` was reading a genuine
    flag, and the "audit finding" that replaced it was wrong.

    Both signals are checked anyway, because `converged` alone is not
    sufficient: `GEE.fit` has three `ConvergenceWarning` paths -- a singular
    matrix in the update, a singular estimated covariance, and an outright
    failure to estimate parameters. The first `break`s out of the loop with
    `del_params` still at its initial -1.0, which satisfies `del_params <
    ctol` and therefore sets `converged = True` on a fit that failed. Taking
    the AND of the flag and the absence of any convergence warning is
    strictly stronger than either signal on its own.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = model.fit(maxiter=60)
    if fit is None:
        # The two "Unable to estimate GEE parameters" paths return None rather
        # than raising. Surfacing that as an exception is right: a caller that
        # went on to read .params would crash less informatively.
        raise RuntimeError(f"[{label}] GEE returned no fit (see warnings above)")
    bad = [w for w in caught if issubclass(
        w.category, (sm.tools.sm_exceptions.IterationLimitWarning,
                     sm.tools.sm_exceptions.ConvergenceWarning))]
    flag = bool(getattr(fit, "converged", False))
    converged = flag and not bad
    # `fit_history` is public on the results object; the private
    # `model._fit_history` was reaching past the API for the same data.
    n_iter = len(fit.fit_history.get("params", []))
    if not converged:
        print(f"  [{label}] *** DID NOT CONVERGE -- flag={flag}, "
              f"warnings={[type(w.message).__name__ for w in bad]}, "
              f"{n_iter} iterations ***")
    return fit, converged, n_iter


class _ReplicateFailure(RuntimeError):
    """A bootstrap replicate that failed STATISTICALLY (non-convergence, or a
    fallback to a different working correlation). Distinct from every other
    exception so the replicate loop can swallow this one and propagate the
    rest, instead of logging a MemoryError as a dropped fit."""


def fit_arm(d, formula, label):
    """Fit one GEE arm. Returns (fit, working-correlation name, converged, n_iter)."""
    try:
        model = smf.gee(formula, groups="player_season", data=d,
                        family=sm.families.Binomial(link=sm.families.links.Logit()),
                        cov_struct=sm.cov_struct.Exchangeable())
        fit, converged, n_iter = _fit_with_convergence_check(model, label)
        return fit, "Exchangeable", converged, n_iter
    except Exception as e:
        print(f"  [{label}] exchangeable failed ({e}); falling back to independence")
        model = smf.gee(formula, groups="player_season", data=d,
                        family=sm.families.Binomial(link=sm.families.links.Logit()),
                        cov_struct=sm.cov_struct.Independence())
        fit, converged, n_iter = _fit_with_convergence_check(model, label)
        return fit, "Independence", converged, n_iter


def terms_of(fit):
    params, ci, pvals = fit.params, fit.conf_int(), fit.pvalues
    out = {}
    for term in params.index:
        if term == "Intercept":
            continue
        out[term] = dict(or_per_sd=float(np.exp(params[term])),
                         ci=[float(np.exp(ci.loc[term, 0])),
                             float(np.exp(ci.loc[term, 1]))],
                         p=float(pvals[term]))
    return out


def _run():

    print("=" * 92)
    print("  PRESPECIFIED ASSOCIATION SCREENING -- BOTH ARMS FITTED ON ONE COHORT")
    print("=" * 92)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # concurrent="include" is passed in exactly one place in this repo: here.
    # The `_c5` covariates read the index game, so every other stage receives a
    # frame that does not contain them at all.
    df = dataio.load(concurrent="include")

    # position arrives one-hot from the build; reconstruct the label for C().
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    df["position"] = (df[pos_cols].idxmax(axis=1)
                      .str.replace("pos_", "", regex=False))

    df["player_season"] = (df["personId"].astype(str) + "_" + df["season"].astype(str))

    # ONE cohort for both arms. Complete cases are taken over the UNION of the
    # two arms' covariates, so the arms are fitted on identical rows and the
    # comparison cannot be confounded by differing missingness: `_c5` has
    # min_periods=1 and is defined wherever a game exists, while `_r5` is
    # undefined on a career-opening game.
    z_covars = SHARED_Z + [c for a in ARMS.values() for c in arm_covars(a["suffix"])]
    needed = z_covars + ["b2b", "player_season", "tissue"]
    df = df.dropna(subset=needed).copy()
    print(f"[COHORT] {len(df):,} athlete-games   "
          f"{df.player_season.nunique():,} athlete-season clusters   "
          f"{df.personId.nunique():,} athletes\n")

    # EXPORTED, not just printed -- added 2026-09-20. The manuscript
    # reported this stage's denominator as 189,430 athlete-games / 4,325
    # clusters long after it had become 216,104 / 4,603. Nothing caught it
    # because this JSON carried no count at all, and verify_no_stale_numbers.py
    # can only check the manuscript against what the JSON contains. A figure
    # that lives only in a printed log is a figure no guard can see.
    cohort = dict(n_rows=int(len(df)),
                  n_clusters=int(df.player_season.nunique()),
                  n_athletes=int(df.personId.nunique()))

    # Kept BEFORE standardisation: the attenuation check below reports the
    # index game's effect on the window in the covariate's own units, which a
    # z-scored frame cannot express.
    raw = df[list(z_covars)
             + [f"{v}_c1" for v in WORKLOAD_BASE] + ["y"]].copy()

    # z-score -> OR per SD, on the shared cohort, so the two arms' odds ratios
    # are on comparable scales.
    for c in z_covars:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    results = {"cohort": cohort,
               "arms": {k: dict(v) for k, v in ARMS.items()},
               "outcomes": {}}
    rows = []

    for code, name in OUTCOMES.items():
        df["target"] = (df["tissue"] == code).astype(int)
        # statsmodels GEE requires contiguous groups
        d = df.sort_values("player_season").reset_index(drop=True)
        n_events = int(d["target"].sum())

        print("-" * 92)
        print(f"  {name}   (events = {n_events:,})")
        print("-" * 92)

        per_arm = {}
        for arm, spec in ARMS.items():
            fit, struct, converged, n_iter = fit_arm(
                d, arm_formula(spec["suffix"]), f"{name}/{arm}")
            terms = terms_of(fit)
            per_arm[arm] = dict(cov_struct=struct,
                                converged=converged,
                                n_iterations=n_iter,
                                feature_definition=spec["definition"],
                                n_rows=int(len(d)),
                                n_clusters=int(d.player_season.nunique()),
                                n_events=n_events,
                                terms=terms)
            print(f"  [{arm:10s}]  working correlation = {struct}   "
                  f"converged={converged} ({n_iter} iter)   "
                  f"({spec['definition']})")
            print(f"  {'covariate':26s} | {'OR/SD':>7s} | {'95% CI':>16s} | {'p':>10s}")
            print("  " + "-" * 68)
            for term, t in terms.items():
                print(f"  {term:26s} | {t['or_per_sd']:7.3f} | "
                      f"[{t['ci'][0]:6.3f}, {t['ci'][1]:6.3f}] | {t['p']:10.2e}")
            print()

        # The workload covariates differ BETWEEN arms (that is the contrast).
        # AGE is the same covariate in both models and is carried here only so
        # the age association the manuscript reports keeps a producer; it moves
        # between the columns through the adjustment, not through its own
        # timing. Flagged in the row so no reader has to infer it.
        pairs = ([(v, v + "_r5", v + "_c5", True) for v in WORKLOAD_BASE]
                 + [(v, v, v, False) for v in COMPARE_SHARED])
        for v, lag_term, con_term, retimed in pairs:
            lag = per_arm["lagged"]["terms"][lag_term]
            con = per_arm["concurrent"]["terms"][con_term]
            rows.append(dict(
                outcome=name, covariate=v, n_events=n_events, retimed=retimed,
                conc_or=con["or_per_sd"],
                conc_ci=f"{con['ci'][0]:.2f}-{con['ci'][1]:.2f}",
                conc_p=con["p"],
                lag_or=lag["or_per_sd"],
                lag_ci=f"{lag['ci'][0]:.2f}-{lag['ci'][1]:.2f}",
                lag_p=lag["p"]))

        results["outcomes"][name] = dict(n_events=n_events, arms=per_arm)

    # ---- the comparison that matters ----------------------------------------
    print("=" * 92)
    print("  CONCURRENT vs LAGGED -- identical models on one cohort, differing only")
    print("  in whether the five-game workload window ends at game t or at t-1")
    print("=" * 92)
    print(f"  {'outcome':16s} | {'covariate':20s} | {'concurrent OR':>13s} | "
          f"{'lagged OR':>10s} | {'95% CI':>14s}")
    print("  " + "-" * 84)
    for r in rows:
        print(f"  {r['outcome']:16s} | {r['covariate']:20s} | "
              f"{r['conc_or']:13.3f} | {r['lag_or']:10.3f} | {r['lag_ci']:>14s}")

    results["comparison"] = rows

    # ---- PAIRED contrast between the arms, cluster bootstrap ---------------
    # Overlapping MARGINAL intervals are not a test of the difference. The two
    # arms are fitted on the same athlete-seasons and differ in one game of one
    # covariate, so their estimates are strongly positively correlated and the
    # marginal intervals say nothing about whether the difference excludes
    # zero. §4.2 asserted they were "not distinguishable" on exactly that bad
    # reasoning until a 2026-09-21 review caught it.
    #
    # Resampling ATHLETE-SEASONS (the clustering unit) and refitting BOTH arms
    # inside each replicate gives a paired interval while leaving the reported
    # point estimates exactly as the separate fits produce them. A stacked
    # joint GEE would be far cheaper but estimates a joint working correlation,
    # which shifts the arm-specific odds ratios off the table's values; Eric
    # chose fidelity to the table over speed (2026-09-21).
    if N_BOOT:
        print("=" * 92)
        print(f"  PAIRED CONTRAST -- cluster bootstrap over athlete-seasons "
              f"({N_BOOT} replicates, seed {BOOT_SEED})")
        print("=" * 92)
        rng = np.random.default_rng(BOOT_SEED)
        clusters = df["player_season"].unique()
        # Row indices per cluster, computed ONCE. The first version copied a
        # DataFrame per cluster and concatenated 4,603 of them per replicate,
        # which cost ~3 min a replicate -- the resample, not the fit, was the
        # bottleneck. Index arrays make it a single take().
        _codes = pd.Categorical(df["player_season"], categories=clusters).codes
        _order = np.argsort(_codes, kind="mergesort")
        _starts = np.searchsorted(_codes[_order], np.arange(len(clusters)))
        _ends = np.searchsorted(_codes[_order], np.arange(len(clusters)), side="right")
        idx_of = [_order[a:b] for a, b in zip(_starts, _ends)]
        df_vals = df.reset_index(drop=True)
        boot = {f"{name}|{v}": [] for name in OUTCOMES.values() for v in WORKLOAD_BASE}
        n_failed = 0
        t_boot = time.time()

        for b in range(N_BOOT):
            picks = rng.integers(0, len(clusters), size=len(clusters))
            chunks = [idx_of[i] for i in picks]
            take = np.concatenate(chunks)
            # Cluster identity must be made unique per draw, or a cluster drawn
            # twice is collapsed into one and the resample is not a resample.
            # Integer labels are used rather than strings: GEE only needs
            # grouping, and the rows are already contiguous by construction.
            new_id = np.repeat(np.arange(len(chunks)),
                               [len(c) for c in chunks])
            bs = df_vals.take(take, axis=0).reset_index(drop=True)
            bs["player_season"] = new_id

            for code, name in OUTCOMES.items():
                bs["target"] = (bs["tissue"] == code).astype(int)
                try:
                    per = {}
                    for arm, spec in ARMS.items():
                        # The working-correlation name is CHECKED, not
                        # discarded. fit_arm falls back from Exchangeable to
                        # Independence on any exception and still reports
                        # converged, so `fit, _, ok, _` would let a replicate
                        # estimated under a different estimator into the
                        # percentile interval while n_failed stayed 0 -- and
                        # "0 failed fits" is quoted in RESULTS.md as evidence
                        # that every replicate was estimated the same way.
                        fit, cov, ok, _ = fit_arm(
                            bs, arm_formula(spec["suffix"]), f"boot{b}/{name}/{arm}")
                        if not ok:
                            raise _ReplicateFailure("replicate did not converge")
                        if cov != "Exchangeable":
                            raise _ReplicateFailure(
                                f"replicate fell back to {cov}; the reported "
                                f"arms are Exchangeable and the two must not "
                                f"be pooled into one interval")
                        per[arm] = fit.params
                    for v in WORKLOAD_BASE:
                        d = float(per["concurrent"][f"{v}_c5"]
                                  - per["lagged"][f"{v}_r5"])
                        boot[f"{name}|{v}"].append(d)
                except _ReplicateFailure as e:
                    # ONLY statistical failure is counted and swallowed. A bare
                    # `except Exception` here counted MemoryError, a patsy
                    # failure on a degenerate resample, and any bug in this loop
                    # identically to non-convergence -- on a run that spent
                    # hours paging, a MemoryError would have been recorded as a
                    # "dropped fit" and the interval computed from the
                    # survivors, with RESULTS.md reporting the count as if it
                    # measured model behavior. Anything else now propagates.
                    n_failed += 1
                    print(f"  [boot {b} {name}] dropped: {e}")

            if (b + 1) % 10 == 0:
                el = time.time() - t_boot
                print(f"  {b + 1}/{N_BOOT} replicates  ({el / (b + 1):.1f}s each, "
                      f"{el / 60:.1f} min elapsed)", flush=True)

        paired = {}
        print()
        print(f"  {'outcome':16s} | {'covariate':20s} | {'diff log-OR':>11s} | "
              f"{'95% CI':>20s} | {'OR ratio':>9s} | excl 0")
        print("  " + "-" * 92)
        for key, vals in boot.items():
            name, v = key.split("|")
            a = np.asarray(vals, dtype=float)
            if a.size < 2:
                continue
            lo, hi = (float(x) for x in np.percentile(a, [2.5, 97.5]))
            # The POINT ESTIMATE is the observed difference from the two fits
            # this stage reports, not the bootstrap mean. The bootstrap supplies
            # the interval only; using its mean would quietly substitute a
            # different (and slightly bias-shifted) estimate for the one in the
            # table, so the table and the contrast could disagree.
            _obs = next(r for r in rows
                        if r["outcome"] == name and r["covariate"] == v)
            pt = float(np.log(_obs["conc_or"]) - np.log(_obs["lag_or"]))
            excl = (lo > 0) or (hi < 0)
            paired[key] = dict(
                outcome=name, covariate=v,
                diff_log_or=pt,
                diff_log_or_bootstrap_mean=float(np.mean(a)),
                diff_log_or_ci=[lo, hi],
                or_ratio=float(np.exp(pt)),
                or_ratio_ci=[float(np.exp(lo)), float(np.exp(hi))],
                excludes_zero=bool(excl),
                n_replicates=int(a.size),
            )
            print(f"  {name:16s} | {v:20s} | {pt:+11.4f} | "
                  f"[{lo:+.4f}, {hi:+.4f}] | {np.exp(pt):9.4f} | "
                  f"{'YES' if excl else 'no'}")
        results["paired_contrast"] = dict(
            method="cluster bootstrap over athlete-seasons; both arms refitted "
                   "inside each replicate, so the reported point estimates are "
                   "unchanged and only the DIFFERENCE gains an interval",
            n_boot=N_BOOT, seed=BOOT_SEED, n_failed_fits=n_failed,
            runtime_seconds=round(time.time() - t_boot, 1),
            contrasts=paired,
        )
        print(f"\n  {n_failed} replicate-outcome fits dropped; "
              f"{(time.time() - t_boot) / 60:.1f} min total.")
        print()


    # ---- why the concurrent arm is ATTENUATED, measured not asserted -------
    # The concurrent window differs from the lagged one by exactly one game:
    # the index game. If the outcome depresses the covariate -- an athlete hurt
    # during a game plays fewer minutes and takes fewer shots -- then adding
    # game t should pull the window DOWN on injured rows and do nothing on
    # healthy ones. That is a testable prediction, and the numbers below are
    # it. They describe the pattern; they do NOT identify its cause, and this
    # is reverse causation rather than the collider structure of S4.9 (the withdrawn availability interpretation,
    # 2026-09-22 -- the two were previously conflated under one name). They
    # are computed on the pre-standardisation values, in the covariate's own
    # units, because an SD-scaled shift is not interpretable here.
    #
    # Reported in S4.2. Without this the direction of the concurrent-vs-lagged
    # difference would be an interpretation rather than a measurement.
    atten = {}
    for v in WORKLOAD_BASE:
        # The index game against the athlete's OWN preceding five-game mean.
        # NOT `_c5 - _r5`: those windows are t-4..t and t-5..t-1, so their
        # difference is (v_t - v_{t-5})/5, which adds game t and drops game
        # t-5. That wrong quantity was reported as "what the index game does
        # to the window" until review caught it on 2026-09-21.
        shift = raw[f"{v}_c1"] - raw[f"{v}_r5"]
        atten[v] = dict(
            definition="index game value minus the athlete's own mean over the "
                       "five games before it (v_t - r5_t)",
            mean_index_game_shift_healthy=float(shift[raw["y"] == 0].mean()),
            mean_index_game_shift_injured=float(shift[raw["y"] == 1].mean()),
            mean_r5_healthy=float(raw.loc[raw["y"] == 0, f"{v}_r5"].mean()),
            mean_r5_injured=float(raw.loc[raw["y"] == 1, f"{v}_r5"].mean()),
            mean_c5_healthy=float(raw.loc[raw["y"] == 0, f"{v}_c5"].mean()),
            mean_c5_injured=float(raw.loc[raw["y"] == 1, f"{v}_c5"].mean()),
        )
    results["index_game_attenuation"] = atten
    print("=" * 92)
    print("  WHY THE CONCURRENT ARM IS ATTENUATED -- the index game against the")
    print("  athlete's own preceding five-game mean, in the covariate's own units")
    print("=" * 92)
    print(f"  {'covariate':22s} | {'shift, healthy':>15s} | {'shift, injured':>15s}")
    print("  " + "-" * 60)
    for v, a in atten.items():
        print(f"  {v:22s} | {a['mean_index_game_shift_healthy']:+15.4f} | "
              f"{a['mean_index_game_shift_injured']:+15.4f}")
    print()
    print("  A negative shift on injured rows and none on healthy rows is")
    print("  consistent with the outcome depressing the covariate (reverse")
    print("  causation). It does NOT identify why index-game activity was")
    print("  lower: these data carry no onset information, so injury during")
    print("  the game, pre-existing symptoms and reporting decisions all fit.")
    print("  Reverse causation is NOT the same structure as conditioning on a")
    print("  common effect, which is the participation selection of S4.9.")
    print()

    print("\n" + "=" * 92)
    print("  INTERPRETATION")
    print("=" * 92)
    print("  Both columns are fitted here, on the SAME rows, with the SAME model and")
    print("  the SAME adjustment set. They differ only in whether the five-game")
    print("  workload window ends at the index game or at the game before it.")
    print()
    print("  Read them against each other: whatever survives the shift to pre-game")
    print("  measurement is a candidate risk factor, and whatever collapses was in")
    print("  part a description of the game in which the athlete was hurt. The")
    print("  direction is NOT assumed -- an athlete hurt during a game plays fewer")
    print("  minutes and takes fewer shots, so concurrent workload can be pulled")
    print("  DOWN by the outcome -- reverse causation, which is distinct from")
    print("  the participation selection of S4.9 and is not identified here.")

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n[EXPORT] {JSON_OUT}")


if __name__ == "__main__":
    main()
