"""audit_feature_timing.py
================================================================================
FEATURE-CONSTRUCTION LEAKAGE CONTROL -- the one the permutation test cannot do.

WHY THIS EXISTS
---------------
`audit_leakage_checks.py` experiment A permutes labels and checks that the model
collapses to chance. That detects PROCEDURAL leakage (preprocessing fitted across
the split, duplicated rows, a grouping variable that leaks an athlete's future).
It is structurally blind to FEATURE-CONSTRUCTION leakage -- the class of defect
this project actually suffered, when covariates were computed from the very game
whose injury they predicted (ROC-AUC 0.7891, an artifact).

The blindness is not a tuning problem. Once labels are permuted, a covariate
built from the outcome game no longer corresponds to its own row's label, so the
model cannot exploit it and discrimination collapses to chance whether or not the
leak is present. THE PERMUTATION CONTROL WOULD HAVE RETURNED PASS ON THE ORIGINAL
LEAKED MODELS.

Until now the only thing standing in that gap was a human reading
`build_predictive_dataset.py`. A defect found by reading is a defect that can be
missed by reading, and nothing re-checked it when the builder changed. This
script replaces the eye with a mechanical, per-column, rerunnable test.

THE PROPERTY UNDER TEST
-----------------------
For every feature column c and every row t:

    c(t) must be a function of rows strictly BEFORE t -- nothing from game t
    itself, nothing after it.

That decomposes into two invariances. Neither closes the gap alone; the pair
does. Both are EXACT-EQUALITY checks, not hypothesis tests.

  (A third, Test C, was added in 2026-09 and is described under SCOPE below:
   it sits one stage earlier than these two.)

  A. FUTURE TRUNCATION. Delete every row dated >= T and rebuild. Features for
     rows before T must be bit-identical to the full-panel build.
     Catches covariates reaching into the FUTURE (`position`).
     Misses, alone: anything reading row t itself -- truncation leaves row t in
     place, so a same-game covariate is unchanged and the test passes.

  B. SAME-ROW PERTURBATION. Replace game t's own measurements with values drawn
     from other rows and rebuild. Features for row t must be unchanged.
     Catches covariates reading the INDEX GAME -- the original leak.
     Misses, alone: covariates reaching past t into later rows.

  At most one row per athlete is perturbed per pass, which guarantees every
  perturbed row's entire look-back history is itself unperturbed. Any difference
  at row t is therefore genuine same-row dependence, not contamination arriving
  through the rolling windows.

WHAT IS DELIBERATELY NOT PERTURBED
----------------------------------
  dt, personId, season -- the schedule is published before tip-off, so a
  covariate depending on when a game is played, who plays it and which season it
  falls in is legitimately pre-game. Perturbing them would manufacture false
  positives.

  AGE, BMI -- biometrics known before tip-off, carried through to the analytic
  table unchanged. They are row-t attributes but not row-t MEASUREMENTS, so the
  same argument applies. Consequence, stated plainly: Test B does not certify
  AGE/BMI. Their timing rests on the same published-in-advance argument as `dt`.

THE POSITIVE CONTROL (what makes this citable)
----------------------------------------------
A control that cannot fail is not a control. Four deliberately-poisoned
covariates are injected into a scratch panel and every test is re-run against
it. Each must be caught by its expected test or this script exits non-zero:

    leak_same_game_minutes    numMinutes at row t         mimics the 0.7891
                                                          artifact       -> B
    leak_career_mean_minutes  athlete mean over the panel mimics position -> A
    leak_future_injury        y at row t+1                outcome peeking -> A
    leak_prior_label_rest     gap to the last appearance  mimics the label-filtered history defect   -> D
                              that was NOT a labelled
                              injury game

The fourth was added 2026-09-18 because Test D had no control at all: the other
three are caught by A, B and C, so a Test D that silently did nothing would have
printed the same all-clear it prints when it works. `leak_prior_label_rest`
reads a label from a row in its own PAST, which is the exact shape of the label-filtered history defect.
A and B are structurally unable to see it -- A only removes future rows, B only
rewrites row t's own measurements -- and that negative is asserted. C is a
near-miss rather than a structural blindness, and is recorded but not asserted;
`prior_label_rest` explains why.

SCOPE -- what this does NOT do
------------------------------
It audits the COVARIATES, not the model pipeline. It says nothing about ROC-AUC,
about the temporal split, or about procedural leakage; `audit_leakage_checks.py`
covers that and is not replaced.

It also cannot see a defect in the SOURCE data -- a corrupted master panel, or
an injury whose recorded date is itself wrong, is invisible here, because every
test compares one build of this pipeline against another build of it.

What it CAN now see, and could not before 2026-09-18, is label information
entering during outcome attachment and cohort selection. Tests A and B both
begin at `load_panel()`; the label-filtered history defect lived upstream of that, in the deletion of
played games because of their injury labels, and this script passed throughout.
Tests C and D both begin at `load_appearances()` instead -- the last frame no
label has touched -- and rewrite injury records before the attachment runs.

**Which of the two actually covers the label-filtered history defect matters, and was got wrong once.**
Test C alone does not: see its docstring. Test D does.

  C. UPSTREAM FUTURE-RECORD PERTURBATION. Rewrite every injury record dated
     >= T and re-run attach_outcomes() and build_features(). Covariates on rows
     before T must be bit-identical.
     Catches FUTURE label information flowing backwards through cohort
     construction.
     Misses, and this was briefly claimed otherwise: the label-filtered history defect itself. That defect
     was row t's rest reading a label on row t-1, which is in its past, and
     Test C perturbs by game date so the previous game's record is never
     touched. Test D is the one that catches it.

  D. LABEL INDEPENDENCE. Rewrite injury records across the WHOLE panel and
     rebuild. Every covariate outside the injury-history block must be
     bit-identical, because rest, workload, exposure, position and biometrics
     are functions of appearances alone.
     Catches a label of any date leaking into feature construction -- which is
     exactly what the label-filtered history defect was.

OUTPUTS
-------
  predictive_model/audit_feature_timing_output.txt
  predictive_model/audit_feature_timing.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

import build_predictive_dataset as B
import dataio
import features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(SCRIPT_DIR, "audit_feature_timing_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "audit_feature_timing.json")

META = ["personId", "Player", "dt", "season", "y", "tissue", "y_concussion"]

# --- configuration -----------------------------------------------------------
# One rebuild of build_features() costs ~22 s on the full panel (measured
# 2026-07-29), so the counts below are ~4 min for the real panel and ~2 min for
# the injected-leak panel. Both are recorded in RESULTS.md: the verdict is only
# reproducible if the number of cutoffs and passes is known.
A_CUTOFF_SEASONS = [2019, 2020, 2021, 2022, 2023]   # truncate at each opener
# Season-opener cutoffs delete WHOLE seasons, so for every compared row all of
# its same-season successors survive the truncation. A covariate that looked
# ahead only within its own season would pass every opener cutoff. Mid-season
# cutoffs (the median game date of the season) are the only ones that can see
# that, so at least one is mandatory -- do not reduce this to openers alone.
A_MIDSEASON_CUTOFF_SEASONS = [2020, 2022]
B_PASSES = 5
B_SEED = 42

# Test C truncation points. Mid-season cutoffs matter more here than for Test A:
# an opener cutoff rewrites whole future seasons, while a mid-season cutoff also
# rewrites the REST of the season the compared rows live in, which is the only
# way to see a covariate that looks ahead within its own season.
C_CUTOFF_SEASONS = [2020, 2022]
C_MIDSEASON_CUTOFF_SEASONS = [2019, 2021, 2023]
C_SEED = 202609

# Test D: label independence. Cheaper than C per pass (no cutoff sweep), and
# two passes with different seeds is plenty -- a covariate that depends on a
# label will move under any relabelling, not just a lucky one.
D_PASSES = 2
D_SEED = 990918

# Covariates known and disclosed to fail, so the stage's exit code can
# distinguish "the known defect" from "a NEW defect".
#
# EMPTY since 2026-09-18, and it must stay empty. It used to hold the four
# `pos_*` dummies, because `position` was the modal startingPosition over the
# athlete's WHOLE panel and therefore read games after the index game --
# limitation 9 in the manuscript, and the reason the abstract's strictly
# pre-game guarantee had to be qualified. Criterion 10 of the repair plan
# replaced it with a prior-only tally, so the exemption has nothing left to
# cover. Anything failing now is a regression in build_features() and fails the
# pipeline.
#
# Do not add to this set to make a run go green. An entry here is a public
# admission that a covariate is not pre-game, and it belongs in the manuscript's
# limitations the same day it is added.
KNOWN_FAILURES = set()

# The injected-leak panel only has to demonstrate detection, not adjudicate 123
# columns, so it runs a cheaper configuration.
INJECT_A_CUTOFF_SEASONS = [2021, 2023]
INJECT_B_PASSES = 2
INJECT_C_MIDSEASON_SEASONS = [2022]
INJECT_D_PASSES = 1

# Row-t measurements Test B replaces. BASE_VARS covers the box score (including
# numMinutes); startingPosition drives `position` and `started_last_game`;
# y/tissue drive the injury-history covariates.
PERTURB_NUMERIC = list(B.BASE_VARS)
PERTURB_CATEGORICAL = ["startingPosition"]
NOT_PERTURBED = ["dt", "personId", "season", "AGE", "BMI"]

# Expected verdicts for the injected leaks: which test MUST catch each.
# Which test MUST catch each injected leak. `leak_future_injury` is expected by
# BOTH A and C: A because truncation removes the row it reads, C because
# rewriting that row's injury record changes what it reads. C is listed
# separately so a Test C that silently stopped working could not hide behind
# A's verdict.
#
# `leak_prior_label_rest` is expected by D, and is asserted in both directions:
# D must catch it, and A and B must NOT (INJECTED_EXPECT_ONLY below). That
# negative is what makes it a control ON TEST D rather than one more leak the
# suite happens to find. C is deliberately left unasserted -- it can reach a
# pre-cutoff label through index-game promotion, so requiring it to be quiet
# would tie the exit code to the data; see `prior_label_rest`.
INJECTED_EXPECT = {
    "leak_same_game_minutes": "B",
    "leak_career_mean_minutes": "A",
    "leak_future_injury": "A",
    "leak_prior_label_rest": "D",
}
INJECTED_EXPECT_C = ["leak_future_injury"]
INJECTED_EXPECT_ONLY = ["leak_prior_label_rest"]


# --- comparison primitives ---------------------------------------------------

def differing(a, b):
    """Element-wise inequality with NaN == NaN, for any dtype pair.

    Exact equality on purpose. A tolerance would let a covariate that depends
    weakly on game t slip through, and "weakly" is not a category this study can
    defend to a reviewer.
    """
    a_null, b_null = pd.isna(a).to_numpy(), pd.isna(b).to_numpy()
    both_null = a_null & b_null
    try:
        eq = (a.to_numpy() == b.to_numpy())
    except (TypeError, ValueError):
        eq = np.array([x == y for x, y in zip(a.to_numpy(), b.to_numpy())])
    eq = np.where(a_null | b_null, False, eq)
    return ~(eq | both_null)


def keyed(frame):
    """Index a build by (personId, dt) so two builds can be aligned row-wise."""
    idx = pd.MultiIndex.from_arrays([frame["personId"].to_numpy(),
                                     frame["dt"].to_numpy()],
                                    names=["personId", "dt"])
    if idx.has_duplicates:
        raise AssertionError(
            "(personId, dt) is not unique in the analytic table, so two builds "
            "cannot be aligned row-wise. Investigate before trusting any verdict.")
    return frame.set_index(idx)


class Verdicts:
    """Per-column tally of differing rows, accumulated across cutoffs/passes."""

    def __init__(self, columns):
        self.diff = {c: 0 for c in columns}
        self.compared = {c: 0 for c in columns}
        self.notes = {}

    def add(self, col, ndiff, ncomp, note=None):
        self.diff[col] += int(ndiff)
        self.compared[col] += int(ncomp)
        if note:
            self.notes.setdefault(col, []).append(note)

    def failed(self):
        return [c for c in self.diff if self.diff[c] > 0]

    def table(self, title, show_pass_count=True):
        bad = self.failed()
        lines = [f"  {title}"]
        if not bad:
            lines.append(f"    PASS -- all {len(self.diff)} columns identical.")
        else:
            lines.append(f"    {len(bad)} of {len(self.diff)} columns FAIL:")
            lines.append(f"      {'column':32s} {'rows differing':>16s} {'fraction':>10s}")
            for c in sorted(bad, key=lambda c: -self.diff[c]):
                n, tot = self.diff[c], max(1, self.compared[c])
                lines.append(f"      {c:32s} {n:>16,} {n / tot:>10.4%}")
                for note in self.notes.get(c, [])[:2]:
                    lines.append(f"        note: {note}")
        if show_pass_count:
            lines.append(f"    PASS: {len(self.diff) - len(bad)}   "
                         f"FAIL: {len(bad)}   (of {len(self.diff)} feature columns)")
        return "\n".join(lines)


def compare(base, got, feature_cols, verdicts):
    """Compare `got` against `base` on the rows they SHARE, column by column.

    Rows present in `base` but absent from `got` are excluded rather than
    reindexed to NaN. Scoring them as differences would turn any row-set
    mismatch into "all 123 columns FAIL", burying the real cause -- a row-set
    mismatch is a structural problem with the rebuild, not evidence about any
    particular covariate. The caller reports the mismatch separately.
    """
    shared = base.index.intersection(got.index)
    base, got = base.loc[shared], got.reindex(shared)
    n = len(shared)
    for c in feature_cols:
        if c not in got.columns:
            verdicts.add(c, n, n, note="column absent from the rebuilt table")
            continue
        verdicts.add(c, differing(base[c], got[c]).sum(), n)


# --- test A: future truncation ----------------------------------------------

def test_a(builder, panel, full_keyed, feature_cols, cutoffs):
    """Rebuild on panel[dt < T]; rows before T must be bit-identical.

    `cutoffs` is a list of (label, timestamp) pairs.
    """
    v = Verdicts(feature_cols)
    detail = []
    for label, T in cutoffs:
        sub = panel[panel["dt"] < T]
        got = keyed(builder(sub))
        base = full_keyed[full_keyed["dt"] < T]
        missing = len(base.index.difference(got.index))
        extra = len(got.index.difference(base.index))
        detail.append({"cutoff_label": label, "cutoff": str(T.date()),
                       "panel_rows": int(len(sub)),
                       "rows_compared": int(len(base.index.intersection(got.index))),
                       "rows_missing_from_rebuild": missing,
                       "rows_extra_in_rebuild": extra})
        print(f"    cutoff {T.date()} ({label}): "
              f"{len(sub):,} panel rows -> comparing {len(base):,} analytic rows"
              + (f"   [ROW-SET DIFFERS: {missing} missing, {extra} extra -- "
                 f"excluded from the comparison, investigate]"
                 if (missing or extra) else ""))
        compare(base, got, feature_cols, v)
    return v, detail


def build_cutoffs(panel, opener_seasons, midseason_seasons):
    """Season-opener and mid-season cutoff dates, as (label, timestamp) pairs."""
    cuts = []
    for s in opener_seasons:
        cuts.append((f"season {s} opener",
                     panel.loc[panel["season"] == s, "dt"].min()))
    for s in midseason_seasons:
        d = panel.loc[panel["season"] == s, "dt"]
        cuts.append((f"season {s} mid-season", d.quantile(0.5)))
    return sorted(cuts, key=lambda p: p[1])


# --- test B: same-row perturbation -------------------------------------------

def perturb(panel, rng, min_prior_games):
    """Replace one row per athlete's own measurements with values from elsewhere.

    Returns (perturbed panel, keys of perturbed rows, fraction of cells that
    actually changed). The fraction is reported because a perturbation that
    happened to redraw the original value would give a PASS with no teeth.
    """
    p = panel.reset_index(drop=True).copy()
    eligible = p.groupby("personId", sort=False).cumcount() >= min_prior_games
    pos = np.flatnonzero(eligible.to_numpy())
    if len(pos) == 0:
        raise AssertionError("no eligible rows to perturb")
    # one row per athlete: shuffle, then keep the first hit per athlete
    order = rng.permutation(pos)
    seen, chosen = set(), []
    for i in order:
        pid = p["personId"].iat[i]
        if pid not in seen:
            seen.add(pid)
            chosen.append(i)
    chosen = np.sort(np.array(chosen))

    n = len(p)
    changed = total = 0
    for col in PERTURB_NUMERIC + PERTURB_CATEGORICAL:
        vals = p[col].to_numpy(copy=True)
        orig = vals[chosen].copy()
        donor = rng.integers(0, n, size=len(chosen))
        new = vals[donor]
        # Redraw where the donor happened to reproduce the original value, so
        # the perturbation actually bites. A few stubborn cells are tolerated
        # and shown in the effective-perturbation fraction below.
        for _ in range(5):
            same = pd.isna(new) & pd.isna(orig)
            with np.errstate(invalid="ignore"):
                same = same | (~pd.isna(new) & ~pd.isna(orig) & (new == orig))
            if not same.any():
                break
            redraw = rng.integers(0, n, size=int(same.sum()))
            new[same] = vals[redraw]
        vals[chosen] = new
        p[col] = vals
        total += len(chosen)
        changed += int((~(pd.isna(new) & pd.isna(orig))
                        & ~((~pd.isna(new)) & (~pd.isna(orig)) & (new == orig))).sum())

    # y/tissue are near-constant (98.9% zero), so a random donor would usually
    # redraw 0 and the perturbation would have no teeth. Force the flip instead:
    # both replacement values occur elsewhere in the column, which is what the
    # invariance requires.
    y = p["y"].to_numpy(copy=True)
    t = p["tissue"].to_numpy(copy=True)
    y[chosen] = 1 - y[chosen]
    t[chosen] = np.where(t[chosen] > 0, 0, 1)
    p["y"], p["tissue"] = y, t
    total += 2 * len(chosen)      # two columns, not one
    changed += 2 * len(chosen)

    # `inj_event_dt` has to move with the label or the perturbation has no
    # teeth for the injury-history block. Since 2026-09-18 those covariates are
    # dated by the injury's own event date rather than by the athlete's next
    # appearance, so flipping `y` alone would leave the event table unchanged
    # and Test B would certify `prior_injuries` on a panel it never perturbed.
    # A row flipped ON gets a plausible report date two calendar days after its
    # game; a row flipped OFF loses its event. Row t's OWN event is dated after
    # row t either way, so this still must not move row t's own history -- which
    # is exactly the property under test.
    if "inj_event_dt" in p.columns:
        ev = p["inj_event_dt"].to_numpy(copy=True)
        newly_on = chosen[y[chosen] == 1]
        newly_off = chosen[y[chosen] == 0]
        ev[newly_on] = (p["dt"].to_numpy()[newly_on].astype("datetime64[D]")
                        + np.timedelta64(2, "D")).astype("datetime64[ns]")
        ev[newly_off] = np.datetime64("NaT")
        p["inj_event_dt"] = ev
        total += len(chosen)
        changed += len(chosen)

    keys = pd.MultiIndex.from_arrays([p["personId"].to_numpy()[chosen],
                                      p["dt"].to_numpy()[chosen]],
                                     names=["personId", "dt"])
    return p, keys, changed / max(1, total)


def test_b(builder, panel, full_keyed, feature_cols, passes):
    """Perturb row t's own measurements; row t's features must not move."""
    v = Verdicts(feature_cols)
    detail = []
    for i in range(passes):
        rng = np.random.default_rng(B_SEED + i)
        pert, keys, frac = perturb(panel, rng, B.MIN_CAREER_GAMES)
        got = keyed(builder(pert))
        keys = keys.intersection(full_keyed.index).intersection(got.index)
        base = full_keyed.loc[keys]
        detail.append({"pass": i, "seed": B_SEED + i,
                       "rows_perturbed": int(len(keys)),
                       "cells_effectively_changed": round(float(frac), 6)})
        print(f"    pass {i} (seed {B_SEED + i}): perturbed {len(keys):,} rows "
              f"(one per athlete); {frac:.2%} of replaced cells actually changed")
        compare(base, got, feature_cols, v)
    return v, detail


# --- test C: upstream future-record perturbation -----------------------------

def test_c(appearances, full_keyed, feature_cols, cutoffs, seed=C_SEED,
           builder=None):
    """Perturb the INJURY RECORDS of future games, then re-run the whole
    label-and-cohort attachment. Covariates before T must not move.

    WHY THIS TEST EXISTS (criterion 11 of the 2026-09-17 repair plan)
    -----------------------------------------------------------------
    Tests A and B both start from `load_panel()`, i.e. from a panel that has
    already had its outcomes attached and (before 2026-09-18) had rows DELETED
    because of those outcomes. They can therefore say nothing whatever about
    label information entering during that attachment -- which is precisely
    where the label-filtered history defect lived. The project's strongest guard was structurally blind to
    the defect it most needed to find, and it passed throughout.

    This test starts one stage earlier, from `load_appearances()`, which is the
    first frame in the pipeline that no label has touched. It rewrites every
    injury record dated at or after T -- category, report date and anatomical
    location, shuffled among themselves and partly deleted -- and then runs
    `attach_outcomes()` and `build_features()` over the result. That changes
    which future rows are positives, which are eligible, and how many rows the
    analytic cohort has after T.

    Nothing before T may move. If anything does, some earlier covariate is a
    function of a later label, whatever route it took to get there.

    Rows AFTER T are expected to differ and are excluded from the comparison;
    only the shared earlier rows are compared, exactly as in Test A.
    """
    if builder is None:
        builder = rebuild
    v = Verdicts(feature_cols)
    detail = []
    for k, (label, T) in enumerate(cutoffs):
        rng = np.random.default_rng(seed + k)
        p = appearances.reset_index(drop=True).copy()
        future = (p["dt"] >= T).to_numpy()
        pos = np.flatnonzero(future)

        # Shuffle the injury record across future rows, then blank a third of
        # them and invent a record on a third of the rows that had none. The
        # point is not realism, it is that the future's labels become different
        # labels.
        for col in ("Injury Category", "Date Injured", "Anatomical_Location"):
            vals = p[col].to_numpy(copy=True)
            vals[pos] = vals[rng.permutation(pos)]
            p[col] = vals

        blank = pos[rng.random(len(pos)) < 0.33]
        for col in ("Injury Category", "Date Injured", "Anatomical_Location"):
            vals = p[col].to_numpy(copy=True)
            vals[blank] = np.nan
            p[col] = vals

        invent = pos[rng.random(len(pos)) < 0.05]
        cats = list(B.TISSUE)
        cat_vals = p["Injury Category"].to_numpy(copy=True)
        inj_vals = pd.to_datetime(p["Date Injured"], errors="coerce").to_numpy(copy=True)
        cat_vals[invent] = rng.choice(cats, size=len(invent))
        inj_vals[invent] = (p["dt"].to_numpy()[invent].astype("datetime64[D]")
                            + np.timedelta64(2, "D")).astype("datetime64[ns]")
        p["Injury Category"] = cat_vals
        p["Date Injured"] = inj_vals

        # `_conc_row` is derived from the two columns just rewritten, so it has
        # to be re-derived or the perturbation would be half-applied.
        p["_conc_row"] = [B.is_concussion(loc, cat) for loc, cat
                          in zip(p["Anatomical_Location"], p["Injury Category"])]

        got = keyed(builder(B.attach_outcomes(p, B.MATCH_WINDOW, verbose=False)))
        base = full_keyed[full_keyed["dt"] < T]
        shared = base.index.intersection(got.index)
        detail.append({"cutoff_label": label, "cutoff": str(T.date()),
                       "future_rows_perturbed": int(len(pos)),
                       "records_blanked": int(len(blank)),
                       "records_invented": int(len(invent)),
                       "rows_compared": int(len(shared))})
        print(f"    cutoff {T.date()} ({label}): rewrote injury records on "
              f"{len(pos):,} future rows "
              f"({len(blank):,} blanked, {len(invent):,} invented) -> "
              f"comparing {len(shared):,} earlier analytic rows")
        compare(base, got, feature_cols, v)
    return v, detail


# --- test D: label independence ----------------------------------------------

def test_d(appearances, full_keyed, feature_cols, passes=D_PASSES, seed=D_SEED,
           builder=None):
    """Rewrite injury records ANYWHERE and rebuild. Non-history covariates
    must not move.

    WHY THIS EXISTS, AND WHY TEST C IS NOT IT (added 2026-09-18, after review)
    --------------------------------------------------------------------------
    Test C was written to satisfy criterion 11 and it does: it perturbs future
    injury records and proves no earlier covariate reads them. But it was then
    described -- in this file, in RESULTS.md and in the plan -- as the test that
    would have caught the label-filtered history defect. **That was wrong, and an external review caught
    it.**

    The label-filtered history defect was not a covariate reading the future. It was `rest_days` on row t
    reading a label attached to row t-1, which is in its PAST. Test C perturbs
    by GAME DATE: every cutoff leaves the previous game's record untouched,
    because the previous game is by definition before the cutoff. Reproduced in
    miniature: games on Jan 1/3/5 with a Jan 6 report matching the last two,
    correct build gives Jan 5 `rest_days = 2` and the buggy pre-filtered build
    gives 4 -- and **no Test C cutoff makes the two disagree**, because at any
    cutoff past the games there is nothing left to perturb.

    This test states the property the label-filtered history defect actually violated:

        a covariate that is not part of the injury-history block must be
        IDENTICAL under any relabelling whatever, anywhere in the panel.

    Rest, rolling workload, cumulative exposure, position and biometrics are
    functions of appearances alone. If changing an injury record moves one of
    them, a label has leaked into feature construction -- whatever direction in
    time it came from.

    The four `features.HISTORY` covariates are EXEMPT and must be: they are
    legitimately derived from labels, and requiring them to be label-invariant
    would be requiring them to be constant. They are covered by Tests A and B,
    which bound WHEN they may read a label rather than WHETHER.

    Rows whose own eligibility the perturbation changed simply leave the
    analytic table; `compare()` intersects, so they are excluded rather than
    scored as differences.
    """
    if builder is None:
        builder = rebuild
    v = Verdicts(feature_cols)
    detail = []
    cats = list(B.TISSUE)
    for i in range(passes):
        rng = np.random.default_rng(seed + i)
        p = appearances.reset_index(drop=True).copy()
        n = len(p)
        idx = np.arange(n)

        # Shuffle every injury record across the WHOLE panel -- no cutoff.
        for col in ("Injury Category", "Date Injured", "Anatomical_Location"):
            vals = p[col].to_numpy(copy=True)
            vals[idx] = vals[rng.permutation(idx)]
            p[col] = vals

        blank = idx[rng.random(n) < 0.30]
        for col in ("Injury Category", "Date Injured", "Anatomical_Location"):
            vals = p[col].to_numpy(copy=True)
            vals[blank] = np.nan
            p[col] = vals

        invent = idx[rng.random(n) < 0.02]
        cat_vals = p["Injury Category"].to_numpy(copy=True)
        inj_vals = pd.to_datetime(p["Date Injured"], errors="coerce").to_numpy(copy=True)
        cat_vals[invent] = rng.choice(cats, size=len(invent))
        inj_vals[invent] = (p["dt"].to_numpy()[invent].astype("datetime64[D]")
                            + np.timedelta64(2, "D")).astype("datetime64[ns]")
        p["Injury Category"] = cat_vals
        p["Date Injured"] = inj_vals
        p["_conc_row"] = [B.is_concussion(loc, cat) for loc, cat
                          in zip(p["Anatomical_Location"], p["Injury Category"])]

        got = keyed(builder(B.attach_outcomes(p, B.MATCH_WINDOW, verbose=False)))
        shared = full_keyed.index.intersection(got.index)
        detail.append({"pass": i, "seed": seed + i,
                       "records_blanked": int(len(blank)),
                       "records_invented": int(len(invent)),
                       "rows_compared": int(len(shared))})
        print(f"    pass {i} (seed {seed + i}): relabelled the whole panel "
              f"({len(blank):,} blanked, {len(invent):,} invented) -> "
              f"comparing {len(shared):,} rows on {len(feature_cols)} "
              f"label-independent covariates")
        compare(full_keyed.loc[shared], got, feature_cols, v)
    return v, detail


# --- injected-leak positive control ------------------------------------------

def rebuild(panel):
    """The builder under test, silenced -- it is called dozens of times."""
    return B.build_features(panel, verbose=False)


def prior_label_rest(p):
    """Days since the athlete's last appearance that was NOT a labelled injury
    game -- the injected leak Test D alone can see.

    This is the label-filtered history defect rebuilt deliberately. The label-filtered history defect computed playing history on a
    panel from which labelled games had already been deleted, so a covariate
    that was supposed to measure the gap between appearances silently measured
    the gap between *uninjured* appearances instead. Here the same thing is done
    openly: the athlete's own game dates are masked wherever `y == 1`, carried
    forward, and differenced.

    Why only D can catch it, stated per test rather than asserted:

      A truncates the panel at T and compares rows before T. Every label this
        covariate reads at row t sits at or before row t, so truncating the
        future changes nothing it reads.
      B rewrites row t's own measurements, including flipping row t's own `y`.
        This covariate never reads row t's label -- only earlier ones -- and B
        compares row t alone, so it cannot move.
      C perturbs injury records dated at or after T and compares rows before T.
        USUALLY untouched, but NOT guaranteed -- see the caveat below, which is
        why C is deliberately excluded from the exclusivity assertion.
      D relabels the WHOLE panel with no cutoff, which is the only perturbation
        that reaches a row's own past.

    THE C CAVEAT (found by an external model review, 2026-09-18, and confirmed
    against `attach_outcomes`): Test C CAN reach a pre-cutoff label, by a route
    that is easy to miss. The index game of an injury event is the appearance
    with the smallest `match_gap` -- i.e. the LATEST one inside the window. When
    an event straddles T, that winner usually sits at or after T. If Test C
    blanks it, the event does not disappear; the next-closest candidate is
    promoted, and that candidate can be dated BEFORE T. Its `y` flips 0 -> 1, and
    a later pre-cutoff row reading it moves. So C's blindness to past labels is
    a property of the data at a given cutoff, not a structural guarantee.

    `INJECTED_EXPECT_ONLY` therefore asserts the negative for A and B ONLY,
    where the argument above is airtight, and records C's result without
    requiring it either way. Asserting C quiet would give this script a
    non-deterministic exit that depends on whether any event happens to straddle
    the chosen cutoff -- a guard that fails for a reason unrelated to what it
    guards, which is how guards get switched off.
    """
    dt = pd.to_datetime(p["dt"])
    clean = dt.where(p["y"].to_numpy() == 0)
    by_person = clean.groupby(p["personId"].to_numpy(), sort=False)
    prev_clean = by_person.shift(1).groupby(p["personId"].to_numpy(),
                                            sort=False).ffill()
    return (dt - prev_clean).dt.days.to_numpy()


def inject_leaks(panel):
    """Attach four deliberately-poisoned covariates to a build.

    Computed from the panel HANDED IN, exactly as a leaky builder would: that is
    what makes them respond to truncation and perturbation.
    """
    p = panel.reset_index(drop=True)
    out = B.build_features(p, verbose=False)
    poison = pd.DataFrame({
        "personId": p["personId"].to_numpy(),
        "dt": p["dt"].to_numpy(),
        "leak_same_game_minutes": p["numMinutes"].to_numpy(),
        "leak_career_mean_minutes":
            p.groupby("personId", sort=False)["numMinutes"].transform("mean").to_numpy(),
        "leak_future_injury":
            p.groupby("personId", sort=False)["y"].shift(-1).to_numpy(),
        "leak_prior_label_rest": prior_label_rest(p),
    })
    return out.merge(poison, on=["personId", "dt"], how="left")


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    t_start = time.time()

    print("=" * 84)
    print("  FEATURE-CONSTRUCTION LEAKAGE CONTROL (per-column, exact-equality)")
    print("=" * 84)
    print("  A. future truncation   -- covariates must not use rows dated >= T")
    print("  B. same-row perturbation -- covariates must not read game t itself")
    print(f"  Not perturbed (legitimately known pre-tip-off): {', '.join(NOT_PERTURBED)}")
    print()

    appearances = B.load_appearances(verbose=False)
    panel = B.attach_outcomes(appearances, B.MATCH_WINDOW, verbose=False)
    full = B.build_features(panel, verbose=False)
    full_keyed = keyed(full)
    feature_cols = [c for c in full.columns if c not in META]
    print(f"  panel {panel.shape[0]:,} rows -> analytic table {full.shape[0]:,} rows "
          f"x {len(feature_cols)} feature columns")
    print()

    cutoffs = build_cutoffs(panel, A_CUTOFF_SEASONS, A_MIDSEASON_CUTOFF_SEASONS)
    print("-" * 84)
    print(f"  TEST A -- FUTURE TRUNCATION ({len(cutoffs)} cutoffs: "
          f"{len(A_CUTOFF_SEASONS)} season openers, "
          f"{len(A_MIDSEASON_CUTOFF_SEASONS)} mid-season)")
    print("-" * 84)
    va, a_detail = test_a(rebuild, panel, full_keyed, feature_cols, cutoffs)
    print()
    print(va.table("Test A verdict:"))
    print()

    print("-" * 84)
    print(f"  TEST B -- SAME-ROW PERTURBATION ({B_PASSES} passes)")
    print("-" * 84)
    vb, b_detail = test_b(rebuild, panel, full_keyed,
                          feature_cols, B_PASSES)
    print()
    print(vb.table("Test B verdict:"))
    print()

    print("-" * 84)
    c_cutoffs = build_cutoffs(panel, C_CUTOFF_SEASONS, C_MIDSEASON_CUTOFF_SEASONS)
    print(f"  TEST C -- UPSTREAM FUTURE-RECORD PERTURBATION ({len(c_cutoffs)} cutoffs)")
    print("-" * 84)
    print("  Rewrites the injury records of games dated >= T and re-runs the")
    print("  whole label-and-cohort attachment. Tests A and B start AFTER that")
    print("  attachment and cannot see it; the label-filtered history defect lived exactly there.")
    vc, c_detail = test_c(appearances, full_keyed, feature_cols, c_cutoffs)
    print()
    print(vc.table("Test C verdict:"))
    print()

    print("-" * 84)
    print(f"  TEST D -- LABEL INDEPENDENCE ({D_PASSES} passes, whole-panel relabelling)")
    print("-" * 84)
    print("  Rewrites injury records across the WHOLE panel and rebuilds. Rest,")
    print("  workload, exposure, position and biometrics must be identical: they")
    print("  are functions of appearances alone. THIS is the test that catches")
    print("  The label-filtered history defect: Test C perturbs by game date, which leaves a row's own past")
    print("  labels intact in the label-filtered history defect configuration. (C is not blind to past")
    print("  labels in general -- it can reach one by index-game promotion -- so")
    print("  the control below asserts D's exclusivity against A and B only.)")
    d_cols = [c for c in feature_cols if c not in set(features.HISTORY)]
    print(f"  exempt (legitimately label-derived): {sorted(set(features.HISTORY))}")
    vd, d_detail = test_d(appearances, full_keyed, d_cols)
    print()
    print(vd.table("Test D verdict:"))
    print()

    # ---- adjudication of every real covariate -------------------------------
    print("=" * 84)
    print("  ADJUDICATION -- every feature column gets a verdict from both tests")
    print("=" * 84)
    fa, fb, fc = set(va.failed()), set(vb.failed()), set(vc.failed())
    fd = set(vd.failed())
    clean = [c for c in feature_cols
             if c not in fa and c not in fb and c not in fc and c not in fd
             and c not in features.CONCURRENT_ONLY]
    print(f"  PASS all four     : {len(clean):>4d} / {len(feature_cols)}")
    print(f"  FAIL A (future)   : {len(fa):>4d}   {sorted(fa)}")
    print(f"  FAIL B (game t)   : {len(fb):>4d}   {sorted(fb)}")
    print(f"  FAIL C (upstream) : {len(fc):>4d}   {sorted(fc)}")
    print(f"  FAIL D (labels)   : {len(fd):>4d}   {sorted(fd)}")
    print()

    # A failure OUTSIDE the disclosed set is a regression in build_features() --
    # exactly what this stage sits in the runner to catch. It must reach the exit
    # code, or the pipeline would go on to fit models and draw figures on a
    # leaking dataset while this stage printed FAIL and returned success.
    # `<var>_c5` is the S4.2 association comparator: a five-game window ENDING
    # AT the index game. It is deliberately not pre-game, it is stripped from
    # every frame `dataio.load()` returns unless a caller passes
    # concurrent="include", and it is therefore not a covariate any model can
    # see. It is audited here anyway, as a POSITIVE CONTROL rather than an
    # exemption: a concurrent window MUST fail the same-row perturbation test,
    # and if it ever passes, either the window was not built concurrent or
    # Test B has stopped working. Either way the S4.2 comparator would be
    # measuring nothing, so this fails the stage.
    concurrent_cols = [c for c in features.CONCURRENT_ONLY if c in feature_cols]
    concurrent_control = {}
    print("-" * 84)
    print("  CONCURRENT-WINDOW CONTROL -- these covariates MUST fail")
    print("-" * 84)
    for c in concurrent_cols:
        caught = dict(A=c in fa, B=c in fb, C=c in fc, D=c in fd)
        ok = caught["B"]
        concurrent_control[c] = dict(caught=caught, must_fail_test="B", passed=ok)
        print(f"  {c:32s} A={caught['A']!s:5s} B={caught['B']!s:5s} "
              f"C={caught['C']!s:5s} D={caught['D']!s:5s}  -> "
              f"{'OK (detected)' if ok else '*** ESCAPED ***'}")
    concurrent_escaped = [c for c, v in concurrent_control.items() if not v["passed"]]
    if not concurrent_cols:
        print("  none present in this build -- the dataset predates the S4.2 "
              "comparator (2026-09-21)")
    elif concurrent_escaped:
        print(f"  *** {concurrent_escaped} passed the same-row perturbation test.")
        print("  *** A window that includes game t cannot do that. Either the")
        print("  *** window is not concurrent or Test B is not wired to it.")
    print()

    # The concurrent columns are EXPECTED to fail Test B, so they are
    # subtracted from THAT test's failures only -- and only after the control
    # above has confirmed they really did fail it. Subtracting them from the
    # union would excuse a `_c5`/`_c1` column that started failing A, C or D,
    # i.e. one that had begun reaching the future or reading a label: it would
    # still pass the control (B is also true) and produce no `new_failures`
    # entry, so the stage would exit 0 on a genuine leak. Narrowed 2026-09-21
    # after review; the previous line subtracted from `(fa | fb | fc | fd)`.
    excused = set(concurrent_cols)
    regressions = sorted(
        ((fb - excused) | fa | fc | fd) - KNOWN_FAILURES)
    fixed = sorted(KNOWN_FAILURES - (fa | fb | fc | fd))
    if regressions:
        print(f"  *** NEW FAILURES, not previously disclosed: {regressions}")
        print("  *** This is a feature-timing regression in build_features().")
        print("  *** Do not trust any downstream number until it is resolved.")
    else:
        # Phrased against `regressions`, which is post-subtraction, so it must
        # NOT claim the raw verdicts are clean -- the concurrent columns fail
        # Test B by design and are listed just above. Saying "no covariate
        # fails any test" here contradicted the block above it in the same
        # committed log. Corrected 2026-09-21 (and it said "three" tests long
        # after there were four).
        _exp = f", and the {len(concurrent_cols)} concurrent column(s) fail Test B by design"             if concurrent_cols else ""
        print("  No UNDISCLOSED covariate fails any of the four tests"
              f"{_exp}. KNOWN_FAILURES is "
              f"{sorted(KNOWN_FAILURES) or 'empty'}.")
    if fixed:
        print(f"  Note: previously-failing covariates now pass: {fixed}. If that")
        print("  was intended, remove them from KNOWN_FAILURES and update"
              " limitation 9.")
    print()

    # ---- injected-leak positive control -------------------------------------
    print("=" * 84)
    print("  POSITIVE CONTROL -- injected leaks must be caught")
    print("=" * 84)
    print("  A control that cannot fail is not a control. Four poisoned")
    print("  covariates are added to a scratch build and all four tests re-run.")
    print()
    inj_full = inject_leaks(panel)
    inj_keyed = keyed(inj_full)
    inj_cols = list(INJECTED_EXPECT)

    inj_cutoffs = build_cutoffs(panel, INJECT_A_CUTOFF_SEASONS, [])
    print(f"  TEST A on the poisoned panel ({len(inj_cutoffs)} cutoffs)")
    iva, _ = test_a(inject_leaks, panel, inj_keyed, inj_cols, inj_cutoffs)
    print(f"  TEST B on the poisoned panel ({INJECT_B_PASSES} passes)")
    ivb, _ = test_b(inject_leaks, panel, inj_keyed, inj_cols, INJECT_B_PASSES)
    inj_c_cutoffs = build_cutoffs(panel, [], INJECT_C_MIDSEASON_SEASONS)
    print(f"  TEST C on the poisoned panel ({len(inj_c_cutoffs)} cutoffs)")
    # Test C needs its own control for the same reason A and B do. The natural
    # one is `leak_future_injury`, which reads the NEXT row's outcome: rewriting
    # the injury records after T changes those outcomes, so a covariate reading
    # them must move on rows before T. If it does not, Test C is not wired to
    # anything.
    ivc, _ = test_c(appearances, inj_keyed, inj_cols, inj_c_cutoffs,
                    builder=inject_leaks)
    print(f"  TEST D on the poisoned panel ({INJECT_D_PASSES} passes)")
    # Test D's own control, added 2026-09-18. Until it existed, D was the only
    # test whose all-clear rested on nothing: a D wired to the wrong builder, or
    # perturbing a column the builder no longer reads, would print PASS exactly
    # as it does when it works. `leak_prior_label_rest` is the covariate that
    # makes D's PASS mean something.
    ivd, _ = test_d(appearances, inj_keyed, inj_cols, passes=INJECT_D_PASSES,
                    builder=inject_leaks)
    print()

    print(f"  {'injected covariate':28s} {'expect':>7s} {'caught by A':>13s} "
          f"{'caught by B':>13s} {'caught by C':>13s} {'caught by D':>13s} "
          f"{'verdict':>10s}")
    escaped, control_rows = [], {}
    for c in inj_cols:
        ca, cb = iva.diff[c] > 0, ivb.diff[c] > 0
        cc, cd = ivc.diff[c] > 0, ivd.diff[c] > 0
        caught = {"A": ca, "B": cb, "C": cc, "D": cd}
        want = INJECTED_EXPECT[c]
        ok = caught[want]
        if c in INJECTED_EXPECT_C:
            ok = ok and cc
        # An exclusive control must also stay exclusive: if a test other than
        # the expected one starts catching it, the control has stopped
        # isolating that test and the ESCAPED exit is the right alarm, even
        # though nothing "escaped" in the usual direction.
        #
        # C is EXEMPT from this check by design -- see the C caveat in
        # `prior_label_rest`. C can legitimately reach a pre-cutoff label when
        # an injury event straddles the cutoff and its index game is promoted
        # backwards, so requiring C to stay quiet would make this script's exit
        # code depend on the data rather than on the property under test. C's
        # result is recorded in the JSON either way.
        others_quiet = True
        if c in INJECTED_EXPECT_ONLY:
            others_quiet = not any(v for k, v in caught.items()
                                   if k != want and k != "C")
            ok = ok and others_quiet
        if not ok:
            escaped.append(c)
        control_rows[c] = {"expected_test": want,
                           "also_expected_by_C": c in INJECTED_EXPECT_C,
                           "expected_by_this_test_only":
                               c in INJECTED_EXPECT_ONLY,
                           "a_and_b_quiet": bool(others_quiet),
                           "caught_by_A": bool(ca),
                           "caught_by_B": bool(cb),
                           "caught_by_C": bool(cc),
                           "caught_by_D": bool(cd),
                           "rows_differing_A": int(iva.diff[c]),
                           "rows_differing_B": int(ivb.diff[c]),
                           "rows_differing_C": int(ivc.diff[c]),
                           "rows_differing_D": int(ivd.diff[c]),
                           "detected": bool(ok)}
        print(f"  {c:28s} {want:>7s} {str(ca):>13s} {str(cb):>13s} "
              f"{str(cc):>13s} {str(cd):>13s} "
              f"{'DETECTED' if ok else '*** ESCAPED ***':>10s}")
    print()

    elapsed = time.time() - t_start
    summary = {
        "config": {"A_cutoff_seasons": A_CUTOFF_SEASONS,
                   "A_midseason_cutoff_seasons": A_MIDSEASON_CUTOFF_SEASONS,
                   "B_passes": B_PASSES, "B_seed": B_SEED,
                   "C_cutoff_seasons": C_CUTOFF_SEASONS,
                   "C_midseason_cutoff_seasons": C_MIDSEASON_CUTOFF_SEASONS,
                   "C_seed": C_SEED,
                   "D_passes": D_PASSES, "D_seed": D_SEED,
                   "D_exempt": sorted(features.HISTORY),
                   "inject_A_cutoff_seasons": INJECT_A_CUTOFF_SEASONS,
                   "inject_B_passes": INJECT_B_PASSES,
                   "inject_C_midseason_seasons": INJECT_C_MIDSEASON_SEASONS,
                   "inject_D_passes": INJECT_D_PASSES,
                   "inject_expect": INJECTED_EXPECT,
                   "inject_expect_only": INJECTED_EXPECT_ONLY,
                   "not_perturbed": NOT_PERTURBED,
                   "known_failures": sorted(KNOWN_FAILURES)},
        "n_feature_columns": len(feature_cols),
        "n_pass_both": len(clean),
        "fail_test_A": sorted(fa),
        "fail_test_B": sorted(fb),
        "fail_test_C": sorted(fc),
        "fail_test_D": sorted(fd),
        "new_failures": regressions,
        "test_A_detail": a_detail,
        "test_B_detail": b_detail,
        "test_C_detail": c_detail,
        "test_D_detail": d_detail,
        "positive_control": control_rows,
        "positive_control_escaped": escaped,
        "concurrent_control": concurrent_control,
        "concurrent_control_escaped": concurrent_escaped,
        "runtime_seconds": round(elapsed, 1),
    }
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 84)
    print(f"  Done in {elapsed / 60:.1f} min. Machine-readable summary: {JSON_OUT}")
    print("=" * 84)

    failures = []
    if escaped:
        failures.append(f"injected leaks escaped detection: {escaped}")
    if regressions:
        failures.append(f"undisclosed feature-timing failures: {regressions}")
    if concurrent_escaped:
        failures.append(
            "concurrent association covariates passed the same-row "
            f"perturbation test, so they are not concurrent: {concurrent_escaped}")
    # Returned, not printed. Until 2026-09-20 these lines went to
    # `tee.terminal`, so they reached the console and NOT the tracked log --
    # a failing run left audit_feature_timing_output.txt byte-identical to a
    # passing one. `dataio.run_stage` prints the verdict while stdout is still
    # the Tee, so the log records it and the exit code still carries it.
    if failures:
        return 1, "FAIL: " + "; ".join(failures)
    return None


if __name__ == "__main__":
    sys.exit(main())
