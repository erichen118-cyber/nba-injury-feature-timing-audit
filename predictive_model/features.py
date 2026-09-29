"""Feature groups — the single definition, imported by every stage that uses them.

These lists name the strictly pre-game covariate blocks the models are built from.
Until 2026-09-06 each of `train_and_evaluate.py`, `absence_confound_analysis.py`,
`sensitivity_analyses.py` and `window_sensitivity.py` carried its own literal copy,
so adding or renaming a feature meant editing four files and a missed one would
silently change which covariates a stage saw — with no error, and no way to tell from
the JSON that two stages had disagreed about the ablation they were reporting. That is
the same drift class `RESULTS.md` exists to catch downstream, caught here instead.

`build_predictive_dataset.py`, which CREATES these columns, imports from here too --
added 2026-09-06 after a review pointed out that the producer still carried its own
copy, so the module's claim to be the single definition was not yet true. That was
the copy that mattered most: a covariate added to the builder alone falls into the
residual workload block downstream and is reported inside the ablation concluding
that workload adds nothing.

Nothing in this module reads data or depends on pandas; it is names only, so importing
it cannot change any result.
"""

# Box-score bases for the lagged rolling covariates ---------------------------
#
# Each of these gets six lagged aggregates in the builder -- `_r3`, `_r5`,
# `_r10`, `_sd5`, `_std`, `_acr` -- so this list times six is the workload
# block, and `len(BASE_VARS) * 6 + len(PARS) + len(HISTORY)` is the model
# covariate count the builder's `[SHAPE]` line reports.
#
# MOVED HERE 2026-09-22, from `build_predictive_dataset.BASE_VARS`. It was the
# last literal feature list outside this module, and `window_sensitivity.py`
# carried a SECOND hand-copied duplicate of it -- so the rebound-percentage drop below had
# to be made in two files that nothing reconciled, which is precisely the drift
# this module's docstring says it exists to prevent. One definition now; the
# builder that creates the columns and the sensitivity stage that re-selects
# them both read it.
#
# 2026-09-22 -- THREE COLUMNS REMOVED, and the removal is the point:
#
#   reboundPercentage, defensiveReboundPercentage, offensiveReboundPercentage
#
# The vendor emitted these as identically 0.00 on ~64% of 2015-16 and 2016-17
# active appearances that recorded a nonzero rebound COUNT -- 48,773 analytic
# rows, 22.6% of the cohort -- and rolling windows carried the contamination
# into 2017-18. The three are availability-denominated (share of rebounds
# available while the athlete was on court), and the coherent reading is that
# the vendor had no on-court opponent-rebound denominator for those two seasons.
# They cannot be reconstructed from this panel at any price, so they are dropped
# rather than repaired: 123 covariates -> 105.
#
# THE SIX COUNT-DERIVED REBOUND COVARIATES STAY. `reboundsTotal` is still here,
# deliberately. The counts are the trustworthy side of the contradiction, and
# `check_source_validity.py`'s 15-pair sweep found no defect in any
# count-derived column, including the team-denominated rebound shares built
# from these same counts. Dropping them too would have been the tidy-looking
# overreaction; it would have cost six admissible covariates for no reason.
BASE_VARS = [
    "numMinutes", "fieldGoalsAttempted", "reboundsTotal", "points", "assists",
    "turnovers", "foulsPersonal", "possessions", "usagePercentage",
    "playerImpactEstimate", "pace",
    "threePointersAttempted", "freeThrowsAttempted", "plusMinusPoints",
]

# The six suffixes the builder derives from every entry in BASE_VARS. Named so
# a reader can count the workload block without reading the builder, and so a
# stage that needs to recognize a lagged column has one place to ask.
LAG_SUFFIXES = ["_r3", "_r5", "_r10", "_sd5", "_std", "_acr"]

# Schedule -------------------------------------------------------------------
# AVAIL is the availability block: what the schedule did TO the athlete. It is the
# block the restricted-cohort experiment removes, so it is named separately rather
# than sliced out of SCHEDULE at each call site.
AVAIL = ["rest_days", "b2b", "season_opener", "games_last_7d", "games_last_14d"]

# EXPOSURE is accumulated participation: how much the athlete has actually played.
EXPOSURE = ["panel_games", "season_games", "season_minutes", "started_last_game"]

SCHEDULE = AVAIL + EXPOSURE

# Biometrics -----------------------------------------------------------------
BIO_BASE = ["AGE", "BMI", "AGE_missing", "BMI_missing"]

# Position dummies. Since 2026-09-18 `position` is the modal starting position
# over games STRICTLY BEFORE the index game, so it is genuinely pre-game and no
# longer carries limitation 9. `pos_Unstarted` replaces `pos_NeverStarted`: the
# old name asserted the athlete never started in the whole panel, which is a
# statement about the future. The builder reindexes its dummies onto exactly
# this list, so every build carries all four columns whether or not a category
# is populated -- which is what makes indexing a frame by `PARS` safe.
POSITION = ["pos_C", "pos_F", "pos_G", "pos_Unstarted"]

BIO = BIO_BASE + POSITION

# Injury history -------------------------------------------------------------
HISTORY = ["prior_injuries", "days_since_injury", "ever_injured", "injuries_last_365d"]

# Parsimonious specification: schedule + biometrics, no injury history.
#
# This list is unfiltered, and that is now safe. `train_and_evaluate.py` builds
# its own equivalent as `[c for c in SCHEDULE + bio if c in df.columns]`, with
# the position dummies read off the frame, while `window_sensitivity.py` and
# `absence_confound_analysis.py` index the frame with PARS directly. The two
# used to agree only because get_dummies happened to emit every position column
# on every dataset, so a narrow sensitivity rebuild missing one category would
# have raised a KeyError in one stage and quietly fitted a 16-covariate model
# reported as 17 in another. Reconciled 2026-09-18 at the producer instead: the
# builder reindexes onto POSITION, so the columns always exist.
PARS = SCHEDULE + BIO


# Association-only covariates -- NOT predictors, NOT pre-game ----------------
#
# `<var>_c5` is a five-game mean ENDING AT the index game, so it INCLUDES game
# t. It exists for one purpose: the S4.2 GEE comparator needs a like-for-like
# concurrent arm to set against the `_r5` mean ending at t-1. Pairing a
# five-game concurrent window against a five-game lagged window isolates
# TIMING; pairing a bare index-game value against a five-game lagged mean would
# confound timing with aggregation, which is the defect the historical
# specification had.
#
# Because it reads game t, it is a leakage hazard in any forecasting model and
# must never enter a predictor set. `dataio.load()` strips these columns unless
# a caller passes concurrent="include", and `audit_feature_timing.py` asserts
# that each of them FAILS its SAME-ROW PERTURBATION test (Test B) -- an
# exclusion that is demonstrated rather than trusted. If one ever passes, the
# window was not built concurrent.
#
# Test B, not the future-truncation test (Test A), is the right one and these
# columns correctly PASS Test A: a window ending at game t reads nothing dated
# after t, so truncating the panel later cannot move it. The defect is
# same-game dependence, not future dependence. Three files asserted Test A
# here until review corrected them on 2026-09-21.
# `<var>_c1` is the INDEX GAME'S OWN VALUE. It exists only to measure the
# mechanism behind the S4.2 comparator, and it is the most dangerous column in
# the build -- it is the raw same-game measurement the whole paper is about.
# It is gated exactly like `_c5`.
#
# It is needed because `_c5 - _r5` is NOT "what the index game does to the
# window". The windows are t-4..t and t-5..t-1, so their difference is
# (v_t - v_{t-5})/5: it adds game t AND drops game t-5. Measuring the collider
# requires the index game against the athlete's own preceding baseline, which
# is `_c1` against `_r5`. Caught in review 2026-09-21 after the wrong quantity
# had already reached the manuscript.
# 2026-09-22: `reboundPercentage` removed. §4.2's rebound arm rested on
# a covariate that is invalid for two of the nine seasons, and the plan's
# decision 3 drops it outright rather than restricting it to 2017-18+ (which
# would fit the table's two rows on two different cohorts -- the exact
# non-comparability the association-arm redesign existed to remove) or absorbing it into a season term
# (which would hide invalid data in a nuisance parameter). §4.2 is now
# field-goal attempts and AGE, and the paired contrast carries 3 cells, not 6.
CONCURRENT_ONLY_BASE = ["fieldGoalsAttempted"]
CONCURRENT_ONLY = ([f"{v}_c5" for v in CONCURRENT_ONLY_BASE]
                   + [f"{v}_c1" for v in CONCURRENT_ONLY_BASE])
