"""
build_predictive_dataset.py
================================================================================
Construct a STRICTLY PRE-GAME feature matrix for injury forecasting.

DESIGN PRINCIPLE
----------------
The prediction task is: *before tip-off of game t, what is the probability that
this athlete sustains a tissue injury in or immediately after game t?*

That question forbids the use of any quantity produced BY game t. The original
pipeline violated this in three ways, all corrected here:

  1. Same-game box score (points, minutes, shots, rebound%) was used as a
     feature. These are outcomes of the game being predicted. -> REMOVED.
     Only lagged rolling aggregates of past games survive.

  2. `*_Acute` rolling means were computed with `min_periods=1` over a window
     INCLUDING the current game. -> All rolling windows are computed on the
     player-shifted series, so game t never enters its own features.

  3. `Total_Prior_Injuries_Any` was populated from the injury record matched to
     row t, then forward-filled. It therefore increments ON the injury row:
     empirically it rises on 70.8% of injury rows and 0.0009% of healthy rows.
     This is target leakage. -> REBUILT causally as the cumulative count of
     injuries strictly BEFORE game t.

     `Prior_Injuries_to_this_Body_Part` is deleted outright: "this body part"
     refers to the body part of the injury being predicted, which is unknown
     before tip-off and undefined for a healthy row.

A second audit (2026-07-14) found three further defects, all corrected here:

  4. `rest_days` was differenced across the whole career, so every season opener
     carried the offseason (~150-200 days) as "rest", clipped to the 30-day
     ceiling. Openers were 73% of all ceiling rows and pooled into the 5+ rest
     bucket with genuine mid-season absences -- the one bucket where the rest
     gradient reverses. -> Rest is now differenced WITHIN season; openers carry
     NaN and an explicit `season_opener` flag.

  5. `pos_C/F/G` were dummies of `startingPosition`, which is populated ONLY for
     starters: 47.8% of rows had all three set to zero. They encoded "started
     game t at position X" -- a property of game t -- while the manuscript
     described them as biometrics, and they sat in the 12-feature headline model.
     -> Split into `position` and `started_last_game` (lagged, and therefore
     admissible pre-tip-off). `position` was career-modal until 2026-09-18,
     which was itself a defect -- see 13 below.

  6. `career_games` counted games IN PANEL, not in a career (r=0.52 with season).
     -> Renamed `panel_games`. AGE/BMI missingness (12.3%) is now carried as
     explicit indicator columns rather than being silently median-imputed.

A third audit (2026-08-16) found one more, corrected here:

  7. `rest_days` differenced TIMESTAMPS, so it counted elapsed 24-hour periods
     rather than calendar days between games. NBA tip-offs are in the evening,
     so a genuine back-to-back is typically 22-23.5 h apart and truncated to
     ZERO, while a game two calendar days later can be 45 h and truncate to 1.
     Which side of that boundary a back-to-back fell on was decided by tip-off
     times -- time zones and broadcast slots -- not by recovery. Consequences on
     the old panel: the bin labeled "1 (back-to-back)" was 64% NOT
     back-to-backs (44,910 of 70,550 rows were two calendar days apart); `b2b`
     was wrong on ~54,000 of 216,074 rows; and 8,739 rows sat at 0, which fell
     outside every reporting bin because `pd.cut` starts at (0, 1], so they
     vanished from the manuscript's rest table entirely.
     -> Dates are normalized to midnight before differencing, making
     `rest_days` the number of calendar days between games -- the definition
     used by the NBA schedule, the load-management literature, and this
     manuscript's own prose. This changes reported rest-bin numbers; it does not
     change the exogenous/endogenous decomposition, which keys off team games
     missed and never reads `rest_days`.

A fourth audit -- an external review, 2026-09-17 -- found six more. They are
the reason this file was reorganized rather than patched, and five of the six
are corrected here (the sixth, the withdrawn availability interpretation, is an inferential question about the
manuscript and is out of scope for the builder):

  8.  The label-filtered history defect, THE STRUCTURAL ONE. `load_panel()` DELETED played games because
      of their injury labels -- non-biomechanical categories, non-index matched
      games, matched games outside the window -- and `build_features()` then
      computed rest, rolling workload and cumulative exposure on what survived.
      A deleted game is not a game the athlete missed, so a positive row's rest
      was inflated BY VIRTUE OF BEING POSITIVE: 1,567 cohort rows carried a
      wrong `rest_days`, 773 of them among the 2,354 positives. The headline
      rest gradient was largely manufactured by it -- recomputed from actual
      appearances, gap-4 risk falls from 4.877% to 1.243% and the availability
      contrast (8.798% vs 1.289%) collapses to 1.160% vs 1.280%.
      -> The pipeline is now three stages, not two. `load_appearances()`
      applies only exposure filters and never reads a label;
      `attach_outcomes()` attaches labels and records analytic exclusions as an
      `eligible` MASK rather than deleting rows; `build_features()` computes
      every covariate on the full appearance panel and applies the mask last.
      `verify_history_reconstruction.py` reconciles the result against the
      master independently of this file.

  9.  The matching-window defect. The matching window was documented as 0-2 days and implemented as
      `(Date Injured - gameDateTimeEst).dt.days`, which floors an elapsed
      duration. At an evening tip-off that made the real window calendar gaps
      1-3, with same-day reports flooring to -1 and never matching.
      -> Both sides are normalized to midnight and MATCH_WINDOW is a calendar
      gap. Numerically unchanged: raw 0/1/2 map exactly onto calendar 1/2/3.

  10. The biometrics join defect. AGE and BMI were keyed on calendar year against a season-labeled
      source, so Oct-Dec games matched nothing and Jan-Aug games matched the
      wrong season. 27,739 cohort rows carried no biometrics that in fact
      existed.
      -> Joined on (personId, season) in `attach_biometrics()`, with a
      uniqueness assertion. AGE coverage went from 87.7% to 100%.

  11. The season-opener shift defect. `*_std` and `season_minutes` shifted across the whole career and
      only then grouped by season, so every season opener inherited the
      previous season's final game as its entire season-to-date.
      -> Shift and accumulate within (personId, season). An opener's `*_std` is
      NaN and its `season_minutes` is 0. The r3/r5/r10/sd5 windows still cross
      the boundary, deliberately.

  12. The injury-dating defect. The 365-day injury lookback lagged `y` by one GAME, dating an
      injury at the athlete's NEXT APPEARANCE. An injury on 2021-04-09 whose
      next appearance was 2021-10-20 was still "within the last year" on
      2022-10-01, so a row could carry `injuries_last_365d = 1` beside
      `days_since_injury = 540`.
      -> All three history covariates are dated by the injury's own event date,
      with strictly-before boundaries, and the contradiction is asserted away.

  13. The career-modal position defect. `position` was the modal startingPosition over the athlete's
      WHOLE panel, future included -- limitation 9, and the reason the
      abstract's strictly-pre-game guarantee had to be qualified.
      -> A prior-only tally. `KNOWN_FAILURES` in audit_feature_timing.py is now
      empty, and the guarantee is true as written.

LABEL COVERAGE (critical)
-------------------------
The injury database (`nba_injury_database_localized_refined.csv`) ends on
2024-05-27; the last injury that matches a regular-season game is 2024-04-14.
The box-score panel, however, runs through 2026. Seasons 2024-25 and 2025-26
therefore contain 52,806 active regular-season rows with ZERO injury records --
they are not healthy, they are UNLABELED. They are dropped.

Scraped coverage also drifts sharply upward over time:

    season   rows    injuries   rate
    2015-16  25,791     126     0.489%
    2016-17  25,881     219     0.846%
    ...
    2023-24  24,457     620     2.535%

A 5x rise in nine years is coverage improvement, not epidemiology. Early seasons
therefore contain unlabeled injuries (false negatives) in the healthy class.
This is recorded as a limitation and motivates the high-coverage sensitivity
cohort (2017-18 onward).

COHORT -- two different things, deliberately kept apart
------
  THE APPEARANCE PANEL (load_appearances) is every game the athlete played. It
  is filtered only by properties of the exposure, never by an outcome:
      Regular season only (preseason injuries are not indexed by the scraper;
      All-Star games are exhibitions).
      Active exposures only (numMinutes > 0) -- the athlete must have played.
      Seasons 2015-16 through 2023-24 (the labeled span).
  Everything an athlete's history is made of is computed on THIS.

  THE ANALYTIC COHORT is the subset that may be SCORED. It removes rows because
  of their labels, which is legitimate for fitting and fatal for feature
  construction, so it is applied only at the very end of build_features():
      Non-biomechanical categories excluded (illness, suspension, rest,
      unknown) -- concussion index games excepted; they get their own label.
      Matched games outside the window, and non-index matched games.
      Each athlete's first 5 IN-PANEL games (no stable history).

TARGET  (one positive per injury, not per matched game)
------
The calendar 1-3 day matching window causes a single injury to match up to TWO
games. Left uncorrected the duplicates double-weight a third of injuries and
induce spurious label autocorrelation between consecutive games.

Those duplicate positives are near-identical feature vectors describing one
event. Left uncorrected they double-weight 32% of injuries and induce spurious
label autocorrelation between consecutive games.

We therefore define the INDEX GAME of an injury as the game with the smallest
non-negative gap to the injury date -- i.e. the last game the athlete played
before the injury was recorded.

  y = 1  on the index game of each distinct (player, injury-date) event
  non-index matched games are DROPPED, not relabeled 0: the athlete may already
  have been symptomatic, so their true label is unknown.

  `tissue` retains the 3-class label for tissue-specific models.

CHECKS THAT GUARD THIS FILE
---------------------------
  predictive_model/test_feature_construction.py
      Hand-built fixtures for the four defects above that are VALUES rather
      than timings -- the whole-panel audits cannot state them. Runs in a
      second; there is no excuse for not running it.
  predictive_model/verify_history_reconstruction.py
      Reconciles rest, season openers and missed team games against the master,
      without reusing any code from this file.
  predictive_model/audit_feature_timing.py
      Per-column timing invariance. Test C starts at load_appearances(), which
      is the only one of the three that can see label information entering
      during cohort construction.

OUTPUT
------
  predictive_model/predictive_dataset.parquet  (or .csv.gz fallback)
  predictive_model/build_dataset_output.txt
"""

import os
import re
import sys
import datetime
import warnings

import numpy as np
import pandas as pd

import dataio
import features

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ANALYSIS_DIR, "Data", "master_model_ready_data_corrected.csv")

TISSUE = {
    "1. Ligament / Joint Injuries": 1,
    "2. Muscle / Tendon Injuries": 2,
    "3. Bone / Contusion Injuries": 3,
}
EXCLUDE_CATEGORIES = [
    "5. Exclusions",
    "4. Load Management / Non-Structural",
    "Unknown / General Injury",
]

# ---------------------------------------------------------------------------
# CONCUSSION -- a fourth, separate outcome boundary (added 2026-09-04)
# ---------------------------------------------------------------------------
# A concussion has no tissue designation, so `Injury Category` files it under
# "Unknown / General Injury" and the cohort filter above dropped the exposure
# row entirely: neither positive nor negative, a different outcome discarded
# through a category that misdescribes it. Four concussions had additionally
# been filed as "3. Bone / Contusion Injuries" and were being POOLED INTO THE
# MSK POSITIVES (Beal 2015-16, Embiid 2017-18, Korver 2019-20, Hayes 2021-22).
#
# FOUR COUNTS, FOUR DIFFERENT QUANTITIES -- they are easy to confuse and have
# been confused already:
#   143  concussion EXPOSURE ROWS retained by the cohort filter (matched rows,
#        including non-index games of the same event)
#   113  concussion EVENTS in the window, i.e. index games (109 filed Unknown,
#        4 filed Bone/Contusion)
#   110  concussion events surviving the MIN_CAREER_GAMES filter -- the analytic
#        stratum, and the only one of these that reaches the manuscript
#   109  events that were previously dropped as "Unknown / General Injury";
#        the other 4 were previously MSK positives, not dropped
#
# Concussion is a MECHANISM, not a tissue and not an anatomical location. It
# therefore gets its own label column -- never a fourth TISSUE code, which
# would let a tissue loop sweep it up by accident -- and the MSK models drop
# its index games from their risk set rather than scoring them as healthy.
# This mirrors the NFL adapter's `y_concussion` in the cross-sport replication.
#
# Detection is on `Anatomical_Location`, where the location field IS the
# designation; `comment` is null on every one of these rows, so no note text is
# read and this is a re-coding, not a re-annotation. Neck (41 records, mostly
# Muscle/Tendon) is musculoskeletal and deliberately stays in the MSK primary.
HEAD_REGION_VALUES = {"Concussion", "Head", "Left Concussion"}

# Any location naming the head region at all. Wider than the matcher on purpose:
# it is the tripwire for a new vendor value ("Facial fracture", "Concussion
# protocol") that the matcher below would silently drop. "Neck" is not in this
# net -- it is a known musculoskeletal location, not a head one.
_HEAD_LIKE = re.compile(r"concus|head|skull|brain|cranial|face|facial|nose|jaw|eye|temple",
                        re.IGNORECASE)


def is_concussion(location, category):
    """True when a record designates a concussion. NaN-safe on both fields.

    DESCRIPTOR OUTRANKS ANATOMY -- the rule this project established on its own
    data and must not contradict here. `ascertainment_analysis.py` found that
    where a source note names a tissue, the assigned `Injury Category` matches
    it in 76 of 76 audited records: a descriptor ("strain", "contusion") governs
    over the anatomical site. So:

      "Concussion" / "Left Concussion"  -> concussion. The LOCATION field is
          itself the diagnosis, whatever the category says. Two such records
          were filed "3. Bone / Contusion Injuries" (Beal, Embiid); that is a
          mis-assignment, and the location overrides it.

      bare "Head" with NO tissue category -> concussion. Nothing competes with
          it, and the record is otherwise discarded as Unknown / General Injury.

      bare "Head" WITH a tissue category -> NOT a concussion. "Head" is a site,
          not a diagnosis, and the category is positive evidence that the source
          note carried a tissue descriptor -- a head contusion, an orbital
          fracture. Two records are in this position (Korver 2019-11-08, Hayes
          2022-03-17) and they stay musculoskeletal.

    The middle and last cases are the whole point: an earlier revision matched
    any head-region location and so overrode an explicit tissue designation on
    those two records, which inverted the 76/76 rule (found in review,
    2026-09-04; decided by the author).
    """
    if not isinstance(location, str):
        return False
    v = location.strip()
    if re.search(r"concus", v, re.IGNORECASE):
        return True
    if v.lower() != "head":
        return False
    # Bare "Head": a concussion only when no tissue designation competes.
    return not (isinstance(category, str) and category.strip() in TISSUE)

# Base per-game statistics from which lagged rolling aggregates are derived.
# Defined in `features.py` since 2026-09-22 -- see the block there for
# why the three rebound-PERCENTAGE bases were removed and the count-derived ones
# kept. Re-exported under the old name so the rest of this module, and
# `audit_feature_timing.py`, read unchanged.
BASE_VARS = features.BASE_VARS

MIN_CAREER_GAMES = 5

# The three values `startingPosition` ever takes in the master (C, F, G; blank
# on the 47.8% of rows that are bench appearances). Named here because the
# prior-only `position` covariate tallies them in this order, and the tie-break
# is "first in this list" -- so the order is part of the definition, not a
# presentation choice.
POSITION_VALUES = ["C", "F", "G"]
# Carried by a row whose athlete has not started ANY earlier game. Deliberately
# not the old "NeverStarted": that label was a claim about the whole panel,
# including the future, which is the very thing criterion 10 removes.
NO_PRIOR_START = "Unstarted"
LAST_LABELED_SEASON = 2023   # season label 2023 == 2023-24
FIRST_SEASON = 2015
EPS = 1e-3

# Injury-to-game matching window, in CALENDAR DAYS (2026-09-18).
#
# It used to be the result of `(inj_dt - dt).dt.days`, which floors an elapsed
# duration rather than counting calendar days. `Date Injured` is a date parsed
# to midnight and `dt` is an evening tip-off, so an injury filed the NEXT
# CALENDAR DAY is ~5 h later and floors to 0. The window documented everywhere
# as "0-2 days" was therefore really calendar gaps 1-3, and a same-day report
# floored to -1 and never matched at all. Confirmed on the master: the raw
# .dt.days values 0/1/2 map exactly onto calendar gaps 1/2/3 with identical
# counts (1,265 / 1,820 / 1,653 on the active labeled panel).
#
# Both sides are now normalized to midnight and the window is stated in the
# units it is described in. MATCH_WINDOW is the MAXIMUM calendar gap:
#
#   MATCH_WINDOW = 3   primary -- calendar gaps 1-3 (the historical "0-2" raw)
#   MATCH_WINDOW = 2   prespecified sensitivity arm -- calendar gaps 1-2
#   MATCH_WINDOW = 1   prespecified sensitivity arm -- calendar gap 1 only
#
# Same-day reports (gap 0) are excluded BY DESIGN, not by accident: only 13 of
# 5,422 source injury records have an appearance on the transaction date, and a
# date-only report cannot be ordered against an evening tip-off. The minimum is
# therefore fixed at 1 and is not a tunable.
#
# The upstream merge in Data/compile_data_corrected.py fixes the maximum at
# calendar gap 3, so this can be NARROWED but not widened without re-running
# that merge against the raw injury database.
#
# A game whose injury falls outside the window is dropped from the analytic
# cohort, not relabeled healthy -- the athlete was hurt within three days either
# way, so their status on that game is unknown. It still contributes its
# APPEARANCE to every other row's playing history (see load_appearances).
# Parsed defensively: this module is imported by audit_feature_timing.py, and an
# unconditional int(sys.argv[1]) would crash that script at import time on any
# flag it was given. Only a bare integer argument is treated as the window.
MIN_MATCH_GAP = 1
_ARGV_WINDOW = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit()
                else None)
MATCH_WINDOW = int(_ARGV_WINDOW) if _ARGV_WINDOW is not None else int(
    os.environ.get("MATCH_WINDOW", "3"))
SUFFIX = "" if MATCH_WINDOW == 3 else f"_w{MATCH_WINDOW}"

# Output paths carry the window suffix, so a sensitivity run at window 1 or 2
# cannot overwrite the primary dataset. (Passing "3" explicitly is not a
# sensitivity run -- it IS the primary build, and writes the primary paths.)
TXT_OUT = os.path.join(SCRIPT_DIR, f"build_dataset_output{SUFFIX}.txt")
PARQUET_OUT = os.path.join(SCRIPT_DIR, f"predictive_dataset{SUFFIX}.parquet")
CSV_OUT = os.path.join(SCRIPT_DIR, f"predictive_dataset{SUFFIX}.csv.gz")

# Longitudinal biometrics, re-joined here BY SEASON (2026-09-18). The upstream
# compile keyed AGE/BMI on calendar year against a season-labeled source, so
# every October-December game took the NEXT season's row or none at all: AGE
# and BMI were missing on all Oct-Dec rows and present on 77-91% of Jan-Aug
# rows, and 27,739 cohort rows carried no biometrics that in fact exist.
BIOMETRICS = os.path.join(ANALYSIS_DIR, "Longitudinal_Biometrics.csv")


def season_of(dt):
    """NBA season label: games from October onward belong to that calendar year."""
    return np.where(dt.dt.month >= 10, dt.dt.year, dt.dt.year - 1)


def load_appearances(verbose=True):
    """Every ACTIVE regular-season appearance in the labeled seasons.

    NOTHING label-dependent happens here. This is the athlete's real playing
    history: the games they turned up for. Outcome eligibility is a separate
    question, answered by `attach_outcomes()` and applied at the very end of
    `build_features()`.

    WHY THE SPLIT EXISTS (2026-09-18)
    -----------------------------------------
    Until this revision `load_panel()` DELETED played games because of the
    injury record attached to them -- non-biomechanical categories, non-index
    matched games, matched games outside the window -- and `build_features()`
    then computed rest, rolling workload and cumulative exposure on what was
    left. A deleted game is not a game the athlete missed, so a positive row's
    rest was inflated BY VIRTUE OF BEING POSITIVE. 1,567 cohort rows carried a
    wrong `rest_days`, 773 of them among the 2,354 positives (33%), and the
    study's headline rest gradient was largely manufactured by it. This is
    label-dependent feature construction -- the same class of defect as the
    2026-07 leak that inverted this study's conclusion, arriving by a different
    route.

    The filters that remain here are properties of the EXPOSURE, not of the
    outcome, so none of them can depend on a label:

      * regular season only  -- preseason injuries are not indexed by the
        scraper and All-Star games are exhibitions
      * numMinutes > 0       -- the athlete has to have played; a DNP is not an
        appearance and must not enter a workload window
      * seasons 2015-16 .. 2023-24 -- the labeled span

    Returns the appearance panel sorted by (personId, dt) with a fresh
    RangeIndex, carrying the raw injury-record columns unfiltered.
    """
    def say(*a):
        if verbose:
            print(*a)

    usecols = list(dict.fromkeys(
        ["personId", "Player", "gameDateTimeEst", "gameType", "numMinutes",
         "Injury Category", "Date Injured", "startingPosition",
         "Year", "Anatomical_Location"] + BASE_VARS
    ))
    df = pd.read_csv(DATA, low_memory=False, usecols=usecols)
    say(f"[LOAD] {len(df):,} rows")

    # Fail loudly on a head-region value this file has never seen. A new vendor
    # string would otherwise be dropped in silence -- exactly the failure that
    # buried 113 concussions under "Unknown / General Injury" for a year.
    seen_head = {v for v in df["Anatomical_Location"].dropna().unique()
                 if _HEAD_LIKE.search(v)}
    if seen_head != HEAD_REGION_VALUES:
        raise AssertionError(
            "Anatomical_Location head-region values changed.\n"
            f"  expected: {sorted(HEAD_REGION_VALUES)}\n"
            f"  found   : {sorted(seen_head)}\n"
            "Adding the new value to HEAD_REGION_VALUES only silences this "
            "check -- it does NOT make the value a concussion. Classification "
            "is done by is_concussion() below, and a value this "
            "tripwire knows about but that matcher rejects is DROPPED from the "
            "cohort, which is how 109 concussions sat under "
            "'Unknown / General Injury' for a year. Decide explicitly, edit "
            "both, and say which you chose in RESULTS.md.")

    # Row-level flag: this exposure carries a concussion record. It becomes the
    # `y_concussion` LABEL in attach_outcomes(), but only on the index game of
    # each distinct event -- the same rule the MSK target uses.
    df["_conc_row"] = [is_concussion(loc, cat) for loc, cat
                       in zip(df["Anatomical_Location"], df["Injury Category"])]
    say(f"[LOAD] head-region rows: "
        + ", ".join(f"{v} {int((df['Anatomical_Location'] == v).sum())}"
                    for v in sorted(HEAD_REGION_VALUES)))

    df["dt"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df = df.dropna(subset=["dt", "personId"])

    # ---- exposure filters (label-independent) -------------------------------
    df = df[dataio.is_regular_season(df["gameType"])]
    say(f"[APPEARANCES] regular season only        : {len(df):,}")

    df = df[(df["numMinutes"] > 0) & df["numMinutes"].notna()]
    say(f"[APPEARANCES] active exposures (min > 0)  : {len(df):,}")

    df["season"] = season_of(df["dt"])
    before = len(df)
    df = df[(df["season"] >= FIRST_SEASON) & (df["season"] <= LAST_LABELED_SEASON)]
    say(f"[APPEARANCES] seasons {FIRST_SEASON}-{LAST_LABELED_SEASON + 1} (labeled)  : "
        f"{len(df):,}   (dropped {before - len(df):,} UNLABELED rows in "
        f"2024-25 / 2025-26)")

    df = df.sort_values(["personId", "dt"]).reset_index(drop=True)
    df = attach_biometrics(df, say)
    return df


def attach_biometrics(df, say):
    """Join AGE and BMI on (personId, NBA SEASON) -- criterion 7.

    The upstream compile keyed these on `gameDateTimeEst.dt.year` against a
    source whose key is a season string ("2019-20" -> 2019). A season spans two
    calendar years, so the calendar-year key matched the January-to-August half
    of a season to the row for the season that STARTED that January -- the
    wrong season -- and matched October-to-December games to nothing at all.
    Measured on the master: AGE and BMI are missing on 100% of Oct/Nov/Dec rows
    and present on 77-91% of Jan-Aug rows, and 27,739 cohort rows carry no
    biometrics that in fact exist in the source.

    Keying on the season identifier both sides already carry removes the
    question. The master's own AGE/BMI columns are not read at all: a
    partly-right join is harder to reason about than a right one.
    """
    bios = pd.read_csv(BIOMETRICS, usecols=["personId", "SEASON", "AGE", "BMI"])
    bios["season"] = bios["SEASON"].str[:4].astype(int)
    dup = int(bios.duplicated(subset=["personId", "season"]).sum())
    if dup:
        raise AssertionError(
            f"Longitudinal_Biometrics.csv has {dup} duplicate (personId, season) "
            "rows, so the join would multiply appearance rows. Resolve the "
            "source before rebuilding -- do not silently keep='first'.")
    out = df.merge(bios[["personId", "season", "AGE", "BMI"]],
                   on=["personId", "season"], how="left", validate="many_to_one")
    assert len(out) == len(df), "biometrics join changed the row count"
    say(f"[BIOMETRICS] season-keyed join: AGE present on "
        f"{out['AGE'].notna().sum():,} / {len(out):,} "
        f"({out['AGE'].notna().mean():.1%}) appearances")
    return out


def attach_outcomes(app, match_window=MATCH_WINDOW, verbose=True):
    """Attach the outcome labels and the analytic-eligibility mask.

    Pure: no IO, does not mutate `app`. Adds

      y              1 on the index game of a tissue-designated injury event
      tissue         3-class tissue code on those rows, else 0
      y_concussion   1 on the index game of a concussion event
      inj_event_dt   the injury's own event date (`Date Injured`) on y == 1 rows
      eligible       True if the row may enter the ANALYTIC cohort

    `eligible` is the whole point of this function. Every rule below removes a
    row from model fitting BECAUSE OF ITS LABEL, and each one used to be a
    physical deletion performed before any covariate was computed. They are now
    recorded as a mask, so the appearance still counts toward other rows'
    playing history.
    """
    def say(*a):
        if verbose:
            print(*a)

    df = app.reset_index(drop=True).copy()

    df["inj_dt"] = pd.to_datetime(df["Date Injured"], errors="coerce")
    matched = df["Injury Category"].notna() & df["inj_dt"].notna()
    say(f"\n[TARGET] matched game-rows            : {int(matched.sum()):,}")

    # CALENDAR-day gap, both sides normalized to midnight. See the MATCH_WINDOW
    # block at the top of this file for why `.dt.days` on the raw timestamps was
    # not this number.
    df["match_gap"] = (df["inj_dt"].dt.normalize()
                       - df["dt"].dt.normalize()).dt.days
    in_window = matched & df["match_gap"].between(MIN_MATCH_GAP, match_window)
    say("[TARGET] calendar gap distribution    : "
        + ", ".join(f"{int(g)}d {int(n):,}" for g, n
                    in df.loc[matched, "match_gap"].value_counts().sort_index().items()))
    say(f"[TARGET] inside window {MIN_MATCH_GAP}-{match_window} days      : "
        f"{int(in_window.sum()):,}   "
        f"({int((matched & ~in_window).sum()):,} matched rows fall outside and "
        f"leave the analytic cohort -- status unknown, not relabeled healthy)")

    m = df[in_window]
    n_events = m.groupby(["personId", "inj_dt"]).ngroups
    say(f"[TARGET] distinct injury events       : {n_events:,}"
        f"   (inflation {in_window.sum() / max(1, n_events):.3f} rows/event)")

    # index game = smallest calendar gap to the injury date within each event
    idx_game = (m.sort_values("match_gap")
                  .drop_duplicates(subset=["personId", "inj_dt"], keep="first").index)
    non_index = m.index.difference(idx_game)

    df["tissue"] = 0
    df.loc[idx_game, "tissue"] = (df.loc[idx_game, "Injury Category"]
                                    .map(TISSUE).fillna(0).astype(int))

    # ---- concussion label ---------------------------------------------------
    conc_index = idx_game[df.loc[idx_game, "_conc_row"].values]
    df["y_concussion"] = 0
    df.loc[conc_index, "y_concussion"] = 1
    reclassified = df.loc[conc_index][df.loc[conc_index, "tissue"] > 0]
    df.loc[conc_index, "tissue"] = 0
    say(f"[TARGET] concussion index games      : {len(conc_index):,}")
    say(f"[TARGET] reclassified OUT of the MSK positives: {len(reclassified)}"
        f"   (were {sorted(reclassified['Injury Category'].unique())})")
    for _, r in reclassified.iterrows():
        say(f"    {r['Player']:<22s} {r['dt']:%Y-%m-%d}  {r['Anatomical_Location']}")

    df["y"] = (df["tissue"] > 0).astype(int)
    df["inj_event_dt"] = df["inj_dt"].where(df["y"] == 1)

    # ---- analytic eligibility ----------------------------------------------
    # Each clause removes a row that must not be SCORED. None of them is a game
    # the athlete failed to play, so every one still contributes its appearance
    # to the playing history of the rows around it.
    excluded_cat = df["Injury Category"].isin(EXCLUDE_CATEGORIES) & ~df["_conc_row"]
    outside = matched & ~in_window
    nonidx = pd.Series(False, index=df.index)
    nonidx.loc[non_index] = True
    unmapped = ((df["tissue"] == 0) & df["Injury Category"].notna()
                & (df["y_concussion"] == 0))

    df["eligible"] = ~(excluded_cat | outside | nonidx | unmapped)
    say(f"\n[ELIGIBILITY] appearances                 : {len(df):,}")
    say(f"[ELIGIBILITY] non-biomechanical category  : -{int(excluded_cat.sum()):,}")
    say(f"[ELIGIBILITY] matched outside the window  : -{int(outside.sum()):,}")
    say(f"[ELIGIBILITY] non-index matched game      : -{int(nonidx.sum()):,}")
    say(f"[ELIGIBILITY] unmapped injury category    : -{int(unmapped.sum()):,}"
        f"   (the four clauses overlap)")
    say(f"[ELIGIBILITY] analytic rows before the >= {MIN_CAREER_GAMES}-game "
        f"filter: {int(df['eligible'].sum()):,}")

    e = df[df["eligible"]]
    say(f"[TARGET] positives: {int(e.y.sum()):,} / {len(e):,} ({e.y.mean():.4%})")
    for name, code in TISSUE.items():
        say(f"    {name:32s}: {int((e.tissue == code).sum()):,}")
    say(f"    {'Concussion (separate stratum)':32s}: {int(e.y_concussion.sum()):,}")
    assert not (df.y & df.y_concussion).any(), \
        "a row is both an MSK positive and a concussion positive"
    assert df.loc[df["y"] == 1, "eligible"].all(), \
        "an MSK positive was marked ineligible -- the masks disagree"

    return df.drop(columns=["_conc_row"])


def load_panel(match_window=MATCH_WINDOW, verbose=True):
    """The full appearance panel with outcomes and eligibility attached.

    This is what `build_features()` expects, and -- since 2026-09-18 -- it is
    NOT pre-filtered by anything label-dependent. `audit_feature_timing.py`
    loads it once and rebuilds features from it many times.
    """
    return attach_outcomes(load_appearances(verbose=verbose),
                           match_window=match_window, verbose=verbose)


def build_features(panel, verbose=True):
    """Build the strictly pre-game covariate matrix from an appearance panel.

    Pure: takes the `load_panel()` frame, returns the analytic table. It does not
    read or write files, and it does not mutate `panel`.

    Every column here must be a function of rows STRICTLY BEFORE the index game.
    `audit_feature_timing.py` tests exactly that property by rebuilding this
    function on truncated and perturbed panels, so keep it free of IO: the
    control calls it dozens of times.

    ORDER, since 2026-09-18: every covariate is computed on the FULL appearance
    panel, and the analytic masks (`eligible`, then MIN_CAREER_GAMES) are
    applied only at the very end. Nothing above that filter may read `eligible`.
    """
    def say(*a):
        if verbose:
            print(*a)

    # Precondition: `panel` is already ordered by (personId, dt) -- load_panel
    # does that once. Re-sorting here would risk permuting same-day ties under
    # pandas' non-stable default and is therefore deliberately NOT done.
    df = panel.reset_index(drop=True).copy()
    if "eligible" not in df.columns:
        raise KeyError(
            "panel has no `eligible` column, so it predates the label-filtered history defect repair "
            "(2026-09-18). Rebuild it through load_panel().")

    pid = df["personId"]
    grp = df.groupby("personId", sort=False)
    season_keys = [pid.values, df["season"].values]
    pxs = df.groupby(["personId", "season"], sort=False)

    # ---- lagged rolling workload features -----------------------------------
    say("\n[FEATURES] Building lagged rolling aggregates (game t excluded)...")
    feat_frames = {}
    for v in BASE_VARS:
        # Career-scoped shift: the r3/r5/r10/sd5 windows DELIBERATELY reach
        # across a season boundary. They describe recent workload, and an
        # athlete's last games of the previous season genuinely are the games
        # before their opener. Intentional, and documented as such.
        prev = grp[v].shift(1)
        pg = prev.groupby(pid.values, sort=False)
        feat_frames[f"{v}_r3"] = pg.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        feat_frames[f"{v}_r5"] = pg.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        feat_frames[f"{v}_r10"] = pg.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        feat_frames[f"{v}_sd5"] = pg.rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
        # Season-to-date mean, strictly prior games WITHIN THE SAME SEASON.
        # The shift has to be season-scoped too (criterion 8): shifting
        # across the career and only then grouping by season handed every
        # season opener the previous season's final game as its entire "season
        # to date", so an opener carried `numMinutes_std = 40` where the
        # correct value is undefined. Season-scoped, an opener's `*_std` is NaN
        # and the second game of the season carries the opener alone.
        prev_in_season = pxs[v].shift(1)
        feat_frames[f"{v}_std"] = (
            prev_in_season.groupby(season_keys, sort=False)
                          .expanding().mean().reset_index(level=[0, 1], drop=True)
        )

    # ---- concurrent window -- ASSOCIATION COMPARATOR ONLY, never a predictor
    # A five-game mean ENDING AT game t (t-4..t), as against the `_r5` mean
    # ending at t-1. It deliberately INCLUDES the index game, which is why it
    # is listed in `features.CONCURRENT_ONLY` and subtracted from every
    # forecasting feature set. Its only consumer is the S4.2 GEE comparator,
    # which needs the two arms to differ in TIMING alone -- a bare index-game
    # value against a five-game lagged mean would confound timing with
    # aggregation. `audit_feature_timing.py` asserts these columns FAIL its
    # same-row perturbation test (Test B), so the exclusion is demonstrated,
    # not assumed. They correctly PASS the future-truncation test: a window
    # ending at t reads nothing after t.
    for v in features.CONCURRENT_ONLY_BASE:
        feat_frames[f"{v}_c5"] = (
            grp[v].rolling(5, min_periods=1).mean()
                  .reset_index(level=0, drop=True)
        )
        # The index game's own value, for the S4.2 mechanism measurement only.
        # `_c5 - _r5` cannot serve: those windows are t-4..t and t-5..t-1, so
        # their difference is (v_t - v_{t-5})/5 and conflates adding game t
        # with dropping game t-5.
        feat_frames[f"{v}_c1"] = df[v]

    feats = pd.DataFrame(feat_frames, index=df.index)

    # acute:chronic ratio -- 5-game load relative to season baseline. NaN on a
    # season opener by construction, because the denominator is undefined there.
    for v in BASE_VARS:
        feats[f"{v}_acr"] = feats[f"{v}_r5"] / (feats[f"{v}_std"].abs() + EPS)
    say(f"  rolling/ratio features: {feats.shape[1]}")

    # ---- schedule load ------------------------------------------------------
    say("[FEATURES] Building schedule-load features...")
    # Rest is measured WITHIN a season, from the athlete's ACTUAL previous
    # appearance, by differencing CALENDAR DATES rather than tip-off timestamps.
    # Two earlier defects here are recorded as 4 and 7 in this file's header;
    # the third is that "the previous appearance" used to be read off a
    # panel from which played games had already been deleted because of their
    # injury labels. It is now read off the full appearance panel.
    _gameday = df["dt"].dt.normalize()
    df["prev_game_dt"] = _gameday.groupby(season_keys, sort=False).shift(1)
    df["rest_days"] = (_gameday - df["prev_game_dt"]).dt.days
    df["season_opener"] = df["rest_days"].isna().astype(int)
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df["rest_days"] = df["rest_days"].clip(upper=30)

    ones = pd.Series(1.0, index=df.index)
    tmp = pd.DataFrame({"dt": df["dt"], "pid": pid, "ones": ones})
    for win, col in (("7D", "games_last_7d"), ("14D", "games_last_14d")):
        r = (tmp.set_index("dt").groupby("pid")["ones"]
                .rolling(win).sum().reset_index(level=0, drop=True))
        df[col] = r.values - 1.0        # exclude the current game

    # Games observed IN PANEL, not games played in a career: an athlete who was
    # already 30 years old in 2015-16 enters this counter at zero. It correlates
    # r=0.52 with season and r=0.48 with age. Named for what it actually is.
    df["panel_games"] = grp.cumcount()
    df["season_games"] = pxs.cumcount()

    # `startingPosition` is recorded ONLY for starters (47.8% of rows are blank),
    # so dummies built from it directly encode "started game t at position X" --
    # a property of game t, not a biometric. Split into its two real parts:
    #   position          - PRIOR-ONLY modal starting position (see below)
    #   started_last_game - role, lagged, and therefore admissible pre-tip-off
    df["is_starter"] = df["startingPosition"].notna().astype(int)
    df["started_last_game"] = grp["is_starter"].shift(1).fillna(0).astype(int)

    # POSITION -- criterion 10. It used to be the modal starting
    # position over the athlete's WHOLE panel, including games after the index
    # game, which is why `pos_*` sat in KNOWN_FAILURES and why the abstract's
    # "strictly pre-game" guarantee had to be qualified by limitation 9. It is
    # now the modal starting position over games STRICTLY BEFORE row t: a
    # running tally of prior starts at each position, argmax'd.
    #
    # Consequences, stated rather than hidden. The covariate is no longer
    # time-invariant within an athlete -- early rows may carry a different modal
    # position from late ones, which is exactly what "known at tip-off" means --
    # and an athlete with no prior start carries "Unstarted" rather than
    # "NeverStarted", which was a claim about the whole panel. Ties break toward
    # the first of C, F, G: deterministic, and reproducible across runs.
    prior_counts = {}
    for v in POSITION_VALUES:
        ind = (df["startingPosition"] == v).astype(float)
        prior = ind.groupby(pid.values, sort=False).shift(1).fillna(0.0)
        prior_counts[v] = prior.groupby(pid.values, sort=False).cumsum()
    pc = pd.DataFrame(prior_counts, index=df.index)
    df["position"] = np.where(pc.sum(axis=1).to_numpy() > 0,
                              pc.idxmax(axis=1).to_numpy(), NO_PRIOR_START)

    prev_min_in_season = pxs["numMinutes"].shift(1)
    df["season_minutes"] = (
        prev_min_in_season.groupby(season_keys, sort=False)
                          .expanding().sum().reset_index(level=[0, 1], drop=True).fillna(0)
    )

    # ---- causal injury history ---------------------------------------------
    # Criterion 9, the injury-dating defect. All three covariates are dated by the injury's own
    # EVENT date (`Date Injured`), not by the athlete's next appearance. The old
    # construction lagged `y` by one GAME, so an injury on 2021-04-09 whose next
    # appearance was 2021-10-20 entered the 365-day window on that October date
    # and was still "within the last year" on 2022-10-01 -- a row could carry
    # `injuries_last_365d = 1` alongside `days_since_injury = 540`.
    #
    # Strictly-before boundaries throughout: an event dated on the same calendar
    # day as the game does not count, which also guarantees the index game's own
    # injury (dated 1-3 days later) can never enter its own history.
    say("[FEATURES] Rebuilding prior-injury history from event dates...")
    gameday = _gameday.to_numpy()
    n = len(df)
    prior_n = np.zeros(n, dtype=float)
    n365 = np.zeros(n, dtype=float)
    last_evt = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    evt = (df.loc[df["y"] == 1, ["personId", "inj_event_dt"]].dropna()
             .sort_values(["personId", "inj_event_dt"]))
    events_by_person = {k: g["inj_event_dt"].to_numpy()
                        for k, g in evt.groupby("personId", sort=False)}
    year = np.timedelta64(365, "D")
    for person, idx in df.groupby("personId", sort=False).indices.items():
        arr = events_by_person.get(person)
        if arr is None:
            continue
        d = gameday[idx]
        hi = np.searchsorted(arr, d, side="left")     # events strictly before d
        lo = np.searchsorted(arr, d - year, side="left")
        prior_n[idx] = hi
        n365[idx] = hi - lo
        has = hi > 0
        last_evt[idx[has]] = arr[hi[has] - 1]

    df["prior_injuries"] = prior_n
    df["injuries_last_365d"] = n365
    dsi = (gameday - last_evt) / np.timedelta64(1, "D")
    df["ever_injured"] = (~np.isnan(dsi)).astype(int)
    df["days_since_injury"] = pd.Series(dsi, index=df.index).fillna(3650).clip(upper=3650)

    # The two cannot disagree: an injury inside the 365-day window is by
    # definition at most 365 days old. That was violable before and was
    # violated; it is now an invariant of the construction, asserted so a future
    # edit cannot quietly reintroduce the disagreement.
    bad = int(((df["injuries_last_365d"] > 0)
               & (df["days_since_injury"] > 365)).sum())
    if bad:
        raise AssertionError(
            f"{bad} rows carry injuries_last_365d > 0 with days_since_injury > "
            "365. The lookback and the recency clock disagree, which means one "
            "of them is not reading the injury event date.")

    # ---- assemble -----------------------------------------------------------
    # AGE/BMI missingness is carried explicitly rather than being silently
    # median-imputed downstream. Recomputed here, after the season-keyed join.
    df["AGE_missing"] = df["AGE"].isna().astype(int)
    df["BMI_missing"] = df["BMI"].isna().astype(int)

    meta = ["personId", "Player", "dt", "season", "y", "tissue", "y_concussion"]
    # From features.py, the one definition. This stage CREATES the columns that
    # every downstream ablation then slices by name, so a list defined here and a
    # list defined there is the drift that matters most: add a schedule covariate
    # to this builder alone and it falls into the residual "lagged workload" block
    # downstream, silently reporting a schedule feature inside the ablation that
    # concludes workload adds nothing. Order is preserved exactly.
    sched = features.SCHEDULE + features.HISTORY
    bio = list(features.BIO_BASE)

    out = pd.concat([df[meta + sched + bio], feats], axis=1)

    # Reindexed to the declared category list rather than to whatever
    # get_dummies happened to emit. `features.PARS` is unfiltered and indexes
    # the frame by name, so a category absent from one build (plausible on a
    # narrow sensitivity rebuild) used to mean a KeyError in one stage and a
    # silently-smaller model in another. Every build now carries all of them.
    pos = (pd.get_dummies(df["position"], prefix="pos", dtype=int)
             .reindex(columns=features.POSITION, fill_value=0))
    out = pd.concat([out, pos], axis=1)

    # ---- analytic masks, applied LAST ---------------------------------------
    before = len(out)
    out = out[df["eligible"].to_numpy()]
    say(f"\n[FILTER] analytic eligibility (label-dependent, applied after every "
        f"covariate): {len(out):,} rows (dropped {before - len(out):,})")
    before = len(out)
    out = out[out["panel_games"] >= MIN_CAREER_GAMES].reset_index(drop=True)
    say(f"[FILTER] require >= {MIN_CAREER_GAMES} prior in-panel games: "
        f"{len(out):,} rows (dropped {before - len(out):,})")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():

    print("=" * 84)
    print("  BUILD STRICTLY PRE-GAME PREDICTIVE DATASET")
    print("=" * 84)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print()

    out = build_features(load_panel(MATCH_WINDOW))
    meta = ["personId", "Player", "dt", "season", "y", "tissue", "y_concussion"]

    # The concurrent `_c5` columns are written to the dataset but are NOT
    # model covariates -- `dataio.load()` strips them unless a caller asks for
    # them, and only the S4.2 GEE comparator does. Counting them here would
    # make this line disagree with every `n_features` the model stages report.
    feature_cols = [c for c in out.columns
                    if c not in meta and c not in features.CONCURRENT_ONLY]
    assoc_cols = [c for c in out.columns if c in features.CONCURRENT_ONLY]
    print(f"[SHAPE] {len(out):,} rows x {len(feature_cols)} features "
          f"(+{len(assoc_cols)} association-only: {assoc_cols})")
    print(f"[TARGET] positives: {out.y.sum():,} ({out.y.mean():.4%})")

    print("\n[SEASON BREAKDOWN]")
    g = out.groupby("season").agg(rows=("y", "size"), pos=("y", "sum"))
    g["rate_pct"] = (g["pos"] / g["rows"] * 100).round(3)
    g["players"] = out.groupby("season")["personId"].nunique()
    print(g.to_string())

    # ---- leakage self-check -------------------------------------------------
    # The ORIGINAL counter was populated FROM the injury matched to row t, so it
    # rose on 70.8% of injury rows vs 0.0009% of healthy rows -- a target leak.
    #
    # The check below used to demand that the counter rise on exactly the rows
    # whose PREVIOUS analytic row was a positive, and it is deliberately no
    # longer stated that way. Since 2026-09-18 an injury is dated by its own
    # event date (`Date Injured`, 1-3 calendar days after the index game), not
    # by the athlete's next appearance, so the increment lands on the first game
    # played after the REPORT rather than on the first game played after the
    # index game. Those are usually the same row and occasionally are not:
    #
    #   * previous row injured, no increment -- the athlete played again before
    #     the report was filed, so at that game the injury was not yet known.
    #     Counting it would have been a small leak, and it is now excluded.
    #   * previous row healthy, increment -- an older report fell due between
    #     the two games, i.e. an intervening appearance separates the index game
    #     from the report.
    #
    # Both are correct under event dating and both are rare. What the leak test
    # actually requires is the asymmetry between healthy and injured rows, so
    # that is what is asserted; the two conditional rates are printed as
    # diagnostics with their expected magnitudes, not as invariants.
    print("\n[SELF-CHECK] prior_injuries counts only reports filed before game t")
    o = out.sort_values(["personId", "dt"])
    d = o["prior_injuries"] - o.groupby("personId")["prior_injuries"].shift(1)
    d = d.dropna()
    yy = o["y"].loc[d.index]
    y_lag = o.groupby("personId")["y"].shift(1).loc[d.index]
    print(f"  P(increase | healthy row)          = {(d[yy == 0] > 0).mean():.6f}")
    print(f"  P(increase | injured row)          = {(d[yy == 1] > 0).mean():.6f}")
    print(f"  P(increase | previous row injured) = {(d[y_lag == 1] > 0).mean():.6f}"
          f"  <- ~1.0; below it, the athlete played before the report")
    print(f"  P(increase | previous row healthy) = {(d[y_lag == 0] > 0).mean():.6f}"
          f"  <- ~0.0; above it, a report fell due between two games")
    print("  (old leaky counter: 0.000009 healthy vs 0.707941 injured)")
    # The counter must never rise ON the index game itself: that would be the
    # row reading its own outcome, which is the original leak exactly.
    own = int((d[yy == 1] > 0).sum() - (d[(yy == 1) & (y_lag == 1)] > 0).sum())
    print(f"  increments on an injured row whose previous row was healthy = "
          f"{own:,}  <- these are older reports falling due, not self-reads")

    print("\n[SELF-CHECK] rest_days never spans the offseason")
    o2 = out.sort_values(["personId", "dt"])
    opener = o2["season_opener"] == 1
    print(f"  season openers                     = {int(opener.sum()):,} (rest_days = NaN)")
    print(f"  max rest_days on non-opener rows   = {o2.loc[~opener, 'rest_days'].max():.0f}"
          f"  <- a genuine mid-season absence")
    print(f"  NaN rest_days off openers          = {int(o2.loc[~opener, 'rest_days'].isna().sum())}"
          f"  <- must be 0")

    # Defect 7 (2026-08-16). rest_days must count CALENDAR days, so it can never
    # be 0 -- two games cannot fall on the same date -- and `b2b` must agree
    # exactly with "the previous game was yesterday". Both were violated while
    # rest_days differenced timestamps: 8,739 rows sat at 0 and vanished from
    # every reporting bin, and b2b was wrong on ~25% of rows. These two lines are
    # what makes that impossible to reintroduce silently.
    print("\n[SELF-CHECK] rest_days counts calendar days, not elapsed 24h periods")
    # Deliberately checked on the STORED column, not by recomputing the date
    # difference here: `out` has already been filtered by MIN_CAREER_GAMES, so a
    # recomputation on this frame would difference against the wrong previous
    # game wherever that game was dropped, and would report false failures.
    #
    # Two games cannot fall on the same calendar date, so a calendar-based
    # rest_days can never be 0. Under the old timestamp differencing it was 0 on
    # 8,739 rows -- evening back-to-backs that were only ~22 h apart -- and those
    # rows then fell outside every reporting bin, because pd.cut starts at (0,1].
    zero = int((o2["rest_days"] == 0).sum())
    b2b_disagree = int(((o2["b2b"] == 1) != (o2["rest_days"] == 1)).sum())
    print(f"  rows with rest_days == 0           = {zero}  <- must be 0")
    print(f"  b2b disagrees with rest_days == 1  = {b2b_disagree}  <- must be 0")
    if zero or b2b_disagree:
        raise AssertionError(
            f"rest_days is not on a calendar-day basis: {zero} rows at 0, "
            f"{b2b_disagree} b2b mismatches. See defect 7 in this file's header.")

    # Position is no longer time-invariant, and must not be: a covariate that
    # is constant across an athlete's whole panel is by construction reading
    # their future. What must hold is that every row lands in exactly one
    # declared category -- which is also what makes indexing by features.PARS
    # safe on any build, however narrow.
    print("\n[SELF-CHECK] position is prior-only and exactly one-hot")
    pos_cols_chk = [c for c in out.columns if c.startswith("pos_")]
    flagged = out[pos_cols_chk].sum(axis=1)
    print(f"  categories                         = {pos_cols_chk}")
    print(f"  declared in features.POSITION      = {features.POSITION}")
    print(f"  rows with exactly one position set = {int((flagged == 1).sum()):,}"
          f" ({(flagged == 1).mean():.1%})  <- must be 100%")
    npos = out.groupby("personId")[pos_cols_chk].nunique().sum(axis=1)
    n_vary = int((npos > len(pos_cols_chk)).sum())
    print(f"  athletes whose position varies     = {n_vary:,}"
          f"  <- expected > 0 now: the tally is prior-only")
    print(f"  rows at '{NO_PRIOR_START}' (no prior start) = "
          f"{int(out[f'pos_{NO_PRIOR_START}'].sum()):,}")
    print(f"  (old encoding: 47.8% of rows all-zero -- it was 'started game t';")
    print(f"   the encoding before today: career-modal, which read the future)")
    if pos_cols_chk != list(features.POSITION) or int((flagged == 1).sum()) != len(out):
        raise AssertionError(
            "position dummies do not match features.POSITION exactly one-hot. "
            f"got {pos_cols_chk}, declared {list(features.POSITION)}.")
    # The line above tests one-hot only, while this block is headed "prior-only
    # and exactly one-hot" -- so until 2026-09-18 the prior-only half was
    # printed and never enforced, and a regression to the career-modal encoding
    # would have passed it. Asserting n_vary > 0 closes that: the career modal
    # is by construction constant within an athlete, so it scores exactly 0
    # here. This is a regression guard, not a proof of prior-only construction;
    # that is what Test D in `audit_feature_timing.py` is for.
    if n_vary == 0:
        raise AssertionError(
            "no athlete's position varies across their panel. A position that is "
            "constant within an athlete is reading their future -- this is the "
            "signature of the career-modal encoding removed on 2026-09-18.")

    print("\n[SELF-CHECK] season-to-date features carry no prior-season game")
    # Criterion 8. An opener has no earlier game IN ITS SEASON, so its
    # season-to-date mean is undefined and its accumulated season minutes are
    # zero. Before the season-scoped shift, an opener inherited the previous
    # season's final game and reported both as if that game had been played
    # this season.
    op = out[out["season_opener"] == 1]
    print(f"  season openers                     = {len(op):,}")
    print(f"  openers with numMinutes_std set    = {int(op['numMinutes_std'].notna().sum())}"
          f"  <- must be 0")
    print(f"  openers with season_minutes != 0   = {int((op['season_minutes'] != 0).sum())}"
          f"  <- must be 0")
    print(f"  openers with season_games != 0     = {int((op['season_games'] != 0).sum())}"
          f"  <- must be 0")
    if (int(op["numMinutes_std"].notna().sum())
            or int((op["season_minutes"] != 0).sum())
            or int((op["season_games"] != 0).sum())):
        raise AssertionError(
            "a season opener carries season-to-date state, so the shift is not "
            "season-scoped. See criterion 8.")

    print("\n[SELF-CHECK] injury history is dated by the event, not the next game")
    # Criterion 9. build_features() already raises on the hard
    # contradiction; this prints the margin so a reader can see it is not
    # vacuous.
    rec = out[out["injuries_last_365d"] > 0]
    print(f"  rows with an injury in the last year = {len(rec):,}")
    print(f"  worst days_since_injury among them   = "
          f"{(rec['days_since_injury'].max() if len(rec) else float('nan')):.0f}"
          f"  <- must be <= 365")
    print(f"  rows ever_injured with days_since_injury == 3650 = "
          f"{int(((out['ever_injured'] == 1) & (out['days_since_injury'] >= 3650)).sum())}"
          f"  <- must be 0")

    print("\n[SELF-CHECK] no feature may correlate with y at |r| > 0.30")
    num = out[feature_cols].select_dtypes(include=[np.number])
    corr = num.corrwith(out["y"]).abs().sort_values(ascending=False)
    print((corr.head(6)).round(4).to_string())
    if (corr > 0.30).any():
        print("  [WARN] suspiciously high correlation -- inspect for leakage")
    else:
        print("  OK: max |r| = {:.4f}".format(corr.max()))

    miss = out[feature_cols].isna().mean().sort_values(ascending=False)
    print(f"\n[MISSINGNESS] worst 5 features:")
    print((miss.head(5) * 100).round(2).to_string())

    export_dataset(out)


def export_dataset(out):
    """Write the analytic dataset, never leaving a stale or partial one behind.

    Extracted from `main()` on 2026-09-18 so the failure path is reachable from
    a test -- `test_feature_construction` case 10. This is the PRIMARY defense
    against downstream stages reading a superseded cohort; the staleness check
    in `dataio.dataset_path()` is the second line.
    """
    # Write to a sidecar and move it into place only on success. `to_parquet`
    # writes IN PLACE, so a mid-write failure -- disk full is the realistic
    # cause, and two runs have already died at ~2 GB free -- used to leave a
    # TRUNCATED file at PARQUET_OUT. The recovery below would then have treated
    # that fragment as "the previous parquet" and moved it over any genuine
    # backup, defeating its own purpose. Writing to `.tmp` first means a failed
    # write never touches the existing dataset at all. Found in review.
    tmp_parquet = PARQUET_OUT + ".tmp"
    try:
        out.to_parquet(tmp_parquet, index=False)
        os.replace(tmp_parquet, PARQUET_OUT)
        print(f"\n[EXPORT] {PARQUET_OUT}")
    except Exception as e:
        if os.path.exists(tmp_parquet):
            try:
                os.remove(tmp_parquet)
            except OSError:
                print(f"[EXPORT] could not remove {tmp_parquet}; delete by hand")
        # A PREVIOUS parquet surviving this failure is the dangerous case, not
        # the failure itself. `dataio.dataset_path()` returns the parquet
        # whenever it exists, so a build that fell back to CSV would succeed,
        # log the CSV it wrote, and every downstream stage would keep reading
        # yesterday's cohort. Reproduced 2026-09-18 with a stale 1%-prevalence
        # parquet beside a fresh 10% CSV: the loader returned the stale one.
        #
        # Move it aside rather than delete it: it is a COMPLETE previous build
        # (the `.tmp` write guarantees that), it may be the only copy, and this
        # repo's data is not all re-downloadable.
        print(f"\n[EXPORT] parquet unavailable ({e}); writing gzip csv")
        if os.path.exists(PARQUET_OUT):
            stale = PARQUET_OUT + ".stale"
            # Guarded: on Windows `os.replace` raises PermissionError when the
            # file is held open by another process (a running stage, an
            # indexer, antivirus). Unguarded, that propagated out of this
            # handler, the CSV below was never written, and the old parquet
            # stayed in place -- re-entering the exact staleness state this
            # block exists to prevent, through the recovery path itself.
            try:
                os.replace(PARQUET_OUT, stale)
                print(f"[EXPORT] *** moved the previous parquet aside -> {stale}")
                print("[EXPORT] *** it predates this build and readers prefer "
                      "parquet, so leaving it would have silently served the "
                      "old cohort to every downstream stage.")
            except OSError as move_err:
                print(f"[EXPORT] *** COULD NOT move the previous parquet aside "
                      f"({move_err}).")
                print(f"[EXPORT] *** {PARQUET_OUT} is STALE and readers prefer "
                      "parquet. Delete or rename it by hand before running any "
                      "downstream stage.")
        out.to_csv(CSV_OUT, index=False, compression="gzip")
        print(f"[EXPORT] {CSV_OUT}")


if __name__ == "__main__":
    main()
