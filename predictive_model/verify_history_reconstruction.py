"""verify_history_reconstruction.py
================================================================================
RECONCILE the analytic dataset's playing-history covariates against an
INDEPENDENT reconstruction from the master box-score panel.

WHY THIS EXISTS
---------------
The 2026-09-17 external review found that `load_panel()` deleted played games
because of their injury labels and `build_features()` then computed rest,
rolling workload and cumulative exposure on what was left. 1,567 cohort rows
carried a wrong `rest_days`, 773 of them among the 2,354 positives, and the
study's headline rest gradient was largely an artifact of it.

The repair reorders the pipeline so covariates are computed before any
label-dependent mask is applied. This script is the check that the repair
actually holds, and -- crucially -- it does NOT reuse the builder's code. It
re-derives each quantity straight from `master_model_ready_data_corrected.csv`
using the exposure definition alone (regular season, numMinutes > 0, labeled
seasons) and compares.

That independence is the point. A check written as `build_features()` a second
time would agree with a wrong builder. This one agrees only with a right one.

WHAT IS CHECKED  (acceptance criteria 2, 3 and 4 of the repair plan)
--------------------------------------------------------------------
  2. every cohort row's `rest_days` equals the gap to the athlete's actual
     previous active regular-season appearance in the same season
  3. no row flagged `season_opener` has an earlier same-season appearance
  4. `negative_control_analyses.py`'s own per-row `prev_date` and
     `team_games_missed`, as WRITTEN BY THAT STAGE, match an independent
     reconstruction

**Check 4 was vacuous until 2026-09-18** and is the reason this warning exists.
It recomputed the quantity here and then inspected its own recomputation,
testing whether an appearance fell strictly between two CONSECUTIVE
appearances. Nothing can, so it reported 0 forever and passed whatever the
analysis stage produced. It was caught by an external review, not by this
file. Two of the three checks here had been broken deliberately and watched
failing; this one had not. **Break every sub-check, not the total.**

RUN
---
  uv run python predictive_model/verify_history_reconstruction.py

Exits non-zero if any reconciliation fails, and names the worst offenders so a
failure is diagnosable rather than merely red.

OUTPUTS
-------
  predictive_model/verify_history_reconstruction_output.txt
  predictive_model/verify_history_reconstruction.json
"""

import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataio                              # noqa: E402
import build_predictive_dataset as B       # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(SCRIPT_DIR, "verify_history_reconstruction_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "verify_history_reconstruction.json")
# Written by negative_control_analyses.py. Criterion 4 is checked against
# THIS, not against a recomputation of it -- see the note in [4] below.
TEAM_GAP = os.path.join(SCRIPT_DIR, "team_gap_attribution.parquet")

# With --require-team-gap, criterion 4 may not be deferred: the artifact must be
# present, current and complete. run_pipeline.py passes it on the invocation that
# runs AFTER negative_control_analyses.py regenerates the artifact.
REQUIRE_TEAM_GAP = "--require-team-gap" in sys.argv

# How many mismatching rows to print when something fails. Enough to diagnose,
# not enough to bury the verdict.
SHOW = 10


def load_master_appearances():
    """Active regular-season appearances, straight from the master.

    Deliberately re-implemented here rather than imported from the builder:
    the whole value of this stage is that it does not share code with the thing
    it checks. The three filters are the exposure definition, none of which can
    depend on an injury label.
    """
    m = pd.read_csv(B.DATA, low_memory=False,
                    usecols=["personId", "gameDateTimeEst", "gameType",
                             "numMinutes", "playerteamId", "playerteamName"])
    m["dt"] = pd.to_datetime(m["gameDateTimeEst"], errors="coerce")
    m = m.dropna(subset=["dt", "personId"])
    m = m[dataio.is_regular_season(m["gameType"])]
    # Same 2021-22 gap as in `negative_control_analyses.py`: the team-game check
    # below groups on `playerteamId`, so without this that season contributed no
    # team schedule at all and its missed-team-game check was vacuous. Must come
    # after the Regular Season filter -- All-Star teams are named for their
    # captains and the same name carries different ids in different years.
    m = dataio.recover_team_ids(m)
    m = m[(m["numMinutes"] > 0) & m["numMinutes"].notna()]
    m["date"] = m["dt"].dt.normalize()
    m["season"] = np.where(m["dt"].dt.month >= 10, m["dt"].dt.year,
                           m["dt"].dt.year - 1)
    m = m[(m["season"] >= B.FIRST_SEASON) & (m["season"] <= B.LAST_LABELED_SEASON)]
    return m.sort_values(["personId", "season", "date"]).reset_index(drop=True)


def previous_appearance(app, keys_person, keys_season, keys_date):
    """For each (person, season, date), the athlete's previous appearance date.

    NaT where there is none, i.e. the athlete's first appearance of that season.
    """
    by = {}
    for (pid_, ssn), g in app.groupby(["personId", "season"], sort=False):
        by[(pid_, ssn)] = g["date"].to_numpy()
    out = np.full(len(keys_date), np.datetime64("NaT"), dtype="datetime64[ns]")
    for i in range(len(keys_date)):
        arr = by.get((keys_person[i], keys_season[i]))
        if arr is None:
            continue
        j = np.searchsorted(arr, keys_date[i], side="left")
        if j > 0:
            out[i] = arr[j - 1]
    return out


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 84)
    print("  PLAYING-HISTORY RECONCILIATION  (criteria 2, 3, 4)")
    print("=" * 84)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("  The analytic dataset is compared against an INDEPENDENT")
    print("  reconstruction from the master. No builder code is reused.")
    print()

    df = dataio.load(concussion="keep", verbose=False)
    df = df.copy()
    df["date"] = df["dt"].dt.normalize()
    print(f"  analytic cohort : {len(df):,} rows, {int(df.y.sum()):,} MSK events")

    app = load_master_appearances()
    print(f"  master appearances: {len(app):,} active regular-season rows "
          f"in seasons {B.FIRST_SEASON}-{B.LAST_LABELED_SEASON}")
    print()

    prev = previous_appearance(app, df["personId"].to_numpy(),
                               df["season"].to_numpy(), df["date"].to_numpy())
    expected_rest = (df["date"].to_numpy() - prev) / np.timedelta64(1, "D")
    expected_rest = pd.Series(expected_rest, index=df.index)
    # The builder clips at 30 days; the reconstruction has to clip identically
    # or every long absence would read as a mismatch of the clip, not of the
    # endpoint. The clip is a reporting decision, not a history claim.
    expected_clipped = expected_rest.clip(upper=30)

    results = {}

    # ---- criterion 2 -------------------------------------------------------
    print("-" * 84)
    print("  [2] rest_days matches the actual previous appearance")
    print("-" * 84)
    got = df["rest_days"]
    both_nan = got.isna() & expected_clipped.isna()
    mismatch = ~(both_nan | (got == expected_clipped))
    n_bad = int(mismatch.sum())
    n_bad_pos = int((mismatch & (df["y"] == 1)).sum())
    print(f"  rows compared                 : {len(df):,}")
    print(f"  rest_days mismatches          : {n_bad:,}   <- must be 0")
    print(f"  of which positives            : {n_bad_pos:,} / {int(df.y.sum()):,}")
    if n_bad:
        cols = ["Player", "dt", "season", "y", "rest_days"]
        bad = df.loc[mismatch, cols].copy()
        bad["expected"] = expected_clipped[mismatch]
        bad["prev_appearance"] = pd.Series(prev, index=df.index)[mismatch]
        print(bad.head(SHOW).to_string())
    results["rest_days_mismatches"] = n_bad
    results["rest_days_mismatches_positive"] = n_bad_pos
    print()

    # ---- criterion 3 -------------------------------------------------------
    print("-" * 84)
    print("  [3] no season opener has an earlier same-season appearance")
    print("-" * 84)
    opener = df["season_opener"] == 1
    false_openers = opener & pd.notna(pd.Series(prev, index=df.index))
    n_false = int(false_openers.sum())
    print(f"  rows flagged season_opener    : {int(opener.sum()):,}")
    print(f"  with an earlier appearance    : {n_false:,}   <- must be 0")
    if n_false:
        bad = df.loc[false_openers, ["Player", "dt", "season", "y"]].copy()
        bad["prev_appearance"] = pd.Series(prev, index=df.index)[false_openers]
        print(bad.head(SHOW).to_string())
    # ...and the converse: a row with no earlier appearance must be flagged.
    missed = (~opener) & pd.isna(pd.Series(prev, index=df.index))
    n_missed = int(missed.sum())
    print(f"  unflagged first-of-season rows: {n_missed:,}   <- must be 0")
    results["false_openers"] = n_false
    results["unflagged_openers"] = n_missed
    print()

    # ---- criterion 4 -------------------------------------------------------
    print("-" * 84)
    print("  [4] a claimed missed team game is a real one")
    print("-" * 84)
    print("  This compares what `negative_control_analyses.py` ACTUALLY WROTE")
    print("  against an independent reconstruction from the master.")
    print()
    print("  It did not, until 2026-09-18. The first version recomputed the")
    print("  quantity here and then inspected its own recomputation, asking")
    print("  whether an appearance fell strictly between two CONSECUTIVE")
    print("  appearances -- which is false by construction, so it returned 0 and")
    print("  passed no matter what the analysis stage produced. An external")
    print("  review caught it. A guard that cannot fail is not a guard.")
    print()
    if not os.path.exists(TEAM_GAP):
        print(f"  SKIP -- {os.path.basename(TEAM_GAP)} not present. Run")
        print("  negative_control_analyses.py first; this check is not optional,")
        print("  and a missing input is a FAIL, not a pass.")
        results["team_gap_rows_checked"] = 0
        results["team_gap_prev_date_mismatches"] = None
        results["team_gap_count_mismatches"] = None
        results["team_gap_rows_missing"] = None
        results["team_gap_rows_extra"] = None
        results["team_gap_input_missing"] = True
        results["team_gap_deferred"] = False
    elif (not REQUIRE_TEAM_GAP
          and os.path.getmtime(TEAM_GAP) < os.path.getmtime(dataio.dataset_path())):
        # ORDERING, not a defect. This stage runs immediately after the build so
        # that criteria 2 and 3 gate every downstream consumer, but criterion 4
        # checks an artifact that `negative_control_analyses.py` writes MUCH
        # later in the pipeline. Whenever the cohort changes -- the concussion
        # exclusion and the playerteamId recovery both did -- the artifact on
        # disk belongs to the previous cohort, and a coverage check against it
        # reports every new row missing. That failure stopped the pipeline
        # before the stage that would have regenerated the artifact could run,
        # so the pipeline could not bootstrap out of it by itself. Found in
        # review.
        #
        # The honest verdict at this point is "not checkable yet", so say that
        # rather than passing or failing. `run_pipeline.py` runs this stage a
        # SECOND time after the artifact is regenerated, with
        # --require-team-gap, where deferral is not permitted and the check is
        # real. A manual out-of-order run gets the honest DEFERRED; a full
        # pipeline run always gets the strict check.
        print("  DEFERRED -- the team-gap artifact predates the current")
        print("  dataset, so it belongs to a previous cohort. This stage runs")
        print("  before negative_control_analyses.py regenerates it. Criterion")
        print("  4 is checked by the later --require-team-gap invocation.")
        results["team_gap_rows_checked"] = 0
        results["team_gap_prev_date_mismatches"] = None
        results["team_gap_count_mismatches"] = None
        results["team_gap_rows_missing"] = None
        results["team_gap_rows_extra"] = None
        results["team_gap_input_missing"] = False
        results["team_gap_deferred"] = True
    else:
        results["team_gap_input_missing"] = False
        results["team_gap_deferred"] = False
        tg = pd.read_parquet(TEAM_GAP)
        tg["date"] = pd.to_datetime(tg["date"])
        tg["prev_date"] = pd.to_datetime(tg["prev_date"])

        # Independent reconstruction of the same two quantities.
        # SORTED and deduplicated. `app` is ordered by (personId, season,
        # date), so a per-team group comes out in athlete order, not
        # chronological -- and np.searchsorted on an unsorted array returns
        # nonsense rather than an error. That mistake produced 9,748 phantom
        # mismatches the first time this check was made real.
        team_dates = {t: np.sort(np.unique(g["date"].to_numpy()))
                      for t, g in app.groupby("playerteamId", sort=False)}
        own_team = (app[["personId", "date", "playerteamId"]]
                    .drop_duplicates(subset=["personId", "date"]))
        chk = tg.merge(own_team, on=["personId", "date"], how="left")
        prev_exp = previous_appearance(app, chk["personId"].to_numpy(),
                                       np.where(chk["date"].dt.month.to_numpy() >= 10,
                                                chk["date"].dt.year.to_numpy(),
                                                chk["date"].dt.year.to_numpy() - 1),
                                       chk["date"].to_numpy())
        miss_exp = np.full(len(chk), np.nan)
        darr = chk["date"].to_numpy()
        tarr = chk["playerteamId"].to_numpy()
        for i in range(len(chk)):
            pv = prev_exp[i]
            if np.isnat(pv) or pd.isna(tarr[i]):
                continue
            dates = team_dates.get(tarr[i])
            if dates is None:
                continue
            lo = np.searchsorted(dates, pv, side="right")
            hi = np.searchsorted(dates, darr[i], side="left")
            miss_exp[i] = max(0, hi - lo)

        got_prev = chk["prev_date"].to_numpy()
        prev_bad = ~((pd.isna(got_prev) & pd.isna(prev_exp)) | (got_prev == prev_exp))
        got_miss = chk["team_games_missed"].to_numpy(dtype=float)
        miss_bad = ~((np.isnan(got_miss) & np.isnan(miss_exp))
                     | (got_miss == miss_exp))
        n_prev_bad = int(prev_bad.sum())
        n_miss_bad = int(miss_bad.sum())

        print(f"  rows written by the analysis   : {len(chk):,}")
        print(f"  prev_date mismatches           : {n_prev_bad:,}   <- must be 0")
        print(f"  team_games_missed mismatches   : {n_miss_bad:,}   <- must be 0")
        print(f"  rows claiming a missed game    : "
              f"{int(np.nansum(got_miss > 0)):,}")
        if n_prev_bad or n_miss_bad:
            bad = chk.loc[prev_bad | miss_bad,
                          ["personId", "date", "prev_date",
                           "team_games_missed"]].head(SHOW).copy()
            bad["expected_prev"] = prev_exp[prev_bad | miss_bad][:SHOW]
            bad["expected_missed"] = miss_exp[prev_bad | miss_bad][:SHOW]
            print(bad.to_string())
        # COVERAGE, not just correctness. Everything above validates the rows
        # the artifact HAPPENS to contain and says nothing about the rows it
        # ought to contain, so an artifact missing a whole season reconciled
        # perfectly and returned PASS -- and so did an EMPTY one, which printed
        # "rows written by the analysis: 0" and passed. Verified by injection
        # 2026-09-18 (exit 0 both times); found in review.
        #
        # That is the same defect this file's own header describes, one layer
        # out: a guard that cannot fail. It matters here because the analysis
        # stage that writes this artifact drops rows on `playerteamId`, which is
        # exactly how 2021-22 went missing from the negative controls for months
        # without any guard noticing.
        # `df` above is loaded with concussion="keep" (216,212 rows), but the
        # analysis stage writes this artifact for the MSK risk set (216,104),
        # so the expectation has to drop the 108 concussion index games or the
        # guard would report them as missing on a perfectly good artifact.
        msk = dataio.drop_concussion(df, verbose=False)
        want = set(zip(msk["personId"].astype("int64").to_numpy(),
                       pd.to_datetime(msk["dt"]).dt.normalize().to_numpy()))
        have = set(zip(tg["personId"].astype("int64").to_numpy(),
                       tg["date"].dt.normalize().to_numpy()))
        missing, extra = want - have, have - want
        print(f"  analytic rows this artifact must cover : {len(want):,}")
        print(f"  rows missing from the artifact         : {len(missing):,}"
              f"   <- must be 0")
        print(f"  rows present that are not in the cohort: {len(extra):,}"
              f"   <- must be 0")
        if missing:
            by_season = {}
            for _, d in missing:
                s = pd.Timestamp(d)
                by_season[s.year - (1 if s.month < 8 else 0)] = \
                    by_season.get(s.year - (1 if s.month < 8 else 0), 0) + 1
            print("    missing rows by season: "
                  + ", ".join(f"{k}: {v:,}" for k, v in sorted(by_season.items())))
        results["team_gap_rows_missing"] = int(len(missing))
        results["team_gap_rows_extra"] = int(len(extra))

        results["team_gap_rows_checked"] = int(len(chk))
        results["team_gap_prev_date_mismatches"] = n_prev_bad
        results["team_gap_count_mismatches"] = n_miss_bad
    print()

    # ---- the diagnostic that started all this ------------------------------
    print("-" * 84)
    print("  Rest gradient as rebuilt (the table the review's diagnostic moved)")
    print("-" * 84)
    grad = []
    for g in (1, 2, 3, 4):
        sub = df[df["rest_days"] == g]
        n, k = len(sub), int(sub.y.sum())
        grad.append({"gap_days": g, "n": n, "events": k,
                     "rate": (k / n if n else float("nan"))})
        print(f"    {g} calendar day(s): {k:>5,} / {n:>7,}   "
              f"{(k / n * 100 if n else float('nan')):.3f}%")
    results["rest_gradient"] = grad
    print()

    team_gap_ok = (results["team_gap_input_missing"] is False
                   and results["team_gap_prev_date_mismatches"] == 0
                   and results["team_gap_count_mismatches"] == 0
                   and results["team_gap_rows_missing"] == 0
                   and results["team_gap_rows_extra"] == 0)
    if results["team_gap_deferred"]:
        # Deferred is neither pass nor fail -- unless the caller said it must be
        # checked now, in which case a deferral IS the failure.
        team_gap_ok = not REQUIRE_TEAM_GAP
    results["team_gap_required"] = bool(REQUIRE_TEAM_GAP)
    ok = (results["rest_days_mismatches"] == 0
          and results["false_openers"] == 0
          and results["unflagged_openers"] == 0
          and team_gap_ok)
    results["pass"] = bool(ok)
    results["n_rows"] = int(len(df))
    results["n_events"] = int(df.y.sum())
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("=" * 84)
    print(f"  {'PASS' if ok else 'FAIL'} -- machine-readable summary: {JSON_OUT}")
    print("=" * 84)
    # Returned rather than printed to `tee.terminal`: the banner above is in
    # the log, and the verdict line belongs there too. `dataio.run_stage`
    # writes it before closing the file.
    if not ok:
        return 1, "FAIL: playing-history reconciliation"
    return None


if __name__ == "__main__":
    sys.exit(main())
