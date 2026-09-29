"""check_source_validity.py
================================================================================
SOURCE SEMANTICS, not feature timing. Does the vendor panel mean what its column
names say?

WHY THIS EXISTS
---------------
On 2026-09-22 an independent review found that the three
"share of available rebounds while the athlete was on court" columns --
`reboundPercentage`, `defensiveReboundPercentage`, `offensiveReboundPercentage`
-- are identically 0.00 on roughly two thirds of the 2015-16 and 2016-17 active
appearances that recorded a nonzero rebound COUNT. A worked example: personId
708, gameId 21500206, 2015-11-23 -- 17 minutes, 10 defensive rebounds, all three
percentages exactly 0.00. Rounding cannot produce that. The coherent reading is
that the vendor had no on-court opponent-rebound denominator for those two
seasons and emitted 0 rather than a null.

**Every guard this project had passed while that was true, and each of them
passed correctly.** `audit_feature_timing.py` certifies WHEN a covariate was
measured. `test_feature_construction.py` certifies that the builder computes
what it claims from whatever it is handed. `verify_no_stale_numbers.py`
certifies that the manuscript matches the JSON. Not one of them looks at
whether the SOURCE VALUE IS MEANINGFUL, so a covariate can be perfectly timed,
perfectly aggregated, perfectly reported and still be invalid. That is the hole
this stage closes, and it is why it runs BEFORE the dataset build: a defect in
the source is not worth propagating through 25 minutes of feature construction.

WHAT IT CHECKS
--------------
One signature, applied to 15 documented count/percentage pairs: a row whose
COUNT is strictly positive while the percentage derived from it is exactly zero.
For a genuine rate that is arithmetically impossible -- a made field goal cannot
give a 0.000 field-goal percentage -- so any occurrence is a source defect, not
a modeling choice.

The sweep is deliberately wider than the known defect. The point of a guard is
to catch the NEXT instance, and the three rebound columns were found by sweeping
families nobody suspected. Adding a pair costs one line here.

HOW AN EXCEPTION WORKS
----------------------
A known defect is not silenced; it is WRITTEN DOWN with its measured counts and
its date, in `KNOWN_DEFECTS` below, and this stage then asserts three things
about it rather than one:

  1. the cell is still defective   -- if it comes back clean, the master has
                                      been repaired or replaced, and the
                                      exception (and the feature drop that
                                      followed from it) must be revisited;
  2. it is defective by EXACTLY the recorded count, AND on exactly the recorded
                                      ROWS -- the master is supposed to be
                                      immutable, so either moving means
                                      something wrote to it. Identity is checked
                                      as well as count because a count alone can
                                      be preserved by a swap: heal one
                                      documented row, break an undocumented one
                                      in the same season, and a count-only guard
                                      reports PASS. Demonstrated against this
                                      file, not imagined;
  3. nothing else in the sweep is defective at all.

So the exception makes the pipeline HONEST about the defect rather than blind
to it. A cell that silently changed in either direction fails this stage.

WHAT IT DOES NOT DO
-------------------
It does not repair anything. `Data/master_model_ready_data_corrected.csv` is
immutable, and the true rebound percentages cannot be reconstructed from this
panel at any price -- REB% needs team and opponent rebounds recorded while the
athlete was on court, which the panel does not carry. The response was
to DROP the 18 rebound-percentage-derived predictors, not to impute them.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import dataio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ANALYSIS_DIR, "Data", "master_model_ready_data_corrected.csv")
TXT_OUT = os.path.join(SCRIPT_DIR, "check_source_validity_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "check_source_validity.json")

# The 15 (count, percentage) pairs. Every one of these is a quantity whose
# percentage is derived from the count on the left, so count > 0 with
# percentage == 0 is a contradiction in the source rather than a small number.
#
# The three availability-denominated rebound shares are listed FIRST because
# they are the defective family; the three TEAM-denominated rebound shares sit
# immediately below them precisely because they are built from the same counts
# and are clean, which is what bounds the defect to the denominator rather than
# the counts.
PAIRS = [
    # availability-denominated rebound shares -- DEFECTIVE, see KNOWN_DEFECTS
    ("reboundsTotal", "reboundPercentage"),
    ("reboundsDefensive", "defensiveReboundPercentage"),
    ("reboundsOffensive", "offensiveReboundPercentage"),
    # team-denominated rebound shares -- same counts, clean in both eras
    ("reboundsTotal", "percentTeamRebounds"),
    ("reboundsOffensive", "percentTeamOffensiveRebounds"),
    ("reboundsDefensive", "percentTeamDefensiveRebounds"),
    # shooting efficiency
    ("fieldGoalsMade", "fieldGoalsPercentage"),
    ("threePointersMade", "threePointersPercentage"),
    ("freeThrowsMade", "freeThrowsPercentage"),
    # team shares of the remaining box-score families
    ("assists", "percentTeamAssists"),
    ("assists", "assistPercentage"),
    ("steals", "percentTeamSteals"),
    ("blocks", "percentTeamBlocks"),
    ("turnovers", "percentTeamTurnovers"),
    ("points", "percentTeamPoints"),
]

# 2026-09-22. (count_col, pct_col, season_label) -> (measured count,
# digest of the offending rows' identities).
#
# Season labels follow the panel convention: 2015 == the 2015-16 season.
# Counts were measured on the master under `dataio.is_regular_season` (the regular-season rule correction: Regular Season + NBA Emirates Cup) and
# `numMinutes > 0`, which is exactly the restriction applied below. Every later
# season is 0 for all three rebound pairs, which is why only those six cells
# appear.
#
# THE DIGEST IS NOT DECORATION, and it was added the same day the stage was
# written, after a review found the hole and a direct test confirmed
# it. The digest is sha256 over the sorted "personId:gameId" of exactly the rows
# in that cell, truncated to 16 hex characters.
#
# A count alone pins HOW MANY rows are defective, not WHICH. So a change that
# heals one documented row and breaks a different, undocumented one in the same
# season leaves the count identical and the whole stage reports PASS. That is
# not hypothetical: healing personId 1642860 / gameId 22500747 and breaking
# personId 2544 / gameId 22500059 -- both blocks == 1, both season 2025 -- was
# run against this file's own logic and produced `unexpected {}`,
# `count_changed {}`, `disappeared []`, verdict PASS. An undocumented defective
# row had been silenced by an exception written for a different row.
#
# The failure matters most exactly where the cells are smallest, because a
# one-row cell needs only a one-row swap to be evaded -- and three of the nine
# cells below hold one or two rows. It is the same coverage failure this project
# has now found repeatedly: the guard was correct about the quantity it checked
# and simply did not check enough. Identity closes it, and costs one hash.
KNOWN_DEFECTS = {
    ("reboundsTotal", "reboundPercentage", 2015): (16570, "421314e5d66eda29"),
    ("reboundsTotal", "reboundPercentage", 2016): (16675, "2332161235d6a303"),
    ("reboundsDefensive", "defensiveReboundPercentage", 2015): (11056, "54f02e2b04a15e54"),
    ("reboundsDefensive", "defensiveReboundPercentage", 2016): (11093, "2915b7335e2ebe74"),
    ("reboundsOffensive", "offensiveReboundPercentage", 2015): (6160, "a1a0e82481645b15"),
    ("reboundsOffensive", "offensiveReboundPercentage", 2016): (6004, "1fd4296a3718667e"),

    # ---- SECOND, UNRELATED EXCEPTION -------------------------------------
    # Found by this stage's first run, 2026-09-22, and NOT part of the rebound-percentage removal. Four
    # isolated rows, each an athlete credited with exactly one block or one
    # turnover whose team share is 0.000:
    #
    #   22400019 Warriors  personId  203110  2024-11-15  1 block
    #   22400249 Suns      personId 1626220  2024-11-18  1 block
    #   22500747 Wizards   personId 1642860  2026-02-07  1 block
    #   22501063 Hornets   personId 1642883  2026-03-26  1 turnover
    #
    # These are single-cell vendor glitches, not a family defect: 4 rows in
    # 277,680 (0.0014%), scattered across three seasons and three columns, with
    # every other row of those columns clean. `percentTeam*` shares are
    # on-court-denominated (they do not sum to 1 across a roster), so they are
    # in the same class as the removed rebound-percentage columns -- but at four rows rather than
    # 64% of two seasons.
    #
    # **They are immaterial to every reported number**, and that is checked
    # rather than asserted: no `percentTeam*` column appears in
    # `build_predictive_dataset.BASE_VARS`, so none of them is a predictor, an
    # association covariate, or an input to any table or figure. They are
    # recorded here so this stage stays strict -- a fifth occurrence fails.
    ("blocks", "percentTeamBlocks", 2024): (2, "89047fa733978d8c"),
    ("blocks", "percentTeamBlocks", 2025): (1, "5b6eb2d4e239dc9f"),
    ("turnovers", "percentTeamTurnovers", 2025): (1, "a0e37e73b91960f9"),
}
KNOWN_DEFECTS_NOTE = (
    "the rebound-percentage removal (2026-09-22): the vendor's availability-denominated rebound shares "
    "are identically 0.00 on ~64% of 2015-16 and 2016-17 active appearances "
    "that recorded a nonzero rebound count. The counts are trustworthy; the "
    "percentages are not. The 18 percentage-derived predictors were dropped; "
    "the 6 count-derived ones were kept. No repair is possible from this panel. "
    "Separately, 4 isolated single-cell glitches in percentTeamBlocks and "
    "percentTeamTurnovers (2024-25, 2025-26) are recorded above; they touch no "
    "predictor and no reported number."
)

# Set by --inject to prove the check can fail. See the block in `_run`.
INJECT = "--inject" in sys.argv


def row_digest(group):
    """Stable fingerprint of WHICH rows are in a defective cell.

    sha256 over the sorted "personId:gameId" of the group, truncated to 16
    hex characters. Sorted so row order cannot change it, and keyed on the
    pair that identifies a player-game rather than on a positional index,
    which a re-sorted master would invalidate for no real reason.
    """
    ids = sorted(f"{int(a)}:{int(b)}"
                 for a, b in zip(group["personId"], group["gameId"]))
    joined = chr(10).join(ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def season_of(dt):
    """NBA season label: games from October onward belong to that calendar year."""
    return np.where(dt.dt.month >= 10, dt.dt.year, dt.dt.year - 1)


def _run():
    # personId and gameId are read for the row-identity digest, not for any
    # analysis. They are the only stable key for a player-game in this panel.
    cols = sorted({c for pair in PAIRS for c in pair}
                  | {"gameDateTimeEst", "gameType", "numMinutes",
                     "personId", "gameId"})
    print("=" * 78)
    print("  SOURCE VALIDITY -- count > 0 while its percentage == 0")
    print("=" * 78)
    print(f"  master: {DATA}")
    print(f"  pairs:  {len(PAIRS)}")
    print()

    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    # The restriction the measured counts were taken under, and the only one
    # under which the question is well posed: a DNP has no rebounds to share,
    # and a playoff/preseason row is outside the analytic panel.
    df = df[dataio.is_regular_season(df["gameType"]) & (df["numMinutes"] > 0)]
    df["season"] = season_of(df["gameDateTimeEst"])
    seasons = sorted(int(s) for s in df["season"].unique())
    print(f"  active regular-season rows: {len(df):,} across seasons "
          f"{seasons[0]}-{seasons[-1]}")
    print()

    if INJECT:
        # Fault injection, run by hand only. Criterion 2 requires this stage to
        # be WATCHED FAILING rather than assumed to work -- a guard is vacuous
        # until you break something and see it complain. This flips one clean
        # cell (points / percentTeamPoints, most recent season) into the
        # defective signature on the IN-MEMORY frame only. Nothing is written,
        # and the master is opened read-only above.
        victim = df.index[(df["points"] > 0)
                          & (df["season"] == seasons[-1])][:7]
        df.loc[victim, "percentTeamPoints"] = 0.0
        print(f"  *** --inject: forced {len(victim)} rows of "
              f"points/percentTeamPoints in {seasons[-1]} to the defective "
              f"signature. This run SHOULD fail.")
        print()

    # ---- the sweep ----------------------------------------------------------
    observed = {}          # (count, pct, season) -> (n, digest)
    for count_col, pct_col in PAIRS:
        bad = df[(df[count_col] > 0) & (df[pct_col] == 0)]
        for season, group in bad.groupby("season"):
            observed[(count_col, pct_col, int(season))] = (
                len(group), row_digest(group))

    # ---- classify -----------------------------------------------------------
    unexpected = {k: v[0] for k, v in observed.items() if k not in KNOWN_DEFECTS}
    # Count AND identity, reported separately because they mean different
    # things: a count change is rows added or removed, while an identity change
    # at the SAME count is a swap -- the case a count-only guard misses whole.
    count_changed = {k: (KNOWN_DEFECTS[k][0], observed[k][0])
                     for k in KNOWN_DEFECTS if k in observed
                     and observed[k][0] != KNOWN_DEFECTS[k][0]}
    rows_changed = {k: (KNOWN_DEFECTS[k][1], observed[k][1])
                    for k in KNOWN_DEFECTS if k in observed
                    and observed[k][0] == KNOWN_DEFECTS[k][0]
                    and observed[k][1] != KNOWN_DEFECTS[k][1]}
    disappeared = [k for k in KNOWN_DEFECTS if k not in observed]

    # ---- report -------------------------------------------------------------
    print("  PER-PAIR TOTALS (all seasons pooled)")
    print(f"  {'count column':<22s} {'percentage column':<30s} {'rows':>9s}")
    for count_col, pct_col in PAIRS:
        total = sum(v[0] for (c, p, _), v in observed.items()
                    if c == count_col and p == pct_col)
        flag = "  <- KNOWN DEFECT" if any(
            k[0] == count_col and k[1] == pct_col for k in KNOWN_DEFECTS) else ""
        print(f"  {count_col:<22s} {pct_col:<30s} {total:>9,}{flag}")
    print()

    print("  DOCUMENTED EXCEPTIONS")
    print(f"  {KNOWN_DEFECTS_NOTE}")
    print()
    for key in sorted(KNOWN_DEFECTS):
        count_col, pct_col, season = key
        exp_n, exp_digest = KNOWN_DEFECTS[key]
        got = observed.get(key)
        if got is None:
            state = "MISSING -- cell is now clean"
        elif got[0] != exp_n:
            state = f"CHANGED -- recorded {exp_n:,}"
        elif got[1] != exp_digest:
            state = (f"SAME COUNT, DIFFERENT ROWS -- recorded {exp_digest}, "
                     f"found {got[1]}")
        else:
            state = "as recorded"
        print(f"  {season}-{str(season + 1)[-2:]}  {count_col:<20s} "
              f"{pct_col:<30s} {0 if got is None else got[0]:>8,}  {state}")
    print()

    failures = []
    if unexpected:
        print("  *** UNDOCUMENTED CONTRADICTIONS")
        for key in sorted(unexpected):
            count_col, pct_col, season = key
            print(f"      {season}-{str(season + 1)[-2:]}  {count_col} / "
                  f"{pct_col}: {unexpected[key]:,} rows")
        print()
        failures.append(
            f"{len(unexpected)} undocumented count>0 & percentage==0 cell(s): "
            + ", ".join(f"{c}/{p} in {s}" for c, p, s in sorted(unexpected)))
    if count_changed:
        failures.append(
            "a documented defect changed size, so the master is not the one "
            "these counts were measured on: "
            + ", ".join(f"{c}/{p} in {s}: recorded {e:,}, found {g:,}"
                        for (c, p, s), (e, g) in sorted(count_changed.items())))
    if rows_changed:
        failures.append(
            "a documented defect has the SAME COUNT but DIFFERENT ROWS, which "
            "is a swap: at least one row this exception silences is not a row "
            "it was written for. "
            + ", ".join(f"{c}/{p} in {yr}: recorded {e}, found {g}"
                        for (c, p, yr), (e, g) in sorted(rows_changed.items())))
    if disappeared:
        failures.append(
            "a documented defect is GONE, so the master was repaired or "
            "replaced and the 18-predictor drop that followed from it must be "
            "revisited: "
            + ", ".join(f"{c}/{p} in {s}" for c, p, s in sorted(disappeared)))

    results = {
        "master": os.path.basename(DATA),
        "restriction": "gameType in {'Regular Season', 'NBA Emirates Cup'} and numMinutes > 0",
        "n_rows": int(len(df)),
        "seasons": seasons,
        "n_pairs": len(PAIRS),
        "pairs": [list(p) for p in PAIRS],
        "known_defects_note": KNOWN_DEFECTS_NOTE,
        "known_defects": {"|".join(map(str, k)): {"n": v[0], "rows_digest": v[1]}
                          for k, v in sorted(KNOWN_DEFECTS.items())},
        "observed": {"|".join(map(str, k)): {"n": v[0], "rows_digest": v[1]}
                     for k, v in sorted(observed.items())},
        "unexpected": {"|".join(map(str, k)): v
                       for k, v in sorted(unexpected.items())},
        "count_changed": {"|".join(map(str, k)): list(v)
                          for k, v in sorted(count_changed.items())},
        "rows_changed": {"|".join(map(str, k)): list(v)
                         for k, v in sorted(rows_changed.items())},
        "disappeared": ["|".join(map(str, k)) for k in sorted(disappeared)],
        "injected": bool(INJECT),
        "pass": not failures,
    }
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[EXPORT] {JSON_OUT}")

    if failures:
        return 1, "FAIL: " + "; ".join(failures)
    print()
    print("  PASS -- every count/percentage contradiction in the master is one "
          "this stage was told to expect.")
    return None


def main():
    return dataio.run_stage(_run, TXT_OUT)


if __name__ == "__main__":
    sys.exit(main())
