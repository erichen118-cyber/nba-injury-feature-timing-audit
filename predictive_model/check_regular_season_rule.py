"""check_regular_season_rule.py
================================================================================
Guard for the regular-season rule correction: the rule reproduces the real NBA schedule, and the
built dataset carries history that includes the NBA Cup games.

WHY THIS EXISTS
---------------
Until 2026-09-25 every stage filtered on `gameType == "Regular Season"`, which
silently dropped the 66 "NBA Emirates Cup" games of 2023-24. Those games count
in the standings, so teams showed 76-78 games instead of 82, and every athlete
who played in them had a wrong rest_days on the following game.

TWO CHECKS, deliberately independent of the builder
---------------------------------------------------
  1. Counting distinct gameIds per team straight from the master, under
     `dataio.REGULAR_SEASON_TYPES`, gives exactly 30 teams x 82 games in
     2023-24, and the one 2025 "Emirates NBA Cup" championship game (which does
     not count in the standings) is outside the rule. This uses no builder
     code; it is the league's own schedule arithmetic.
  2. A named row: LeBron James (personId 2544) on 2023-11-15 has rest_days == 1
     in the built dataset. He played a Cup game on 2023-11-14; under the old
     filter the row stored 7.

Both were watched failing against the old filter before being trusted.

What this does NOT see: a game missing from the master entirely. The 2023-24
NBA Cup championship (2023-12-09) has no rows, and because it does not count in
the standings the 30 x 82 check passes regardless (limitation 2b).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataio  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(SCRIPT_DIR, "..", "Data",
                      "master_model_ready_data_corrected.csv")
TXT_OUT = os.path.join(SCRIPT_DIR, "check_regular_season_rule.txt")

EVAL_SEASON = 2023          # panel convention: 2023 == 2023-24
EXPECTED_TEAMS = 30
EXPECTED_GAMES = 82
LEBRON, LEBRON_DATE, LEBRON_REST = 2544, "2023-11-15", 1


def games_per_team(master, types):
    """Distinct gameIds per team in EVAL_SEASON, among rows whose gameType is in `types`."""
    m = master[master["gameType"].isin(types)]
    m = m[m["season"] == EVAL_SEASON]
    return m.groupby("playerteamName")["gameId"].nunique()


def load_master():
    m = pd.read_csv(MASTER, low_memory=False,
                    usecols=["gameId", "gameDateTimeEst", "gameType",
                             "playerteamName"])
    m["dt"] = pd.to_datetime(m["gameDateTimeEst"], errors="coerce")
    m["season"] = np.where(m["dt"].dt.month >= 10, m["dt"].dt.year,
                           m["dt"].dt.year - 1)
    return m


def _run():
    failures = []
    m = load_master()

    counts = games_per_team(m, dataio.REGULAR_SEASON_TYPES)
    print(f"[82-GAME] 2023-24 teams: {len(counts)}; games per team: "
          f"min {counts.min()}, max {counts.max()}")
    if len(counts) != EXPECTED_TEAMS or not (counts == EXPECTED_GAMES).all():
        failures.append(f"2023-24 is not {EXPECTED_TEAMS} x {EXPECTED_GAMES}: "
                        f"{counts[counts != EXPECTED_GAMES].to_dict()}")

    champ = m[m["gameType"] == "Emirates NBA Cup"]
    print(f"[82-GAME] 'Emirates NBA Cup' games in master: "
          f"{champ['gameId'].nunique()}; inside the rule: "
          f"{bool(dataio.is_regular_season(champ['gameType']).any())}")
    if champ.empty:
        failures.append("no 'Emirates NBA Cup' row found -- the exclusion "
                        "check is vacuous; the master changed")
    elif dataio.is_regular_season(champ["gameType"]).any():
        failures.append("the 'Emirates NBA Cup' championship is inside the rule")

    d = dataio.load(columns=["personId", "dt", "rest_days"], verbose=False)
    row = d[(d["personId"] == LEBRON)
            & (d["dt"].dt.normalize() == pd.Timestamp(LEBRON_DATE))]
    print(f"[NAMED ROW] LeBron James {LEBRON_DATE}: rows {len(row)}, "
          f"rest_days {row['rest_days'].tolist()}")
    if len(row) != 1 or float(row["rest_days"].iat[0]) != LEBRON_REST:
        failures.append(f"LeBron {LEBRON_DATE} rest_days is not {LEBRON_REST}")

    if failures:
        return 1, "FAIL: " + "; ".join(failures)
    return 0, "PASS -- the regular-season rule reproduces 30 x 82 and the named row."


def main():
    return dataio.run_stage(_run, TXT_OUT)


if __name__ == "__main__":
    sys.exit(main())
