"""test_feature_construction.py
================================================================================
FIXTURE TESTS for the covariate builder -- the cases the whole-panel audits
cannot state.

WHY THIS EXISTS
---------------
`audit_feature_timing.py` proves an INVARIANCE on the real panel: no covariate
moves when the future is truncated or when game t's own measurements are
replaced. That is the right test for "does this read the future", and it is
useless for "is this the number it claims to be". Four of the nine defects the
2026-09-17 external review found were of the second kind -- the covariate was
perfectly pre-game and simply wrong:

  - playing history was computed after label-dependent rows were deleted
  - the matching window was described in calendar days and implemented
    as floored elapsed hours
  - a season opener inherited the previous season's final game
  - an injury was dated at the athlete's next appearance, not at the
    injury

Each is checked here on a hand-built panel small enough that the right answer
can be written down by hand. These run in under a second and need no data
files, so there is no excuse for not running them.

RUN
---
  uv run python predictive_model/test_feature_construction.py

Exits non-zero on the first failure, with the case named.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_predictive_dataset as B     # noqa: E402
import dataio                            # noqa: E402
import features                          # noqa: E402

FAILURES = []


def check(case, got, want, note=""):
    """Record a single expectation. NaN equals NaN here, as everywhere else."""
    ok = (pd.isna(got) and pd.isna(want)) if (
        not isinstance(got, (list, tuple, np.ndarray))
        and not isinstance(want, (list, tuple, np.ndarray))
        and pd.isna(got) is not False and pd.isna(want) is not False
    ) else None
    if ok is None:
        if isinstance(want, float) and isinstance(got, (int, float)) and not pd.isna(want):
            ok = abs(float(got) - want) < 1e-9
        else:
            ok = bool(np.all(np.asarray(got) == np.asarray(want)))
    status = "ok  " if ok else "FAIL"
    print(f"    [{status}] {case}: got {got!r}, want {want!r}"
          + (f"   ({note})" if note else ""))
    if not ok:
        FAILURES.append(case)


def panel(rows, tip_hour=19):
    """Build a minimal appearance panel in the shape `load_appearances` returns.

    `rows` is a list of dicts with at least `pid` and `date`; `minutes`,
    `start` (startingPosition), `cat` (Injury Category), `inj` (Date Injured)
    and `loc` (Anatomical_Location) are optional.

    Tip-off defaults to 19:00 because that is the whole point of the window
    fixture: an evening tip-off is what made floored elapsed hours disagree
    with calendar days.
    """
    recs = []
    for r in rows:
        dt = pd.Timestamp(r["date"]) + pd.Timedelta(hours=tip_hour)
        rec = {
            "personId": r["pid"],
            "Player": f"P{r['pid']}",
            "gameDateTimeEst": dt,
            "gameType": "Regular Season",
            "numMinutes": r.get("minutes", 30.0),
            "Injury Category": r.get("cat"),
            "Date Injured": (pd.Timestamp(r["inj"]) if r.get("inj") else pd.NaT),
            "startingPosition": r.get("start"),
            "Year": dt.year,
            "Anatomical_Location": r.get("loc"),
            "dt": dt,
            "AGE": 27.0,
            "BMI": 25.0,
        }
        for v in B.BASE_VARS:
            rec.setdefault(v, r.get(v, 10.0))
        rec["numMinutes"] = r.get("minutes", 30.0)
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["season"] = B.season_of(df["dt"])
    df["_conc_row"] = [B.is_concussion(l, c) for l, c
                       in zip(df["Anatomical_Location"], df["Injury Category"])]
    return df.sort_values(["personId", "dt"]).reset_index(drop=True)


def build(rows, window=3, tip_hour=19):
    """Panel -> outcomes -> features, with the career-games gate disabled.

    MIN_CAREER_GAMES exists to keep athletes with no stable history out of the
    ANALYTIC cohort; it would delete every row of a five-game fixture. Patched
    to 0 for the duration and restored afterwards.
    """
    app = panel(rows, tip_hour=tip_hour)
    out_panel = B.attach_outcomes(app, match_window=window, verbose=False)
    saved = B.MIN_CAREER_GAMES
    B.MIN_CAREER_GAMES = 0
    try:
        feats = B.build_features(out_panel, verbose=False)
    finally:
        B.MIN_CAREER_GAMES = saved
    return out_panel, feats


MSK = "2. Muscle / Tendon Injuries"


# ---------------------------------------------------------------------------
def case_1_history_survives_its_own_exclusion():
    """The label-filtered history defect, criterion 1 / T2.

    One injury report matches TWO played games. The earlier one is not the
    index game, so it leaves the analytic cohort -- and used to be deleted from
    the panel before rest was computed, which handed the index game the rest it
    would have had if the athlete had never played that night.

    Jan 1, Jan 3, Jan 5 are all played. A report dated Jan 6 is 1 calendar day
    after Jan 5 and 3 after Jan 3, so both match and Jan 5 is the index game.
    Jan 5's rest must be 2 days, measured from the game the athlete actually
    played -- not 4 days, measured from Jan 1 after Jan 3 was deleted.
    """
    print("\n  case 1 -- an excluded game still contributes its appearance")
    rows = [
        dict(pid=1, date="2019-01-01", minutes=30),
        dict(pid=1, date="2019-01-03", minutes=30, cat=MSK, inj="2019-01-06"),
        dict(pid=1, date="2019-01-05", minutes=30, cat=MSK, inj="2019-01-06"),
    ]
    p, f = build(rows)
    check("Jan 3 is excluded from the analytic cohort",
          bool(p.loc[p["dt"].dt.day == 3, "eligible"].iat[0]), False)
    check("Jan 5 is the index game", int(f.loc[f["dt"].dt.day == 5, "y"].iat[0]), 1)
    check("Jan 5 rest_days is measured from Jan 3",
          float(f.loc[f["dt"].dt.day == 5, "rest_days"].iat[0]), 2.0,
          "4.0 would mean Jan 3 was deleted before rest was computed")
    check("Jan 3 does not appear in the analytic table",
          int((f["dt"].dt.day == 3).sum()), 0)
    # The same appearance must also count toward the rolling workload window.
    check("Jan 5 sees two prior games in its 7-day window",
          float(f.loc[f["dt"].dt.day == 5, "games_last_7d"].iat[0]), 2.0)


def case_2_matching_window_is_calendar_days():
    """The matching-window defect, criterion 5 / T4.

    At a 19:00 tip-off, `(Date Injured - gameDateTimeEst).dt.days` floors a
    same-day report to -1, a next-day report to 0, and so on -- the historical
    "0-2 day window" was calendar gaps 1-3. Both sides are now normalized to
    midnight, so the number means what it says.
    """
    print("\n  case 2 -- the window is calendar days at an evening tip-off")
    rows = [
        dict(pid=10, date="2019-01-02", minutes=30, cat=MSK, inj="2019-01-02"),
        dict(pid=11, date="2019-01-02", minutes=30, cat=MSK, inj="2019-01-03"),
        dict(pid=12, date="2019-01-02", minutes=30, cat=MSK, inj="2019-01-04"),
        dict(pid=13, date="2019-01-02", minutes=30, cat=MSK, inj="2019-01-05"),
        dict(pid=14, date="2019-01-02", minutes=30, cat=MSK, inj="2019-01-06"),
    ]
    p, f = build(rows)
    gaps = dict(zip(p["personId"], p["match_gap"]))
    check("same-day report has calendar gap 0", gaps[10], 0)
    check("next-day report has calendar gap 1", gaps[11], 1)
    check("two calendar days later", gaps[12], 2)
    check("three calendar days later", gaps[13], 3)
    check("four calendar days later", gaps[14], 4)
    # Raw .dt.days is what the code used to compute. Shown so the fixture
    # records the mapping the review found, not just the corrected value.
    raw = (p["inj_dt"] - p["dt"]).dt.days
    check("raw .dt.days floors the calendar gaps to 0-2",
          list(raw.to_numpy()), [-1, 0, 1, 2, 3],
          "this is the arithmetic the documented '0-2 day window' really used")
    positives = dict(zip(f["personId"], f["y"]))
    check("gap 0 is excluded by design", positives.get(10, 0), 0)
    for pid_ in (11, 12, 13):
        check(f"gap {gaps[pid_]} is inside the primary window",
              positives.get(pid_, 0), 1)
    check("gap 4 is outside the window", positives.get(14, 0), 0)
    check("gap 4 leaves the cohort rather than being scored healthy",
          int((f["personId"] == 14).sum()), 0)

    # The 1-2 sensitivity arm must drop gap 3 and keep gaps 1-2.
    _, f2 = build(rows, window=2)
    pos2 = dict(zip(f2["personId"], f2["y"]))
    check("sensitivity arm 1-2 keeps gap 2", pos2.get(12, 0), 1)
    check("sensitivity arm 1-2 drops gap 3", int((f2["personId"] == 13).sum()), 0)


def case_3_multiple_games_and_reports():
    """The matching-window defect continued: several games, several reports, one index game each."""
    print("\n  case 3 -- two reports over four games pick two index games")
    rows = [
        dict(pid=20, date="2019-01-01", minutes=30),
        dict(pid=20, date="2019-01-03", minutes=30, cat=MSK, inj="2019-01-04"),
        dict(pid=20, date="2019-01-20", minutes=30),
        dict(pid=20, date="2019-01-22", minutes=30, cat=MSK, inj="2019-01-23"),
    ]
    _, f = build(rows)
    check("two index games", int(f["y"].sum()), 2)
    check("both are the game nearest their report",
          sorted(f.loc[f["y"] == 1, "dt"].dt.day.tolist()), [3, 22])


def case_4_season_boundary():
    """The season-opener shift defect, criterion 8 / T7.

    A new season's opener must carry no season-to-date state at all, and the
    second game of the season must carry the opener alone.
    """
    print("\n  case 4 -- season-to-date state does not cross the offseason")
    rows = [
        dict(pid=30, date="2019-03-01", minutes=40),   # season 2018 finale
        dict(pid=30, date="2019-10-22", minutes=20),   # season 2019 opener
        dict(pid=30, date="2019-10-25", minutes=10),   # second game
    ]
    _, f = build(rows)
    opener = f[f["dt"] == pd.Timestamp("2019-10-22 19:00")].iloc[0]
    second = f[f["dt"] == pd.Timestamp("2019-10-25 19:00")].iloc[0]
    check("opener is flagged", int(opener["season_opener"]), 1)
    check("opener season_minutes", float(opener["season_minutes"]), 0.0,
          "40.0 was the previous season's final game")
    check("opener numMinutes_std is undefined", opener["numMinutes_std"], np.nan)
    check("opener acute:chronic is undefined", opener["numMinutes_acr"], np.nan)
    check("opener season_games", int(opener["season_games"]), 0)
    check("opener rest_days is undefined", opener["rest_days"], np.nan)
    check("second game season_minutes is the opener alone",
          float(second["season_minutes"]), 20.0)
    check("second game numMinutes_std is the opener alone",
          float(second["numMinutes_std"]), 20.0)
    # The rolling windows are DELIBERATELY cross-season: they describe recent
    # workload, and the previous season's last games really are the games
    # before the opener. Asserted so a future edit cannot "fix" it silently.
    check("r3 deliberately crosses the season boundary",
          float(opener["numMinutes_r3"]), 40.0,
          "intentional -- see build_features, lagged rolling aggregates")


def case_5_injury_dated_at_the_event():
    """The injury-dating defect, criterion 9 / T9.

    The review's own example. An injury reported 2021-04-09 whose next
    appearance is 2021-10-20 used to enter the 365-day window on that October
    date, so it was still 'in the last year' on 2022-10-01 while
    `days_since_injury` correctly read 540.
    """
    print("\n  case 5 -- the 365-day lookback uses the injury's own date")
    rows = [
        dict(pid=40, date="2021-04-07", minutes=30, cat=MSK, inj="2021-04-09"),
        dict(pid=40, date="2021-10-20", minutes=30),
        dict(pid=40, date="2022-10-01", minutes=30),
    ]
    _, f = build(rows)
    r = f[f["dt"] == pd.Timestamp("2022-10-01 19:00")].iloc[0]
    check("days_since_injury on 2022-10-01", float(r["days_since_injury"]), 540.0)
    check("injuries_last_365d on 2022-10-01", float(r["injuries_last_365d"]), 0.0,
          "1.0 was the defect: the injury was dated at the next appearance")
    check("prior_injuries still counts it", float(r["prior_injuries"]), 1.0)
    check("ever_injured", int(r["ever_injured"]), 1)

    # Both sides of the 365-day boundary, and the index game itself.
    idx = f[f["dt"] == pd.Timestamp("2021-04-07 19:00")].iloc[0]
    check("the index game does not see its own injury",
          float(idx["prior_injuries"]), 0.0,
          "the report is dated two days later, so it cannot be known pre-tip")
    oct20 = f[f["dt"] == pd.Timestamp("2021-10-20 19:00")].iloc[0]
    check("194 days later the injury is inside the window",
          float(oct20["injuries_last_365d"]), 1.0)
    check("and days_since_injury agrees", float(oct20["days_since_injury"]), 194.0)

    rows_edge = [
        dict(pid=41, date="2021-04-07", minutes=30, cat=MSK, inj="2021-04-09"),
        dict(pid=41, date="2022-04-09", minutes=30),   # exactly 365 days
        dict(pid=41, date="2022-04-10", minutes=30),   # 366 days
    ]
    _, fe = build(rows_edge)
    on = fe[fe["dt"] == pd.Timestamp("2022-04-09 19:00")].iloc[0]
    off = fe[fe["dt"] == pd.Timestamp("2022-04-10 19:00")].iloc[0]
    check("exactly 365 days is inside", float(on["injuries_last_365d"]), 1.0)
    check("366 days is outside", float(off["injuries_last_365d"]), 0.0)
    check("and the two covariates never disagree",
          bool((fe["injuries_last_365d"] > 0).eq(fe["days_since_injury"] <= 365).all()),
          True)


def case_6_position_is_prior_only():
    """The career-modal position defect, criterion 10 / T11.

    `position` used to be the modal starting position over the whole panel,
    future included. It is now the modal position over strictly earlier games,
    so a row before the athlete's first start carries `Unstarted` and a switch
    of position shows up only after the games that caused it.
    """
    print("\n  case 6 -- position is a prior-only tally")
    rows = [
        dict(pid=50, date="2019-01-01", minutes=30, start=None),
        dict(pid=50, date="2019-01-03", minutes=30, start="G"),
        dict(pid=50, date="2019-01-05", minutes=30, start="G"),
        dict(pid=50, date="2019-01-07", minutes=30, start="C"),
        dict(pid=50, date="2019-01-09", minutes=30, start="C"),
        dict(pid=50, date="2019-01-11", minutes=30, start="C"),
        dict(pid=50, date="2019-01-13", minutes=30, start="C"),
    ]
    _, f = build(rows)
    got = [c.replace("pos_", "")
           for c in [f.loc[i, B.features.POSITION][f.loc[i, B.features.POSITION] == 1]
                     .index[0] for i in f.index]]
    check("position by game", got,
          ["Unstarted", "Unstarted", "G", "G", "G", "C", "C"],
          "career-modal would have said C on every row, including game 1")
    check("exactly one dummy set on every row",
          int((f[B.features.POSITION].sum(axis=1) == 1).sum()), len(f))
    check("all declared categories present",
          [c in f.columns for c in B.features.POSITION], [True] * 4)


def case_7_dnp_rows_never_enter_a_window():
    """The DNP guardrail, exercised through the REAL loader.

    `load_appearances()` drops `numMinutes == 0` rows before anything else, so
    a DNP can never enter a rolling window or become a rest endpoint.

    The first version of this case asserted that by filtering the rows itself
    and then building from a frame it had already cleaned, which tested the
    fixture rather than the code: an external review pointed out it passed even
    with `load_appearances` replaced by a function that always raises. It now
    writes a miniature master to disk, points the module's own `DATA` and
    `BIOMETRICS` paths at it, and calls the real loader.

    The tiny master has to carry one row for each head-region value, because
    `load_appearances()` has a tripwire that fails loudly on an unfamiliar one.
    That tripwire firing here would itself be a useful signal, so it is fed
    rather than disabled.
    """
    print("\n  case 7 -- a DNP is not an appearance (through load_appearances)")
    import tempfile

    rows = [
        # The athlete under test: plays, sits out, plays again.
        dict(pid=60, date="2019-01-01", minutes=30),
        dict(pid=60, date="2019-01-02", minutes=0),     # DNP
        dict(pid=60, date="2019-01-04", minutes=30),
        # A second athlete carrying the three head-region values the loader's
        # tripwire insists on seeing.
        dict(pid=61, date="2019-01-01", minutes=25, loc="Concussion"),
        dict(pid=61, date="2019-01-03", minutes=25, loc="Head"),
        dict(pid=61, date="2019-01-05", minutes=25, loc="Left Concussion"),
    ]
    raw = panel(rows).drop(columns=["_conc_row", "dt", "season"])
    # A playoff row and an out-of-range season, so the other two exposure
    # filters are exercised by the same call rather than assumed.
    extra = panel([dict(pid=60, date="2019-05-01", minutes=30),
                   dict(pid=60, date="2014-12-01", minutes=30)])
    extra = extra.drop(columns=["_conc_row", "dt", "season"])
    # Select by DATE, not by position: `panel()` sorts by (personId, dt), so
    # `.index[0]` is the 2014 row, not the May one. Flagging the wrong row here
    # made this check pass for the wrong reason on the first attempt.
    is_may = extra["gameDateTimeEst"].dt.strftime("%Y-%m-%d") == "2019-05-01"
    extra.loc[is_may, "gameType"] = "Playoffs"
    raw = pd.concat([raw, extra], ignore_index=True)

    tmp = tempfile.mkdtemp(prefix="nba_fixture_")
    master = os.path.join(tmp, "mini_master.csv")
    bios = os.path.join(tmp, "mini_bios.csv")
    raw.to_csv(master, index=False, encoding="utf-8")
    pd.DataFrame({"personId": [60, 61], "SEASON": ["2018-19", "2018-19"],
                  "AGE": [27.0, 29.0], "BMI": [25.0, 24.0]}
                 ).to_csv(bios, index=False, encoding="utf-8")

    saved_data, saved_bios, saved_min = B.DATA, B.BIOMETRICS, B.MIN_CAREER_GAMES
    B.DATA, B.BIOMETRICS, B.MIN_CAREER_GAMES = master, bios, 0
    try:
        app = B.load_appearances(verbose=False)
        feats = B.build_features(
            B.attach_outcomes(app, 3, verbose=False), verbose=False)
    finally:
        B.DATA, B.BIOMETRICS, B.MIN_CAREER_GAMES = saved_data, saved_bios, saved_min

    a60 = app[app["personId"] == 60]
    check("load_appearances drops the DNP", len(a60), 2,
          "the real loader, not a frame the fixture pre-filtered")
    check("it also drops the playoff row",
          int((~dataio.is_regular_season(app["gameType"])).sum()), 0)
    check("and the out-of-range season",
          int((app["season"] < B.FIRST_SEASON).sum()), 0)
    f60 = feats[feats["personId"] == 60]
    check("rest spans the DNP rather than stopping at it",
          float(f60.loc[f60["dt"].dt.day == 4, "rest_days"].iat[0]), 3.0,
          "2.0 would mean the DNP became a rest endpoint")
    check("the DNP contributes no minutes to season_minutes",
          float(f60.loc[f60["dt"].dt.day == 4, "season_minutes"].iat[0]), 30.0)
    check("biometrics joined by season through the real loader",
          float(app.loc[app["personId"] == 60, "AGE"].iloc[0]), 27.0)


def case_8_team_id_recovery_refuses_to_guess():
    """`dataio.recover_team_ids` must fill from the name and never invent.

    2021-22 carries a null `playerteamId` on 99.9% of its active Regular Season
    rows, so any stage dropping on that column excluded the season whole. The
    recovery closes that, but a recovery that guesses would be worse than the
    gap: it would attribute games to the wrong team silently.

    Three refusals are checked as well as the happy path, because a guard is
    vacuous until something is broken and watched failing:

      - a name mapping to two different ids must raise, not pick one;
      - a name never seen with an id must raise, not get a synthetic one
        (the deleted `gee_cohort_control/` hashed the name into a fake id here);
      - a NULL name must raise. It did not until 2026-09-18: `groupby` drops a
        NaN key so the conflict check never saw it, while `map` matched NaN to
        NaN, so a row with no name and no id silently took an unrelated row's
        id. Found in review; no row in the master has a null name, so no
        reported number was affected.
    """
    print("\n  case 8 -- team-id recovery fills from the name and refuses to guess")

    def outcome(frame):
        try:
            return list(dataio.recover_team_ids(frame, verbose=False)["playerteamId"])
        except AssertionError:
            return "RAISED"

    check("fills a missing id from the name",
          outcome(pd.DataFrame({"playerteamName": ["LAL", "LAL", "BOS", "BOS"],
                                "playerteamId": [1.0, np.nan, 2.0, np.nan]})),
          [1.0, 1.0, 2.0, 2.0])
    check("no-op when nothing is missing",
          outcome(pd.DataFrame({"playerteamName": ["LAL", "BOS"],
                                "playerteamId": [1.0, 2.0]})),
          [1.0, 2.0])
    check("raises when one name carries two ids",
          outcome(pd.DataFrame({"playerteamName": ["LAL", "LAL", "LAL"],
                                "playerteamId": [1.0, 9.0, np.nan]})),
          "RAISED", "must not silently pick one")
    check("raises on a name never seen with an id",
          outcome(pd.DataFrame({"playerteamName": ["LAL", "NEW"],
                                "playerteamId": [1.0, np.nan]})),
          "RAISED", "no synthetic id is ever invented")
    check("raises on a NULL name (NaN)",
          outcome(pd.DataFrame({"playerteamName": [np.nan, np.nan, np.nan],
                                "playerteamId": [1.0, 2.0, np.nan]})),
          "RAISED", "regression: NaN once matched NaN and took row 0's id")
    check("raises on a NULL name (None)",
          outcome(pd.DataFrame({"playerteamName": [None, "LAL", None],
                                "playerteamId": [1.0, 5.0, np.nan]})),
          "RAISED")


def case_9_loader_refuses_a_stale_parquet():
    """`dataio.dataset_path()` must not silently serve an outdated dataset.

    The builder writes parquet and falls back to gzip CSV if that raises. The
    loader prefers parquet whenever it exists. Together those meant a build
    whose parquet write failed would succeed, log the fresh CSV it wrote, and
    leave every downstream stage reading the PREVIOUS build's parquet.

    Reproduced 2026-09-18 with a stale 1%-prevalence parquet beside a fresh
    10% CSV: the loader returned the stale one, so every rate, denominator and
    model would have come from the wrong cohort while the build log described
    the right one. Found in review.

    The builder now moves a superseded parquet aside; this is the second line
    of defense, and the case that matters is the middle one -- a CSV that is
    merely OLD is an ordinary leftover and must NOT trip the guard, or every
    normal run would start failing.
    """
    print("\n  case 9 -- the loader refuses a parquet the CSV has superseded")
    import tempfile, time
    orig = dataio.SCRIPT_DIR
    try:
        with tempfile.TemporaryDirectory() as d:
            dataio.SCRIPT_DIR = d
            pq = os.path.join(d, "predictive_dataset.parquet")
            cs = os.path.join(d, "predictive_dataset.csv.gz")
            pd.DataFrame({"x": [1], "y": [0]}).to_parquet(pq)

            def outcome():
                try:
                    return os.path.basename(dataio.dataset_path())
                except RuntimeError:
                    return "RAISED"

            check("parquet alone is returned", outcome(),
                  "predictive_dataset.parquet")
            pd.DataFrame({"x": [1], "y": [0]}).to_csv(
                cs, index=False, compression="gzip")
            os.utime(cs, (time.time() - 600, time.time() - 600))
            check("an OLDER csv is ignored", outcome(),
                  "predictive_dataset.parquet",
                  "a stale leftover csv must not break a normal run")
            os.utime(cs, (time.time() + 600, time.time() + 600))
            check("a NEWER csv raises", outcome(), "RAISED",
                  "signature of a failed parquet write; refuse rather than guess")
    finally:
        dataio.SCRIPT_DIR = orig


def case_10_failed_parquet_write_never_leaves_a_stale_one():
    """The builder's half of the staleness fix, which case 9 does not cover.

    Case 9 tests `dataio.dataset_path()` against hand-placed files. That is the
    second line of defense. The PRIMARY defense is the builder's export block,
    and until this case existed a future edit could reorder it -- moving
    `to_csv` above the `os.replace`, or dropping the `os.path.exists` guard --
    and break the real protection with every test still green. Flagged by
    a review against this repo's own rule that a guard is vacuous until
    something is broken and watched failing.

    Three properties, exercised against the real export block rather than a
    reimplementation of it:

      1. a SUCCESSFUL write leaves no `.tmp` sidecar behind;
      2. a FAILED write leaves the existing parquet untouched and complete --
         it is never overwritten by the partial write, because the write goes
         to `.tmp` first;
      3. after a failed write the previous parquet is moved to `.stale` and a
         CSV is written, so nothing is left at the path `dataset_path()`
         prefers.
    """
    print("\n  case 10 -- a failed parquet write never leaves a stale one in place")
    import tempfile
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})

    with tempfile.TemporaryDirectory() as d:
        pq = os.path.join(d, "predictive_dataset.parquet")
        cs = os.path.join(d, "predictive_dataset.csv.gz")
        saved = (B.PARQUET_OUT, B.CSV_OUT)
        B.PARQUET_OUT, B.CSV_OUT = pq, cs
        try:
            B.export_dataset(frame)
            check("a successful write produces the parquet",
                  os.path.exists(pq), True)
            check("and leaves no .tmp sidecar",
                  os.path.exists(pq + ".tmp"), False)

            before = os.path.getsize(pq)
            orig = pd.DataFrame.to_parquet

            def boom(self, *a, **k):
                raise OSError("simulated parquet write failure")

            pd.DataFrame.to_parquet = boom
            try:
                B.export_dataset(frame)
            finally:
                pd.DataFrame.to_parquet = orig

            check("the previous parquet is moved aside, not left to win",
                  os.path.exists(pq), False,
                  "dataset_path() prefers parquet, so leaving one serves stale data")
            check("it is preserved as .stale rather than deleted",
                  os.path.exists(pq + ".stale"), True)
            check("and preserved INTACT -- the partial write never touched it",
                  os.path.getsize(pq + ".stale"), before,
                  "to_parquet writes in place; the .tmp sidecar is what protects it")
            check("the csv fallback is written", os.path.exists(cs), True)
            check("no .tmp sidecar survives the failure",
                  os.path.exists(pq + ".tmp"), False)
        finally:
            B.PARQUET_OUT, B.CSV_OUT = saved


def case_11_run_stage_always_writes_its_verdict_to_the_log():
    """The stage log must record the verdict -- `dataio.run_stage`.

    Thirteen stages used to decide for themselves whether the verdict was
    printed before or after the log was closed, and `audit_feature_timing.py`
    chose `file=tee.terminal`: a FAILING run left a log byte-identical to a
    passing one, so the tracked artifact could not distinguish them. That is
    the defect this helper exists to make impossible, so it is checked here on
    all three exits -- clean, non-zero return, and raise -- rather than read
    off the code.

    Note this test does not merely assert the exit code; it reads the FILE.
    An exit code that is right while the log is silent is the exact failure.
    """
    print("\n[case 11] run_stage writes the verdict INSIDE the log")

    import tempfile

    saved_stdout = sys.stdout
    tmp = tempfile.mkdtemp(prefix="run_stage_")

    # ---- clean exit ------------------------------------------------------
    p_ok = os.path.join(tmp, "ok.txt")
    code_ok = dataio.run_stage(lambda: print("body ran"), p_ok)
    log_ok = open(p_ok, encoding="utf-8").read()

    # ---- non-zero return -------------------------------------------------
    p_bad = os.path.join(tmp, "bad.txt")
    code_bad = dataio.run_stage(lambda: 1, p_bad)
    log_bad = open(p_bad, encoding="utf-8").read()

    # ---- explicit (code, message) ----------------------------------------
    p_msg = os.path.join(tmp, "msg.txt")
    code_msg = dataio.run_stage(lambda: (0, "Done -- completion status only."),
                                p_msg)
    log_msg = open(p_msg, encoding="utf-8").read()

    # ---- raise -----------------------------------------------------------
    def boom():
        print("partial output")
        raise ValueError("deliberate")

    p_raise = os.path.join(tmp, "raise.txt")
    raised = None
    try:
        dataio.run_stage(boom, p_raise)
    except ValueError as exc:
        raised = str(exc)
    log_raise = open(p_raise, encoding="utf-8").read()

    # stdout must be back to the real stream on every path above, or every
    # check below would be writing into a closed file.
    restored = sys.stdout is saved_stdout

    check("stdout is restored after every path", restored, True,
          "a leaked Tee would silently swallow the rest of the run")
    check("clean run exits 0", code_ok, 0)
    check("clean run's log ends with the verdict",
          log_ok.strip().endswith("Done."), True,
          "the ten simple stages printed this AFTER closing the file")
    check("the body's own output is in the log", "body ran" in log_ok, True)

    check("a non-zero return becomes the exit code", code_bad, 1)
    check("and the log says so", "FAIL:" in log_bad, True,
          "audit_feature_timing.py's log could not say this")

    check("an explicit message is used verbatim",
          log_msg.strip().endswith("Done -- completion status only."), True)
    check("with its own exit code", code_msg, 0)

    check("a raise propagates", raised, "deliberate",
          "run_stage records the failure, it does not swallow it")
    check("a raise still leaves output in the log",
          "partial output" in log_raise, True)
    check("and the log names the exception",
          "FAIL: stage raised ValueError: deliberate" in log_raise, True,
          "otherwise a crash is indistinguishable from a kill or a truncation")
    check("the log is closed after a raise",
          _is_closed(p_raise), True)


def case_12_concurrent_window_includes_the_index_game():
    """`_c5` ends AT game t; `_r5` ends at t-1. The S4.2 comparator needs both.

    Written as a fixture rather than trusted from the builder's comment,
    because the whole point of the comparator is that the two arms differ in
    TIMING ALONE. If `_c5` were built with the same shift as `_r5` the table
    would compare a window against itself and report a null that means nothing,
    with every downstream checker green.

    EIGHT games, not five. The first version of this case used exactly five,
    so neither window ever slid off a game -- and with both windows still
    filling, `c5 - r5` happens to equal the index game's own contribution.
    That coincidence is exactly what let the claim "the two windows differ by
    exactly one game" reach the manuscript. Once the windows are full they
    span t-4..t and t-5..t-1, so their difference is (v_t - v_{t-5})/5, and
    the assertions below pin that rather than the convenient special case.
    """
    print("\n  case 12 -- the concurrent window includes game t")
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    rows = [dict(pid=90, date=f"2019-01-{i + 1:02d}",
                 fieldGoalsAttempted=v)
            for i, v in enumerate(vals)]
    _, f = build(rows)
    f = f.sort_values("dt").reset_index(drop=True)

    c5 = f["fieldGoalsAttempted_c5"].tolist()
    r5 = f["fieldGoalsAttempted_r5"].tolist()
    c1 = f["fieldGoalsAttempted_c1"].tolist()

    # Concurrent: mean of games t-4..t. Lagged: mean of games t-5..t-1.
    want_c5 = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
    want_r5 = [None, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    check("case 12 c5 window", [round(v, 6) for v in c5], want_c5,
          "five-game mean ending AT game t")
    check("case 12 r5 window",
          [None if pd.isna(v) else round(v, 6) for v in r5], want_r5,
          "five-game mean ending at game t-1")

    # `_c1` is the index game itself -- the column the S4.2 mechanism
    # measurement uses, because `c5 - r5` is NOT the index game's effect.
    check("case 12 c1 is the index game", [round(v, 6) for v in c1], vals,
          "the raw game-t value")

    # Once BOTH windows are full (t >= 5 here), their difference must equal
    # (v_t - v_{t-5})/5 -- adding game t AND dropping game t-5.
    for t in range(5, len(vals)):
        check(f"case 12 c5-r5 at t={t}", round(c5[t] - r5[t], 6),
              round((vals[t] - vals[t - 5]) / 5, 6),
              "differs by TWO games, not one")

    # EVERY association-only covariate must be built, not just the first.
    #
    # This was a literal assertion about `reboundPercentage_c5` until
    # 2026-09-22, when the rebound-percentage removal dropped that covariate and the check became a
    # KeyError rather than a test. Derived from `features.CONCURRENT_ONLY` now,
    # so it keeps its original meaning -- "the builder does not stop after the
    # head of the list" -- whatever that list holds, and additionally pins the
    # set EXACTLY: a stray concurrent column nobody declared is as much a
    # defect as a missing one, since `dataio.load()` withholds these by name.
    built = sorted(c for c in f.columns
                   if c.endswith("_c5") or c.endswith("_c1"))
    check("case 12 every concurrent covariate is built",
          built, sorted(features.CONCURRENT_ONLY),
          "the builder emits exactly the declared CONCURRENT_ONLY set")

    # Every row where both windows are defined must differ, not merely one.
    both = [(a, b) for a, b in zip(c5, r5) if not pd.isna(b)]
    check("case 12 arms differ on every comparable row",
          all(abs(a - b) > 1e-9 for a, b in both), True,
          "a concurrent arm identical to the lagged one measures nothing")


def _is_closed(path):
    """True if `path` can be reopened for writing -- i.e. nothing holds it.

    On Windows an open handle blocks this, so it is a real check rather than a
    tautology. `run_stage` closes in a `finally`, so a raise must not leak the
    handle; a leaked one is how a partially written log survives a run.
    """
    try:
        with open(path, "a", encoding="utf-8"):
            return True
    except OSError:
        return False


def main():
    print("=" * 84)
    print("  FIXTURE TESTS -- covariate construction")
    print("=" * 84)
    for fn in (case_1_history_survives_its_own_exclusion,
               case_2_matching_window_is_calendar_days,
               case_3_multiple_games_and_reports,
               case_4_season_boundary,
               case_5_injury_dated_at_the_event,
               case_6_position_is_prior_only,
               case_7_dnp_rows_never_enter_a_window,
               case_8_team_id_recovery_refuses_to_guess,
               case_9_loader_refuses_a_stale_parquet,
               case_10_failed_parquet_write_never_leaves_a_stale_one,
               case_11_run_stage_always_writes_its_verdict_to_the_log,
               case_12_concurrent_window_includes_the_index_game):
        fn()
    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} expectation(s) not met:")
        for c in FAILURES:
            print(f"  - {c}")
        return 1
    print("PASS -- every fixture expectation met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
