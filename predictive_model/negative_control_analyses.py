"""
negative_control_analyses.py
================================================================================
Does a gap between games carry risk because of its LENGTH, or because of its
CAUSE?

WHY THIS SCRIPT EXISTS
----------------------
The study's central claim is that `rest_days` is not a recovery measure but a
proxy for ABSENCE: an athlete who has not played for four days has usually not
been resting, they have been unavailable. The evidence for that claim in the
previous revision was a single contrast -- the season opener, maximal rest with
no preceding absence, injured at 0.664% against 6.870% at four days' rest --
resting on 6 events in 904 openers.

That contrast is correct but underpowered, and it is only one of three natural
experiments available in these data at no cost. This script runs all three.

  [1] EXOGENOUS vs ENDOGENOUS REST  (the decisive decomposition)

      For every athlete-game, days of rest can be decomposed by asking a second
      question the box score already answers: how many games did the athlete's
      TEAM play during that gap?

          team games missed = 0   the whole roster was off. The gap is the
                                  league schedule. Nobody was withheld.
          team games missed >= 1  the team played and this athlete did not.
                                  Somebody decided they would not play.

      Both arms hold rest LENGTH fixed and vary only its CAUSE. The availability
      account predicts risk rises only in the endogenous arm; a recovery or
      deconditioning account predicts the gradient tracks length in both.

      This replaces an n=6 comparison with one spanning tens of thousands of
      exposures.

  [2] THE ALL-STAR BREAK  (a league-wide exogenous gap)

      Every athlete not selected for the All-Star Game receives a 5-9 day gap
      whose cause is the calendar. It is the exogenous arm of [1] delivered to
      the whole league simultaneously, and it is immune to the objection that
      teams choose their own schedule gaps around athletes they are managing.

  [3] SEASON OPENERS, POOLED  (the previous revision's contrast, powered)

      The opener comparison is repeated on the full panel and on the
      high-ascertainment cohort rather than on the two test seasons alone.

NEGATIVE CONTROL FRAMING
------------------------
[2] and [3] are negative control exposures in the sense of Lipsitch et al.
(Epidemiology 2010): exposures that share the hypothesised confounding structure
but cannot plausibly cause the outcome through the mechanism under test. A long
gap that carries no injury information should be harmless if absence is the
active ingredient, and hazardous if gap length is.

OUTPUTS
-------
  predictive_model/negative_control_output.txt
  predictive_model/negative_controls.json
"""

import os
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import dataio

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
MASTER = os.path.join(ANALYSIS_DIR, "Data", "master_model_ready_data_corrected.csv")
TXT_OUT = os.path.join(SCRIPT_DIR, "negative_control_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "negative_controls.json")
TEAM_GAP_OUT = os.path.join(SCRIPT_DIR, "team_gap_attribution.parquet")

TEST_SEASONS = [2022, 2023]
HIGH_ASCERTAINMENT_FROM = 2017

# Rest strata. The 3-4 day band is where the endogenous gradient peaks and is
# therefore where the two accounts diverge most sharply.
REST_BINS = [0, 1, 2, 4, 6, 100]
REST_LABELS = ["1 (back-to-back)", "2", "3-4", "5-6", "7+"]


def wilson(k, n, z=1.959963985):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_row(sub):
    n = int(len(sub))
    k = int(sub.y.sum())
    lo, hi = wilson(k, n)
    return dict(n=n, injuries=k, rate=(k / n if n else np.nan), lo=lo, hi=hi)


def fmt_rate(r):
    if r["n"] == 0:
        return f"{'--':>9s}"
    return f"{r['rate']*100:6.3f}%"


def fmt_ci(r):
    if r["n"] == 0:
        return "--"
    return f"{r['lo']*100:.3f}-{r['hi']*100:.3f}%"


def risk_ratio(a, b):
    """Risk ratio a/b with a Katz log-normal 95% interval."""
    if a["injuries"] == 0 or b["injuries"] == 0 or a["n"] == 0 or b["n"] == 0:
        return dict(rr=np.nan, lo=np.nan, hi=np.nan)
    rr = a["rate"] / b["rate"]
    se = np.sqrt(1 / a["injuries"] - 1 / a["n"] + 1 / b["injuries"] - 1 / b["n"])
    return dict(rr=float(rr), lo=float(rr * np.exp(-1.959964 * se)),
                hi=float(rr * np.exp(1.959964 * se)))


# --------------------------------------------------------------------------
# Team schedule reconstruction
# --------------------------------------------------------------------------
def season_of(dt):
    return np.where(dt.dt.month >= 10, dt.dt.year, dt.dt.year - 1)


def load_team_schedule():
    """Every (team, date) on which a regular-season game was played.

    Built from the full box-score panel rather than the analytic cohort: the
    analytic cohort excludes athletes who did not play, so counting team games
    from it would undercount exactly the games this analysis needs to see.
    """
    cols = ["personId", "gameDateTimeEst", "gameType", "playerteamId",
            "playerteamName", "numMinutes"]
    m = pd.read_csv(MASTER, usecols=cols, low_memory=False)
    m = m[dataio.is_regular_season(m["gameType"])].copy()
    # 2021-22 carries a null `playerteamId` on 99.9% of its active Regular
    # Season rows, so the dropna below used to exclude that season whole. The
    # name is populated everywhere and maps unambiguously to an id; recover
    # before any team-dependent frame is built. Raises rather than guessing.
    #
    # MUST come after the Regular Season filter. All-Star teams are named for
    # their captains ("LeBron", "Giannis") and the same captain name carries a
    # different id in different years, so recovering on the unfiltered frame
    # raises on an ambiguity that is real but irrelevant here.
    m = dataio.recover_team_ids(m)
    m["dt"] = pd.to_datetime(m["gameDateTimeEst"], errors="coerce")
    m = m.dropna(subset=["dt", "personId"])
    m["date"] = m["dt"].dt.normalize()
    m["season"] = season_of(m["dt"])

    # An athlete's APPEARANCE history does not need a team, and must not be
    # filtered by one. The team-dependent frames below drop rows with no
    # `playerteamId`; `appearances` deliberately does not.
    appearances = (m.loc[(m["numMinutes"] > 0) & m["numMinutes"].notna(),
                         ["personId", "season", "date"]]
                   .drop_duplicates()
                   .sort_values(["personId", "season", "date"])
                   .reset_index(drop=True))

    # ---- team attribution is NOT complete, and one whole season is missing --
    # Found 2026-09-18. Every active row of 2021-22 carries a NULL
    # `playerteamId` in the master: 25,805 of 25,805. Nothing else is affected
    # -- no model covariate reads the team -- but this analysis does, so the
    # exogenous/endogenous decomposition has silently excluded that entire
    # season since it was written. It shows up only as the "team gap resolved
    # for 87.2%" line below, which reads like ordinary attrition and is not.
    #
    # Printed as a per-season table so a whole missing season can never again
    # hide inside an aggregate percentage.
    no_team = m[(m["numMinutes"] > 0) & m["numMinutes"].notna()
                & m["playerteamId"].isna()]
    if len(no_team):
        by_season = no_team.groupby("season").size()
        tot = (m[(m["numMinutes"] > 0) & m["numMinutes"].notna()]
               .groupby("season").size())
        audit_lines = []
        for ssn, k in by_season.items():
            frac = k / tot.loc[ssn]
            flag = "   <-- ENTIRE SEASON" if frac > 0.999 else ""
            audit_lines.append(f"    season {int(ssn)}: {k:,} / {tot.loc[ssn]:,} "
                               f"active rows have no playerteamId ({frac:.1%}){flag}")
        print("[WARN] team attribution is incomplete in the master:")
        for line in audit_lines:
            print(line)
        print("       Rows with no team cannot enter the exogenous/endogenous")
        print("       decomposition at all. A season at 100% is excluded whole.")

    # Unreachable since the recovery above, which raises on any residual null.
    # Kept as a guard so a future change that bypasses the recovery cannot
    # silently reintroduce whole-season exclusion.
    m = m.dropna(subset=["playerteamId"])

    sched = (m[["playerteamId", "season", "date"]]
             .drop_duplicates()
             .sort_values(["playerteamId", "date"])
             .reset_index(drop=True))

    # Which team did each athlete play for on each date (for trade handling)?
    roster = (m[["personId", "date", "playerteamId"]]
              .drop_duplicates(subset=["personId", "date"])
              .rename(columns={"playerteamId": "team"}))

    league = (m[["season", "date"]].drop_duplicates()
              .sort_values("date").reset_index(drop=True))

    # `appearances` was built above, BEFORE the playerteamId filter
    # (2026-09-18, criterion 4). `attach_team_gap` used to take an
    # athlete's previous game from the ANALYTIC frame it was handed, which has
    # had rows removed because of their injury labels. A removed row is not a
    # game the athlete missed, so the gap was measured from the wrong endpoint
    # and `team_games_missed` then counted team games the athlete had in fact
    # played in. The endpoint now comes from the full appearance history.
    return sched, roster, league, appearances


def attach_team_gap(df, sched, roster, appearances):
    """Attach, for each athlete-game, the number of games the athlete's team
    played strictly inside the gap since that athlete last played.

    team_games_missed == 0  -> the gap is schedule-mandated (exogenous)
    team_games_missed >= 1  -> the athlete sat out games the team played
                               (endogenous)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["dt"]).dt.normalize()
    df = df.merge(roster, on=["personId", "date"], how="left")

    df = df.sort_values(["personId", "season", "date"]).reset_index(drop=True)

    # PREVIOUS APPEARANCE, from the full box-score panel rather than from this
    # frame. Taking `.shift(1)` here would skip every game deleted from the
    # analytic cohort by a label-dependent rule -- a non-index matched game, an
    # excluded category -- and so measure the gap from a game the athlete had
    # actually played through. See load_team_schedule().
    prev_by_person = {}
    for (pid_, ssn), g in appearances.groupby(["personId", "season"], sort=False):
        prev_by_person[(pid_, ssn)] = g["date"].to_numpy()
    prev_dates = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    date_vals = df["date"].to_numpy()
    for i, (pid_, ssn) in enumerate(zip(df["personId"].to_numpy(),
                                        df["season"].to_numpy())):
        arr = prev_by_person.get((pid_, ssn))
        if arr is None:
            continue
        j = np.searchsorted(arr, date_vals[i], side="left")
        if j > 0:
            prev_dates[i] = arr[j - 1]
    df["prev_date"] = prev_dates

    # The team the athlete played for at that previous APPEARANCE -- used to
    # detect trades inside the gap, where "games the team played" is ambiguous.
    prev_team_lookup = (roster.rename(columns={"date": "prev_date",
                                               "team": "prev_team"}))
    df = df.merge(prev_team_lookup, on=["personId", "prev_date"], how="left")
    df["traded_in_gap"] = (df["team"] != df["prev_team"]) & df["prev_team"].notna()

    # Cumulative team-game index: games played by team T strictly before date d
    # is searchsorted into that team's sorted date array.
    counts = np.full(len(df), np.nan)
    team_rest = np.full(len(df), np.nan)
    by_team = {t: g["date"].values for t, g in sched.groupby("playerteamId")}

    team_arr = df["team"].values
    date_arr = df["date"].values
    prev_arr = df["prev_date"].values

    for i in range(len(df)):
        t = team_arr[i]
        if pd.isna(t):
            continue
        dates = by_team.get(t)
        if dates is None:
            continue
        d = date_arr[i]
        # team games strictly between prev game and this game
        p = prev_arr[i]
        if not pd.isna(p):
            lo = np.searchsorted(dates, p, side="right")
            hi = np.searchsorted(dates, d, side="left")
            counts[i] = max(0, hi - lo)
        # days since the team's own previous game
        j = np.searchsorted(dates, d, side="left")
        if j > 0:
            team_rest[i] = (d - dates[j - 1]) / np.timedelta64(1, "D")

    df["team_games_missed"] = counts
    df["team_rest_days"] = team_rest

    # Persist the per-row attribution so it can be CHECKED (2026-09-18).
    # `verify_history_reconstruction.py` used to "verify" this quantity by
    # recomputing it and inspecting its own recomputation, which is vacuous:
    # it never read what this stage actually produced, so a regression here
    # could not fail it. External review caught that. The verifier now reads
    # this file and compares row for row.
    out = df[["personId", "date", "prev_date", "team_games_missed",
              "team_rest_days", "traded_in_gap"]].copy()
    out.to_parquet(TEAM_GAP_OUT, index=False)
    print(f"[EXPORT] per-row gap attribution -> {TEAM_GAP_OUT}")
    return df


# --------------------------------------------------------------------------
# [1] Exogenous vs endogenous rest
# --------------------------------------------------------------------------
def analysis_exogenous(df, res, label, tee_note=""):
    print("=" * 96)
    print(f"  [1] EXOGENOUS vs ENDOGENOUS REST  --  {label}")
    print("=" * 96)
    if tee_note:
        print(f"  {tee_note}\n")
    print("  Rest length is held fixed within each row. The only thing that varies")
    print("  across the two columns is WHY the athlete did not play.\n")

    sub = df[df["team_games_missed"].notna() & df["rest_days"].notna()].copy()
    sub = sub[~sub["traded_in_gap"].astype(bool)]      # ambiguous team attribution
    sub["rest_bin"] = pd.cut(sub["rest_days"], REST_BINS, labels=REST_LABELS)
    sub["endogenous"] = (sub["team_games_missed"] >= 1).astype(int)

    store = {}
    print(f"  {'rest before game':>18s} | {'EXOGENOUS (team was off)':>34s} | "
          f"{'ENDOGENOUS (athlete sat out)':>36s} | {'RR endo/exo':>18s}")
    print(f"  {'':>18s} | {'n':>9s} {'inj':>5s} {'rate':>8s} {'':>8s} | "
          f"{'n':>9s} {'inj':>5s} {'rate':>8s} {'':>8s} |")
    print("  " + "-" * 112)

    for lab in REST_LABELS:
        s = sub[sub["rest_bin"] == lab]
        exo = rate_row(s[s.endogenous == 0])
        endo = rate_row(s[s.endogenous == 1])
        rr = risk_ratio(endo, exo)
        store[lab] = dict(exogenous=exo, endogenous=endo, rr=rr)
        rrs = "--" if np.isnan(rr["rr"]) else f"{rr['rr']:.2f}x ({rr['lo']:.2f}-{rr['hi']:.2f})"
        print(f"  {lab:>18s} | {exo['n']:>9,} {exo['injuries']:>5,} {fmt_rate(exo)} "
              f"{'':>8s} | {endo['n']:>9,} {endo['injuries']:>5,} {fmt_rate(endo)} "
              f"{'':>8s} | {rrs:>18s}")

    # Pooled 3+ day gaps: the band where the accounts diverge
    s34 = sub[sub["rest_days"] >= 3]
    exo34 = rate_row(s34[s34.endogenous == 0])
    endo34 = rate_row(s34[s34.endogenous == 1])
    rr34 = risk_ratio(endo34, exo34)
    store["pooled_3plus"] = dict(exogenous=exo34, endogenous=endo34, rr=rr34)

    print("\n  Pooled gaps of 3 or more days -- the band where the two accounts diverge:")
    print(f"    exogenous  (team was off)     : {exo34['injuries']:,} / {exo34['n']:,} = "
          f"{exo34['rate']*100:.3f}%  (95% CI {fmt_ci(exo34)})")
    print(f"    endogenous (athlete sat out)  : {endo34['injuries']:,} / {endo34['n']:,} = "
          f"{endo34['rate']*100:.3f}%  (95% CI {fmt_ci(endo34)})")
    if not np.isnan(rr34["rr"]):
        print(f"    risk ratio                    : {rr34['rr']:.2f}x "
              f"(95% CI {rr34['lo']:.2f}-{rr34['hi']:.2f})")

    # A normally-rested reference: 1-2 days, no games missed
    ref = rate_row(sub[(sub["rest_days"] <= 2) & (sub["endogenous"] == 0)])
    store["reference_normal_exogenous"] = ref
    print(f"\n  Reference -- normal schedule (1-2 days) with no games missed:")
    print(f"    {ref['injuries']:,} / {ref['n']:,} = {ref['rate']*100:.3f}%  "
          f"(95% CI {fmt_ci(ref)})")
    print(f"    exogenous 3+ day gap vs this reference : "
          f"{risk_ratio(exo34, ref)['rr']:.2f}x")
    print(f"    endogenous 3+ day gap vs this reference: "
          f"{risk_ratio(endo34, ref)['rr']:.2f}x")

    # Formal test: does gap LENGTH still carry risk once CAUSE is in the model?
    print("\n  Formal test -- logistic regression, SEs clustered on athlete:")
    print("    y ~ rest_days + games_missed   (both continuous, gaps of 2+ days)")
    mdl = sub[sub["rest_days"] >= 2].copy()
    mdl["gm"] = mdl["team_games_missed"].clip(upper=6)
    X = sm.add_constant(mdl[["rest_days", "gm"]].astype(float))
    try:
        fit = sm.GLM(mdl["y"].astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": mdl["personId"].values})
        coefs = {}
        for name in ["rest_days", "gm"]:
            b = fit.params[name]
            lo, hi = fit.conf_int().loc[name]
            coefs[name] = dict(or_=float(np.exp(b)), lo=float(np.exp(lo)),
                               hi=float(np.exp(hi)), p=float(fit.pvalues[name]))
            print(f"      {name:12s}  OR {np.exp(b):6.3f}  "
                  f"(95% CI {np.exp(lo):.3f}-{np.exp(hi):.3f})  p = {fit.pvalues[name]:.3g}")
        store["logit_length_vs_cause"] = coefs
        print("\n    Read: `gm` is the number of team games the athlete missed inside the")
        print("    gap. `rest_days` is the gap's length. If length carried the risk, its")
        print("    odds ratio should survive adjustment for cause.")
    except Exception as e:                                    # pragma: no cover
        print(f"      [model failed: {e}]")

    res[label] = store
    print()
    return sub


# --------------------------------------------------------------------------
# [2] All-Star break
# --------------------------------------------------------------------------
def find_all_star_breaks(league):
    """The All-Star break is the longest league-wide gap that resumes in February.

    Restricting the search window matters: the longest raw gap in 2019-20 is the
    141-day COVID-19 suspension, which is not an ordinary scheduled break and
    carries an entirely different absence structure.
    """
    breaks = {}
    for season, g in league.groupby("season"):
        d = g["date"].sort_values().reset_index(drop=True)
        gaps = d.diff().dt.days
        # candidate resumption dates inside the canonical All-Star window
        cand = [(gaps.iloc[i], i) for i in range(1, len(d))
                if pd.notna(gaps.iloc[i])
                and pd.Timestamp(year=d.iloc[i].year, month=1, day=20)
                <= d.iloc[i]
                <= pd.Timestamp(year=d.iloc[i].year, month=3, day=15)
                and 4 <= gaps.iloc[i] <= 12]
        if not cand:
            continue
        gap, i = max(cand)
        breaks[int(season)] = dict(gap_days=float(gap),
                                   last_before=d.iloc[i - 1],
                                   first_after=d.iloc[i])
    return breaks


def analysis_all_star(df, league, res, label):
    print("=" * 96)
    print(f"  [2] THE ALL-STAR BREAK AS A LEAGUE-WIDE EXOGENOUS GAP")
    print("=" * 96)
    breaks = find_all_star_breaks(league)
    print("  Detected breaks (longest 4-12 day league-wide gap resuming in Jan 20 - Mar 15):")
    for s in sorted(breaks):
        b = breaks[s]
        print(f"    {s}-{str(s+1)[-2:]}  {b['gap_days']:.0f} days   "
              f"{b['last_before'].date()} -> {b['first_after'].date()}")

    sub = df[df["rest_days"].notna() & df["team_games_missed"].notna()].copy()
    sub = sub[~sub["traded_in_gap"].astype(bool)]
    sub["date"] = pd.to_datetime(sub["dt"]).dt.normalize()

    # An athlete's FIRST game of the season on or after the resumption date, whose
    # gap spans the break. Anything later is an ordinary mid-season exposure.
    sub["post_break"] = 0
    for s, b in breaks.items():
        m = (sub["season"] == s) & (sub["date"] >= b["first_after"])
        if not m.any():
            continue
        idx = sub[m].sort_values("date").groupby("personId").head(1).index
        spans = sub.loc[idx, "prev_date"] <= b["last_before"]
        sub.loc[idx[spans.values], "post_break"] = 1

    pb = sub[sub.post_break == 1]
    # The clean cell: crossed the break having missed NO team games either side.
    pb_exo = pb[pb.team_games_missed == 0]
    pb_endo = pb[pb.team_games_missed >= 1]
    mid = sub[(sub.post_break == 0) & (sub.rest_days >= 5)]
    mid_exo = mid[mid.team_games_missed == 0]
    mid_endo = mid[mid.team_games_missed >= 1]
    normal = sub[(sub.rest_days <= 2) & (sub.team_games_missed == 0)]

    rows = [("post-break, missed no team games (EXOGENOUS)", rate_row(pb_exo)),
            ("post-break, also sat out games (endogenous)", rate_row(pb_endo)),
            ("mid-season gap >= 5d, no games missed (EXOGENOUS)", rate_row(mid_exo)),
            ("mid-season gap >= 5d, sat out games (endogenous)", rate_row(mid_endo)),
            ("normal schedule (1-2 days, no games missed)", rate_row(normal))]

    print(f"\n  {'exposure':>50s} | {'n':>8s} | {'inj':>5s} | {'rate':>8s} | {'95% CI':>16s}")
    print("  " + "-" * 102)
    for nm, r in rows:
        print(f"  {nm:>50s} | {r['n']:>8,} | {r['injuries']:>5,} | {fmt_rate(r)} | {fmt_ci(r):>16s}")

    r_norm = rate_row(normal)
    print(f"\n  Risk ratio vs a normal schedule:")
    out_rr = {}
    for nm, r in rows[:-1]:
        rr = risk_ratio(r, r_norm)
        out_rr[nm] = rr
        if not np.isnan(rr["rr"]):
            print(f"    {nm:>50s} : {rr['rr']:5.2f}x (95% CI {rr['lo']:.2f}-{rr['hi']:.2f})")

    print("\n  Every athlete in the first row took the same 6-8 day league-mandated gap")
    print("  and missed nothing. They are the cleanest available negative control for")
    print("  gap length: maximal rest, zero absence.")

    res[label] = dict(
        breaks={str(k): {kk: (str(vv) if isinstance(vv, pd.Timestamp) else vv)
                         for kk, vv in v.items()} for k, v in breaks.items()},
        post_break_exogenous=rate_row(pb_exo), post_break_endogenous=rate_row(pb_endo),
        mid_season_exogenous=rate_row(mid_exo), mid_season_endogenous=rate_row(mid_endo),
        normal=r_norm, risk_ratios=out_rr)
    print()


# --------------------------------------------------------------------------
# [3] Season openers, pooled
# --------------------------------------------------------------------------
def team_opener_mask(df, sched):
    """True where the row is the athlete's first appearance AND falls on their
    team's first regular-season game of the season.

    2026-09-25: `season_opener` marks the ATHLETE's first appearance,
    which is the right pre-game covariate but the wrong negative control: 1,095
    of 3,373 athlete openers (full panel) came after the team had already played
    -- late returns, some of them documented injury returns -- so the group was
    not "schedule-imposed, no games missed" as Figure 7C labeled it. The control
    keeps only openers on the team's own first game, where no game can have been
    missed. The `season_opener` predictor is unchanged.
    """
    first = (sched.groupby(["playerteamId", "season"])["date"].min()
             .rename("team_first_date").reset_index()
             .rename(columns={"playerteamId": "team"}))
    key = df[["team", "season"]].merge(first, on=["team", "season"], how="left")
    return ((df["season_opener"].to_numpy() == 1)
            & (df["date"].to_numpy() == key["team_first_date"].to_numpy()))


def analysis_openers(df, sched, res):
    print("=" * 96)
    print("  [3] SEASON OPENERS, POOLED ACROSS SEASONS")
    print("=" * 96)
    print("  The control is the athlete's first appearance ON THE TEAM'S FIRST GAME of")
    print("  the season: a long gap imposed by the schedule, with no team game missed.")
    print("  Athletes whose first appearance came after the team had already played")
    print("  (late returns, some documented injury returns) are excluded.\n")
    df = df.assign(team_opener=team_opener_mask(df, sched))
    ath = int((df["season_opener"] == 1).sum())
    n_team = int(df["team_opener"].sum())
    print(f"  athlete first appearances: {ath:,}; on the team's first game: "
          f"{n_team:,}; excluded as late starts: {ath - n_team:,}\n")

    cohorts = {
        "test seasons only (2022-23, 2023-24)": df[df.season.isin(TEST_SEASONS)],
        "high-ascertainment panel (2017-18 on)": df[df.season >= HIGH_ASCERTAINMENT_FROM],
        "full panel (2015-16 on)": df,
    }
    store = {}
    for name, sub in cohorts.items():
        clean = sub[sub["team_games_missed"].notna() & ~sub["traded_in_gap"].astype(bool)]
        op = rate_row(sub[sub.team_opener])
        late = rate_row(sub[(sub.season_opener == 1) & ~sub.team_opener])
        peak = rate_row(sub[(sub.rest_days >= 3) & (sub.rest_days <= 4)])
        b2b = rate_row(sub[sub.rest_days == 1])
        exo3 = rate_row(clean[(clean.rest_days >= 3) & (clean.team_games_missed == 0)])
        endo3 = rate_row(clean[(clean.rest_days >= 3) & (clean.team_games_missed >= 1)])
        rr = risk_ratio(peak, op)
        store[name] = dict(opener=op, late_start_opener=late,
                           rest_3_4=peak, back_to_back=b2b,
                           exogenous_3plus=exo3, endogenous_3plus=endo3,
                           rr_peak_vs_opener=rr,
                           rr_opener_vs_exogenous=risk_ratio(op, exo3))

        print(f"  {name}")
        for lab, r in ((("opener, team's first game"), op), ("3-4 days rest", peak),
                       ("back-to-back", b2b),
                       ("3+ days, EXOGENOUS (team off)", exo3),
                       ("3+ days, ENDOGENOUS (sat out)", endo3)):
            print(f"    {lab:>30s} : {r['injuries']:>4,} / {r['n']:>7,} = "
                  f"{fmt_rate(r)}  (95% CI {fmt_ci(r)})")
        if not np.isnan(rr["rr"]):
            print(f"    {'RR (3-4 days / opener)':>30s} : {rr['rr']:.2f}x "
                  f"(95% CI {rr['lo']:.2f}-{rr['hi']:.2f})")
        ro = store[name]["rr_opener_vs_exogenous"]
        if not np.isnan(ro["rr"]):
            print(f"    {'RR (opener / exogenous 3+)':>30s} : {ro['rr']:.2f}x "
                  f"(95% CI {ro['lo']:.2f}-{ro['hi']:.2f})   <- coherence check")
        print()

    # This used to be a fixed sentence asserting the opener "sits close
    # to" other exogenous gaps. Restricting the control to the team's first game
    # moved that ratio, and a hard-coded conclusion would have kept printing the
    # old one. The verdict is now read off the CI.
    ro = store["high-ascertainment panel (2017-18 on)"]["rr_opener_vs_exogenous"]
    if np.isnan(ro["rr"]):
        verdict = "cannot be estimated"
    elif ro["hi"] < 1:
        verdict = "is LOWER than other exogenous 3+ day gaps (CI excludes 1)"
    elif ro["lo"] > 1:
        verdict = "is HIGHER than other exogenous 3+ day gaps (CI excludes 1)"
    else:
        verdict = "is not distinguishable from other exogenous 3+ day gaps (CI spans 1)"
    print(f"  COHERENCE (high-ascertainment panel). The team-first-game opener {verdict}:")
    print(f"  RR {ro['rr']:.2f} ({ro['lo']:.2f}-{ro['hi']:.2f}).\n")
    res["season_opener_coherence"] = dict(verdict=verdict, **ro)

    res["season_openers"] = store


def manuscript_table(res):
    """Emit Table 2(b) of the manuscript exactly as it should be typeset, so no
    risk ratio in the paper is computed by hand."""
    print("=" * 96)
    print("  [4] MANUSCRIPT TABLE 2(b) -- negative control exposures vs a normal schedule")
    print("=" * 96)
    asb = res["all_star_break"]
    dec = res["high-ascertainment panel"]
    op = res["season_openers"]["high-ascertainment panel (2017-18 on)"]["opener"]
    ref = asb["normal"]

    rows = [
        ("Normal schedule (1-2 d, no games missed)", ref),
        ("Season opener", op),
        ("After All-Star break, no games missed", asb["post_break_exogenous"]),
        ("Mid-season gap >= 5 d, no games missed", asb["mid_season_exogenous"]),
        ("3-4 d gap, athlete missed games", dec["3-4"]["endogenous"]),
    ]
    print(f"  {'exposure':44s} | {'n':>8s} | {'rate (95% CI)':>22s} | {'RR vs normal':>20s}")
    print("  " + "-" * 104)
    out = {}
    for name, r in rows:
        rr = risk_ratio(r, ref)
        out[name] = dict(**r, rr_vs_normal=rr)
        rrs = "1.00 (ref)" if r is ref else (
            "--" if np.isnan(rr["rr"]) else
            f"{rr['rr']:.2f} ({rr['lo']:.2f}-{rr['hi']:.2f})")
        print(f"  {name:44s} | {r['n']:>8,} | "
              f"{r['rate']*100:6.2f}% ({r['lo']*100:.2f}-{r['hi']*100:.2f}) | {rrs:>20s}")
    res["manuscript_table_2b"] = out
    print()


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    print("=" * 96)
    print("  NEGATIVE CONTROL ANALYSES -- is it the LENGTH of a gap, or its CAUSE?")
    print("=" * 96)
    print(f"  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = dataio.load()
    print(f"[LOAD] analytic cohort: {len(df):,} exposures, {int(df.y.sum()):,} injuries "
          f"({df.y.mean():.4%})")

    sched, roster, league, appearances = load_team_schedule()
    print(f"[LOAD] team schedule: {len(sched):,} team-games, "
          f"{sched.playerteamId.nunique()} teams, {league.date.nunique():,} league dates")

    df = attach_team_gap(df, sched, roster, appearances)
    ok = df["team_games_missed"].notna()
    print(f"[LINK] team gap resolved for {ok.sum():,} / {len(df):,} exposures "
          f"({ok.mean():.1%})")
    print(f"[LINK] traded inside the gap (excluded from [1]): "
          f"{int(df['traded_in_gap'].sum()):,}\n")

    res = {}
    analysis_exogenous(df[df.season.isin(TEST_SEASONS)], res, "test seasons",
                       "Held-out seasons 2022-23 and 2023-24.")
    analysis_exogenous(df[df.season >= HIGH_ASCERTAINMENT_FROM], res,
                       "high-ascertainment panel",
                       "Seasons 2017-18 onward, where recorded incidence exceeds ~1%.")
    analysis_exogenous(
        df[(df.season >= HIGH_ASCERTAINMENT_FROM) & (df.injuries_last_365d == 0)], res,
        "no injury recorded in prior 365 days",
        "Sensitivity: athletes carrying no injury record in the preceding year, so the\n"
        "  endogenous arm cannot be a mechanical restatement of a documented injury.")
    # NB: distinct key -- analysis_exogenous already wrote
    # "high-ascertainment panel", and reusing the label would overwrite it.
    analysis_all_star(df[df.season >= HIGH_ASCERTAINMENT_FROM], league, res,
                      "all_star_break")
    analysis_openers(df, sched, res)

    manuscript_table(res)

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"[EXPORT] {JSON_OUT}")


if __name__ == "__main__":
    main()
