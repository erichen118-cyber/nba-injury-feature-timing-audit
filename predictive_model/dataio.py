"""
dataio.py
================================================================================
One place that knows how to find and load the analytic dataset.

WHY THIS EXISTS
---------------
`build_predictive_dataset.py` writes parquet when pyarrow is available and falls
back to gzipped CSV when it is not, so downstream scripts that hard-coded one
extension broke whenever the environment changed. They now all call `load()`.

Parquet is preferred: it is roughly an order of magnitude faster to read than the
100 MB gzipped CSV and it preserves dtypes, so `dt` needs no re-parsing.
"""

import os
import sys

import pandas as pd

import features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rest-day reporting bins, in ONE place. absence_confound_analysis.py and
# train_and_evaluate.py both report a rest gradient and the manuscript reads
# them side by side (§4.6 against §4.10a), so the two must bin identically.
# They were two independent copies with a comment asserting they matched, which
# nothing enforced: splitting a bin in one file would have left the manuscript
# silently comparing different gradients.
REST_BINS = [0, 1, 2, 3, 4, 6, 13, 100]
REST_LABELS = ["1 (back-to-back)", "2", "3", "4", "5-6", "7-13", "14+"]

# Regular-season membership, in ONE place (2026-09-25). The master
# labels the 66 in-season-tournament games of 2023-24 "NBA Emirates Cup"
# (group play, quarterfinals, semifinals); every one counts in the standings,
# and with them all 30 teams reach 82 games. Filtering on "Regular Season"
# alone dropped them from every athlete's history, so retained rows carried a
# wrong rest_days (LeBron James 2023-11-15 stored 7 days' rest; truth is 1).
# The single "Emirates NBA Cup" championship game (2025-12-16) does not count
# in the standings and stays out. Every stage filters through this function.
# The 2023-24 championship (2023-12-09) would be excluded the same way, but it
# is not in the master at all -- a source gap, stated as limitation 2b; the
# finalists' next game carries rest measured from the 12-07 semifinal.
REGULAR_SEASON_TYPES = frozenset({"Regular Season", "NBA Emirates Cup"})


def is_regular_season(game_type):
    """Boolean mask: True where `game_type` is a standings-counting game."""
    return game_type.isin(REGULAR_SEASON_TYPES)


class StaleDatasetError(RuntimeError):
    """A build for this suffix exists in two formats and the parquet is older.

    Its own type, not a bare RuntimeError, so callers can tell it apart from a
    genuine crash. `window_sensitivity.py` skips ONE arm on a missing or stale
    build and carries on to the primary window; a bare RuntimeError sailed past
    its `except FileNotFoundError` and killed the stage before the primary arm
    ran. Subclasses RuntimeError so any existing broad handler still catches it.
    """


def dataset_path(suffix=""):
    """Return the analytic dataset path, preferring parquet.

    `suffix` selects a matching-window sensitivity build, e.g. "_w1".
    """
    parquet = os.path.join(SCRIPT_DIR, f"predictive_dataset{suffix}.parquet")
    csv_gz = os.path.join(SCRIPT_DIR, f"predictive_dataset{suffix}.csv.gz")
    if os.path.exists(parquet):
        # Second line of defense for the staleness path the builder now closes
        # by moving a superseded parquet aside. If both exist and the CSV is
        # NEWER, the parquet is from an older build and returning it would serve
        # a stale cohort silently -- so refuse rather than guess which the
        # caller wanted.
        if (os.path.exists(csv_gz)
                and os.path.getmtime(csv_gz) > os.path.getmtime(parquet) + 1):
            raise StaleDatasetError(
                f"Refusing to choose between two analytic datasets for suffix "
                f"'{suffix}'. The gzip CSV is NEWER than the parquet:\n"
                f"  {csv_gz}  (newer)\n  {parquet}  (older, but preferred by "
                f"this loader)\n"
                "That is the signature of a build whose parquet write failed "
                "and fell back to CSV. Delete or rename whichever is stale, "
                "then re-run. Do not assume the parquet is current.")
        return parquet
    if os.path.exists(csv_gz):
        return csv_gz
    raise FileNotFoundError(
        f"No analytic dataset for suffix '{suffix}'. Expected one of:\n"
        f"  {parquet}\n  {csv_gz}\n"
        f"Run: python predictive_model/build_predictive_dataset.py"
        + (f" {suffix.lstrip('_w')}" if suffix else ""))


def load(suffix="", columns=None, concussion="exclude",
         concurrent="exclude", verbose=True):
    """Load the analytic dataset with `dt` as datetime64.

    `columns` reads only the named columns. The dataset is 216k rows by ~130
    columns -- 123 model covariates, the meta columns, and the association-only
    columns described under `concurrent` below -- so a caller that needs two
    columns should say so rather than materialising all of them; parquet reads
    only the requested column chunks off disk.

    `concussion` decides what happens to concussion index games, which the
    builder labels `y_concussion = 1`, `y = 0`, `tissue = 0`:

      "exclude" (default) -- drop those rows and drop the `y_concussion`
          column. This is the MSK analytic cohort, and it is the default
          because the alternative failure is silent: a stage that kept them
          would score a concussed athlete as a healthy control, and nothing
          downstream would say so. Excluding here rather than in each of the
          eight MSK stages also means a stage added tomorrow is safe without
          having to remember. Dropping the column too keeps `y_concussion` out
          of every `[c for c in df.columns if c not in META]` feature list.

      "keep" -- return every row with `y_concussion` intact. Only the
          concussion stratum in `train_and_evaluate.py` wants this.

    `concurrent` decides whether the caller receives the index-game-inclusive
    association covariates (`features.CONCURRENT_ONLY`: the `_c5` five-game
    windows ending AT game t, and the `_c1` index-game values):

      "exclude" (default) -- drop those columns entirely. They read game t and
          are a leakage hazard in any forecasting model, so the default is to
          make them invisible rather than to rely on every stage remembering
          to subtract them from `[c for c in df.columns if c not in META]`.
          Asking for one by name in `columns` while excluding is a
          contradiction and raises.

      "include" -- return them. Exactly ONE caller in this repo passes this:
          `gee_lagged_association.py`, whose S4.2 comparator is the reason
          they are built at all. If you are adding a second, that is a
          decision to document, not a convenience.

    A dataset built before 2026-09-04 has no `y_concussion` column. That is an
    error rather than a shrug: proceeding would silently restore the old
    behavior, in which 109 concussion events were dropped from the cohort and a
    further 4 were counted as musculoskeletal positives -- 113 events in all,
    110 of which survive the MIN_CAREER_GAMES filter and form the stratum. See
    the count glossary in build_predictive_dataset.py's CONCUSSION block.
    """
    if concussion not in ("exclude", "keep"):
        raise ValueError(f"concussion must be 'exclude' or 'keep', got {concussion!r}")
    if concurrent not in ("exclude", "include"):
        raise ValueError(
            f"concurrent must be 'exclude' or 'include', got {concurrent!r}")
    if concurrent == "exclude" and columns is not None:
        asked = [c for c in columns if c in features.CONCURRENT_ONLY]
        if asked:
            raise ValueError(
                f"load(columns={asked!r}) asks for index-game-inclusive "
                "covariates while concurrent='exclude'. Pass "
                "concurrent='include' and say why -- these columns read game "
                "t and may not enter a forecasting model.")

    path = dataset_path(suffix)
    # `y_concussion` is appended for BOTH modes, not just "exclude". Guarding
    # this on the mode meant load(columns=[...], concussion="keep") came back
    # without the column and then tripped the staleness check below, telling the
    # caller to spend 25 minutes rebuilding a dataset that was perfectly current.
    read_cols = columns
    if columns is not None and "y_concussion" not in columns:
        read_cols = list(columns) + ["y_concussion"]

    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=read_cols)
    else:
        # parse_dates on a column that was not read raises; only ask when present.
        parse = ["dt"] if (read_cols is None or "dt" in read_cols) else None
        df = pd.read_csv(path, low_memory=False, usecols=read_cols,
                         parse_dates=parse)
    if "dt" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["dt"]):
        df["dt"] = pd.to_datetime(df["dt"])

    if "y_concussion" not in df.columns:
        raise KeyError(
            f"{path} has no `y_concussion` column, so it predates the "
            "concussion stratum (2026-09-04). Rebuild it:\n"
            f"  uv run python predictive_model/build_predictive_dataset.py"
            + (f" {suffix.lstrip('_w')}" if suffix else ""))

    if concurrent == "exclude":
        drop = [c for c in features.CONCURRENT_ONLY if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
    else:
        # Which columns SHOULD be here: all of them for a full read, or just
        # the requested ones when `columns` was given. Guarding only the
        # `columns is None` case meant a stale dataset read with an explicit
        # column list either raised a bare pandas KeyError or -- worse --
        # succeeded and returned a frame with no concurrent covariates at all,
        # despite the caller explicitly asking to include them.
        _want = ([c for c in features.CONCURRENT_ONLY if c in columns]
                 if columns is not None else list(features.CONCURRENT_ONLY))
        missing = [c for c in _want if c not in df.columns]
        if missing:
            raise KeyError(
                f"{path} has none of {missing!r}, so it predates the S4.2 "
                "concurrent comparator (2026-09-21). Rebuild it:\n"
                f"  uv run python predictive_model/build_predictive_dataset.py"
                + (f" {suffix.lstrip('_w')}" if suffix else ""))

    if concussion == "exclude":
        df = drop_concussion(df, verbose=verbose)
    return df


def drop_concussion(df, verbose=True):
    """Return the musculoskeletal view of a frame loaded with `concussion="keep"`.

    Split out of `load()` so a caller that needs BOTH frames -- the concussion
    stratum does -- can read the file once and derive the second view, instead
    of holding two near-identical 216k x 124 frames from two reads. That
    doubling landed on the pipeline's heaviest stage, which is where two runs
    died of memory exhaustion on 2026-09-04.
    """
    n = int(df["y_concussion"].sum())
    out = df[df["y_concussion"] == 0].drop(columns=["y_concussion"]).reset_index(drop=True)
    if verbose:
        print(f"[COHORT] dropped {n} concussion index games from the MSK "
              f"risk set -> {len(out):,} rows")
    return out


def recover_team_ids(df, id_col="playerteamId", name_col="playerteamName",
                     verbose=True):
    """Fill a missing team id from the team NAME, and refuse to guess.

    WHY THIS EXISTS
    ---------------
    2021-22 carries a null `playerteamId` on 25,805 of 25,826 active Regular
    Season rows (99.9%); every other season is 0. Any stage that drops rows on
    this column therefore excludes that season whole. `negative_control_analyses.py`
    does exactly that, and the exogenous/endogenous decomposition -- the evidence
    the participation-collider claim rested on -- has been computed without one
    of nine seasons since it was written.

    `playerteamName` is populated on every row in every season, and the name-to-id
    relation is an unambiguous function (30 names, one id each, zero conflicts),
    so the repair is a lookup rather than an inference.

    WHAT THIS DELIBERATELY DOES NOT DO
    ----------------------------------
    The deleted `gee_cohort_control/gee_cohort_analysis.py:78` filled anything the
    map missed with `float(hash(name) % 10**8)` -- a synthetic id invented from a
    string hash. That fabricates team identities, and Python salts string `hash()`
    per process unless PYTHONHASHSEED is fixed, so it is not even reproducible
    across runs. There is no fallback here: if a row cannot be resolved, this
    raises and names the offenders.

    Raises on an ambiguous map as well. Silently picking one id for a name that
    has two would corrupt team scheduling without any visible symptom.
    """
    out = df.copy()
    # Both columns must be non-null to teach the map. Dropping only on `id_col`
    # lets a NULL-NAMED row into it, and the two pandas calls below then
    # disagree about nulls: `groupby` silently drops a NaN key, so the conflict
    # check never sees it, while `drop_duplicates`/`to_dict`/`map` keep it and
    # happily match NaN to NaN. The result was that a row with no name AND no id
    # got filled with an arbitrary other null-named row's id instead of raising.
    # Verified 2026-09-18 on names [NaN, NaN, NaN] with ids [737, 738, NaN]: the
    # third row silently became 737. Found in review. No row in the
    # current master has a null name, so no reported number was affected.
    known = out.dropna(subset=[id_col, name_col])
    conflicts = known.groupby(name_col)[id_col].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise AssertionError(
            f"{name_col} -> {id_col} is not a function; cannot recover. "
            f"Ambiguous: {conflicts.to_dict()}")

    name_to_id = (known.drop_duplicates(subset=[name_col])
                       .set_index(name_col)[id_col].to_dict())
    missing_before = int(out[id_col].isna().sum())
    out[id_col] = out[id_col].fillna(out[name_col].map(name_to_id))
    missing_after = int(out[id_col].isna().sum())

    if missing_after:
        bad = sorted(out.loc[out[id_col].isna(), name_col].dropna().unique())
        raise AssertionError(
            f"{missing_after:,} rows still have no {id_col} after recovery. "
            f"Unresolvable {name_col} values: {bad[:20]}. No synthetic id is "
            "invented -- fix the source or exclude these rows deliberately.")

    if verbose and missing_before:
        print(f"[TEAM] recovered {missing_before:,} missing {id_col} from "
              f"{name_col} ({len(name_to_id)} teams, 0 unresolved)")
    return out


# =============================================================================
# STAGE LOGGING
# =============================================================================
# Every stage that produces a manuscript number writes a plain-text log next to
# itself, so the run that produced each value is on disk. Until 2026-09-20 each
# of the thirteen such stages carried its OWN byte-identical copy of the `Tee`
# class below and hand-rolled the open / redirect / try / restore / close dance
# in its `main()`. `leakage_reconstruction.py` imported `Tee` from
# `train_and_evaluate.py` -- one analysis module reaching into another purely
# because there was nowhere better to put it.
#
# The duplication was not the problem. The problem is that each copy
# independently decided WHERE THE VERDICT IS PRINTED RELATIVE TO THE LOG BEING
# CLOSED, and they gave four different answers:
#
#   audit_feature_timing.py     verdict to `tee.terminal` only -- so a FAILING
#                               run left a log byte-identical to a passing one.
#   verify_history_reconstruction.py  banner in the log, plus a second line to
#                               `tee.terminal`.
#   audit_leakage_checks.py     verdict inside the `try` (fixed 2026-09-18),
#                               plus a `FAIL:` line to `tee.terminal` after the
#                               close -- the same defect reintroduced one layer
#                               out, on the same day it was fixed.
#   the other ten               `print("Done.")` AFTER `tee.file.close()`, so
#                               the tracked log never recorded that the stage
#                               finished at all.
#
# Four answers to one question in code that was otherwise identical, and every
# new stage got to choose again. `run_stage()` below removes the choice: it
# writes the verdict while stdout is still the Tee, and closes the file only
# afterwards. A stage cannot opt out without deleting the helper.
#
# `print(..., file=tee.terminal)` was never the right call anyway. A Tee writes
# to the terminal AND the file, so printing to `sys.stdout` already reaches the
# console; naming `tee.terminal` only SKIPS the file. And `run_pipeline.py`
# captures stage stdout and discards it on exit 0, so a "Done." printed to the
# real stdout after the close is written nowhere a human will ever read it.


class Tee:
    """`sys.stdout` duplicated to a UTF-8 log file.

    Deliberately not a `contextlib` wrapper: `run_stage()` needs the original
    stream back on the way out even when the body raises, and an explicit
    try/finally there is easier to read than a nested context manager.
    """

    def __init__(self, path):
        self.terminal = sys.stdout
        # buffering=1 -- line-buffered, added 2026-09-20 alongside the same fix
        # in `run_pipeline.py`. The thirteen private copies of this class all
        # used the default, which for a regular file is a 8 KiB block buffer, so
        # a stage's log stayed at 0 BYTES until the stage exited. During an
        # 18-minute audit there was no way to tell a progressing stage from a
        # wedged one, and the project's run instructions had to tell readers to
        # watch artifact mtimes because the logs were useless as progress. It
        # also means a killed stage now leaves the output it had actually
        # produced rather than an empty file.
        self.file = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, m):
        self.terminal.write(m)
        self.file.write(m)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def _verdict(result):
    """Normalize a stage function's return value to `(exit_code, message)`.

    Accepted forms, chosen to cover what the thirteen stages already returned:

        None          -> (0, "Done.")          the nine analysis stages
        int           -> (n, ...)              the three exit-code stages
        (int, str)    -> as given              a stage with its own wording,
                                               e.g. score_human_audit.py
    """
    if result is None:
        return 0, "Done."
    if isinstance(result, tuple):
        code, message = result
        return int(code), str(message)
    code = int(result)
    return code, "Done." if code == 0 else f"FAIL: stage returned exit {code}"


def run_stage(fn, txt_out):
    """Run `fn` with stdout teed to `txt_out`; return its process exit code.

    THE WHOLE POINT: the verdict line is printed while `sys.stdout` is still
    the Tee, so it lands in the tracked log as well as on the console, and the
    file is closed only after. A failing run can therefore never leave a log
    that looks like a passing one -- which is exactly what
    `audit_feature_timing.py` did until this helper existed.

    A raise gets the same treatment. The traceback still goes to stderr (and to
    `run_pipeline.py`, which prints both streams on a non-zero exit), but the
    log now ends with an explicit `FAIL:` line rather than simply stopping,
    which is indistinguishable from a kill or a truncation.

    The caller decides what to do with the returned code. Most stages ignore it
    and rely on the raise; the three audit stages `sys.exit()` on it.
    """
    tee = Tee(txt_out)
    sys.stdout = tee
    try:
        try:
            code, message = _verdict(fn())
        except BaseException as exc:  # noqa: BLE001 -- re-raised immediately
            print(f"FAIL: stage raised {type(exc).__name__}: {exc}")
            raise
        print(message)
    finally:
        sys.stdout = tee.terminal
        tee.close()
    return code
