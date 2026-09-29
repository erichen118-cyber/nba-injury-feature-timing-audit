"""
run_pipeline.py
================================================================================
Reproduce the study end to end, in order. Each stage depends on the one before.

    python run_pipeline.py

Every stage writes a plain-text log next to itself; every number in the
manuscript is traceable to the run that produced it.

PUBLIC RELEASE
--------------
This is the analysis-and-audit subset of the study's working repository. The
figure, table and manuscript-checking stages are omitted (see _OMITTED below),
as is the earlier leaked-model analysis, which is described in the manuscript
but deliberately not distributed. The input data are not included; see DATA.md.
"""

import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

# This runner echoes captured stage output, so its OWN stdout must accept the
# same characters. When the runner is itself piped (CI, tee, a log file) Python
# picks cp1252 here too, and the runner dies reporting a stage that succeeded.
#
# line_buffering=True is the other half, added 2026-09-20. Python block-buffers
# stdout whenever it is not a terminal, so redirecting a 75-minute run to a log
# -- the only sane way to run it -- left that log at 0 bytes until the process
# exited. A progressing run and a wedged one looked identical, and a crash or a
# kill left no partial log at all. Line buffering makes each stage's start and
# its [OK]/[FAIL] line land in the file as it happens. It costs nothing: this
# runner prints a few dozen lines over more than an hour.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                           line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace",
                           line_buffering=True)
except (AttributeError, OSError, ValueError):
    pass  # already UTF-8, or a stream that cannot be reconfigured

# Stage output is captured through a pipe, so Python picks the locale encoding
# for the child's stdout -- cp1252 on this machine. Any stage that prints a
# non-ASCII character (a check mark, an en dash, a player name) then dies with
# UnicodeEncodeError *after* having already written its outputs, which is how a
# successful run gets reported as a failure. Force UTF-8 for every child.
# GEE_N_BOOT is PINNED to the value that produced the published §4.2 paired
# contrast. gee_lagged_association.py defaults N_BOOT to 500, and this runner
# passes no per-stage environment, so before 2026-09-22 a full pipeline run
# would silently launch a 500-replicate bootstrap: at the 258 s/replicate this
# box measured, roughly 36 HOURS, and it would then overwrite
# gee_lagged_association.json with different percentile bounds and fail
# verify_no_stale_numbers.py on six numbers that were never wrong. Pinning it
# here means a pipeline run REPRODUCES the manuscript rather than re-estimating
# it. An explicit GEE_N_BOOT in the caller's environment still wins, so a
# deliberate re-estimation is one variable away.
STAGE_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
STAGE_ENV.setdefault("GEE_N_BOOT", "200")

# (subdir, script, args, why). Every stage that feeds a manuscript table belongs
# here: four scripts (calibration, negative controls, window sensitivity, human
# audit) were outside this list until 2026-07-28, so their outputs silently
# predated the 2026-07-27 rerun and their tables were never actually verified.
STAGES = [
    ("Data", "compile_data_corrected.py", [],
     "Build the source player-game panel from box scores, biometrics, injuries."),
    # Fixture tests before the build, not after: they run in a second, need no
    # data files, and they state the four defects of the 2026-09 review that
    # are VALUES rather than timings. Failing here costs a second; failing
    # after the build costs 25 minutes and a dataset nobody should trust.
    ("predictive_model", "test_feature_construction.py", [],
     "Hand-built fixtures for rest, the matching window, season boundaries "
     "and injury dating."),
    # Added 2026-09-22, and it belongs HERE -- after the master exists,
    # before anything is built from it. Every other guard in this list certifies
    # WHEN a covariate was measured, whether the builder computes what it claims,
    # or whether the manuscript matches the JSON. None of them asks whether the
    # SOURCE VALUE MEANS ANYTHING, which is how two seasons of identically-zero
    # rebound percentages survived nine prior review findings with every check
    # passing. A defect in the source is not worth propagating through 25
    # minutes of feature construction, so this fails the run before the build.
    ("predictive_model", "check_source_validity.py", [],
     "Source semantics: no count > 0 with its own percentage == 0, bar "
     "documented exceptions."),
    ("predictive_model", "build_predictive_dataset.py", [],
     "Construct the strictly pre-game covariate matrix and event-level outcome."),
    # Reconciles the built dataset against the master WITHOUT reusing builder
    # code. This is the check that would have caught the label-filtered history defect, and it sits
    # immediately after the build so nothing downstream consumes a dataset
    # whose playing history has not been independently confirmed.
    ("predictive_model", "verify_history_reconstruction.py", [],
     "Reconcile rest, season openers and missed team games against the master."),
    # The regular-season rule gives 30 x 82 in 2023-24 (counted from the
    # master, no builder code) and the built dataset has the Cup-corrected rest.
    ("predictive_model", "check_regular_season_rule.py", [],
     "Confirm the regular-season rule reproduces 30 x 82 and a named Cup-affected row."),
    # The narrowed matching windows are inputs to window_sensitivity.py. They were
    # never built by this runner, so that table compared datasets of different
    # vintages. Outputs carry a _w1/_w2 suffix and cannot overwrite the primary.
    ("predictive_model", "build_predictive_dataset.py", ["1"],
     "Rebuild the covariate matrix under a 1-day injury-matching window."),
    ("predictive_model", "build_predictive_dataset.py", ["2"],
     "Rebuild the covariate matrix under a 1-2 day injury-matching window."),
    ("predictive_model", "train_and_evaluate.py", [],
     "Temporal split, validation-based selection, calibration, seeds, ablations."),
    # Added 2026-09-20. Separates feature timing from split design in the
    # 0.7891 -> 0.6743 comparison, which was confounded on three axes. It sits
    # immediately after train_and_evaluate.py because it anchors its cell 1
    # against that stage's metrics.json and fails if the two disagree -- an
    # anchor read from a stale JSON would check nothing.
    ("predictive_model", "leakage_reconstruction.py", [],
     "2x2 reconstruction: same-game features x random/temporal split."),
    ("predictive_model", "calibration_and_utility.py", [],
     "Calibration slope/intercept, O:E, Brier skill, decision curve."),
    ("predictive_model", "absence_confound_analysis.py", [],
     "Rest gradient and restricted-cohort analysis -- the central experiment."),
    ("predictive_model", "gee_lagged_association.py", [],
     "Association screening: concurrent vs strictly pre-game covariates."),
    ("predictive_model", "sensitivity_analyses.py", [],
     "High-ascertainment cohort, positional stratification, workload x rest."),
    ("predictive_model", "negative_control_analyses.py", [],
     "Negative controls, including shuffled-label collapse to chance."),
    # Criterion 4 checks an artifact negative_control_analyses.py has only just
    # written, so the stage-4 invocation above defers it whenever the cohort has
    # changed under it. This second, strict run is where criterion 4 is actually
    # enforced: --require-team-gap makes deferral a failure, so the coverage
    # check cannot be skipped for the whole pipeline. Added 2026-09-18 after
    # A review found the stage-4 run could deadlock the pipeline on any
    # cohort change -- it failed on a stale artifact, which stopped the run
    # before the stage that regenerates that artifact.
    ("predictive_model", "verify_history_reconstruction.py",
     ["--require-team-gap"],
     "Re-check the team-gap reconciliation against the artifact just written."),
    ("predictive_model", "window_sensitivity.py", [],
     "Sensitivity to the matching window (calendar 1-1, 1-2, 1-3)."),
    ("predictive_model", "score_human_audit.py", [],
     "Score the hand-audited team/injury samples against the pipeline labels."),
    # Added 2026-08-30. Quantifies what the tissue-designation requirement
    # excludes -- a third of the categorized injury corpus, at a rate that
    # drifts across the temporal split. Those numbers reached §3.2 and
    # limitation 2a, so they are pipeline output and belong here rather than in
    # a chat transcript; stage 19 now checks the manuscript against this JSON.
    ("predictive_model", "ascertainment_analysis.py", [],
     "Tissue-designation exclusion, its drift across the split, descriptor test."),
    # Validity check, not a results producer. It is in the runner deliberately:
    # a negative control that is not rerun is a negative control that rots, which
    # is how the four stages above came to hold outputs nobody had verified.
    ("predictive_model", "audit_leakage_checks.py", [],
     "Label-permutation negative control and position-covariate sensitivity."),
    # The stage above covers PROCEDURAL leakage only and would have passed the
    # original leaked models. This one covers FEATURE-CONSTRUCTION leakage -- the
    # defect this project actually had -- by rebuilding the covariates on
    # truncated and perturbed panels and demanding bit-identical output. Same
    # reasoning for keeping it in the runner: it re-checks the builder every time
    # the builder changes, which no human reading can promise.
    ("predictive_model", "audit_feature_timing.py", [],
     "Per-column feature-timing control: future-truncation and same-row invariance."),
    ("predictive_model", "make_figures_final.py", [],
     "Figures 1-10 at 600 dpi."),
    ("predictive_model", "make_dag_figure.py", [],
     "Figure 11, the causal DAG."),
    # Added 2026-08-16. This was outside the runner, so a full pipeline run
    # produced every manuscript figure except this one -- and nobody could tell,
    # because the stale PNG sat in figures_final/ looking current.
    ("predictive_model", "make_figure_negative_controls.py", [],
     "Figure 12, exogenous vs endogenous rest."),
    # Added 2026-08-23, straight after the three figure stages. Figure 12 once
    # shipped at 3913 x 8459 px -- four times its intended height, from a single
    # annotation placed outside the axis limits with bbox_inches="tight" -- and
    # the broken PNG reached the Detroit submission packet because nothing
    # checked figure geometry. This asserts every figure is a real journal
    # column width, that no text lies outside its canvas, and that the style
    # ladder holds across all three generators.
    ("predictive_model", "check_figure_geometry.py", ["--render"],
     "Assert figure geometry and style are consistent. Fails the run if not."),
    # Added 2026-09-06, and it must precede the stage below. tables_final/ is the
    # authority the manuscript is now checked against, and it was NOT regenerated
    # by this runner -- so a full run rewrote every JSON and then compared the
    # manuscript against tables built on some earlier day. That is the same defect
    # this list already records twice (four stages outside it in 2026-07-28,
    # make_figure_negative_controls.py in 2026-08-16), one level further out: the
    # guard's own input was the artifact nobody refreshed.
    ("predictive_model", "make_tables_final.py", [],
     "Rebuild tables_final/ from the JSONs -- the authority for the stage below."),
    # Added 2026-08-17, and it belongs LAST: it reads every JSON the stages above
    # write and asserts the manuscripts agree with them. It was outside the runner
    # for its whole life, which is how its own expectations went stale while it
    # sat there looking like a working guard -- the same failure this stage list
    # already records for make_figure_negative_controls.py, one level up.
    # Stale-number regression, run first: if the per-occurrence check cannot fail on
    # a corrupted abstract, the manuscript check below it proves nothing.
    ("predictive_model", "test_claim_consistency.py", [],
     "Regression: a stale abstract beside a correct table must fail."),
    ("predictive_model", "verify_no_stale_numbers.py", [],
     "Assert both manuscripts match the pipeline JSON and every generated table."),
]

# Public release: these stages build figures, tables and manuscript checks, not
# reported numbers, and depend on files that are not distributed here.
_OMITTED = {
    "score_human_audit.py", "make_figures_final.py", "make_dag_figure.py",
    "make_figure_negative_controls.py", "check_figure_geometry.py",
    "make_tables_final.py", "test_claim_consistency.py",
    "verify_no_stale_numbers.py",
}
STAGES = [s for s in STAGES if s[1] not in _OMITTED]


def main():
    # --from N resumes at stage N. Stage 1 rewrites the master panel, which is
    # slow and was already proven byte-identical; skipping it is legitimate only
    # when nothing upstream of N has changed. Say so in the log either way.
    #
    # THE ARGUMENT IS VALIDATED BEFORE ANYTHING RUNS, and a bad one exits 2.
    # `--from 999` used to print "RESUMING AT STAGE 999 -- stages 1-998 NOT
    # rerun", attempt nothing, report "0/0 stages attempted succeeded" and
    # **exit 0** -- so a typo'd stage number, or a stage list that shrank under
    # a script, reported success having done no work. Anything calling this
    # runner (a wrapper, a CI job, a shell `&&`) would have read that as green.
    # Same family as the guards this project keeps finding: a check that cannot
    # fail. `--from` with no value raised a bare IndexError and a non-integer a
    # bare ValueError; both now name the flag.
    start = 1
    if "--from" in sys.argv:
        i = sys.argv.index("--from")
        if i + 1 >= len(sys.argv):
            print(f"error: --from requires a stage number (1-{len(STAGES)})",
                  file=sys.stderr)
            return 2
        raw = sys.argv[i + 1]
        try:
            start = int(raw)
        except ValueError:
            print(f"error: --from expects an integer stage number "
                  f"(1-{len(STAGES)}), got {raw!r}", file=sys.stderr)
            return 2
        if not 1 <= start <= len(STAGES):
            print(f"error: --from {start} is outside the stage list; this "
                  f"pipeline has {len(STAGES)} stages, so --from must be "
                  f"1-{len(STAGES)}. Nothing was run.", file=sys.stderr)
            return 2

    print("=" * 78)
    print("  NBA PRE-GAME INJURY FORECASTING -- FULL PIPELINE")
    print("=" * 78)
    print(f"  {len(STAGES)} stages, run in order.")
    if start > 1:
        print(f"  RESUMING AT STAGE {start} -- stages 1-{start - 1} NOT rerun. "
              f"Their outputs are assumed current.")
    print()

    results = []
    for i, (subdir, script, args, why) in enumerate(STAGES, 1):
        if i < start:
            continue
        path = os.path.join(BASE, subdir, script)
        label = " ".join([script] + args)
        print(f"[{i}/{len(STAGES)}] {subdir}/{label}")
        print(f"        {why}")

        if not os.path.exists(path):
            print("        [SKIP] not found\n")
            results.append((label, "MISSING", 0.0))
            continue

        t0 = time.time()
        proc = subprocess.run([sys.executable, script] + args,
                              cwd=os.path.join(BASE, subdir),
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              env=STAGE_ENV)
        dt = time.time() - t0

        if proc.returncode == 0:
            print(f"        [OK] {dt:.1f}s\n")
            results.append((label, "OK", dt))
        else:
            print(f"        [FAIL] exit {proc.returncode} after {dt:.1f}s")
            # Show BOTH streams. This was `proc.stderr or proc.stdout`, which
            # picks stderr whenever it is non-empty -- and sklearn and xgboost
            # routinely emit FutureWarning/UserWarning there on an otherwise
            # clean run. Stages print their verdict ("FAIL: permuted-label
            # discrimination is outside the prespecified band for: [...]") to
            # STDOUT, so the operator saw `[FAIL] exit 1` followed by an
            # irrelevant deprecation warning and never the reason. Found in
            # review.
            # `stream_name`, NOT `label`: this loop used to bind `label`,
            # shadowing the stage label built above, so the SUMMARY table
            # reported a failed stage as "stderr" instead of naming the script
            # that failed. Only reachable on a failure, which is why it
            # survived -- the one path where the stage's name matters most.
            for stream_name, stream in (("stdout", proc.stdout),
                                        ("stderr", proc.stderr)):
                text = (stream or "").strip()
                if not text:
                    continue
                lines = text.splitlines()[-8:]
                print(f"          {stream_name}:")
                for line in lines:
                    print(f"          | {line}")
            print()
            results.append((label, f"FAIL({proc.returncode})", dt))
            # Stages are strictly sequential: a failure invalidates everything
            # downstream, so stop rather than emit results built on a broken input.
            print("  Stopping: later stages consume this stage's output.")
            break

    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for script, status, dt in results:
        print(f"  {status:12s} {dt:7.1f}s  {script}")

    failed = [r for r in results if r[1].startswith("FAIL")]
    missing = [r for r in results if r[1] == "MISSING"]
    # Count against stages ATTEMPTED, not the full list: on a --from resume the
    # latter reads as a failure ("4/14 succeeded") when nothing actually failed.
    print(f"\n  {len(results) - len(failed)}/{len(results)} stages attempted succeeded.")
    if start > 1:
        print(f"  ({start - 1} earlier stages were skipped by --from {start}.)")

    # A run that did no work is not a successful run. `--from` is validated
    # above so this should be unreachable by that route, but a stage whose
    # script has been moved or renamed is recorded MISSING and was NOT counted
    # as a failure -- so a pipeline that skipped every stage still exited 0.
    # Both now exit non-zero, because the only honest answer to "did the
    # pipeline succeed?" after running nothing is no.
    if not results:
        print("\n  NOTHING RAN. Exiting non-zero: a run that attempts no "
              "stages has not succeeded.")
        return 2
    if missing:
        print(f"\n  {len(missing)} stage script(s) NOT FOUND: "
              f"{', '.join(r[0] for r in missing)}. Exiting non-zero -- a "
              f"stage that cannot be found has not run, and the artifacts "
              f"downstream of it are stale rather than absent.")
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
