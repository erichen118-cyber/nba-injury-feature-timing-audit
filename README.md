# NBA injury forecasting: feature-timing audit and analysis code

Code accompanying *An Executable Feature-Timing Audit for Injury-Prediction Models, with a
Nine-Season National Basketball Association Cohort as the Demonstration* (manuscript in
preparation).

The study asks two questions: how well a tissue-designated musculoskeletal injury can be
forecast from information available before tip-off, and how much of a previously reported
figure came from same-game information. Its main contribution is an **executable audit** that
fails the pipeline when a covariate draws on the game it is meant to predict. That audit is
the part of this repository we would most like other groups to reuse.

This is the analysis-and-audit subset of the authors' working repository. It contains the
code that produced every reported number and the aggregate results it wrote. It does **not**
contain the input data (see [`DATA.md`](DATA.md)), the figure and manuscript-building code,
or the earlier uncontrolled analysis that the manuscript describes and supersedes.

## The audit

Each component is a pipeline stage that exits non-zero on failure. The feature-timing,
permutation and reconstruction controls have each been broken deliberately and watched to
fail (manuscript §3.6a).

| Component | Script | What it checks |
|---|---|---|
| Feature-timing invariance (Tests A–D) | `predictive_model/audit_feature_timing.py` | Rebuilds the covariate matrix on truncated and perturbed panels (future games deleted, the index game's values replaced, future labels rewritten, all labels rewritten) and requires bit-identical output. Four injected leaks act as positive controls, and two association-only columns that include the index game are *required* to fail. |
| Label-permutation control | `predictive_model/audit_leakage_checks.py` | Refits on permuted outcomes; discrimination must fall inside a prespecified chance band. This catches procedural leakage only: it cannot see leakage built into a covariate. |
| Independent history reconstruction | `predictive_model/verify_history_reconstruction.py` | Recomputes playing history from the source panel without reusing builder code, checking both values and coverage. |
| Source-validity control | `predictive_model/check_source_validity.py` | Sweeps count/percentage column pairs for contradictions in the vendor data. |
| Builder fixtures | `predictive_model/test_feature_construction.py` | Synthetic-data regression cases for the covariate builder. **Runs without the study data.** |

The core rule is in `features.py` and `build_predictive_dataset.py`: every covariate is built
from games strictly before the index game, and anything computed from the outcome sits
downstream of the eligibility filter.

## Analysis scripts and results

Each script writes a JSON of aggregate results next to itself. The JSONs from the run
reported in the manuscript are included, so the numbers can be checked without the data.

| Script | Result file | Manuscript section |
|---|---|---|
| `build_predictive_dataset.py` | *(the analytic dataset, not distributed)* | §3.2–3.4 |
| `train_and_evaluate.py` | `metrics.json` | §4.3, §4.5, §4.8, §4.10 |
| `leakage_reconstruction.py` | `leakage_reconstruction.json` | §4.3a (2 × 2 factorial reconstruction) |
| `calibration_and_utility.py` | `calibration_utility.json` | §4.4, §4.8a |
| `gee_lagged_association.py` | `gee_lagged_association.json` | §4.2 |
| `absence_confound_analysis.py` | `absence_confound.json` | §4.6, §4.7 |
| `negative_control_analyses.py` | `negative_controls.json` | §4.6a, §4.6b |
| `sensitivity_analyses.py` | `sensitivity_analyses.json` | §4.11–4.14 |
| `window_sensitivity.py` | `window_sensitivity.json` | §4.11a |
| `ascertainment_analysis.py` | `ascertainment.json` | §4.1; limitation 2a |
| `check_regular_season_rule.py` | *(log only)* | §3.2 cohort rule |

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14. Versions are pinned exactly in
`pyproject.toml` and `uv.lock`, because they are the versions that produced the reported
numbers.

```bash
uv sync

# Runs without any study data (synthetic fixtures):
cd predictive_model && uv run python test_feature_construction.py

# Full pipeline, once the inputs in DATA.md are in place:
uv run python run_pipeline.py            # all stages
uv run python run_pipeline.py --from 2   # skip stage 1 (source compilation)
```

A full run from stage 2 took 6–8 hours on the authors' laptop, about 5 hours of it the GEE
cluster bootstrap.

## Citation

Please cite the manuscript once it is published. Citation details and an archived DOI will be
added here.

## License

Code is released under the MIT License (see `LICENSE`). The license covers this code only, not
any third-party data it is run on.
