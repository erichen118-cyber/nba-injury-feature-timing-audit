# Data

**No study data are distributed in this repository.** The biometric source's terms of use
prohibit republication, and the analytic panel combines all three sources below. Each source
is publicly accessible, and the scripts expect the files at the paths shown (relative to the
repository root).

## Sources

| Data | Source | Terms |
|---|---|---|
| Player box scores | *NBA Dataset: Box Scores and Stats (1947 - Today)* on Kaggle, [`eoinamoore/historical-nba-data-and-player-box-scores`](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores), file `PlayerStatisticsExtended.csv`. Compiled by its author from NBA.com. | CC0 1.0 |
| Injury events | Injured-list transactions from [Pro Sports Transactions](https://www.prosportstransactions.com/), collected by the authors for 2013-10-29 to 2024-06-30 | See the site's terms |
| Age and body-mass index | NBA.com Stats player biographical data (`LeagueDashPlayerBioStats` endpoint), retrieved season by season with the [`nba_api`](https://github.com/swar/nba_api) Python package | [NBA.com Terms of Use](https://www.nba.com/termsofuse): no republication without permission |

All three were retrieved no later than 23 May 2026. The Kaggle dataset is updated continuously,
and the other two sources revise their records, so re-collection will not reproduce the
analyzed panel exactly.

## Expected files

| File | Used by | What it is |
|---|---|---|
| `PlayerStatisticsExtended.csv` | stage 1, `Data/compile_data_corrected.py` | Box scores (838,269 rows) |
| `nba_injury_database_localized_refined.csv` | stage 1 | Categorized injury events (5,412 records) |
| `Longitudinal_Biometrics.csv` | stage 1; `build_predictive_dataset.py` | Age and body-mass index by athlete-season |
| `Data/master_model_ready_data_corrected.csv` | stage 2 onward | The linked player-game panel that stage 1 writes |
| `Data/temporal_injury_data.csv` | `ascertainment_analysis.py` | Injury records for the ascertainment analysis |
| `human audit/*.csv` | `ascertainment_analysis.py` | Source note text for a sample of injury records (manuscript limitation 2a) |

## What is included instead

The aggregate result JSONs in `predictive_model/` are the outputs of the run reported in the
manuscript. They contain summary statistics only, with no athlete-level rows.
`predictive_model/test_feature_construction.py` exercises the covariate builder on synthetic
data and runs without any of the files above.
