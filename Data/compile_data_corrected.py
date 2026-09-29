"""
compile_data_corrected.py
============================================
Non-destructive re-compilation of the NBA injury-analytics master dataset.

PURPOSE:
--------
The original Compile_Data.py has a catastrophic selection-bias bug:
  It merges injuries to player-game logs on `Player` name alone (left join),
  which creates a many-to-many cartesian product. It then keeps only rows
  whose Date Injured is null ("healthy") or within a 2-day window ("injured").
  But because the left join assigns a non-null Date Injured to EVERY row for
  any player who was ever injured, zero healthy rows survive for those players.

  Empirical proof: of 4,713 rows for 644 injured players in the original
  master_model_ready_data.csv, only 43 are labeled healthy (Is_Injured == 0).

FIX:
----
This script performs a non-destructive merge:
  1. Load the three source files (stats, biometrics, injuries).
  2. For each injury record, find the closest game within a 0–2 day window
     using a sorted merge_asof approach (efficient, O(n log n)).
  3. Map injury metadata onto only those matched games.
  4. All other game rows remain labeled as healthy (Is_Injured == 0).
  5. Recompute all derived features (spikes, game numbers, minutes).

OUTPUTS:
--------
  Data/master_model_ready_data_corrected.csv
  Data/compile_audit_report.txt
"""
import pandas as pd
import numpy as np
import os
import warnings
import time

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root, where the three raw inputs live. This was
# dirname(dirname(SCRIPT_DIR)) while the tree sat one level deeper;
# after the 2026-07-19 relocation to the repo root that resolved one level too
# high (to ~/research), so stage 1 could not find its inputs at all.
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

audit_lines = []  # Collect audit notes for the verification report

def audit(msg):
    """Print and log an audit line."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))
    audit_lines.append(msg)


def load_source_data():
    """Load the three source CSV files."""
    audit("=" * 70)
    audit("[STEP 1] Loading Source Data Files")
    audit("=" * 70)

    stats_df = pd.read_csv(os.path.join(BASE_DIR, "PlayerStatisticsExtended.csv"), low_memory=False)
    audit(f"  PlayerStatisticsExtended.csv : {len(stats_df):,} rows, {stats_df['personId'].nunique()} unique players")

    bios_df = pd.read_csv(os.path.join(BASE_DIR, "Longitudinal_Biometrics.csv"))
    audit(f"  Longitudinal_Biometrics.csv  : {len(bios_df):,} rows")

    injury_df = pd.read_csv(
        os.path.join(BASE_DIR, "nba_injury_database_localized_refined.csv"),
        on_bad_lines='skip'
    )
    # Strip whitespace from column names (the CSV has space-padded headers)
    injury_df.columns = injury_df.columns.str.strip()
    # Strip whitespace from string columns
    for col in injury_df.select_dtypes(include='object').columns:
        injury_df[col] = injury_df[col].str.strip()
    audit(f"  nba_injury_database_localized_refined.csv : {len(injury_df):,} injury records, {injury_df['Player'].nunique()} unique players")

    return stats_df, bios_df, injury_df


def prepare_player_names(stats_df, injury_df):
    """
    Build a Player name column in stats_df and create a name-mapping
    lookup to handle alias mismatches in the injury DB.
    """
    audit("\n" + "=" * 70)
    audit("[STEP 2] Name Matching & Alias Resolution")
    audit("=" * 70)

    stats_df['Player'] = stats_df['firstName'] + ' ' + stats_df['lastName']
    stats_players = set(stats_df['Player'].unique())
    inj_players = set(injury_df['Player'].unique())

    # Direct matches
    direct = stats_players.intersection(inj_players)
    audit(f"  Direct name matches: {len(direct)} / {len(inj_players)} injury-DB players")

    # Build alias map for unmatched names
    unmatched = inj_players - stats_players
    alias_map = {}
    for name in unmatched:
        # Handle patterns like "(James) Mike Scott" → "Mike Scott"
        if '(' in name and ')' in name:
            # Extract the part after the parenthetical
            clean = name.split(')')[1].strip() if ')' in name else name
            if clean in stats_players:
                alias_map[name] = clean
                continue
        # Handle patterns like "J.R. Smith" → try without periods
        clean = name.replace('.', '')
        if clean in stats_players:
            alias_map[name] = clean
            continue
        # Handle " / " separator patterns (take each variant)
        if ' / ' in name:
            for variant in name.split(' / '):
                variant = variant.strip()
                if variant in stats_players:
                    alias_map[name] = variant
                    break

    audit(f"  Alias-resolved matches: {len(alias_map)}")
    remaining = unmatched - set(alias_map.keys())
    audit(f"  Remaining unmatched injury players: {len(remaining)}")
    if remaining:
        audit(f"    Sample: {list(remaining)[:10]}")

    # Apply alias mapping to injury_df
    injury_df['Player_Matched'] = injury_df['Player'].map(alias_map).fillna(injury_df['Player'])

    return stats_df, injury_df


def merge_biometrics(stats_df, bios_df):
    """Merge biometric data (AGE, BMI) by personId + Year.

    KNOWN WRONG, and deliberately left in place (2026-09-18).

    `Year` here is `gameDateTimeEst.dt.year` on the left and the first four
    characters of a SEASON string on the right. An NBA season spans two
    calendar years, so those two keys do not name the same thing: games played
    in October-December match nothing at all, and games played in January-
    August match the row for the season that STARTED that January -- the
    following one. Measured on the master, AGE and BMI are missing on 100% of
    Oct/Nov/Dec rows and present on 77-91% of Jan-Aug rows, and 27,739 cohort
    rows carry no biometrics that in fact exist in the source.

    It is not fixed here because this stage cannot currently be re-run: its two
    source inputs (`nba_injury_database_localized_refined.csv` and
    `PlayerStatisticsExtended.csv`) are not at their expected paths, and it
    overwrites the master in place. Editing it would leave the master
    disagreeing with the code that claims to have produced it.

    The join is done correctly downstream instead, in
    `build_predictive_dataset.attach_biometrics()`, which keys on
    (personId, season) with a uniqueness assertion and does not read the AGE
    and BMI columns this function writes. Coverage there is 100%.

    If this stage is ever re-run, fix the key here first and delete this
    paragraph -- do not leave two joins disagreeing.
    """
    audit("\n" + "=" * 70)
    audit("[STEP 3] Merging Biometrics (AGE, BMI)")
    audit("=" * 70)

    stats_df['gameDateTimeEst'] = pd.to_datetime(stats_df['gameDateTimeEst'], errors='coerce')
    stats_df = stats_df.dropna(subset=['Player', 'gameDateTimeEst'])
    stats_df['Year'] = stats_df['gameDateTimeEst'].dt.year
    bios_df['Year'] = bios_df['SEASON'].str[:4].astype(int)

    master = pd.merge(
        stats_df, bios_df[['personId', 'Year', 'AGE', 'BMI']],
        on=['personId', 'Year'], how='left'
    )
    audit(f"  After biometrics merge: {len(master):,} rows")
    audit(f"  AGE coverage: {master['AGE'].notna().sum():,} / {len(master):,} ({master['AGE'].notna().mean():.1%})")

    return master


def merge_injuries_nondestructive(master, injury_df):
    """
    NON-DESTRUCTIVE injury merge using vectorized date-window matching.

    Strategy:
    ---------
    1. Build a lookup: for each player, their list of injury records.
    2. For each player's game log, find injuries where:
         game_date + 1 calendar day <= injury_date <= game_date + 3
    3. Map injury metadata onto matched games only.
    4. All unmatched games remain healthy (Is_Injured == 0).

    This preserves ALL game rows for every player.
    """
    audit("\n" + "=" * 70)
    audit("[STEP 4] Non-Destructive Injury Merge (date-window matching)")
    audit("=" * 70)

    rows_before = len(master)

    # Prepare injury dates
    injury_df['Date Injured'] = pd.to_datetime(injury_df['Date Injured'], errors='coerce')
    injury_df = injury_df.dropna(subset=['Player_Matched', 'Date Injured'])

    # Select the injury columns we need
    inj_cols = ['Player_Matched', 'Date Injured', 'Anatomical_Location',
                'Injury Category', 'Prior_Injuries_to_this_Body_Part',
                'Total_Prior_Injuries_Any']
    inj_subset = injury_df[inj_cols].copy()
    inj_subset = inj_subset.rename(columns={'Player_Matched': 'Player'})
    audit(f"  Total injury records to match: {len(inj_subset):,}")

    # Strategy: Instead of merging ALL 835K stats rows with 5.4K injuries
    # (which creates an enormous cartesian product), we:
    # 1. Identify which players appear in the injury DB
    # 2. Only merge the stats rows for those players (much smaller subset)
    # 3. Find matching game-injury pairs within the 2-day window
    # 4. Map results back to the master dataframe by index

    injured_players_set = set(inj_subset['Player'].unique())
    master_injured_mask = master['Player'].isin(injured_players_set)
    n_injured_rows = master_injured_mask.sum()
    audit(f"  Stats rows for injured players: {n_injured_rows:,} / {len(master):,}")

    # Step 1: Extract only the subset of stats rows for injured players
    subset_df = master.loc[master_injured_mask, ['Player', 'gameDateTimeEst']].copy()
    subset_df['_orig_idx'] = subset_df.index  # preserve original position in master

    # Step 2: Inner merge only the subset with injury records
    audit(f"  Performing targeted inner merge...")
    temp = pd.merge(
        subset_df,
        inj_subset,
        on='Player',
        how='inner'
    )
    audit(f"  Cartesian product size: {len(temp):,} (filtered from full dataset)")
    
    # Step 3: filter to the matching window, in CALENDAR DAYS (2026-09-18).
    #
    # This line used to read `(Date Injured - gameDateTimeEst).dt.days` and
    # keep 0-2. `.dt.days` FLOORS an elapsed duration, and the two operands are
    # not the same kind of thing: `Date Injured` is a date parsed to midnight
    # while `gameDateTimeEst` carries an evening tip-off. A report filed the
    # next calendar morning is about 5 hours later and floors to 0; one filed
    # the same day is negative and never matched at all.
    #
    # So the window documented throughout this project as "0-2 days" was in
    # fact calendar gaps 1-3, with same-day reports excluded. Both sides are
    # now normalized to midnight and the bounds are stated in the units the
    # documentation uses. This is a RESTATEMENT, not a change: verified on the
    # master, the raw values 0/1/2 map onto calendar gaps 1/2/3 with identical
    # counts (1,265 / 1,820 / 1,653 on the active labeled panel).
    #
    # MIN_GAP is 1 by design, not by oversight. A date-only report cannot be
    # ordered against an evening tip-off, so a same-day match cannot be said to
    # follow the game; only 13 of 5,422 source records are in that position.
    MIN_GAP, MAX_GAP = 1, 3
    temp['Days_Diff'] = (temp['Date Injured'].dt.normalize()
                         - temp['gameDateTimeEst'].dt.normalize()).dt.days
    matched = temp[(temp['Days_Diff'] >= MIN_GAP)
                   & (temp['Days_Diff'] <= MAX_GAP)].copy()
    audit(f"  Game-injury pairs within the calendar {MIN_GAP}-{MAX_GAP} day "
          f"window: {len(matched):,}")
    audit("  calendar gap distribution: "
          + ", ".join(f"{int(g)}d {int(n):,}" for g, n
                      in matched['Days_Diff'].value_counts().sort_index().items()))
    
    # Step 4: If a game matched multiple injuries, keep the closest one
    matched = matched.sort_values('Days_Diff').drop_duplicates(subset='_orig_idx', keep='first')
    audit(f"  Unique games with matched injuries: {len(matched):,}")
    
    # Step 5: Build a clean injury lookup from matched pairs, then merge by index
    injury_merge_cols = ['Anatomical_Location', 'Injury Category',
                         'Prior_Injuries_to_this_Body_Part', 'Total_Prior_Injuries_Any',
                         'Date Injured']
    
    # Create a clean lookup: original_index -> injury columns
    injury_lookup = matched.set_index('_orig_idx')[injury_merge_cols].copy()
    # Convert all columns to standard Python types to avoid StringArray issues
    for col in injury_lookup.columns:
        injury_lookup[col] = injury_lookup[col].astype(object)
    
    # Left-merge the injury lookup back onto master using index alignment
    master = master.join(injury_lookup, how='left', rsuffix='_TEMP')
    
    # If there were conflicting column names, use the new ones
    for col in injury_merge_cols:
        temp_col = col + '_TEMP'
        if temp_col in master.columns:
            master[col] = master[temp_col]
            master = master.drop(columns=[temp_col])

    rows_after = len(master)
    audit(f"\n  Rows before merge: {rows_before:,}")
    audit(f"  Rows after merge:  {rows_after:,}")
    assert rows_before == rows_after, f"MERGE CHANGED ROW COUNT! {rows_before} -> {rows_after}"
    audit(f"  Row count preserved (no rows added or deleted)")

    # Create Is_Injured flag
    master['Is_Injured'] = master['Injury Category'].notna().astype(int)

    # Audit injury distribution
    n_inj = master['Is_Injured'].sum()
    n_healthy = (master['Is_Injured'] == 0).sum()
    audit(f"\n  Injury Distribution:")
    audit(f"    Injured rows:  {n_inj:,} ({n_inj / len(master):.2%})")
    audit(f"    Healthy rows:  {n_healthy:,} ({n_healthy / len(master):.2%})")

    # CRITICAL AUDIT: Check that injured players now have healthy rows
    injured_player_ids = master[master['Is_Injured'] == 1]['personId'].unique()
    subset = master[master['personId'].isin(injured_player_ids)]
    healthy_for_injured = (subset['Is_Injured'] == 0).sum()
    audit(f"\n  CRITICAL CHECK: Healthy rows for injured players: {healthy_for_injured:,}")
    audit(f"    (Was 43 in the buggy dataset; must be >> 0 now)")

    return master


def engineer_features(df):
    """Recompute all derived features on the corrected dataset."""
    audit("\n" + "=" * 70)
    audit("[STEP 5] Feature Engineering (Spikes, Game Numbers, Workload)")
    audit("=" * 70)

    # Sort chronologically
    df = df.sort_values(['personId', 'gameDateTimeEst']).reset_index(drop=True)

    # Forward-fill prior injury counts within player groups
    df['Total_Prior_Injuries_Any'] = df.groupby('Player')['Total_Prior_Injuries_Any'].ffill().fillna(0)
    df['Prior_Injuries_to_this_Body_Part'] = df.groupby('Player')['Prior_Injuries_to_this_Body_Part'].ffill().fillna(0)

    # Acute/Chronic spike ratios
    tracking_vars = ['numMinutes', 'fieldGoalsAttempted', 'reboundsTotal']
    for var in tracking_vars:
        audit(f"  Computing spikes for: {var}")
        df[f'{var}_Acute'] = df.groupby('Player')[var].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f'{var}_Chronic'] = df.groupby(['Player', 'Year'])[var].transform(
            lambda x: x.expanding().mean()
        )
        df[f'{var}_Spike'] = df[f'{var}_Acute'] / (df[f'{var}_Chronic'] + 0.001)

    audit(f"  ✓ Spike features computed for {len(tracking_vars)} variables")
    return df


def apply_filters_and_save(df):
    """Filter to post-2015, save the corrected dataset, and write audit report."""
    audit("\n" + "=" * 70)
    audit("[STEP 6] Applying Post-2015 Filter & Saving")
    audit("=" * 70)

    pre_filter = len(df)
    df = df[df['gameDateTimeEst'] >= '2015-10-01'].copy()
    audit(f"  Pre-filter rows:  {pre_filter:,}")
    audit(f"  Post-2015 rows:   {len(df):,}")
    audit(f"  Removed (pre-2015): {pre_filter - len(df):,}")

    # Final injury distribution
    n_inj = df['Is_Injured'].sum()
    n_healthy = (df['Is_Injured'] == 0).sum()
    audit(f"\n  Final Injury Distribution (Post-2015):")
    audit(f"    Injured rows:  {n_inj:,} ({n_inj / len(df):.2%})")
    audit(f"    Healthy rows:  {n_healthy:,} ({n_healthy / len(df):.2%})")
    audit(f"    Total rows:    {len(df):,}")
    audit(f"    Unique players: {df['personId'].nunique():,}")

    # Final critical check
    injured_pids = df[df['Is_Injured'] == 1]['personId'].unique()
    subset = df[df['personId'].isin(injured_pids)]
    healthy_for_injured = (subset['Is_Injured'] == 0).sum()
    injured_for_injured = (subset['Is_Injured'] == 1).sum()
    audit(f"\n  ✓ FINAL CRITICAL CHECK (Post-2015, Injured Players Only):")
    audit(f"    Total rows:    {len(subset):,}")
    audit(f"    Healthy rows:  {healthy_for_injured:,} ({healthy_for_injured / len(subset):.1%})")
    audit(f"    Injured rows:  {injured_for_injured:,} ({injured_for_injured / len(subset):.1%})")

    # Spot-check a specific player
    sample_pid = injured_pids[0] if len(injured_pids) > 0 else None
    if sample_pid is not None:
        player_rows = df[df['personId'] == sample_pid]
        pname = player_rows['Player'].iloc[0]
        audit(f"\n  Spot Check — Player: {pname} (personId={sample_pid})")
        audit(f"    Total games: {len(player_rows)}")
        audit(f"    Healthy:     {(player_rows['Is_Injured'] == 0).sum()}")
        audit(f"    Injured:     {(player_rows['Is_Injured'] == 1).sum()}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "master_model_ready_data_corrected.csv")
    df.to_csv(out_path, index=False)
    audit(f"\n  ✓ Saved corrected dataset to: {out_path}")
    audit(f"    File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Write audit report
    report_path = os.path.join(OUTPUT_DIR, "compile_audit_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(audit_lines))
    print(f"\n  ✓ Audit report saved to: {report_path}")

    return df


def main():
    t0 = time.time()
    audit("=" * 70)
    audit("  Non-Destructive Data Compilation Pipeline")
    audit("=" * 70)

    # Load
    stats_df, bios_df, injury_df = load_source_data()

    # Name matching
    stats_df, injury_df = prepare_player_names(stats_df, injury_df)

    # Biometrics
    master = merge_biometrics(stats_df, bios_df)

    # Non-destructive injury merge
    master = merge_injuries_nondestructive(master, injury_df)

    # Feature engineering
    master = engineer_features(master)

    # Filter and save
    master = apply_filters_and_save(master)

    elapsed = time.time() - t0
    audit(f"\n  Total execution time: {elapsed:.1f} seconds")
    audit("=" * 70)
    audit("  COMPILATION COMPLETE")
    audit("=" * 70)

    return master


if __name__ == '__main__':
    main()
