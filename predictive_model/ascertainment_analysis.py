"""
ascertainment_analysis.py
================================================================================
Quantify what the tissue-designation requirement costs the outcome definition.

WHY THIS SCRIPT EXISTS
----------------------
`build_predictive_dataset.py` maps three tissue categories to a positive label
and drops every other categorized injury row, `Unknown / General Injury`
included (see its EXCLUDE_CATEGORIES, and the `eligible` mask built in
attach_outcomes).
Those rows leave the panel entirely: they are neither positives nor negatives.

That exclusion is far larger than the manuscript's phrasing implied. It was
disclosed only as the word "uncategorised" inside a list with illness,
suspension and rest, which reads as a trivial residual. It is a third of the
categorized matched injury events. The study's outcome is therefore not "injury" but
"injury whose tissue was named in the source record", and the difference is
material enough that it belongs in Limitations with numbers attached.

This script produces those numbers so they have a provenance row rather than
living in a chat message. It changes no model and no existing output.

WHAT IS REPORTED
----------------
  1. Categorized matched injury events by category, and the unknown share.
  2. That share across the temporal split blocks. It is not constant, so the
     effective case definition differs between training and test seasons --
     which is the part a reviewer of a temporally-validated model will ask about.
  3. Whether the labelers were following a descriptor-first rule, tested on the
     `human audit/` templates -- the only records in this repository that retain
     the source note text. This distinguishes "the labeling is unreliable" from
     "the source text carries no tissue", which are different problems with
     different remedies.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not reclassify anything. A reclassification rule was planned and
abandoned; see `docs/archive/plans/tissue-reclass-sensitivity-plan.md` for why, in
short: the recoverable rows cannot move a metric whose confidence interval is
already +/-0.020, and the site-concordance rule that would have recovered more
of them is contaminated by the very inconsistency documented in (3) below.

The note text exists for 200 audited records, of which only some fall inside the
matched event set; coverage is reported as k of N after checking each record's
membership. Everything in (3) is therefore an audited-sample estimate and is
reported as one.

THE UNIT IS THE EVENT (2026-09-25)
------------------------------------------
`temporal_injury_data.csv` holds one row per matched GAME, so an injury matched
to several games was counted several times. Counts here are distinct athlete /
report-date events ("matched injury events"), restricted to regular-season
appearances under `dataio.is_regular_season`. The file is a matched subset of
the injury source, not the source, so it is not called a corpus. Game-row
counts stay in the JSON for traceability only.

OUTPUTS
-------
  predictive_model/ascertainment_output.txt
  predictive_model/ascertainment.json
"""

import os
import re
import json
import glob
import datetime
import warnings

import numpy as np
import pandas as pd

import dataio

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
INJURY = os.path.join(ANALYSIS_DIR, "Data", "temporal_injury_data.csv")
AUDIT_DIR = os.path.join(ANALYSIS_DIR, "human audit")
TXT_OUT = os.path.join(SCRIPT_DIR, "ascertainment_output.txt")
JSON_OUT = os.path.join(SCRIPT_DIR, "ascertainment.json")

UNKNOWN = "Unknown / General Injury"
TISSUE_CATS = ["1. Ligament / Joint Injuries",
               "2. Muscle / Tendon Injuries",
               "3. Bone / Contusion Injuries"]

# Split blocks mirror train_and_evaluate.py exactly: seasons 2015-2020 train,
# 2021 calibration, 2022-2023 test. Written as the source file's season labels.
TRAIN_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
VAL_SEASONS = ["2021-22"]
TEST_SEASONS = ["2022-23", "2023-24"]

# Descriptor vocabularies. These name a TISSUE, which is the thing the category
# scheme encodes; they are deliberately not a general injury-word list.
DESCRIPTORS = {
    "Muscle/Tendon": r"strain|torn|tear|tendin|tendon|fascii|rupture|pull",
    "Ligament/Joint": r"sprain|ligament|\bacl\b|\bmcl\b|dislocat|labr|menisc",
    "Bone/Contusion": r"contusion|fracture|broken|bruise|\bbone\b|stress reaction",
}


def classify_note(note):
    """Which tissue, if any, the source note names.

    Ligament/Joint is tested BEFORE Muscle/Tendon: "torn meniscus" contains
    "torn" but a meniscus is fibrocartilage, and reading it as muscle/tendon
    on the strength of the verb is exactly the error this ordering prevents.
    """
    n = str(note).lower()
    for tissue in ("Ligament/Joint", "Muscle/Tendon", "Bone/Contusion"):
        if re.search(DESCRIPTORS[tissue], n):
            return tissue
    return None


def short_cat(value):
    """Map a full category string to a short label, or 'Unknown'."""
    mapping = {TISSUE_CATS[0]: "Ligament/Joint",
               TISSUE_CATS[1]: "Muscle/Tendon",
               TISSUE_CATS[2]: "Bone/Contusion"}
    return mapping.get(value, "Unknown")


def main():
    return dataio.run_stage(_run, TXT_OUT)


def _run():
    out = {"generated": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"}

    print("=" * 84)
    print("  INJURY ASCERTAINMENT: WHAT THE TISSUE REQUIREMENT EXCLUDES")
    print("=" * 84)

    inj = pd.read_csv(INJURY, encoding="utf-8", low_memory=False)
    rows = inj[inj["Injury Category"].notna()].copy()
    key = ["personId", "Date Injured"]

    # ---- 0. the unit ---------------------------------------------------------
    # The file holds one row per matched GAME. First reproduce the
    # 2026-09-24 review's all-game-type event figures, as a check that this
    # dedup key is the one it used, THEN apply the regular-season restriction.
    ev_all = rows.drop_duplicates(key)
    unit = {
        "matched_game_rows_all_types": int(len(rows)),
        "events_all_types": int(len(ev_all)),
        "unknown_events_all_types": int((ev_all["Injury Category"] == UNKNOWN).sum()),
        "events_with_conflicting_category": int(
            (rows.groupby(key)["Injury Category"].nunique() > 1).sum()),
    }
    print(f"\n[UNIT] all game types: {unit['matched_game_rows_all_types']:,} "
          f"matched game rows = {unit['events_all_types']:,} distinct athlete/"
          f"report-date events ({unit['unknown_events_all_types']:,} unknown "
          f"tissue); events with conflicting categories: "
          f"{unit['events_with_conflicting_category']}")
    rs_rows = rows[dataio.is_regular_season(rows["gameType"])]
    cat = rs_rows.drop_duplicates(key).copy()
    unit["matched_game_rows_regular_season"] = int(len(rs_rows))
    unit["events_regular_season"] = int(len(cat))
    print(f"[UNIT] regular season (dataio.is_regular_season): {len(rs_rows):,} "
          f"rows = {len(cat):,} events. Everything below counts these events.")
    out["unit"] = unit

    # ---- 1. matched injury events ------------------------------------------
    print(f"\n[EVENTS] categorized matched injury events: {len(cat):,}")
    counts = cat["Injury Category"].value_counts()
    for name, n in counts.items():
        print(f"    {name:38s} {n:6,}  ({100 * n / len(cat):5.2f}%)")

    n_unknown = int((cat["Injury Category"] == UNKNOWN).sum())
    pct_unknown = 100 * n_unknown / len(cat)
    print(f"\n[EXCLUDED] {UNKNOWN}: {n_unknown:,} of {len(cat):,} events = {pct_unknown:.1f}%")
    print("           These events are dropped from the panel entirely -- neither")
    print("           positive nor negative. The outcome is 'injury whose tissue")
    print("           was named', not 'injury'.")

    out["events"] = {
        "categorized_events": int(len(cat)),
        "by_category": {str(k): int(v) for k, v in counts.items()},
        "unknown_events": n_unknown,
        "unknown_pct": round(pct_unknown, 2),
    }

    # ---- 2. drift across the temporal split ---------------------------------
    print("\n[DRIFT] unknown share by season")
    per = cat.groupby("Season").agg(n=("Injury Category", "size"))
    per["unknown"] = cat[cat["Injury Category"] == UNKNOWN].groupby("Season").size()
    per["unknown"] = per["unknown"].fillna(0).astype(int)
    per["pct"] = (100 * per["unknown"] / per["n"]).round(1)
    print(per.to_string())

    blocks = {}
    print("\n[DRIFT] unknown share by split block")
    for label, seasons in (("train", TRAIN_SEASONS), ("validation", VAL_SEASONS),
                           ("test", TEST_SEASONS)):
        sub = cat[cat["Season"].isin(seasons)]
        k = int((sub["Injury Category"] == UNKNOWN).sum())
        p = 100 * k / len(sub)
        blocks[label] = {"events": int(len(sub)), "unknown": k, "pct": round(p, 1)}
        print(f"    {label:11s} events={len(sub):5,}  unknown={k:5,}  {p:5.1f}%")
    print("    The case definition is therefore not identical across the split.")

    out["drift"] = {
        "by_season": {str(s): {"events": int(r.n), "unknown": int(r.unknown),
                               "pct": float(r.pct)} for s, r in per.iterrows()},
        "by_block": blocks,
    }

    # ---- 3. was a descriptor-first rule being followed? ---------------------
    print("\n" + "=" * 84)
    print("  DESCRIPTOR TEST (audited sample -- the only records retaining notes)")
    print("=" * 84)

    files = [f for f in sorted(glob.glob(os.path.join(AUDIT_DIR, "*.csv")))
             if "corrected" not in os.path.basename(f)]
    aud = pd.concat([pd.read_csv(f, encoding="utf-8") for f in files],
                    ignore_index=True)
    aud = (aud.dropna(subset=["Raw Notes"])
              .drop_duplicates(subset=["Player", "Date Injured", "Raw Notes"])
              .reset_index(drop=True))
    aud["descriptor"] = aud["Raw Notes"].map(classify_note)
    aud["assigned"] = aud["Injury Category"].map(short_cat)

    # Coverage is k of N, with k checked record by record -- an audited
    # record counts only if its (name, injury date) is one of the events above.
    # The old len(aud) / len(cat) divided 200 deduplicated records by a game-row
    # count and included records outside the event set (pre-panel seasons,
    # reports never matched to a game).
    ev_keys = set(zip(cat["Player"],
                      pd.to_datetime(cat["Date Injured"]).dt.normalize()))
    k_cov = sum((p, d) in ev_keys for p, d in
                zip(aud["Player"],
                    pd.to_datetime(aud["Date Injured"]).dt.normalize()))
    print(f"\n[SAMPLE] unique audited injury records with note text: {len(aud):,};"
          f" inside the matched event set: {k_cov} of {len(cat):,} events")

    tab = pd.crosstab(aud["descriptor"].fillna("(none)"), aud["assigned"])
    print("\n[TABLE] source descriptor vs assigned category")
    print(tab.to_string())

    has = aud[aud["descriptor"].notna()]
    non = aud[aud["descriptor"].isna()]

    agree = int((has["descriptor"] == has["assigned"]).sum())
    conflict = has[(has["assigned"] != "Unknown")
                   & (has["descriptor"] != has["assigned"])]
    n_resolved = int((has["assigned"] != "Unknown").sum())
    print(f"\n[DESCRIPTOR PRESENT] {len(has):,} records")
    print(f"    assigned a tissue category      : {n_resolved:,}")
    print(f"    of those, matching the descriptor: {agree:,} "
          f"({100 * agree / max(n_resolved, 1):.1f}%)")
    print(f"    contradicting the descriptor     : {len(conflict):,}")
    print(f"    left Unknown despite a descriptor: "
          f"{int((has['assigned'] == 'Unknown').sum()):,}")

    unk_non = int((non["assigned"] == "Unknown").sum())
    mt_non = int((non["assigned"] == "Muscle/Tendon").sum())
    print(f"\n[NO DESCRIPTOR] {len(non):,} records")
    print(f"    left Unknown                     : {unk_non:,} "
          f"({100 * unk_non / len(non):.0f}%)")
    print(f"    assigned Muscle/Tendon anyway    : {mt_non:,} "
          f"({100 * mt_non / len(non):.0f}%)")
    print("    Same note shape, opposite fate: one becomes a positive, the")
    print("    other is deleted from the dataset.")

    # Sites where descriptor-free notes went BOTH ways -- the inconsistency is
    # within site, so it cannot be explained as site-driven judgement.
    non = non.copy()
    non["site"] = (non["Anatomical_Location"].fillna("")
                   .str.replace("^(Left|Right) ", "", regex=True))
    st = pd.crosstab(non["site"], non["assigned"])
    both = []
    if "Muscle/Tendon" in st.columns and "Unknown" in st.columns:
        split = st[(st["Muscle/Tendon"] > 0) & (st["Unknown"] > 0)]
        both = sorted(split.index.tolist())
        print(f"\n[WITHIN-SITE SPLIT] sites where descriptor-free notes were "
              f"assigned BOTH ways: {len(both)}")
        print(split.to_string())

    out["descriptor_test"] = {
        "audited_records": int(len(aud)),
        "audited_in_event_set": int(k_cov),
        "event_set_size": int(len(cat)),
        "descriptor_present": {
            "records": int(len(has)),
            "assigned_a_tissue": n_resolved,
            "matching_descriptor": agree,
            "contradicting_descriptor": int(len(conflict)),
            "left_unknown": int((has["assigned"] == "Unknown").sum()),
        },
        "descriptor_absent": {
            "records": int(len(non)),
            "left_unknown": unk_non,
            "assigned_muscle_tendon": mt_non,
            "pct_left_unknown": round(100 * unk_non / len(non), 1),
        },
        "within_site_split_sites": both,
    }

    # ---- 4. what a reclassification could and could not have moved ----------
    print("\n" + "=" * 84)
    print("  WHY NO RECLASSIFICATION WAS RUN")
    print("=" * 84)
    site = (cat["Anatomical_Location"].fillna("")
            .str.replace("^(Left|Right) ", "", regex=True))
    at_ach = cat[site == "Achilles"]
    ach = at_ach[at_ach["Injury Category"] == UNKNOWN]
    ach_classified = at_ach[at_ach["Injury Category"].isin(TISSUE_CATS)]
    ach_mt = int((ach_classified["Injury Category"] == TISSUE_CATS[1]).sum())
    print(f"\n  Achilles is the strongest anatomical case: a tendon by definition,")
    print(f"  and {ach_mt} of {len(ach_classified)} classified rows at that site are")
    print(f"  muscle/tendon ({100 * ach_mt / max(len(ach_classified), 1):.0f}%).")
    print(f"  Unknown-tissue events at Achilles: {len(ach):,} across nine seasons.")
    # Computed, not typed (2026-09-28 review): this used to hard-code "~15
    # events would join a test set of 757 ... move ROC-AUC by 0.014", figures
    # from the old game-row count and the pre-correction test fold.
    k_test = int(ach["Season"].isin(TEST_SEASONS).sum())
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), encoding="utf-8") as fh:
        met = json.load(fh)
    sel = met["models"][met["best_model"]]
    n_pos = int(met["split"]["test_pos"])
    auc, (ci_lo, ci_hi) = sel["roc_auc"], sel["roc_ci"]
    # Upper bound: every test-season event joins the test set, and each is
    # ranked below every negative, which lowers ROC-AUC by auc * k / (P + k).
    shift = auc * k_test / (n_pos + k_test)
    print(f"  {k_test} of them fall in the test seasons. Even if all {k_test} joined")
    print(f"  the {n_pos}-event test set and were ranked below every negative, ROC-AUC")
    print(f"  would fall by at most {shift:.3f}, against a confidence interval of")
    print(f"  +/-{(ci_hi - ci_lo) / 2:.3f} on the selected model.")
    out["achilles_reclassification_bound"] = {
        "test_season_events": k_test, "test_events": n_pos,
        "max_roc_auc_shift": round(shift, 4),
        "roc_ci_half_width": round((ci_hi - ci_lo) / 2, 4)}
    out["achilles_unknown_events"] = int(len(ach))

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[EXPORT] {JSON_OUT}")
    print(f"[EXPORT] {TXT_OUT}")



if __name__ == "__main__":
    main()
