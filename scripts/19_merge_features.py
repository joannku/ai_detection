#!/usr/bin/env python3
"""
Merge all meta-features into a single dataset for the meta-classifier.

Sources:
  - Coherence + style similarity  (output/generate_features/features.csv)
  - LIWC-22 categories            (output/liwc/liwc.csv)
  - Readability metrics           (output/readability/readability.csv)
  - NLI consistency               (output/nli_consistency/nli_consistency.csv)
  - BLEU (participant-level)      (output/bleu/participant_bleu.csv)
  - High-confidence subset        (output/high_confidence_subset/high_confidence_subset.csv)

Output:
  - output/merge_features/meta_features.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_OUTPUT = ROOT / "output" / "merge_features"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)

# Feature-set variant. Controls which coherence/style + NLI inputs are read
# and the suffix applied to the merged output. Only coherence, style
# similarity and the NLI rates change between variants; LIWC, readability and
# BLEU are identical across variants.
#   "orig"      -> original adjacent-sentence metrics (0-imputed short entries)
#   "sw"        -> sliding-window metrics
#   "drop"      -> original metrics with <2-sentence entries dropped (NaN)
#   "drop_nowc" -> same inputs as "drop" but the word_count column is removed
#                  from the feature matrix (ablation of raw length signal)
VARIANT = "drop"

# "drop_nowc" reuses the "drop" per-entry inputs, so its *data* suffix is
# "_drop" while its *output* suffix is the distinct "_drop_nowc".
_DATA_VARIANT = "drop" if VARIANT == "drop_nowc" else VARIANT
DATA_SUFFIX = "" if _DATA_VARIANT == "orig" else f"_{_DATA_VARIANT}"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"
EXCLUDE_WC = VARIANT == "drop_nowc"

LIWC_META_COLS = {"Row ID", "Segment"}

# Exclude LIWC summary variables that use proprietary internal algorithms
# (Analytic, Clout, Authentic, Tone) and length / dictionary-coverage proxies
# already captured by dedicated length features (WC, WPS) or that are not
# word categories at all (Dic = % of tokens matched by the LIWC dictionary).
# Matches the words-only LASSO exclusion set in scripts/15_analyze_liwc.py.
# `BigWords` is *kept* because it measures the proportion of long words,
# which our length features and readability metrics do not capture directly.
LIWC_EXCLUDE = {"Analytic", "Clout", "Authentic", "Tone", "WC", "WPS", "Dic"}


def load_high_confidence():
    path = ROOT / "output" / "high_confidence_subset" / "high_confidence_subset.csv"
    df = pd.read_csv(path)
    print(f"High-confidence subset: {len(df)} entries")
    return df


def load_coherence_style():
    """Load per-entry coherence and style similarity from the (variant) features CSV."""
    path = ROOT / "output" / f"generate_features{DATA_SUFFIX}" / f"features{DATA_SUFFIX}.csv"
    if not path.exists():
        print(f"WARNING: {path.name} not found — coherence & style similarity will be missing.")
        print(f"  Run `python scripts/11_generate_features{DATA_SUFFIX}.py` first, then re-run this script.")
        return None

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["JournalUniqueID"], keep="first")
    # Variant CSVs carry suffixed metric names (e.g. coherence_sw); normalise
    # them back to canonical names so the merged matrix is variant-agnostic.
    df = df.rename(columns={
        f"coherence{DATA_SUFFIX}": "coherence",
        f"style_similarity{DATA_SUFFIX}": "style_similarity",
    })
    cols = ["JournalUniqueID", "coherence", "style_similarity",
            "word_count", "sentence_count", "character_count"]
    present = [c for c in cols if c in df.columns]
    print(f"Coherence/style: loaded {len(df)} entries from {path.name}")
    return df[present]


def load_liwc():
    path = ROOT / "output" / "liwc" / "liwc.csv"
    df = pd.read_csv(path)
    drop = [c for c in LIWC_META_COLS if c in df.columns]
    df = df.drop(columns=drop)
    exclude = [c for c in LIWC_EXCLUDE if c in df.columns]
    df = df.drop(columns=exclude)
    if exclude:
        print(f"  Excluded LIWC columns: {exclude}")
    df = df.drop_duplicates(subset=["JournalUniqueID"], keep="first")
    liwc_feature_cols = [c for c in df.columns if c not in ("JournalUniqueID",)]
    df = df.rename(columns={c: f"liwc_{c}" for c in liwc_feature_cols})
    print(f"LIWC: {len(df)} entries, {len(liwc_feature_cols)} features")
    return df


def load_readability():
    path = ROOT / "output" / "readability" / "readability.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["final_label"], errors="ignore")
    read_cols = [c for c in df.columns if c not in ("JournalUniqueID",)]
    df = df.rename(columns={c: f"read_{c}" for c in read_cols})
    print(f"Readability: {len(df)} entries, {len(read_cols)} features")
    return df


def load_nli():
    path = ROOT / "output" / f"nli_consistency{DATA_SUFFIX}" / f"nli_consistency{DATA_SUFFIX}.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["final_label"], errors="ignore")
    # Normalise variant column names back to the canonical schema so the merged
    # matrix is identical across variants (only the values differ). The sliding
    # window variant also renames its window-based bookkeeping columns so they
    # land in the same DROP_FEATURES slots downstream.
    rename_map = {
        f"consistency_score{DATA_SUFFIX}": "consistency_score",
        f"contradiction_rate{DATA_SUFFIX}": "contradiction_rate",
        f"entailment_rate{DATA_SUFFIX}": "entailment_rate",
        f"neutral_rate{DATA_SUFFIX}": "neutral_rate",
        "n_windows": "n_sentences",
        "n_adj_pairs": "n_pairs",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    nli_cols = [c for c in df.columns if c not in ("JournalUniqueID",)]
    df = df.rename(columns={c: f"nli_{c}" for c in nli_cols})
    print(f"NLI consistency: {len(df)} entries, {len(nli_cols)} features")
    return df


def load_bleu():
    """BLEU is per-participant; merge via ParticipantID."""
    path = ROOT / "output" / "bleu" / "participant_bleu.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"bleu": "bleu_participant"})
    df = df[["ParticipantID", "bleu_participant"]]
    print(f"BLEU: {len(df)} participants")
    return df


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Variant: {VARIANT} (output suffix: '{SUFFIX}')")

    hc = load_high_confidence()
    hc_ids = set(hc["JournalUniqueID"])

    journals = pd.read_csv(ROOT / "output" / "export" / "final_journals.csv")
    id_to_participant = journals.set_index("JournalUniqueID")["ParticipantID"].to_dict()

    print()
    coherence = load_coherence_style()
    liwc = load_liwc()
    readability = load_readability()
    nli = load_nli()
    bleu = load_bleu()

    # Start with high-confidence subset (JournalUniqueID + label)
    merged = hc[["JournalUniqueID", "final_label"]].copy()
    merged["ParticipantID"] = merged["JournalUniqueID"].map(id_to_participant)

    feature_sources = [("liwc", liwc), ("readability", readability), ("nli", nli)]
    if coherence is not None:
        feature_sources.insert(0, ("coherence", coherence))

    for name, df in feature_sources:
        before = len(merged)
        merged = merged.merge(df, on="JournalUniqueID", how="left")
        matched = merged["JournalUniqueID"].isin(df["JournalUniqueID"]).sum()
        print(f"  Merged {name}: {matched}/{before} matched")

    # BLEU via ParticipantID
    before = len(merged)
    merged = merged.merge(bleu, on="ParticipantID", how="left")
    matched = merged["bleu_participant"].notna().sum()
    print(f"  Merged BLEU: {matched}/{before} matched")

    # drop_nowc ablation: remove the raw word_count length feature.
    if EXCLUDE_WC and "word_count" in merged.columns:
        merged = merged.drop(columns=["word_count"])
        print("  Excluded word_count from feature matrix (drop_nowc variant)")

    # Move identifiers and label to the front
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = sorted(c for c in merged.columns if c not in id_cols)
    merged = merged[id_cols + feature_cols]

    n_features = len(feature_cols)
    n_missing = merged[feature_cols].isna().sum().sum()
    n_cells = len(merged) * n_features

    out_path = REVIEWS_OUTPUT / f"meta_features{SUFFIX}.csv"
    merged.to_csv(out_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"Merged meta-features dataset")
    print(f"  Entries    : {len(merged)}")
    print(f"  Features   : {n_features}")
    print(f"  Missing    : {n_missing}/{n_cells} cells ({n_missing/n_cells:.1%})")
    print(f"  Labels     : ai={sum(merged.final_label=='ai')}, human={sum(merged.final_label=='human')}")
    print(f"  Output     : {out_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
