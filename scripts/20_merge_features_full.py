#!/usr/bin/env python3
"""
Build the full-dataset meta-feature matrix (all valid entries, not just
the high-confidence subset).

Uses the same feature sources as merge_features.py but starts from ALL
valid rated entries rather than the high-confidence subset.

Output:
  - output/merge_features_full/meta_features_full.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_OUTPUT = ROOT / "output" / "merge_features_full"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)

LIWC_META_COLS = {"Row ID", "Segment"}

# Exclude LIWC summary variables that use proprietary internal algorithms
# (Analytic, Clout, Authentic, Tone) and length / dictionary-coverage proxies
# already captured by dedicated length features (WC, WPS) or that are not
# word categories at all (Dic = % of tokens matched by the LIWC dictionary).
# Matches the words-only LASSO exclusion set in scripts/15_analyze_liwc.py.
# `BigWords` is *kept* because it measures the proportion of long words,
# which our length features and readability metrics do not capture directly.
LIWC_EXCLUDE = {"Analytic", "Clout", "Authentic", "Tone", "WC", "WPS", "Dic"}


def load_all_valid():
    ratings = pd.read_csv(ROOT / "ratings" / "all_ratings.csv")
    valid = ratings[
        ~ratings["excluded_invalid_majority"]
        & ratings["final_label"].isin(["ai", "human"])
    ].copy()
    valid = valid[["JournalUniqueID", "final_label"]].drop_duplicates(
        subset=["JournalUniqueID"], keep="first"
    )
    print(f"All valid entries: {len(valid)}")
    print(f"  ai={sum(valid.final_label=='ai')}, human={sum(valid.final_label=='human')}")
    return valid


def load_coherence_style():
    path = ROOT / "output" / "generate_features" / "features.csv"
    if not path.exists():
        print("WARNING: features.csv not found — coherence & style similarity will be missing.")
        return None
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["JournalUniqueID"], keep="first")
    cols = ["JournalUniqueID", "coherence", "style_similarity",
            "word_count", "sentence_count", "character_count"]
    present = [c for c in cols if c in df.columns]
    print(f"Coherence/style: loaded {len(df)} entries")
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
    liwc_feature_cols = [c for c in df.columns if c != "JournalUniqueID"]
    df = df.rename(columns={c: f"liwc_{c}" for c in liwc_feature_cols})
    print(f"LIWC: {len(df)} entries, {len(liwc_feature_cols)} features")
    return df


def load_readability():
    path = ROOT / "output" / "readability" / "readability.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["final_label"], errors="ignore")
    read_cols = [c for c in df.columns if c != "JournalUniqueID"]
    df = df.rename(columns={c: f"read_{c}" for c in read_cols})
    print(f"Readability: {len(df)} entries, {len(read_cols)} features")
    return df


def load_nli():
    path = ROOT / "output" / "nli_consistency" / "nli_consistency.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["final_label"], errors="ignore")
    nli_cols = [c for c in df.columns if c != "JournalUniqueID"]
    df = df.rename(columns={c: f"nli_{c}" for c in nli_cols})
    print(f"NLI consistency: {len(df)} entries, {len(nli_cols)} features")
    return df


def load_bleu():
    path = ROOT / "output" / "bleu" / "participant_bleu.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"bleu": "bleu_participant"})
    df = df[["ParticipantID", "bleu_participant"]]
    print(f"BLEU: {len(df)} participants")
    return df


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)

    valid = load_all_valid()

    journals = pd.read_csv(ROOT / "output" / "export" / "final_journals.csv")
    id_to_participant = journals.set_index("JournalUniqueID")["ParticipantID"].to_dict()

    print()
    coherence = load_coherence_style()
    liwc = load_liwc()
    readability = load_readability()
    nli = load_nli()
    bleu = load_bleu()

    merged = valid[["JournalUniqueID", "final_label"]].copy()
    merged["ParticipantID"] = merged["JournalUniqueID"].map(id_to_participant)

    feature_sources = [("liwc", liwc), ("readability", readability), ("nli", nli)]
    if coherence is not None:
        feature_sources.insert(0, ("coherence", coherence))

    for name, df in feature_sources:
        before = len(merged)
        merged = merged.merge(df, on="JournalUniqueID", how="left")
        matched = merged["JournalUniqueID"].isin(df["JournalUniqueID"]).sum()
        print(f"  Merged {name}: {matched}/{before} matched")

    before = len(merged)
    merged = merged.merge(bleu, on="ParticipantID", how="left")
    matched = merged["bleu_participant"].notna().sum()
    print(f"  Merged BLEU: {matched}/{before} matched")

    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = sorted(c for c in merged.columns if c not in id_cols)
    merged = merged[id_cols + feature_cols]

    n_features = len(feature_cols)
    n_missing = merged[feature_cols].isna().sum().sum()
    n_cells = len(merged) * n_features

    out_path = REVIEWS_OUTPUT / "meta_features_full.csv"
    merged.to_csv(out_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"Full-dataset meta-features")
    print(f"  Entries    : {len(merged)}")
    print(f"  Features   : {n_features}")
    print(f"  Missing    : {n_missing}/{n_cells} cells ({n_missing/n_cells:.1%})")
    print(f"  Labels     : ai={sum(merged.final_label=='ai')}, human={sum(merged.final_label=='human')}")
    print(f"  Output     : {out_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
