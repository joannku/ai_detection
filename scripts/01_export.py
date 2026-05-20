#!/usr/bin/env python3
"""Export analysis journals: written only, valid labels, not invalid by majority."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    journals = pd.read_csv(ROOT / "input/journals.csv", dtype=str).drop_duplicates("JournalUniqueID")
    ratings = pd.read_csv(ROOT / "ratings/all_ratings.csv")
    
    # Filter: written only, valid labels, not invalid-majority
    written = journals[journals["Type"].str.lower() != "voice"]
    valid_ratings = ratings[~ratings["excluded_invalid_majority"] & ratings["final_label"].isin(["ai", "human"])]
    
    df = written.merge(valid_ratings[["JournalUniqueID", "final_label"]], on="JournalUniqueID")
    df = df.rename(columns={"JournalAnonymised": "text", "final_label": "label"})
    df = df[["JournalUniqueID", "ParticipantID", "text", "label"]].dropna(subset=["text"])
    
    out_dir = ROOT / "output" / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_journals.csv"
    df.to_csv(out, index=False)
    
    total_words = df["text"].str.split().str.len().sum()
    print(f"✓ {len(df)} entries, {total_words:,} words → {out.name} (ai={sum(df.label=='ai')}, human={sum(df.label=='human')})")


if __name__ == "__main__":
    main()
