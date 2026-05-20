#!/usr/bin/env python3
"""Combine rater files, normalize labels, majority-vote, write all_ratings.csv."""

import re
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RATINGS_DIR = PROJECT_DIR / "ratings"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_CSV = RATINGS_DIR / "all_ratings.csv"

# Label token sets for fuzzy matching
AI_TOKENS = {"fraudulent", "fraud", "ai", "aigenerated", "fake", "chatgpt", "gpt",
             "fradulent", "fadulent", "fraudelent", "frauduelnt"}
HUMAN_TOKENS = {"authentic", "human", "real", "humanwritten", "ok", "accept",
                "authetic", "authetntic", "authentc", "authenic", "authntic", "humsn"}
INVALID_TOKENS = {"invalid", "invaild", "skip", "n/a", "na", "notapplicable", "exclude", "ni"}


def normalize_journal_id(raw_id) -> Optional[str]:
    """Convert ID to consistent string format."""
    if pd.isna(raw_id): 
        return None
    if isinstance(raw_id, float) and raw_id.is_integer(): 
        return str(int(raw_id))
    return str(raw_id).strip()


def normalize_rating_label(raw_label) -> Optional[str]:
    """Normalize rating label to 'human', 'ai', 'invalid', or None."""
    if pd.isna(raw_label): 
        return None

    # Handle numeric encodings
    if isinstance(raw_label, (int, float)) and not isinstance(raw_label, bool):
        return {-1: "invalid", 1: "human", 0: "ai"}.get(raw_label)
    
    normalized_text = re.sub(r"[^a-z0-9]", "", str(raw_label).strip().lower())
    
    # Single-letter shorthands
    if normalized_text == "i": return "invalid"
    if normalized_text == "f": return "ai"
    if normalized_text in {"a", "h"}: return "human"
    if normalized_text in {"af", "fa"}: return None  # Ambiguous
    
    # Token matching
    if "invalid" in normalized_text or normalized_text in INVALID_TOKENS: 
        return "invalid"
    
    contains_ai_token = "fraud" in normalized_text or normalized_text in AI_TOKENS
    contains_human_token = "human" in normalized_text or "authent" in normalized_text or normalized_text in HUMAN_TOKENS
    
    if contains_ai_token and contains_human_token: 
        return None  # Ambiguous
    if contains_ai_token: 
        return "ai"
    if contains_human_token: 
        return "human"
    return None


def find_column_by_candidates(dataframe: pd.DataFrame, candidate_names: List[str]) -> Optional[str]:
    """Find column name matching any candidate (case-insensitive, ignoring special chars)."""
    normalized_to_original = {re.sub(r"[^a-z0-9]", "", col.lower()): col for col in dataframe.columns}
    for candidate in candidate_names:
        if candidate in normalized_to_original: 
            return normalized_to_original[candidate]
        return None


def load_rater_file(file_path: Path) -> Tuple[str, pd.DataFrame]:
    """Load a rater file, normalize IDs and ratings."""
    # Extract rater ID from filename
    rater_match = re.search(r"rater[_-]?([a-z0-9]+)", file_path.stem, re.I)
    rater_id = rater_match.group(1) if rater_match else file_path.stem
    
    # Load file
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        excel_sheets = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
        rater_df = max(excel_sheets.values(), key=len, default=pd.DataFrame())
    else:
        rater_df = pd.read_csv(file_path)

    # Detect columns
    id_column = find_column_by_candidates(rater_df, ["journaluniqueid", "journalid", "entryid", "id"])
    rating_column = find_column_by_candidates(rater_df, ["manualreview", "rating", "label"])
    
    if not id_column or not rating_column:
        return rater_id, pd.DataFrame(columns=["JournalUniqueID", f"rating_{rater_id}"])
    
    # Build normalized output
    rater_ratings = pd.DataFrame({
        "JournalUniqueID": rater_df[id_column].apply(normalize_journal_id),
        f"rating_{rater_id}": rater_df[rating_column].apply(normalize_rating_label)
    }).drop_duplicates(subset=["JournalUniqueID"], keep="last")

    print(f"{file_path.name}: {len(rater_ratings)} entries")
    return rater_id, rater_ratings


def get_valid_journal_ids() -> Tuple[set, int, int]:
    """Get valid IDs from journals.csv, excluding empty and duplicate content."""
    journals_path = INPUT_DIR / "journals.csv"
    if not journals_path.exists(): 
        return set(), 0, 0
    
    journals_df = pd.read_csv(journals_path, dtype=str)
    journals_df["JournalUniqueID"] = journals_df["JournalUniqueID"].apply(normalize_journal_id)
    
    text_column = "JournalAnonymised" if "JournalAnonymised" in journals_df.columns else "JournalContent"
    if text_column not in journals_df.columns or "ParticipantID" not in journals_df.columns:
        return set(journals_df["JournalUniqueID"].dropna()), 0, 0
    
    journals_df["content_stripped"] = journals_df[text_column].fillna("").str.strip()
    
    # Count and exclude empty entries
    num_empty = (journals_df["content_stripped"] == "").sum()
    journals_df = journals_df[journals_df["content_stripped"] != ""]
    
    # Mark and count content duplicates
    journals_df["is_duplicate"] = journals_df.duplicated(subset=["ParticipantID", "content_stripped"], keep="first")
    num_duplicates = journals_df["is_duplicate"].sum()
    
    valid_ids = set(journals_df.loc[~journals_df["is_duplicate"], "JournalUniqueID"].dropna())
    return valid_ids, num_duplicates, num_empty


def get_sample_stats() -> dict:
    """Compute sample statistics for paper text."""
    journals_path = INPUT_DIR / "journals.csv"
    journals_df = pd.read_csv(journals_path, dtype=str)
    journals_df["JournalUniqueID"] = journals_df["JournalUniqueID"].apply(normalize_journal_id)
    journals_df = journals_df.drop_duplicates(subset=["JournalUniqueID"])
    
    text_column = "JournalAnonymised" if "JournalAnonymised" in journals_df.columns else "JournalContent"
    journals_df["content"] = journals_df[text_column].fillna("").str.strip()
    journals_df["word_count"] = journals_df["content"].apply(lambda t: len(t.split()) if t else 0)
    
    # Filter: non-empty, non-duplicate content
    valid_entries = journals_df[journals_df["content"] != ""]
    if "ParticipantID" in valid_entries.columns:
        valid_entries = valid_entries[~valid_entries.duplicated(subset=["ParticipantID", "content"], keep="first")]
    
    # Load ratings for labelling stats
    ratings_df = pd.read_csv(OUTPUT_CSV)
    text_entry_ids = set(valid_entries.loc[valid_entries.get("Type", pd.Series()).str.lower() != "voice", "JournalUniqueID"])
    labelled = ratings_df[ratings_df["JournalUniqueID"].isin(text_entry_ids)]
    
    return {
        "total_participants": journals_df["ParticipantID"].nunique(),
        "participants_with_entries": valid_entries["ParticipantID"].nunique(),
        "total_entries": len(valid_entries),
        "text_entries": len(valid_entries) - (valid_entries["Type"].str.lower().eq("voice").sum() if "Type" in valid_entries.columns else 0),
        "voice_entries": valid_entries["Type"].str.lower().eq("voice").sum() if "Type" in valid_entries.columns else 0,
        "mean_words": valid_entries["word_count"].mean(),
        "mean_sentences": valid_entries["content"].apply(lambda t: len(re.split(r"[.!?]+", t.strip()))).mean(),
        "mean_chars": valid_entries["content"].apply(len).mean(),
        "total_labelled": len(labelled),
        "human_count": int((labelled["final_label"] == "human").sum()),
        "ai_count": int((labelled["final_label"] == "ai").sum()),
        "invalid_count": int((labelled["final_label"] == "invalid").sum()),
        "invalid_majority": int(labelled["excluded_invalid_majority"].sum()),
    }


def compute_majority_vote(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Add majority vote columns to ratings dataframe."""
    rating_columns = [col for col in ratings_df.columns if col.startswith("rating_")]
    
    def count_labels_for_entry(row):
        labels = [row[col] for col in rating_columns if pd.notna(row[col])]
        return pd.Series({
            "num_raters": len(labels),
            "num_ai": sum(label == "ai" for label in labels),
            "num_human": sum(label == "human" for label in labels),
            "num_invalid": sum(label == "invalid" for label in labels),
        })
    
    label_counts = ratings_df.apply(count_labels_for_entry, axis=1)
    ratings_df = pd.concat([ratings_df, label_counts], axis=1)
    
    def determine_final_label(row):
        if row["num_raters"] == 0: return "invalid"
        if row["num_invalid"] > row["num_raters"] / 2: return "invalid"
        if row["num_human"] > row["num_raters"] / 2: return "human"
        if row["num_ai"] > row["num_raters"] / 2: return "ai"
        return "invalid"  # No clear majority
    
    ratings_df["final_label"] = ratings_df.apply(determine_final_label, axis=1)
    ratings_df["excluded_invalid_majority"] = ratings_df["num_invalid"] > ratings_df["num_raters"] / 2
    return ratings_df


def main():
    # Step 1: Get valid journal IDs
    valid_journal_ids, num_duplicates, num_empty = get_valid_journal_ids()
    print(f"✓ {len(valid_journal_ids):,} valid IDs ({num_empty} empty, {num_duplicates} duplicates excluded)")
    
    # Step 2: Load all rater files
    rater_files = sorted(
        path for path in RATINGS_DIR.iterdir() 
        if path.suffix.lower() in {".csv", ".xlsx"} and "all_ratings" not in path.stem.lower()
    )
    rater_data = [load_rater_file(path) for path in rater_files]
    
    # Step 3: Merge all raters
    combined_ratings = None
    for rater_id, rater_ratings in rater_data:
        if rater_ratings.empty: 
            continue
        if combined_ratings is None:
            combined_ratings = rater_ratings
        else:
            combined_ratings = combined_ratings.merge(rater_ratings, on="JournalUniqueID", how="outer")
    
    if combined_ratings is None:
        print("No data to combine")
        return
    
    # Step 4: Filter to valid IDs
    entries_before_filter = len(combined_ratings)
    combined_ratings = combined_ratings[combined_ratings["JournalUniqueID"].isin(valid_journal_ids)]
    entries_removed = entries_before_filter - len(combined_ratings)
    print(f"✓ Removed {entries_removed} entries not in valid IDs")
    
    # Step 5: Compute majority vote
    combined_ratings = compute_majority_vote(combined_ratings)
    
    # Step 6: Order columns and save
    output_columns = ["JournalUniqueID", "num_raters", "num_ai", "num_human", "num_invalid", 
                      "excluded_invalid_majority", "final_label"]
    output_columns += sorted(col for col in combined_ratings.columns if col.startswith("rating_"))
    combined_ratings = combined_ratings[[col for col in output_columns if col in combined_ratings.columns]]
    combined_ratings.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Wrote {len(combined_ratings):,} rows to {OUTPUT_CSV}")

    # ── Detailed label breakdown (matches the paper's classification-outcomes
    # paragraph exactly so wording stays in sync with the data).
    n_total = len(combined_ratings)
    n_human = int((combined_ratings["final_label"] == "human").sum())
    n_ai = int((combined_ratings["final_label"] == "ai").sum())
    n_invalid_total = int((combined_ratings["final_label"] == "invalid").sum())

    # Invalid sub-breakdown:
    #   1. invalid-majority: >50% of raters labelled the entry invalid.
    #   2. no-clear-majority: no label (human / AI / invalid) crossed 50%,
    #      so determine_final_label() falls through to "invalid".
    invalid_majority_mask = combined_ratings["excluded_invalid_majority"]
    n_invalid_majority = int(invalid_majority_mask.sum())
    n_no_majority = n_invalid_total - n_invalid_majority

    n_after_excl = n_human + n_ai

    def _pct(num: int, denom: int) -> str:
        return f"{100 * num / denom:.1f}%" if denom else "n/a"

    print("\nLabel breakdown (majority vote, " f"n={n_total:,} entries):")
    print(f"  Human                : {n_human:>5,}  ({_pct(n_human, n_total)})")
    print(f"  AI                   : {n_ai:>5,}  ({_pct(n_ai, n_total)})")
    print(f"  Invalid (total)      : {n_invalid_total:>5,}  ({_pct(n_invalid_total, n_total)})")
    print(f"    └─ invalid-majority: {n_invalid_majority:>5,}  "
          f"({_pct(n_invalid_majority, n_total)} of all; "
          f"{_pct(n_invalid_majority, n_invalid_total)} of invalid)")
    print(f"    └─ no clear majority: {n_no_majority:>4,}  "
          f"({_pct(n_no_majority, n_total)} of all; "
          f"{_pct(n_no_majority, n_invalid_total)} of invalid)")

    print(f"\nAfter excluding invalid entries: "
          f"{n_after_excl:,} entries retained for analysis")
    print(f"  Human : {n_human:>5,}  ({_pct(n_human, n_after_excl)} of retained)")
    print(f"  AI    : {n_ai:>5,}  ({_pct(n_ai, n_after_excl)} of retained)")

    # ── Coverage: how many raters voted per entry, and how many entries each
    # rater contributed.
    rater_count_dist = combined_ratings["num_raters"].value_counts().sort_index()
    print("\nRater coverage (raters per entry):")
    for k, v in rater_count_dist.items():
        print(f"  {int(k)} rater(s) : {int(v):>5,} entries  ({_pct(int(v), n_total)})")

    rating_cols = sorted(c for c in combined_ratings.columns if c.startswith("rating_"))
    if rating_cols:
        print(f"\nPer-rater contributions ({len(rating_cols)} raters):")
        for col in rating_cols:
            series = combined_ratings[col]
            n_rated = int(series.notna().sum())
            n_h = int((series == "human").sum())
            n_a = int((series == "ai").sum())
            n_i = int((series == "invalid").sum())
            n_other = n_rated - n_h - n_a - n_i
            extra = f", other={n_other}" if n_other else ""
            print(f"  {col[len('rating_'):]:>4s}: {n_rated:>5,} ratings  "
                  f"(human={n_h}, ai={n_a}, invalid={n_i}{extra})")


if __name__ == "__main__":
    main()
