#!/usr/bin/env python3
"""
Build a high-confidence subset of journal entries for the meta-classifier.

Selection criteria (configurable):
  1. Rater agreement: ≥ RATER_AGREE_THRESH of non-invalid raters agree on the label.
  2. GPTZero alignment: GPTZero's predicted probability is confident in the
     same direction as the rater majority label
     (ai_prob > GPTZERO_AI_THRESH for AI, ai_prob < GPTZERO_HUMAN_THRESH for human).

Outputs:
  - output/high_confidence_subset/high_confidence_subset.csv  (the subset with metadata)
  - output/high_confidence_subset/high_confidence_summary.txt (human-readable report)
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_OUTPUT = ROOT / "output" / "high_confidence_subset"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Configurable thresholds ─────────────────────────────────────────────────
RATER_AGREE_THRESH = 0.80   # fraction of non-invalid raters that must agree
GPTZERO_AI_THRESH = 0.80    # GPTZero ai_prob must exceed this for AI entries
GPTZERO_HUMAN_THRESH = 0.20 # GPTZero ai_prob must be below this for human entries


def load_data():
    ratings = pd.read_csv(ROOT / "ratings/all_ratings.csv")
    gptzero = pd.read_csv(ROOT / "output" / "gptzero" / "gptzero_results.csv")
    journals = pd.read_csv(ROOT / "output" / "export" / "final_journals.csv")
    return ratings, gptzero, journals


def compute_agreement(ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep valid (non-excluded, ai/human) entries and compute agreement %."""
    df = ratings[
        ~ratings["excluded_invalid_majority"]
        & ratings["final_label"].isin(["ai", "human"])
    ].copy()

    df["valid_votes"] = df["num_ai"] + df["num_human"]
    df["majority_votes"] = df[["num_ai", "num_human"]].max(axis=1)
    df["agree_pct"] = df["majority_votes"] / df["valid_votes"]

    return df


def apply_gptzero_filter(df: pd.DataFrame, gptzero: pd.DataFrame) -> pd.DataFrame:
    """Merge GPTZero results and flag entries where GPTZero is confident AND agrees."""
    df = df.merge(
        gptzero[["JournalUniqueID", "gptzero_ai_prob", "gptzero_class"]],
        on="JournalUniqueID",
        how="inner",
    )

    df["gptzero_confident"] = (
        ((df["final_label"] == "ai") & (df["gptzero_ai_prob"] > GPTZERO_AI_THRESH))
        | ((df["final_label"] == "human") & (df["gptzero_ai_prob"] < GPTZERO_HUMAN_THRESH))
    )

    return df


def build_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply both thresholds to produce the high-confidence subset."""
    return df[
        (df["agree_pct"] >= RATER_AGREE_THRESH) & df["gptzero_confident"]
    ].copy()


def _valid_votes_block(title: str, df: pd.DataFrame) -> list[str]:
    """Coverage stats on `valid_votes` (= num_ai + num_human, i.e. ratings
    after invalid ratings are excluded).

    With 1 or 2 valid ratings the 80% agreement threshold is trivially
    satisfied (the only option is 100% agreement), so we explicitly flag
    those rows."""
    vv = df["valid_votes"]
    n = len(df)
    n_trivial = int((vv <= 2).sum())
    lines = [
        "",
        title,
        f"  N entries        : {n:,}",
        f"  Mean (SD)        : {vv.mean():.2f} ({vv.std():.2f})",
        f"  Median [IQR]     : {vv.median():.0f} "
        f"[{vv.quantile(0.25):.0f}–{vv.quantile(0.75):.0f}]",
        f"  Min / Max        : {int(vv.min())} / {int(vv.max())}",
        "  Distribution:",
    ]
    for k, count in vv.value_counts().sort_index().items():
        marker = "  (80% threshold trivial)" if int(k) <= 2 else ""
        lines.append(
            f"    {int(k)} valid rater(s): "
            f"{int(count):>5,} entries ({count/n:.1%}){marker}"
        )
    lines.append(
        f"  Entries with <= 2 valid ratings: "
        f"{n_trivial:,} ({n_trivial/n:.1%})"
    )
    return lines


def write_summary(full_df, subset_df, out_path):
    n_ai = (subset_df["final_label"] == "ai").sum()
    n_human = (subset_df["final_label"] == "human").sum()

    lines = [
        "High-Confidence Subset Summary",
        "=" * 50,
        "",
        "Thresholds",
        f"  Rater agreement  : >= {RATER_AGREE_THRESH:.0%}",
        f"  GPTZero AI prob  : > {GPTZERO_AI_THRESH} (for AI entries)",
        f"  GPTZero AI prob  : < {GPTZERO_HUMAN_THRESH} (for human entries)",
        "",
        "Counts",
        f"  Full valid pool  : {len(full_df):,}",
        f"  High-confidence  : {len(subset_df):,}  ({len(subset_df)/len(full_df):.1%} of pool)",
        f"    AI entries     : {n_ai:,}",
        f"    Human entries  : {n_human:,}",
        f"    Class ratio    : {n_human/n_ai:.1f}:1 (human:ai)" if n_ai > 0 else "",
        "",
        "Agreement distribution in subset",
    ]

    for pct in [0.80, 0.85, 0.90, 0.95, 1.00]:
        n = (subset_df["agree_pct"] >= pct).sum()
        lines.append(f"  >= {pct:.0%}: {n:,}")

    lines += _valid_votes_block(
        "Valid-rater coverage (full valid pool)", full_df
    )
    lines += _valid_votes_block(
        "Valid-rater coverage (high-confidence subset)", subset_df
    )

    lines += [
        "",
        "GPTZero probability in subset",
        f"  AI entries   — mean ai_prob: {subset_df.loc[subset_df.final_label=='ai', 'gptzero_ai_prob'].mean():.3f}",
        f"  Human entries — mean ai_prob: {subset_df.loc[subset_df.final_label=='human', 'gptzero_ai_prob'].mean():.4f}",
    ]

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)

    ratings, gptzero, journals = load_data()
    valid = compute_agreement(ratings)
    valid = apply_gptzero_filter(valid, gptzero)
    subset = build_subset(valid)

    # Merge with journal text for downstream use
    subset = subset.merge(
        journals[["JournalUniqueID", "text"]],
        on="JournalUniqueID",
        how="inner",
    )

    keep_cols = [
        "JournalUniqueID", "final_label", "text",
        "num_raters", "num_ai", "num_human", "num_invalid",
        "valid_votes", "majority_votes", "agree_pct",
        "gptzero_ai_prob", "gptzero_class",
    ]
    subset = subset[[c for c in keep_cols if c in subset.columns]]

    out_csv = REVIEWS_OUTPUT / "high_confidence_subset.csv"
    subset.to_csv(out_csv, index=False)
    print(f"\n✓ Saved {len(subset):,} entries → {out_csv}\n")

    write_summary(valid, subset, REVIEWS_OUTPUT / "high_confidence_summary.txt")


if __name__ == "__main__":
    main()
