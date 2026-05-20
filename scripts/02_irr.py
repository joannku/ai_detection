#!/usr/bin/env python3
"""Inter-rater reliability for the 21-rater journal-labelling task.

Two complementary metrics are computed from `ratings/all_ratings.csv`:

1. **Krippendorff's alpha (nominal, 3-way).**
   How much do raters agree on the *full* label set human / AI / invalid?
   Alpha corrects for chance and naturally handles missing ratings, so we
   pass it the entire 21 raters x 2,808 items matrix and let it ignore
   cells where a rater did not see an item.

2. **Cohen's kappa (binary, pairwise).**
   For every pair of raters, how much do they agree on the *human vs AI*
   call alone? Invalid labels are dropped first because we only care about
   the substantive disagreement, and the same-pair sample sizes vary, so
   we report both an unweighted mean and an overlap-weighted mean across
   the 77 pairs that actually share items.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
from krippendorff import alpha as kripp_alpha
from sklearn.metrics import cohen_kappa_score

pd.set_option("future.no_silent_downcasting", True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
ALL_RATINGS_PATH = PROJECT_DIR / "ratings" / "all_ratings.csv"

# Numeric encoding used by Krippendorff (nominal, so the integers are
# arbitrary; only equality matters).
LABEL_TO_INT = {"human": 0, "ai": 1, "invalid": 2}
# Binary encoding used by Cohen kappa (invalid is dropped before mapping).
LABEL_TO_BINARY = {"human": 0, "ai": 1}


# ── Loading ─────────────────────────────────────────────────────────────────

def _load_ratings() -> tuple[pd.DataFrame, list[str]]:
    """Read the merged ratings file and return (df, list of rating columns)."""
    df = pd.read_csv(ALL_RATINGS_PATH)
    rating_cols = [c for c in df.columns if c.startswith("rating_")]
    return df, rating_cols


# ── Krippendorff's alpha (3-way: human / AI / invalid) ─────────────────────

def krippendorff_alpha(df: pd.DataFrame, rating_cols: list[str]
                       ) -> tuple[float | None, int]:
    """Return (alpha, n_items_with_>=2_ratings).

    The `krippendorff` package expects a 2D array of shape (raters, items)
    with NaN for cells where a given rater did not label a given item.
    Items with fewer than 2 non-NaN ratings contribute nothing to alpha.
    """
    numeric = (df[rating_cols]
               .replace(LABEL_TO_INT)
               .apply(pd.to_numeric, errors="coerce"))
    n_eligible = int((numeric.notna().sum(axis=1) >= 2).sum())
    if n_eligible == 0:
        return None, 0

    matrix = numeric.T.to_numpy(dtype=float)  # raters × items
    alpha = kripp_alpha(reliability_data=matrix, level_of_measurement="nominal")
    return (None if alpha is None else float(alpha)), n_eligible


# ── Cohen's kappa (binary human/AI, pairwise) ──────────────────────────────

def _drop_invalid(s: pd.Series) -> pd.Series:
    """Replace 'invalid' with NaN so that pair-wise overlap excludes it."""
    return s.where(~s.isin(["invalid"]))


def cohen_kappa_pair(rater_a: pd.Series, rater_b: pd.Series
                     ) -> tuple[float | None, int, float]:
    """Cohen kappa + raw % agreement on items both raters labelled human/AI.

    Returns (kappa, overlap_n, percent_agreement). If the pair shares zero
    items after dropping invalid / NaN cells, returns (None, 0, NaN).
    """
    overlap = (rater_a.notna() & rater_b.notna()
               & (rater_a != "invalid") & (rater_b != "invalid"))
    if not overlap.any():
        return None, 0, math.nan

    a = rater_a[overlap].map(LABEL_TO_BINARY).astype(int)
    b = rater_b[overlap].map(LABEL_TO_BINARY).astype(int)
    return float(cohen_kappa_score(a, b)), len(a), float((a.values == b.values).mean())


def pairwise_kappa_table(df: pd.DataFrame, rating_cols: list[str]
                         ) -> pd.DataFrame:
    """One row per (rater_a, rater_b) pair with overlap, kappa, agreement.

    Invalid labels are dropped first so kappa is computed on human/AI only.
    Pairs that share no items at all are omitted. Kappa and percent
    agreement are kept at full precision; round only when displaying.
    """
    df = df.copy()
    for col in rating_cols:
        df[col] = _drop_invalid(df[col])

    rows = []
    for a, b in combinations(rating_cols, 2):
        kappa, overlap, pct = cohen_kappa_pair(df[a], df[b])
        if overlap == 0:
            continue
        rows.append({
            "rater_a": a.removeprefix("rating_"),
            "rater_b": b.removeprefix("rating_"),
            "overlap": overlap,
            "percent_agreement": pct,
            "cohen_kappa": kappa,
        })
    return pd.DataFrame(rows)


def _aggregate_kappa(table: pd.DataFrame) -> tuple[float | None, float | None]:
    """Mean kappa across pairs, plus an overlap-weighted mean.

    Weighting by overlap stops a pair with 5 shared items from counting
    the same as a pair with 800 shared items.
    """
    if table.empty:
        return None, None
    mean = table["cohen_kappa"].mean()
    weighted = (table["cohen_kappa"] * table["overlap"]).sum() / table["overlap"].sum()
    return float(mean), float(weighted)


# ── Public API consumed by 99_paper_text.py ─────────────────────────────────

def get_irr_stats() -> dict:
    """Bundle the headline numbers used in the Inter-rater reliability paragraph.

    Returns a dict with: alpha, mean_kappa, weighted_kappa, num_pairs,
    total_entries, entries_2plus, entries_single.
    """
    df, rating_cols = _load_ratings()
    alpha, n_alpha = krippendorff_alpha(df, rating_cols)
    pairs = pairwise_kappa_table(df, rating_cols)
    mean_kappa, weighted_kappa = _aggregate_kappa(pairs)

    if "num_raters" in df.columns:
        per_item = df["num_raters"].astype(int)
    else:
        per_item = df[rating_cols].notna().sum(axis=1)

    return {
        "alpha": alpha,
        "mean_kappa": mean_kappa,
        "weighted_kappa": weighted_kappa,
        "num_pairs": len(pairs),
        "total_entries": len(per_item),
        "entries_2plus": int((per_item >= 2).sum()),
        "entries_single": int((per_item == 1).sum()),
    }


# ── CLI: human-readable diagnostic report ──────────────────────────────────

def _fmt(value: float | None, ndigits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{ndigits}f}"


def main() -> None:
    if not ALL_RATINGS_PATH.exists():
        print(f"Not found: {ALL_RATINGS_PATH}")
        return

    mtime = datetime.fromtimestamp(os.stat(ALL_RATINGS_PATH).st_mtime)
    print(f"Reading: {ALL_RATINGS_PATH}")
    print(f"Last modified: {mtime}")

    df, rating_cols = _load_ratings()
    if not rating_cols:
        print("No rating_* columns found.")
        return

    alpha, n_alpha = krippendorff_alpha(df, rating_cols)
    pairs = pairwise_kappa_table(df, rating_cols)
    mean_kappa, weighted_kappa = _aggregate_kappa(pairs)

    print(f"\nRaters: {len(rating_cols)}")
    print(f"Items with >=2 ratings (any label, used for alpha): {n_alpha:,}")

    if alpha is not None:
        print(f"Krippendorff alpha (nominal, 3-way: human/AI/invalid): "
              f"{alpha:.4f}  (paper rounds to {round(alpha, 3):.3f})")
    else:
        print("Krippendorff alpha: NA")

    if not pairs.empty:
        print(f"Cohen kappa (binary human/AI, invalid dropped): "
              f"mean={mean_kappa:.4f}  weighted={weighted_kappa:.4f}  "
              f"over {len(pairs)} pairs  "
              f"(paper rounds to {round(mean_kappa, 3):.3f} / "
              f"{round(weighted_kappa, 3):.3f})")

        print("\nPairwise (top 10 by overlap):")
        top = (pairs.sort_values("overlap", ascending=False)
                    .head(10)
                    .assign(percent_agreement=lambda d: d["percent_agreement"].round(4),
                            cohen_kappa=lambda d: d["cohen_kappa"].round(4)))
        with pd.option_context("display.width", 120, "display.max_columns", None):
            print(top.to_string(index=False))
    else:
        print("Cohen kappa: NA (no overlapping pairs)")


if __name__ == "__main__":
    main()
