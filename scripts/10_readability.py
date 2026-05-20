#!/usr/bin/env python3
"""Readability metrics (Flesch, Gunning, SMOG, ARI, Coleman-Liau).

For every journal entry we compute seven readability indices using the
`textstat` library (matching the methods description in 4experiments.tex):

  flesch_reading_ease            higher = simpler text       (0-100 scale)
  flesch_kincaid_grade           higher = more complex       (US grade level)
  gunning_fog                    higher = more complex       (US grade level)
  smog_index                     higher = more complex       (US grade level)
  automated_readability_index    higher = more complex       (US grade level)
  coleman_liau_index             higher = more complex       (US grade level)
  syllable_count                 raw count

Outputs (under output/readability/):
  readability.csv             one row per JournalUniqueID with all 7 metrics
  readability_by_label.csv    summary (mean and SD) per label
  readability_cache_meta.json manifest tying the above to the entry set

Running
-------
There are no CLI flags. The single switch is `RECOMPUTE` near the top of
this file (or the `recompute=` kwarg of `main()`).

  * Run as a script  -> `python scripts/10_readability.py`
  * Run in an interactive window -> `%run scripts/10_readability.py`,
    or import and call `main()` / `main(recompute=True)`.

Caching
-------
Per-entry incremental: we read the existing readability.csv and only score
entries whose `JournalUniqueID` is not already present. Adding a few new
entries therefore costs only the new entries, not a full re-run.

Setting RECOMPUTE = True (or passing recompute=True to main) forces every
current entry to be re-scored from scratch.

Paper paragraph cross-check
---------------------------
After every run the script prints the numbers from the paper next to their
paragraph role, e.g.:

  AI Flesch-Kincaid grade  = 11.4 +/- 3.1   (-> paragraph: '11.4 +/- 3.1')
  Human Flesch-Kincaid grade = 8.8 +/- 7.3  (-> paragraph: '8.8 +/- 7.3')
  AI Reading Ease          = 48.2          (-> paragraph: '48.2')
  Human Reading Ease       = 70.1          (-> paragraph: '70.1')
  AI Coleman-Liau          = 11.2          (top LASSO feature, +2.08 coefficient)

`get_readability_stats()` (unchanged signature) is consumed by
`scripts/99_paper_text.py` and is therefore kept as a public API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import textstat
except ImportError:
    raise ImportError("Install textstat: pip install textstat")


# ── Configuration ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "input"
RATINGS_CSV = ROOT_DIR / "ratings" / "all_ratings.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "readability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_CSV = OUTPUT_DIR / "readability.csv"
SUMMARY_CSV = OUTPUT_DIR / "readability_by_label.csv"
META_JSON = OUTPUT_DIR / "readability_cache_meta.json"

# Single switch:
#   False -> incremental: score only entries not yet in readability.csv
#   True  -> always rescore every current entry from scratch
RECOMPUTE: bool = False

METRICS = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "automated_readability_index",
    "coleman_liau_index",
    "syllable_count",
]


# ── Per-entry computation ─────────────────────────────────────────────────

def compute_readability(text: str) -> dict[str, Optional[float]]:
    """Run all seven textstat metrics on one text. Empty input -> all None."""
    if not text or not text.strip():
        return {m: None for m in METRICS}
    return {
        "flesch_reading_ease":         textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade":        textstat.flesch_kincaid_grade(text),
        "gunning_fog":                 textstat.gunning_fog(text),
        "smog_index":                  textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index":          textstat.coleman_liau_index(text),
        "syllable_count":              textstat.syllable_count(text),
    }


# ── Loading ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load journals + final labels, keep only valid AI/Human entries."""
    journals = pd.read_csv(INPUT_DIR / "journals.csv", dtype=str)
    journals = journals.drop_duplicates(subset=["JournalUniqueID"])

    ratings = pd.read_csv(RATINGS_CSV)
    if "excluded_invalid_majority" in ratings.columns:
        excluded = (ratings["excluded_invalid_majority"]
                      .astype(str).str.lower()
                      .isin(["true", "1", "1.0"]))
        ratings = ratings[~excluded].copy()
    ratings = ratings[ratings["final_label"].isin(["ai", "human"])].copy()

    text_col = ("JournalAnonymised" if "JournalAnonymised" in journals.columns
                else "JournalContent")
    cols = ["JournalUniqueID", "ParticipantID", text_col]
    merged = (journals[cols]
                .merge(ratings[["JournalUniqueID", "final_label"]],
                       on="JournalUniqueID")
                .rename(columns={text_col: "text"})
                .dropna(subset=["text"]))
    merged = merged[merged["text"].str.strip() != ""]

    n_ai = int((merged["final_label"] == "ai").sum())
    n_human = int((merged["final_label"] == "human").sum())
    print(f"Loaded {len(merged)} entries (AI: {n_ai}, Human: {n_human})")
    return merged


# ── Cache: tie outputs to the exact entry set ─────────────────────────────

def cache_key(df: pd.DataFrame) -> dict:
    """Manifest of the input set readability was/will be computed on."""
    def _hash(ids: list[str]) -> str:
        return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()

    ai_ids = sorted(df.loc[df["final_label"] == "ai", "JournalUniqueID"]
                       .astype(str).tolist())
    hu_ids = sorted(df.loc[df["final_label"] == "human", "JournalUniqueID"]
                       .astype(str).tolist())
    return {
        "n_ai": len(ai_ids),
        "n_human": len(hu_ids),
        "ai_ids_hash": _hash(ai_ids),
        "human_ids_hash": _hash(hu_ids),
        "metrics": METRICS,
    }


def cached_ids(entry_csv: Path = ENTRY_CSV) -> set[str]:
    """JournalUniqueIDs that already have readability scores on disk."""
    if not entry_csv.exists():
        return set()
    try:
        prior = pd.read_csv(entry_csv, usecols=["JournalUniqueID"])
        return set(prior["JournalUniqueID"].astype(str))
    except Exception:
        return set()


def coverage(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Return (already-scored ID set, IDs that still need scoring)."""
    have = cached_ids()
    needed = set(df["JournalUniqueID"].astype(str))
    return have & needed, needed - have


def write_cache_meta(current_key: dict) -> None:
    META_JSON.write_text(json.dumps(current_key, indent=2))


# ── Incremental processing ────────────────────────────────────────────────

def process_entries(df_to_score: pd.DataFrame) -> pd.DataFrame:
    """Score every entry in `df_to_score` for the seven readability metrics."""
    if df_to_score.empty:
        return pd.DataFrame()
    print(f"\nScoring readability for {len(df_to_score)} entries...")
    rows: list[dict] = []
    for i, (_, row) in enumerate(df_to_score.iterrows(), start=1):
        metrics = compute_readability(row["text"])
        rows.append({
            "JournalUniqueID": row["JournalUniqueID"],
            "final_label": row["final_label"],
            **metrics,
        })
        # textstat is fast (microseconds per call), so a coarse progress is fine.
        if i % 500 == 0:
            print(f"  {i}/{len(df_to_score)} entries scored")
    return pd.DataFrame(rows)


def merge_with_existing(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append `new_rows` to readability.csv, deduping on JournalUniqueID."""
    if ENTRY_CSV.exists() and not new_rows.empty:
        prior = pd.read_csv(ENTRY_CSV)
        combined = (pd.concat([prior, new_rows], ignore_index=True)
                      .drop_duplicates("JournalUniqueID", keep="last"))
    elif ENTRY_CSV.exists():
        combined = pd.read_csv(ENTRY_CSV)
    else:
        combined = new_rows
    combined.to_csv(ENTRY_CSV, index=False)
    return combined


def summarise_by_label(entries: pd.DataFrame) -> pd.DataFrame:
    """Mean / SD per (final_label, metric); saved to readability_by_label.csv."""
    rows = []
    # Drop syllable_count from the displayed paper-style summary (it's in the
    # per-entry CSV but not part of the headline reporting).
    summary_metrics = [m for m in METRICS if m != "syllable_count"]
    for label in ("ai", "human"):
        s = entries[entries["final_label"] == label]
        if s.empty:
            continue
        row = {"label": label, "n": len(s)}
        for m in METRICS:                     # save mean/std for *all* metrics
            row[f"{m}_mean"] = float(s[m].mean())
            row[f"{m}_std"]  = float(s[m].std())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


# ── Reporting ──────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_groups(summary: pd.DataFrame) -> None:
    """Per-label readability summary."""
    _print_header("Readability by label")
    for _, r in summary.iterrows():
        print(f"\n  {r['label'].upper()} (n={int(r['n'])}):")
        print(f"    Flesch Reading Ease   : {r['flesch_reading_ease_mean']:6.2f} "
              f"\u00b1 {r['flesch_reading_ease_std']:.2f}    "
              "(higher = easier)")
        print(f"    Flesch-Kincaid grade  : {r['flesch_kincaid_grade_mean']:6.2f} "
              f"\u00b1 {r['flesch_kincaid_grade_std']:.2f}    "
              "(US grade level)")
        print(f"    Gunning Fog index     : {r['gunning_fog_mean']:6.2f} "
              f"\u00b1 {r['gunning_fog_std']:.2f}")
        print(f"    SMOG index            : {r['smog_index_mean']:6.2f} "
              f"\u00b1 {r['smog_index_std']:.2f}")
        print(f"    Automated Readability : "
              f"{r['automated_readability_index_mean']:6.2f} "
              f"\u00b1 {r['automated_readability_index_std']:.2f}")
        print(f"    Coleman-Liau index    : {r['coleman_liau_index_mean']:6.2f} "
              f"\u00b1 {r['coleman_liau_index_std']:.2f}    "
              "(top LASSO feature)")


def report_paper_paragraph(summary: pd.DataFrame) -> None:
    """Print every readability number that appears in the paper next to its
    paragraph role, so a glance verifies §Readability and the LASSO claim
    about Coleman-Liau being the top positive coefficient.
    """
    if summary.empty:
        return
    by_label = summary.set_index("label")
    if not {"ai", "human"}.issubset(by_label.index):
        return
    ai = by_label.loc["ai"]
    hu = by_label.loc["human"]

    _print_header("Paper paragraph cross-check")
    print("Numbers used in the readability paragraph and "
          "tab:feature_means_semantic:\n")
    print(f"  AI Flesch-Kincaid grade    = "
          f"{ai['flesch_kincaid_grade_mean']:.1f} \u00b1 "
          f"{ai['flesch_kincaid_grade_std']:.1f}  "
          "(\u2192 paragraph: '11.4 \u00b1 3.1')")
    print(f"  Human Flesch-Kincaid grade = "
          f"{hu['flesch_kincaid_grade_mean']:.1f} \u00b1 "
          f"{hu['flesch_kincaid_grade_std']:.1f}  "
          "(\u2192 paragraph: '8.8 \u00b1 7.3')")
    print(f"  AI Reading Ease            = "
          f"{ai['flesch_reading_ease_mean']:.1f}  "
          "(\u2192 paragraph + table: '48.2')")
    print(f"  Human Reading Ease         = "
          f"{hu['flesch_reading_ease_mean']:.1f}  "
          "(\u2192 paragraph + table: '70.1')")
    print(f"  Reading Ease difference    = "
          f"{ai['flesch_reading_ease_mean']-hu['flesch_reading_ease_mean']:.1f}  "
          "(\u2192 table: '\u221221.9')")
    print(f"  AI Coleman-Liau            = "
          f"{ai['coleman_liau_index_mean']:.2f} \u00b1 "
          f"{ai['coleman_liau_index_std']:.2f}  "
          "(top LASSO feature, +2.08 coefficient)")
    print(f"  Human Coleman-Liau         = "
          f"{hu['coleman_liau_index_mean']:.2f} \u00b1 "
          f"{hu['coleman_liau_index_std']:.2f}")


# ── Public API consumed by 99_paper_text.py (do not rename) ───────────────

def get_readability_stats() -> dict:
    """Return the readability stats used by `scripts/99_paper_text.py`.

    Reads the on-disk summary so callers don't have to re-run the pipeline.
    Empty dict if the summary CSV has not been produced yet.
    """
    if not SUMMARY_CSV.exists():
        return {}
    df = pd.read_csv(SUMMARY_CSV)
    if not {"ai", "human"}.issubset(set(df["label"])):
        return {}
    ai = df[df["label"] == "ai"].iloc[0]
    human = df[df["label"] == "human"].iloc[0]
    return {
        "fk_grade_ai":         ai["flesch_kincaid_grade_mean"],
        "fk_grade_human":      human["flesch_kincaid_grade_mean"],
        "fk_grade_ai_std":     ai["flesch_kincaid_grade_std"],
        "fk_grade_human_std":  human["flesch_kincaid_grade_std"],
        "fre_ai":              ai["flesch_reading_ease_mean"],
        "fre_human":           human["flesch_reading_ease_mean"],
        "fre_ai_std":          ai["flesch_reading_ease_std"],
        "fre_human_std":       human["flesch_reading_ease_std"],
        "gunning_ai":          ai["gunning_fog_mean"],
        "gunning_human":       human["gunning_fog_mean"],
    }


# ── Driver ─────────────────────────────────────────────────────────────────

def main(recompute: bool = RECOMPUTE) -> None:
    """Pipeline driver.

    Decides which entries (if any) need readability scoring, runs only
    those, refreshes the per-label summary, prints group-level results
    and the paper-paragraph cross-check. Pass `recompute=True` (or set
    the module-level constant `RECOMPUTE` to True) to force re-scoring
    every current entry.
    """
    df = load_data()
    current_key = cache_key(df)

    if recompute:
        print("[cache] recompute=True requested; re-scoring every current entry.")
        to_score = df
    else:
        have, missing = coverage(df)
        extras = (cached_ids() - set(df["JournalUniqueID"].astype(str))
                  if ENTRY_CSV.exists() else set())
        if not missing:
            print(f"[cache] valid: all {len(have):,} current entries already "
                  "have readability scores on disk.")
            print("[cache] skipping recomputation; reading from disk.")
            to_score = df.iloc[0:0]   # empty
        else:
            if have:
                msg = (f"[cache] partial: {len(have):,} entries already scored, "
                       f"{len(missing):,} new to score")
                if extras:
                    msg += (f" (and {len(extras):,} cached entries no longer in "
                            "current set, kept on disk)")
                print(msg + ".")
            else:
                print(f"[cache] empty; scoring all {len(missing):,} entries.")
            to_score = df[df["JournalUniqueID"].astype(str).isin(missing)]

    new_rows = process_entries(to_score)
    all_entries = merge_with_existing(new_rows)

    # Summary is computed over the CURRENT entry set only (not stray IDs).
    current_ids = set(df["JournalUniqueID"].astype(str))
    current_entries = all_entries[
        all_entries["JournalUniqueID"].astype(str).isin(current_ids)
    ]
    summary = summarise_by_label(current_entries)
    write_cache_meta(current_key)

    report_groups(summary)
    report_paper_paragraph(summary)

    print(f"\n\u2713 Per-entry results: {ENTRY_CSV}")
    print(f"\u2713 Per-label summary : {SUMMARY_CSV}")
    print(f"\u2713 Cache manifest    : {META_JSON}")


if __name__ == "__main__":
    main()
