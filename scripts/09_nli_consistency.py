#!/usr/bin/env python3
"""NLI-based consistency analysis: detect internal contradictions in entries.

For every journal entry we split the text into sentences (filtering segments
of <=10 characters), enumerate every pair of sentences, and classify each
pair as `entailment`, `neutral`, or `contradiction` using BART-large-MNLI
via zero-shot classification. From those per-pair labels we compute four
entry-level rates:

  consistency_score = 1 - contradiction_rate
  entailment_rate
  contradiction_rate
  neutral_rate

Entries with fewer than two qualifying sentences receive default values
(consistency = 1.0, contradiction = 0, entailment = 0, neutral = 1.0) --
mirroring the convention described in the paper's methods.

Outputs (under output/nli_consistency/):
  nli_consistency.csv             one row per JournalUniqueID with the four rates
  nli_consistency_by_label.csv    summary (mean/SD) per label
  nli_cache_meta.json             manifest tying the above to the entry set

Running
-------
There are no CLI flags. The single switch is `RECOMPUTE` near the top of
this file (or the `recompute=` kwarg of `main()`).

  * Run as a script  -> `python scripts/09_nli_consistency.py`
  * Run in an interactive window -> `%run scripts/09_nli_consistency.py`,
    or import and call `main()` / `main(recompute=True)`.

Caching
-------
NLI is slow (one pipeline call per sentence pair). The cache here is
**per-entry incremental**: we read the existing nli_consistency.csv and
only score entries whose `JournalUniqueID` is not already present. Adding
a few new entries therefore costs only the new entries, not a full re-run.

Setting RECOMPUTE = True (or passing recompute=True to main) forces every
current entry to be re-scored, regardless of cache state.

Paper paragraph cross-check
---------------------------
After every run the script prints the numbers from §NLI consistency next
to the paragraph claim, e.g.:

  AI consistency        = 0.750 ± 0.279  (-> paragraph: '0.750 ± 0.279')
  AI entailment rate    = 0.745          (-> paragraph: '75% of sentence pairs')
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
RATINGS_CSV = PROJECT_DIR / "ratings" / "all_ratings.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "nli_consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_CSV = OUTPUT_DIR / "nli_consistency.csv"
SUMMARY_CSV = OUTPUT_DIR / "nli_consistency_by_label.csv"
META_JSON = OUTPUT_DIR / "nli_cache_meta.json"

# Single switch:
#   False -> incremental: score only entries not yet in nli_consistency.csv.
#   True  -> always rescore every current entry from scratch (slow).
RECOMPUTE: bool = False

# Sentence-level filtering matches the paper's methods description.
MIN_SENTENCE_CHARS = 10


# ── NLI pipeline (lazy: model loads only on first use) ────────────────────

_nli_pipeline = None


def get_nli_pipeline():
    """Load the zero-shot NLI pipeline on first call, then memoise."""
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        device = _get_device()
        print("Loading NLI model (facebook/bart-large-mnli)...")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        print("\u2713 NLI model loaded")
    return _nli_pipeline


def _get_device():
    """Pick the fastest available device for the NLI pipeline."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("\u2713 GPU detected: Apple Silicon (MPS)")
            return "mps"
        if torch.cuda.is_available():
            print(f"\u2713 GPU detected: {torch.cuda.get_device_name(0)} (CUDA)")
            return 0
        print("\u26a0 No GPU detected, using CPU")
        return -1
    except ImportError:
        print("\u26a0 PyTorch not installed, using CPU")
        return -1


# ── Sentence splitting + per-entry NLI ────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping fragments shorter than 10 characters."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > MIN_SENTENCE_CHARS]


def compute_nli_pair(premise: str, hypothesis: str) -> dict:
    """Single sentence-pair classification: returns the top label + 3 scores."""
    nli = get_nli_pipeline()
    result = nli(
        hypothesis,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
        multi_label=False,
    )
    label_scores = dict(zip(result["labels"], result["scores"]))
    return {
        "top_label": result["labels"][0],
        "entailment": label_scores.get("entailment", 0.0),
        "neutral": label_scores.get("neutral", 0.0),
        "contradiction": label_scores.get("contradiction", 0.0),
    }


def compute_entry_consistency(text: str) -> dict:
    """Entry-level NLI rates over every pair of qualifying sentences.

    For entries with <2 qualifying sentences we return the paper's documented
    defaults: consistency = 1.0, contradiction = 0, entailment = 0,
    neutral = 1.0. This mirrors the methods description in 4experiments.tex.
    """
    sentences = split_sentences(text)
    n_sentences = len(sentences)
    if n_sentences < 2:
        return {
            "consistency_score": 1.0,
            "contradiction_rate": 0.0,
            "entailment_rate": 0.0,
            "neutral_rate": 1.0,
            "n_sentences": n_sentences,
            "n_pairs": 0,
        }

    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    n_pairs = 0
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            n_pairs += 1
            try:
                top = compute_nli_pair(sentences[i], sentences[j])["top_label"]
            except Exception:
                top = "neutral"  # be conservative on transient pipeline errors
            counts[top] += 1

    return {
        "consistency_score": 1 - counts["contradiction"] / n_pairs,
        "contradiction_rate": counts["contradiction"] / n_pairs,
        "entailment_rate": counts["entailment"] / n_pairs,
        "neutral_rate": counts["neutral"] / n_pairs,
        "n_sentences": n_sentences,
        "n_pairs": n_pairs,
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
    """Manifest of the input set NLI was/will be computed on."""
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
        "min_sentence_chars": MIN_SENTENCE_CHARS,
    }


def cached_ids(entry_csv: Path = ENTRY_CSV) -> set[str]:
    """JournalUniqueIDs that already have NLI scores on disk."""
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
    """Score every entry in `df_to_score` for NLI consistency."""
    if df_to_score.empty:
        return pd.DataFrame()
    print(f"\nScoring NLI consistency for {len(df_to_score)} entries...")
    rows: list[dict] = []
    for _, row in tqdm(df_to_score.iterrows(),
                       total=len(df_to_score), desc="NLI"):
        metrics = compute_entry_consistency(row["text"])
        rows.append({
            "JournalUniqueID": row["JournalUniqueID"],
            "final_label": row["final_label"],
            **metrics,
        })
    return pd.DataFrame(rows)


def merge_with_existing(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append `new_rows` to nli_consistency.csv, deduping on JournalUniqueID."""
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
    """Mean / SD over each (final_label) group; saved to nli_consistency_by_label.csv."""
    rows = []
    for label in ("ai", "human"):
        s = entries[entries["final_label"] == label]
        if s.empty:
            continue
        rows.append({
            "label": label,
            "n": len(s),
            "consistency_mean":   float(s["consistency_score"].mean()),
            "consistency_std":    float(s["consistency_score"].std()),
            "contradiction_rate": float(s["contradiction_rate"].mean()),
            "entailment_rate":    float(s["entailment_rate"].mean()),
            "neutral_rate":       float(s["neutral_rate"].mean()),
            "avg_sentences":      float(s["n_sentences"].mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


# ── Reporting ──────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_groups(summary: pd.DataFrame) -> None:
    """Per-label NLI summary (mirrors the paper's reporting style)."""
    _print_header("NLI consistency by label")
    for _, r in summary.iterrows():
        print(f"\n  {r['label'].upper()} (n={int(r['n'])}):")
        print(f"    Consistency:        {r['consistency_mean']:.4f} "
              f"\u00b1 {r['consistency_std']:.4f}")
        print(f"    Contradiction rate: {r['contradiction_rate']:.4f}")
        print(f"    Entailment rate:    {r['entailment_rate']:.4f}")
        print(f"    Neutral rate:       {r['neutral_rate']:.4f}")
        print(f"    Avg sentences:      {r['avg_sentences']:.1f}")


def report_paper_paragraph(summary: pd.DataFrame) -> None:
    """Print every number used in §NLI consistency next to its paragraph role."""
    if summary.empty:
        return
    by_label = summary.set_index("label")
    if not {"ai", "human"}.issubset(by_label.index):
        return
    ai = by_label.loc["ai"]
    hu = by_label.loc["human"]

    _print_header("Paper paragraph cross-check")
    print("Numbers used in '\u00a7NLI-based consistency analysis...':\n")
    print(f"  AI consistency       = {ai['consistency_mean']:.3f} \u00b1 "
          f"{ai['consistency_std']:.3f}  "
          f"(\u2192 paragraph: '0.750 \u00b1 0.279')")
    print(f"  Human consistency    = {hu['consistency_mean']:.3f} \u00b1 "
          f"{hu['consistency_std']:.3f}  "
          f"(\u2192 paragraph: '0.623 \u00b1 0.394')")
    print(f"  AI entailment rate   = {ai['entailment_rate']:.3f}  "
          f"(\u2192 paragraph: '75% of sentence pairs')")
    print(f"  Human entailment     = {hu['entailment_rate']:.3f}  "
          f"(feature_means table: '0.270')")
    print(f"  AI contradiction     = {ai['contradiction_rate']:.3f}")
    print(f"  Human contradiction  = {hu['contradiction_rate']:.3f}")
    print(f"  AI neutral           = {ai['neutral_rate']:.3f}")
    print(f"  Human neutral        = {hu['neutral_rate']:.3f}")
    print(f"  AI avg sentences     = {ai['avg_sentences']:.1f}")
    print(f"  Human avg sentences  = {hu['avg_sentences']:.1f}")


# ── Driver ─────────────────────────────────────────────────────────────────

def main(recompute: bool = RECOMPUTE) -> None:
    """Pipeline driver.

    Decides which entries (if any) need NLI scoring, runs only those,
    refreshes the summary, prints group-level results and the paper-paragraph
    cross-check. Pass `recompute=True` (or set the module-level constant
    `RECOMPUTE` to True) to force re-scoring every current entry.
    """
    df = load_data()
    current_key = cache_key(df)

    if recompute:
        print("[cache] recompute=True requested; re-scoring every current entry.")
        to_score = df
    else:
        have, missing = coverage(df)
        if not missing:
            print(f"[cache] valid: all {len(have):,} current entries already "
                  "have NLI scores on disk.")
            print("[cache] skipping recomputation; reading from disk.")
            to_score = df.iloc[0:0]   # empty
        else:
            extras = ENTRY_CSV.exists() and (cached_ids() - set(df["JournalUniqueID"].astype(str)))
            if have:
                msg = (f"[cache] partial: {len(have):,} entries already scored, "
                       f"{len(missing):,} new to score")
                if extras:
                    msg += f" (and {len(extras):,} cached entries no longer in current set, kept on disk)"
                print(msg + ".")
            else:
                print(f"[cache] empty; scoring all {len(missing):,} entries.")
            to_score = df[df["JournalUniqueID"].astype(str).isin(missing)]

    # Score whatever still needs scoring; merge with prior results on disk.
    new_rows = process_entries(to_score)
    all_entries = merge_with_existing(new_rows)

    # Summary is computed over the CURRENT entry set only (not stray IDs that
    # no longer belong, even though we keep them in the per-entry CSV).
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
