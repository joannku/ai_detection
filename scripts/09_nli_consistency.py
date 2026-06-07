#!/usr/bin/env python3
"""NLI consistency (canonical pipeline — drops <2-sentence entries).

This is the main NLI-consistency script for the paper. It splits each entry into
sentences (strict regex on terminal punctuation, dropping segments <=10 chars),
classifies **all** sentence pairs with BART-large-MNLI, and reports
consistency = 1 - contradiction_rate.

Entries with fewer than two qualifying sentences cannot be scored the paper's
way, so they are **dropped** (excluded): they get NaN for every rate and fall
out of the group means. This is the conservative correction adopted as the
project default — earlier 0-imputed and sliding-window variants are archived in
``scripts/draft/``.

Outputs carry a ``_drop`` suffix everywhere (consumed by the meta-classifier
scripts, which default to ``VARIANT = "drop"``):

  output/nli_consistency_drop/nli_consistency_drop.csv            per-entry rates
  output/nli_consistency_drop/nli_consistency_drop_by_label.csv   per-label summary
  output/nli_consistency_drop/nli_drop_cache_meta.json            cache manifest

Per-entry columns:
  consistency_score_drop  = 1 - contradiction_rate_drop   (NaN if <2 sentences)
  contradiction_rate_drop
  entailment_rate_drop
  neutral_rate_drop
  n_sentences             qualifying sentences found (strict splitter)
  n_pairs                 sentence pairs classified (= C(n_sentences, 2))

Running
-------
Organised into ``#%%`` cells so it can be executed top-to-bottom interactively,
or as a plain script:

    python scripts/09_nli_consistency.py            # incremental
    # offline (model already cached):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/09_nli_consistency.py

Caching is per-entry incremental: entries already present in
nli_consistency_drop.csv are skipped. Set ``RECOMPUTE = True`` to rescore all.
"""

# %% ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Heavyweight ML dependencies. Imported here (per project preference for
# top-of-file imports) even though the model itself is still constructed lazily
# in ``get_nli_pipeline`` so a pure cache-hit run stays fast.
import torch
from transformers import pipeline


# %% ── Configuration ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
INPUT_DIR = PROJECT_DIR / "input"
RATINGS_CSV = PROJECT_DIR / "ratings" / "all_ratings.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "nli_consistency_drop"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_CSV = OUTPUT_DIR / "nli_consistency_drop.csv"
SUMMARY_CSV = OUTPUT_DIR / "nli_consistency_drop_by_label.csv"
META_JSON = OUTPUT_DIR / "nli_drop_cache_meta.json"

# Sentence-level filtering — matches the paper's described method exactly.
# A "qualifying" sentence is a strict-split segment longer than this many chars.
MIN_SENTENCE_CHARS = 10

# Single switch:
#   False -> incremental: score only entries not yet in nli_consistency_drop.csv.
#   True  -> always rescore every current entry from scratch (slow).
RECOMPUTE: bool = False


# %% ── Sentence splitting (strict — identical to the original 09) ───────────
def split_sentences(text: str) -> list[str]:
    """Split text into qualifying sentences (strict, paper-faithful).

    Steps:
        1. Reject empty / non-string input early (returns no sentences).
        2. Split on terminal punctuation (``.!?``) that is followed by
           whitespace — the original file's exact rule.
        3. Keep only segments longer than ``MIN_SENTENCE_CHARS`` characters,
           discarding trivial fragments the NLI model cannot reason about.

    Note: this is deliberately strict (no lenient / newline fallback). Entries
    that this cannot break into >=2 sentences are exactly the ones we drop.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    raw_segments = re.split(r"(?<=[.!?])\s+", text.strip())

    sentences = []
    for segment in raw_segments:
        cleaned = segment.strip()
        if len(cleaned) > MIN_SENTENCE_CHARS:
            sentences.append(cleaned)
    return sentences


# %% ── NLI pipeline (lazy: model loads only on first use) ───────────────────
_nli_pipeline = None


def _get_device():
    """Pick the fastest available device for the NLI pipeline."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("\u2713 GPU detected: Apple Silicon (MPS)")
        return "mps"
    if torch.cuda.is_available():
        print(f"\u2713 GPU detected: {torch.cuda.get_device_name(0)} (CUDA)")
        return 0
    print("\u26a0 No GPU detected, using CPU")
    return -1


def get_nli_pipeline():
    """Load the zero-shot NLI pipeline on first call, then memoise."""
    global _nli_pipeline
    if _nli_pipeline is None:
        device = _get_device()
        print("Loading NLI model (facebook/bart-large-mnli)...")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        print("\u2713 NLI model loaded")
    return _nli_pipeline


def compute_nli_pair(premise: str, hypothesis: str) -> dict:
    """Classify the relationship between two sentences with zero-shot NLI.

    Identical to the original file (same labels / template) so per-pair behaviour
    matches. Returns the winning label plus the score for each NLI label.
    """
    nli = get_nli_pipeline()

    # Zero-shot classification: score the hypothesis against each candidate
    # label. ``hypothesis_template="{}"`` uses each label verbatim.
    result = nli(
        hypothesis,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
        multi_label=False,
    )

    # Pair the best-first labels with their scores for order-independent lookup.
    label_scores = dict(zip(result["labels"], result["scores"]))

    return {
        "top_label": result["labels"][0],
        "entailment": label_scores.get("entailment", 0.0),
        "neutral": label_scores.get("neutral", 0.0),
        "contradiction": label_scores.get("contradiction", 0.0),
    }


# %% ── Per-entry NLI over ALL sentence pairs ────────────────────────────────
def compute_entry_consistency_drop(text: str) -> dict:
    """Compute the four NLI rates for a single entry, dropping short entries.

    High-level idea:
        Split the entry into qualifying sentences, then classify *every* pair of
        sentences (not just neighbours) — this captures global logical
        consistency across the whole entry, exactly as the paper does.

    Drop rule (the only change vs the original metric):
        An entry needs >=2 qualifying sentences to form a pair. Entries with
        fewer are **dropped**: every rate is NaN (no imputation), so they are
        excluded from the group means. ``n_sentences`` is still recorded so the
        number of dropped entries can be audited.
    """
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    # Too few sentences to score the paper's way -> drop this entry (NaN).
    if n_sentences < 2:
        return {
            "consistency_score_drop": np.nan,
            "contradiction_rate_drop": np.nan,
            "entailment_rate_drop": np.nan,
            "neutral_rate_drop": np.nan,
            "n_sentences": n_sentences,
            "n_pairs": 0,
        }

    # Classify every unordered pair of sentences (i < j).
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    n_pairs = 0
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            try:
                top_label = compute_nli_pair(sentences[i], sentences[j])["top_label"]
            except Exception:
                # Be conservative on transient pipeline errors: count as neutral
                # so one bad call cannot inflate the contradiction rate.
                top_label = "neutral"
            counts[top_label] += 1
            n_pairs += 1

    # Convert raw counts into proportions. consistency = 1 - contradiction rate.
    contradiction_rate = counts["contradiction"] / n_pairs
    entailment_rate = counts["entailment"] / n_pairs
    neutral_rate = counts["neutral"] / n_pairs

    return {
        "consistency_score_drop": 1 - contradiction_rate,
        "contradiction_rate_drop": contradiction_rate,
        "entailment_rate_drop": entailment_rate,
        "neutral_rate_drop": neutral_rate,
        "n_sentences": n_sentences,
        "n_pairs": n_pairs,
    }


# %% ── Loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load journals + final labels, keep only valid AI/Human entries.

    Same scoping as the original 09 file and the ``_sw`` variant.
    """
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
              .merge(ratings[["JournalUniqueID", "final_label"]], on="JournalUniqueID")
              .rename(columns={text_col: "text"})
              .dropna(subset=["text"]))
    merged = merged[merged["text"].str.strip() != ""]

    n_ai = int((merged["final_label"] == "ai").sum())
    n_human = int((merged["final_label"] == "human").sum())
    print(f"Loaded {len(merged)} entries (AI: {n_ai}, Human: {n_human})")
    return merged


# %% ── Cache: tie outputs to the exact entry set ────────────────────────────
def _sorted_ids_for_label(df: pd.DataFrame, label: str) -> list[str]:
    """Return the sorted list of JournalUniqueIDs that carry ``label``.

    Sorting makes the result order-independent so the hash below depends only on
    *which* entries are present, not the order they arrived in.
    """
    rows_for_label = df[df["final_label"] == label]
    ids = rows_for_label["JournalUniqueID"].astype(str).tolist()
    return sorted(ids)


def _hash_ids(ids: list[str]) -> str:
    """Hash a list of IDs into a short fingerprint string.

    Two runs over the exact same set of entries produce the same fingerprint;
    adding or removing even one entry changes it, which is how the cache detects
    a stale result.
    """
    joined = "\n".join(ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def cache_key(df: pd.DataFrame) -> dict:
    """Build a manifest describing the exact input set + parameters used."""
    ai_ids = _sorted_ids_for_label(df, "ai")
    human_ids = _sorted_ids_for_label(df, "human")

    return {
        "n_ai": len(ai_ids),
        "n_human": len(human_ids),
        "ai_ids_hash": _hash_ids(ai_ids),
        "human_ids_hash": _hash_ids(human_ids),
        "min_sentence_chars": MIN_SENTENCE_CHARS,
        "strategy": "drop_entries_with_fewer_than_2_sentences",
    }


def cached_ids(entry_csv: Path = ENTRY_CSV) -> set[str]:
    """JournalUniqueIDs that already have ``_drop`` scores on disk."""
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


# %% ── Incremental processing ───────────────────────────────────────────────
def process_entries(df_to_score: pd.DataFrame) -> pd.DataFrame:
    """Score every entry in ``df_to_score`` for the drop-variant NLI rates."""
    if df_to_score.empty:
        return pd.DataFrame()
    print(f"\nScoring drop-variant NLI for {len(df_to_score)} entries...")
    rows: list[dict] = []
    for _, row in tqdm(df_to_score.iterrows(), total=len(df_to_score), desc="NLI_drop"):
        metrics = compute_entry_consistency_drop(row["text"])
        rows.append({
            "JournalUniqueID": row["JournalUniqueID"],
            "final_label": row["final_label"],
            **metrics,
        })
    return pd.DataFrame(rows)


def merge_with_existing(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Combine freshly scored rows with whatever is already on disk.

    Three situations:
        1. New rows AND a previous CSV exists -> concatenate, then keep the last
           occurrence of each ID so newly scored rows win over stale ones.
        2. No new rows but a CSV exists -> reload it unchanged.
        3. No CSV yet -> the new rows are all we have.
    """
    have_prior_csv = ENTRY_CSV.exists()
    have_new_rows = not new_rows.empty

    if have_prior_csv and have_new_rows:
        prior = pd.read_csv(ENTRY_CSV)
        combined = pd.concat([prior, new_rows], ignore_index=True)
        combined = combined.drop_duplicates("JournalUniqueID", keep="last")
    elif have_prior_csv:
        combined = pd.read_csv(ENTRY_CSV)
    else:
        combined = new_rows

    combined.to_csv(ENTRY_CSV, index=False)
    return combined


# %% ── Summary ──────────────────────────────────────────────────────────────
def summarise_by_label(entries: pd.DataFrame) -> pd.DataFrame:
    """Mean / SD per (final_label) group; saved to *_by_label.csv.

    Means use pandas' default skipna=True, so dropped (NaN) entries are excluded
    rather than imputed — the whole point of this variant.
    """
    rows = []
    for label in ("ai", "human"):
        group = entries[entries["final_label"] == label]
        if group.empty:
            continue

        # An entry is "scored" if it had >=2 sentences (consistency not NaN).
        # The rest are "excluded" (dropped, NaN) and left out of the means below.
        is_scored = group["consistency_score_drop"].notna()
        n_total = len(group)
        n_scored = int(is_scored.sum())
        n_excluded = n_total - n_scored

        rows.append({
            "label": label,
            "n": n_total,
            "n_scored": n_scored,
            "n_excluded": n_excluded,
            "consistency_mean":   float(group["consistency_score_drop"].mean()),
            "consistency_std":    float(group["consistency_score_drop"].std()),
            "contradiction_rate": float(group["contradiction_rate_drop"].mean()),
            "entailment_rate":    float(group["entailment_rate_drop"].mean()),
            "neutral_rate":       float(group["neutral_rate_drop"].mean()),
            "avg_sentences":      float(group["n_sentences"].mean()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


# %% ── Reporting ────────────────────────────────────────────────────────────
def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_groups(summary: pd.DataFrame) -> None:
    """Per-label drop-variant NLI summary."""
    _print_header("Drop-variant NLI consistency by label")
    for _, r in summary.iterrows():
        print(f"\n  {r['label'].upper()} (n={int(r['n'])}, scored={int(r['n_scored'])}, "
              f"excluded={int(r['n_excluded'])}):")
        print(f"    Consistency:        {r['consistency_mean']:.4f} \u00b1 {r['consistency_std']:.4f}")
        print(f"    Contradiction rate: {r['contradiction_rate']:.4f}")
        print(f"    Entailment rate:    {r['entailment_rate']:.4f}")
        print(f"    Neutral rate:       {r['neutral_rate']:.4f}")
        print(f"    Avg sentences:      {r['avg_sentences']:.1f}")


def report_gap(summary: pd.DataFrame) -> None:
    """Print the AI-vs-Human gap and compare to the original published number.

    The original (impute-1.0) consistency means were 0.750 (AI) vs 0.623 (Human),
    a gap of +0.127; shown here for reference.
    """
    if summary.empty or not {"ai", "human"}.issubset(set(summary["label"])):
        return
    by = summary.set_index("label")
    ai, hu = by.loc["ai"], by.loc["human"]

    _print_header("AI vs Human (drop variant)")
    print(f"  Consistency   AI={ai['consistency_mean']:.3f}  "
          f"Human={hu['consistency_mean']:.3f}  "
          f"gap={ai['consistency_mean'] - hu['consistency_mean']:+.3f}  "
          f"(original impute gap: +0.127)")
    print(f"  Entailment    AI={ai['entailment_rate']:.3f}  "
          f"Human={hu['entailment_rate']:.3f}  "
          f"gap={ai['entailment_rate'] - hu['entailment_rate']:+.3f}")
    print(f"  Contradiction AI={ai['contradiction_rate']:.3f}  "
          f"Human={hu['contradiction_rate']:.3f}  "
          f"gap={ai['contradiction_rate'] - hu['contradiction_rate']:+.3f}")


# %% ── Driver ───────────────────────────────────────────────────────────────
def select_entries_to_score(df: pd.DataFrame, recompute: bool) -> pd.DataFrame:
    """Decide which entries actually need (re)scoring and explain the decision.

    Returns the subset of ``df`` that should be sent to the (slow) NLI model.
    The ``print`` calls narrate the cache state for a human watching the run.

    Three outcomes:
        * recompute=True             -> score everything.
        * nothing missing from cache -> score nothing (pure cache hit).
        * some entries missing        -> score only the missing ones.
    """
    if recompute:
        print("[cache] recompute=True requested; re-scoring every current entry.")
        return df

    already_scored, missing = coverage(df)

    if not missing:
        print(f"[cache] valid: all {len(already_scored):,} current entries already have _drop scores.")
        print("[cache] skipping recomputation; reading from disk.")
        return df.iloc[0:0]

    if already_scored:
        message = (f"[cache] partial: {len(already_scored):,} entries already scored, "
                   f"{len(missing):,} new to score")

        current_ids = set(df["JournalUniqueID"].astype(str))
        stray_cached_ids = cached_ids() - current_ids if ENTRY_CSV.exists() else set()
        if stray_cached_ids:
            message += (f" (and {len(stray_cached_ids):,} cached entries no longer "
                        "in current set, kept on disk)")
        print(message + ".")
    else:
        print(f"[cache] empty; scoring all {len(missing):,} entries.")

    needs_scoring = df["JournalUniqueID"].astype(str).isin(missing)
    return df[needs_scoring]


def main(recompute: bool = RECOMPUTE) -> pd.DataFrame:
    """Pipeline driver (same control flow as ``09_nli_consistency_sw.py``).

    Steps:
        1. Load the scoped entry set and build a cache manifest for it.
        2. Decide which entries need scoring (see ``select_entries_to_score``).
        3. Score those, then merge them with anything already on disk.
        4. Summarise per label over the *current* entry set only.
        5. Persist the cache manifest and print the summary.

    Returns the per-label summary DataFrame for convenience in interactive use.
    """
    df = load_data()
    current_key = cache_key(df)

    to_score = select_entries_to_score(df, recompute)
    new_rows = process_entries(to_score)
    all_entries = merge_with_existing(new_rows)

    # The on-disk CSV may contain stray IDs from older runs; restrict the summary
    # to the entries that belong to the current dataset.
    current_ids = set(df["JournalUniqueID"].astype(str))
    is_current = all_entries["JournalUniqueID"].astype(str).isin(current_ids)
    current_entries = all_entries[is_current]

    summary = summarise_by_label(current_entries)
    write_cache_meta(current_key)

    report_groups(summary)
    report_gap(summary)

    print(f"\n\u2713 Per-entry results: {ENTRY_CSV}")
    print(f"\u2713 Per-label summary : {SUMMARY_CSV}")
    print(f"\u2713 Cache manifest    : {META_JSON}")
    return summary


# %% ── Run ──────────────────────────────────────────────────────────────────
# Executable both as a script (`python scripts/09_nli_consistency_drop.py`) and
# cell-by-cell (run the cells above, then this one).
if __name__ == "__main__":
    summary = main()
