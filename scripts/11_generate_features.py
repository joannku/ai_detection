#!/usr/bin/env python3
"""Meta-features: coherence & style similarity (canonical — drops <2-sentence entries).

This is the main feature-generation script for the paper. It measures two
embedding metrics on *adjacent sentences*:

  coherence         mean cosine of adjacent sentence embeddings (all-mpnet-base-v2)
  style_similarity  mean cosine of adjacent sentence embeddings (StyleDistance)

Both need the sentence splitter to find >=2 sentences. Entries it cannot split
are **dropped** (excluded): they get NaN for both metrics and fall out of the
group means. This is the conservative correction adopted as the project default —
earlier 0-imputed and sliding-window variants are archived in ``scripts/draft/``.

Outputs carry a ``_drop`` suffix everywhere (consumed by the meta-classifier
scripts, which default to ``VARIANT = "drop"``):

  output/generate_features_drop/features_drop.csv             per-entry features
  output/generate_features_drop/features_drop_by_label.csv    per-label summary
  output/generate_features_drop/features_drop_cache_meta.json cache manifest

Per-entry columns:
  word_count             whitespace tokens                        (length count)
  sentence_count         rule-based split on terminal punctuation (length count)
  character_count        raw len(text)                            (length count)
  coherence_drop         mean adjacent-sentence cosine, all-mpnet-base-v2 (NaN if <2)
  style_similarity_drop  mean adjacent-sentence cosine, StyleDistance     (NaN if <2)
  final_label            'ai' / 'human'

Scope note
----------
Like ``11_generate_features_sw.py``, this scopes to valid AI/Human entries only
(not the original 11's "no clear majority" NaN-label rows), so the ``_sw`` and
``_drop`` files cover the same entry set.

Running
-------
Organised into ``#%%`` cells so it can be executed top-to-bottom interactively,
or as a plain script:

    python scripts/11_generate_features.py            # incremental
    # offline (models already cached):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/11_generate_features.py

Caching is per-entry incremental: entries already present in features_drop.csv
are skipped. Set ``RECOMPUTE = True`` to rescore all.
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

# Heavyweight ML dependency. Imported here (per project preference for
# top-of-file imports) even though the two models are still constructed lazily
# in ``get_models`` so a pure cache-hit run stays fast.
from sentence_transformers import SentenceTransformer


# %% ── Configuration ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
INPUT_DIR = PROJECT_DIR / "input"
RATINGS_CSV = PROJECT_DIR / "ratings" / "all_ratings.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "generate_features_drop"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_CSV = OUTPUT_DIR / "features_drop.csv"
SUMMARY_CSV = OUTPUT_DIR / "features_drop_by_label.csv"
META_JSON = OUTPUT_DIR / "features_drop_cache_meta.json"

# Embedding models (same as the original file).
COHERENCE_MODEL_NAME = "all-mpnet-base-v2"            # semantic coherence
STYLE_MODEL_NAME = "StyleDistance/styledistance"     # stylistic consistency

# Single switch:
#   False -> incremental: score only entries not yet in features_drop.csv.
#   True  -> always rescore every current entry from scratch (slow).
RECOMPUTE: bool = False

# The order columns are written to disk in.
OUTPUT_COLUMNS = [
    "JournalUniqueID",
    "word_count", "sentence_count", "character_count",
    "coherence_drop", "style_similarity_drop",
    "final_label",
]


# %% ── Length counts + sentence splitting (reproduced from the original 11) ──
def split_sentences(text: str) -> list[str]:
    """Rule-based splitter on terminal punctuation (matches the original file).

    Splits on ``.!?`` followed by whitespace and keeps every non-empty segment.
    This is the SAME splitter the original 11 uses both for ``sentence_count``
    and for the adjacent-sentence cosine, so "has <2 sentences" (the drop rule)
    and "sentence_count < 2" always agree.
    """
    cleaned = (text or "").strip()
    pieces = re.split(r"(?<=[.!?])\s+", cleaned)
    return [piece for piece in pieces if piece]


def word_count(text: str) -> int:
    """Number of whitespace-separated tokens."""
    return len((text or "").split())


def character_count(text: str) -> int:
    """Raw character length of the text."""
    return len(text or "")


def sentence_count(text: str) -> int:
    """Number of rule-based sentences (descriptive length feature)."""
    return len(split_sentences(text or ""))


# %% ── Embedding models (lazy: loaded only when scoring is needed) ──────────
_models: tuple[SentenceTransformer, SentenceTransformer] | None = None


def get_models() -> tuple[SentenceTransformer, SentenceTransformer]:
    """Load both SentenceTransformer models on first call, then memoise.

    Returns ``(coherence_model, style_model)``. These are heavy to load, so on a
    pure cache-hit run we never call this at all.
    """
    global _models
    if _models is None:
        print(f"Loading coherence model: {COHERENCE_MODEL_NAME}")
        coherence_model = SentenceTransformer(COHERENCE_MODEL_NAME)
        print(f"Loading style model:     {STYLE_MODEL_NAME}")
        style_model = SentenceTransformer(STYLE_MODEL_NAME)
        _models = (coherence_model, style_model)
        print("\u2713 Models loaded")
    return _models


# %% ── Per-entry adjacent-sentence cosine ───────────────────────────────────
def adjacent_sentence_cosine(text: str, model: SentenceTransformer) -> float:
    """Mean cosine similarity between embeddings of adjacent sentences.

    High-level idea:
        Split the entry into sentences, embed each sentence, then average the
        cosine similarity of every neighbouring pair. High values mean
        consecutive sentences sit close together in the model's space (coherent /
        stylistically uniform); low values mean they drift apart.

    Drop rule (the only change vs the original metric):
        At least 2 sentences are needed to have a single neighbouring pair.
        Entries with fewer are **dropped**: we return NaN (no imputation),
        mirroring the NLI ``_drop`` file.

    Note on cosine:
        We ask the model for L2-normalised embeddings, so the cosine similarity
        of two vectors is simply their dot product.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return np.nan

    # Embed all sentences for this entry at once (normalised -> dot == cosine).
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # Average the similarity of each adjacent sentence pair.
    pair_similarities = []
    for i in range(len(embeddings) - 1):
        cosine = float(np.dot(embeddings[i], embeddings[i + 1]))
        pair_similarities.append(cosine)

    return float(np.mean(pair_similarities))


def compute_entry_features_drop(text: str,
                                coherence_model: SentenceTransformer,
                                style_model: SentenceTransformer) -> dict:
    """Compute all per-entry features for one entry.

    Length counts are cheap and always defined; the two adjacent-sentence cosine
    metrics are NaN when the entry has fewer than 2 sentences (dropped).
    """
    return {
        "word_count": word_count(text),
        "sentence_count": sentence_count(text),
        "character_count": character_count(text),
        "coherence_drop": adjacent_sentence_cosine(text, coherence_model),
        "style_similarity_drop": adjacent_sentence_cosine(text, style_model),
    }


# %% ── Loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load journals + final labels, keep only valid AI/Human entries.

    Same scoping as ``11_generate_features_sw.py``: drop entries excluded by the
    invalid-majority flag, keep final_label in {ai, human}, drop empty text.
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
        "coherence_model": COHERENCE_MODEL_NAME,
        "style_model": STYLE_MODEL_NAME,
        "strategy": "drop_entries_with_fewer_than_2_sentences",
    }


def cached_ids(entry_csv: Path = ENTRY_CSV) -> set[str]:
    """JournalUniqueIDs that already have ``_drop`` features on disk."""
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
    """Score every entry in ``df_to_score`` for the drop-variant features.

    Loads the two embedding models lazily (skipped entirely when there is
    nothing to score, i.e. on a pure cache hit).
    """
    if df_to_score.empty:
        return pd.DataFrame()

    coherence_model, style_model = get_models()
    print(f"\nScoring drop-variant features for {len(df_to_score)} entries...")

    rows: list[dict] = []
    for _, row in tqdm(df_to_score.iterrows(), total=len(df_to_score), desc="features_drop"):
        metrics = compute_entry_features_drop(row["text"], coherence_model, style_model)
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

    # Keep a stable column order where possible (extra columns, if any, trail).
    ordered = [c for c in OUTPUT_COLUMNS if c in combined.columns]
    remaining = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + remaining]

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

        # An entry is "scored" if it had >=2 sentences (coherence not NaN).
        # coherence and style share the same NaN pattern, so one count suffices.
        is_scored = group["coherence_drop"].notna()
        n_total = len(group)
        n_scored = int(is_scored.sum())
        n_excluded = n_total - n_scored

        rows.append({
            "label": label,
            "n": n_total,
            "n_scored": n_scored,
            "n_excluded": n_excluded,
            "coherence_mean":        float(group["coherence_drop"].mean()),
            "coherence_std":         float(group["coherence_drop"].std()),
            "style_similarity_mean": float(group["style_similarity_drop"].mean()),
            "style_similarity_std":  float(group["style_similarity_drop"].std()),
            "word_count_mean":       float(group["word_count"].mean()),
            "sentence_count_mean":   float(group["sentence_count"].mean()),
            "character_count_mean":  float(group["character_count"].mean()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


# %% ── Reporting ────────────────────────────────────────────────────────────
def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_groups(summary: pd.DataFrame) -> None:
    """Per-label drop-variant feature summary."""
    _print_header("Drop-variant features by label")
    for _, r in summary.iterrows():
        print(f"\n  {r['label'].upper()} (n={int(r['n'])}, scored={int(r['n_scored'])}, "
              f"excluded={int(r['n_excluded'])}):")
        print(f"    Coherence (drop):        {r['coherence_mean']:.4f} \u00b1 {r['coherence_std']:.4f}")
        print(f"    Style similarity (drop): {r['style_similarity_mean']:.4f} \u00b1 {r['style_similarity_std']:.4f}")
        print(f"    Word count:              {r['word_count_mean']:.1f}")
        print(f"    Sentence count:          {r['sentence_count_mean']:.1f}")
        print(f"    Character count:         {r['character_count_mean']:.1f}")


def report_gap(summary: pd.DataFrame) -> None:
    """Print the AI-vs-Human gap and compare to the original published numbers.

    The original (0-imputed) means were coherence 0.468 (AI) vs 0.237 (Human)
    and style similarity 0.862 (AI) vs 0.559 (Human); shown here for reference.
    """
    if summary.empty or not {"ai", "human"}.issubset(set(summary["label"])):
        return
    by = summary.set_index("label")
    ai, hu = by.loc["ai"], by.loc["human"]

    _print_header("AI vs Human (drop variant)")
    print(f"  Coherence        AI={ai['coherence_mean']:.3f}  "
          f"Human={hu['coherence_mean']:.3f}  "
          f"gap={ai['coherence_mean'] - hu['coherence_mean']:+.3f}  "
          f"(original 0-imputed gap: +0.231)")
    print(f"  Style similarity AI={ai['style_similarity_mean']:.3f}  "
          f"Human={hu['style_similarity_mean']:.3f}  "
          f"gap={ai['style_similarity_mean'] - hu['style_similarity_mean']:+.3f}  "
          f"(original 0-imputed gap: +0.304)")


# %% ── Driver ───────────────────────────────────────────────────────────────
def select_entries_to_score(df: pd.DataFrame, recompute: bool) -> pd.DataFrame:
    """Decide which entries actually need (re)scoring and explain the decision.

    Returns the subset of ``df`` that should be sent to the (slow) embedding
    models. The ``print`` calls narrate the cache state for a human watching.

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
        print(f"[cache] valid: all {len(already_scored):,} current entries already have _drop features.")
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
    """Pipeline driver (same control flow as ``11_generate_features_sw.py``).

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
# Executable both as a script (`python scripts/11_generate_features_drop.py`) and
# cell-by-cell (run the cells above, then this one).
if __name__ == "__main__":
    summary = main()
