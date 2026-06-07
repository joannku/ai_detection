#!/usr/bin/env python3
"""Per-entry meta-features: length counts + semantic coherence + style similarity.

For every journal entry that has a valid (non-`excluded_invalid_majority`)
rating we compute six features that feed the meta-classifiers:

  word_count        whitespace tokens
  sentence_count    rule-based split on terminal punctuation
  character_count   raw len(text)
  coherence         mean cosine of adjacent sentence embeddings
                    using all-mpnet-base-v2  (entries with <2 sentences -> 0.0)
  style_similarity  same, but with StyleDistance/styledistance
                    (entries with <2 sentences -> 0.0)
  label             'AI' / 'Human' / NaN (NaN = "no clear majority")

Outputs (under output/generate_features/):
  features.csv               one row per JournalUniqueID, schema as above
  features_cache_meta.json   manifest tying features.csv to the entry set

Running
-------
There are no CLI flags. The single switch is `RECOMPUTE` near the top of
this file (or the `recompute=` kwarg of `main()`).

  * Run as a script  -> `python scripts/11_generate_features.py`
  * Run in an interactive window -> `%run scripts/11_generate_features.py`,
    or import and call `main()` / `main(recompute=True)`.

Caching
-------
Per-entry incremental: we read the existing features.csv and only score
entries whose `JournalUniqueID` is not already present. The two
SentenceTransformer models are heavyweight, so they are *lazy-loaded* —
only fetched when there are missing entries to score. A pure cache hit is
therefore very fast and never touches HuggingFace at all.

Setting `RECOMPUTE = True` (or passing `recompute=True` to `main`) forces
every current entry to be re-scored from scratch.

Paper paragraph cross-check
---------------------------
After every run the script prints AI vs Human means for the four numbers
that appear in the paper's tab:feature_means_semantic and the prose
around line 139-145 of 5results.tex:

  Word count        87.4 (Human) vs 190.3 (AI)
  Sentence count     4.8 (Human) vs  10.0 (AI)
  Coherence        0.237 (Human) vs 0.468 (AI)
  Style similarity 0.559 (Human) vs 0.862 (AI)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parent.parent
INPUT_DIR   = ROOT_DIR / "input"
OUTPUT_DIR  = ROOT_DIR / "output" / "generate_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOURNALS_CSV = INPUT_DIR / "journals.csv"
RATINGS_CSV  = ROOT_DIR / "ratings" / "all_ratings.csv"
FEATURES_CSV = OUTPUT_DIR / "features.csv"
META_JSON    = OUTPUT_DIR / "features_cache_meta.json"

COHERENCE_MODEL_NAME = "all-mpnet-base-v2"
STYLE_MODEL_NAME     = "StyleDistance/styledistance"

# Single switch:
#   False -> incremental: score only entries not yet in features.csv
#   True  -> always rescore every current entry from scratch
RECOMPUTE: bool = False

OUTPUT_COLUMNS = [
    "JournalUniqueID",
    "word_count", "sentence_count", "character_count",
    "coherence", "style_similarity",
    "label",
]


# ── Per-entry computation ─────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Rule-based splitter on terminal punctuation (matches paper §Length counts)."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s for s in sentences if s]


def word_count(text: str) -> int:
    return len((text or "").split())


def character_count(text: str) -> int:
    return len(text or "")


def sentence_count(text: str) -> int:
    return len(split_sentences(text or ""))


def _adjacent_cosine_mean(text: str, model) -> float:
    """Mean cosine similarity between embeddings of adjacent sentences.

    Returns 0.0 when fewer than 2 sentences (matches paper §Semantic coherence
    and §Stylistic consistency).
    """
    from sentence_transformers import util as st_util
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.0
    emb = model.encode(sentences, normalize_embeddings=True)
    sims = [
        st_util.cos_sim(emb[i], emb[i + 1]).item()
        for i in range(len(emb) - 1)
    ]
    return sum(sims) / len(sims)


def coherence_score(text: str, model) -> float:
    """Mean cosine of adjacent sentences using all-mpnet-base-v2."""
    return _adjacent_cosine_mean(text, model)


def style_similarity_score(text: str, model) -> float:
    """Mean cosine of adjacent sentences using StyleDistance/styledistance."""
    return _adjacent_cosine_mean(text, model)


def load_models() -> tuple:
    """Lazy-load both SentenceTransformer models. Heavy: only call when needed."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading coherence model:  {COHERENCE_MODEL_NAME}")
    coh = SentenceTransformer(COHERENCE_MODEL_NAME)
    print(f"Loading style model:      {STYLE_MODEL_NAME}")
    sty = SentenceTransformer(STYLE_MODEL_NAME)
    return coh, sty


# ── Loading ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Journal text + label, scoped to non-excluded majority-vote ratings.

    Mirrors the original script's behaviour:
      * drop rows where excluded_invalid_majority is True;
      * keep everything else (including 'invalid' / "no clear majority"
        rows whose label maps to NaN);
      * map final_label ai/human -> AI/Human, drop final_label.
    """
    if "final_label" not in pd.read_csv(RATINGS_CSV, nrows=1).columns:
        raise ValueError("ratings/all_ratings.csv must include 'final_label'")

    ratings = pd.read_csv(RATINGS_CSV)
    if "excluded_invalid_majority" in ratings.columns:
        excluded = (ratings["excluded_invalid_majority"]
                      .astype(str).str.lower()
                      .isin(["true", "1", "1.0"]))
        ratings = ratings[~excluded].copy()
    ratings = ratings[["JournalUniqueID", "final_label"]].copy()
    ratings["label"] = ratings["final_label"].map({"ai": "AI", "human": "Human"})
    ratings = ratings.drop(columns=["final_label"])

    df = pd.read_csv(JOURNALS_CSV)
    # journals.csv contains a small number of duplicate JournalUniqueIDs with
    # identical text; downstream consumers already dedupe defensively, and the
    # original script silently inflated row counts. Dedupe at the source.
    df = df.drop_duplicates(subset=["JournalUniqueID"], keep="first")
    df = df[df["JournalUniqueID"].isin(ratings["JournalUniqueID"])].copy()
    df["JournalAnonymised"] = df["JournalAnonymised"].fillna("").astype(str)
    df = df.merge(ratings, on="JournalUniqueID", how="left")

    n_ai    = int((df["label"] == "AI").sum())
    n_human = int((df["label"] == "Human").sum())
    n_nan   = int(df["label"].isna().sum())
    print(f"Loaded {len(df)} entries "
          f"(AI: {n_ai}, Human: {n_human}, no-clear-majority: {n_nan})")
    return df


# ── Cache: tie outputs to the exact entry set ─────────────────────────────

def cache_key(df: pd.DataFrame) -> dict:
    """Manifest of the input set features were/will be computed on."""
    def _hash(ids: list[str]) -> str:
        return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()

    all_ids = sorted(df["JournalUniqueID"].astype(str).tolist())
    by_label = {
        "AI":    sorted(df.loc[df["label"] == "AI",    "JournalUniqueID"].astype(str).tolist()),
        "Human": sorted(df.loc[df["label"] == "Human", "JournalUniqueID"].astype(str).tolist()),
    }
    return {
        "n_total":          len(all_ids),
        "n_ai":             len(by_label["AI"]),
        "n_human":          len(by_label["Human"]),
        "all_ids_hash":     _hash(all_ids),
        "ai_ids_hash":      _hash(by_label["AI"]),
        "human_ids_hash":   _hash(by_label["Human"]),
        "coherence_model":  COHERENCE_MODEL_NAME,
        "style_model":      STYLE_MODEL_NAME,
    }


def cached_ids(features_csv: Path = FEATURES_CSV) -> set[str]:
    """JournalUniqueIDs that already have features on disk."""
    if not features_csv.exists():
        return set()
    try:
        prior = pd.read_csv(features_csv, usecols=["JournalUniqueID"])
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
    """Compute the six features for every entry in `df_to_score`.

    Loads the SentenceTransformer models lazily (skipped entirely when
    `df_to_score` is empty, i.e. on a pure cache hit).
    """
    if df_to_score.empty:
        return pd.DataFrame()

    coh_model, sty_model = load_models()
    print(f"\nScoring features for {len(df_to_score)} entries...")
    rows: list[dict] = []
    for i, (_, row) in enumerate(df_to_score.iterrows(), start=1):
        text = row["JournalAnonymised"]
        rows.append({
            "JournalUniqueID":  row["JournalUniqueID"],
            "word_count":       word_count(text),
            "sentence_count":   sentence_count(text),
            "character_count":  character_count(text),
            "coherence":        coherence_score(text, coh_model),
            "style_similarity": style_similarity_score(text, sty_model),
            "label":            row.get("label"),
        })
        if i % 100 == 0 or i == len(df_to_score):
            print(f"  {i}/{len(df_to_score)} entries scored")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def merge_with_existing(new_rows: pd.DataFrame,
                        current_labels: pd.DataFrame) -> pd.DataFrame:
    """Append `new_rows` to features.csv and refresh the `label` column.

    Labels (AI/Human/NaN) can change between runs as raters add or revise
    ratings, even when the per-entry metric values do not. We therefore
    overwrite the `label` column for every cached row from
    `current_labels` (which is `load_data()`'s scoped frame). On a pure
    cache hit with stable labels this is a no-op rewrite at most.
    """
    if new_rows.empty and not FEATURES_CSV.exists():
        return new_rows

    if FEATURES_CSV.exists():
        prior = pd.read_csv(FEATURES_CSV)
        # Drop any duplicate JID rows from older runs (we now write deduped).
        prior = prior.drop_duplicates(subset=["JournalUniqueID"], keep="first")
        if new_rows.empty:
            combined = prior
        else:
            new_ids = set(new_rows["JournalUniqueID"].astype(str))
            prior = prior[~prior["JournalUniqueID"].astype(str).isin(new_ids)]
            combined = pd.concat([prior, new_rows], ignore_index=True)
    else:
        combined = new_rows

    # Refresh labels from the current ratings frame (one row per JID).
    labels = (current_labels[["JournalUniqueID", "label"]]
                .drop_duplicates(subset=["JournalUniqueID"], keep="first"))
    if "label" in combined.columns:
        combined = combined.drop(columns=["label"])
    combined = combined.merge(labels, on="JournalUniqueID", how="left")
    combined = combined[OUTPUT_COLUMNS]
    combined.to_csv(FEATURES_CSV, index=False)
    return combined


def summarise_by_label(entries: pd.DataFrame) -> pd.DataFrame:
    """Mean per (label, feature). Used for the paper cross-check."""
    rows = []
    metrics = ["word_count", "sentence_count", "character_count",
               "coherence", "style_similarity"]
    for label in ("AI", "Human"):
        s = entries[entries["label"] == label]
        if s.empty:
            continue
        row = {"label": label, "n": len(s)}
        for m in metrics:
            row[f"{m}_mean"] = float(s[m].mean())
            row[f"{m}_std"]  = float(s[m].std())
        rows.append(row)
    return pd.DataFrame(rows)


# ── Reporting ──────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_groups(summary: pd.DataFrame) -> None:
    """Per-label summary of the six features."""
    _print_header("Features by label")
    if summary.empty:
        print("  (no AI/Human entries to summarise)")
        return
    for _, r in summary.iterrows():
        print(f"\n  {r['label']} (n={int(r['n'])}):")
        print(f"    Word count       : {r['word_count_mean']:7.2f} "
              f"\u00b1 {r['word_count_std']:.2f}")
        print(f"    Sentence count   : {r['sentence_count_mean']:7.2f} "
              f"\u00b1 {r['sentence_count_std']:.2f}")
        print(f"    Character count  : {r['character_count_mean']:7.2f} "
              f"\u00b1 {r['character_count_std']:.2f}")
        print(f"    Coherence        : {r['coherence_mean']:7.3f} "
              f"\u00b1 {r['coherence_std']:.3f}")
        print(f"    Style similarity : {r['style_similarity_mean']:7.3f} "
              f"\u00b1 {r['style_similarity_std']:.3f}")


def report_paper_paragraph(summary: pd.DataFrame) -> None:
    """Cross-check vs §Results around line 139-145 + tab:feature_means_semantic."""
    if summary.empty:
        return
    by_label = summary.set_index("label")
    if not {"AI", "Human"}.issubset(by_label.index):
        return
    ai = by_label.loc["AI"]
    hu = by_label.loc["Human"]

    _print_header("Paper paragraph cross-check")
    print("Numbers used in §Results (lines ~139-145) and "
          "tab:feature_means_semantic:\n")
    print(f"  Word count        Human={hu['word_count_mean']:6.1f}    "
          f"AI={ai['word_count_mean']:6.1f}    "
          f"diff={ai['word_count_mean']-hu['word_count_mean']:+6.1f}  "
          "(\u2192 paragraph: '190.3 vs. 87.4'; table: '+102.9')")
    print(f"  Sentence count    Human={hu['sentence_count_mean']:6.1f}    "
          f"AI={ai['sentence_count_mean']:6.1f}    "
          f"diff={ai['sentence_count_mean']-hu['sentence_count_mean']:+6.1f}  "
          "(\u2192 table: '4.8 vs 10.0', diff '+5.2')")
    print(f"  Coherence         Human={hu['coherence_mean']:6.3f}    "
          f"AI={ai['coherence_mean']:6.3f}    "
          f"diff={ai['coherence_mean']-hu['coherence_mean']:+6.3f}  "
          "(\u2192 paragraph + table: '0.468 vs 0.237', diff '+0.231')")
    print(f"  Style similarity  Human={hu['style_similarity_mean']:6.3f}    "
          f"AI={ai['style_similarity_mean']:6.3f}    "
          f"diff={ai['style_similarity_mean']-hu['style_similarity_mean']:+6.3f}  "
          "(\u2192 paragraph + table: '0.862 vs 0.559', diff '+0.304')")


# ── Driver ─────────────────────────────────────────────────────────────────

def main(recompute: bool = RECOMPUTE) -> None:
    """Pipeline driver.

    Decides which entries (if any) need scoring, loads SentenceTransformer
    models lazily, runs only those, refreshes labels for cached rows,
    prints the per-label summary and the paper-paragraph cross-check.
    Pass `recompute=True` (or set the module-level constant `RECOMPUTE`
    to True) to force re-scoring every current entry.
    """
    df = load_data()
    current_key = cache_key(df)

    if recompute:
        print("[cache] recompute=True requested; re-scoring every current entry.")
        to_score = df
    else:
        have, missing = coverage(df)
        extras = (cached_ids() - set(df["JournalUniqueID"].astype(str))
                  if FEATURES_CSV.exists() else set())
        if not missing:
            print(f"[cache] valid: all {len(have):,} current entries already "
                  "have features on disk.")
            print("[cache] skipping recomputation; reading from disk.")
            to_score = df.iloc[0:0]
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
    all_entries = merge_with_existing(new_rows, current_labels=df)

    # Summary is computed over the CURRENT entry set only (not stray IDs).
    current_ids = set(df["JournalUniqueID"].astype(str))
    current_entries = all_entries[
        all_entries["JournalUniqueID"].astype(str).isin(current_ids)
    ]
    summary = summarise_by_label(current_entries)
    write_cache_meta(current_key)

    report_groups(summary)
    report_paper_paragraph(summary)

    print(f"\n\u2713 Per-entry features: {FEATURES_CSV}")
    print(f"\u2713 Cache manifest    : {META_JSON}")


if __name__ == "__main__":
    main()
