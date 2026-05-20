#!/usr/bin/env python3
"""BLEU-based diversity analysis: AI vs Human entries.

For each label we compute pairwise sentence-BLEU between every pair of
entries in that group ("self-BLEU"). The headline statistic is the *mean*
pairwise BLEU per label: higher means entries are more similar to each
other, i.e. less lexically diverse.

Outputs (under output/bleu/):
  bleu_by_label.csv               headline mean BLEU per label
  bleu_pairwise_distribution.csv  full distribution stats per label (mean, sd, median, min, max, n_pairs)
  participant_bleu.csv            within-participant BLEU (one row per
                                  participant with >= MIN_ENTRIES entries)
  bleu_cache_meta.json            manifest tying the above to the exact
                                  set of JournalUniqueIDs scored

Running
-------
There are no CLI flags. To use this file:

  * Run as a script  -> `python scripts/08_bleu.py`
  * Run in an interactive window (Jupyter, IPython, VS Code "# %%" cells,
    PyCharm interactive console) -> just import or `%run` it.

Either way the single switch is the constant `RECOMPUTE_DEFAULT` at the
top of this file, or the `recompute=` kwarg of `main()`:

    from importlib import import_module
    bleu = import_module("scripts.08_bleu")
    bleu.main()                  # use cache when valid (default)
    bleu.main(recompute=True)    # force a re-run

Caching
-------
BLEU is O(n^2) over pairs and slow. Re-running for the same input set is
wasteful, so this script caches per-group results plus a manifest of the
JournalUniqueIDs that produced them (hashed). On startup we recompute
only when the manifest is missing, the per-group entry counts changed, or
the hashed ID list changed -- i.e. we genuinely need to score new rows.

Paper paragraph cross-check
---------------------------
After every run (cached or fresh) the script prints the numbers from §BLEU
analysis next to their paragraph role, e.g.:

  AI BLEU                = 0.0184  (-> paragraph: '0.018')
  Human BLEU             = 0.0106  (-> paragraph: '0.011')
  Approx. % more overlap = 74.4%   (-> paragraph: 'approximately 74%')
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU


# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
RATINGS_CSV = PROJECT_DIR / "ratings" / "all_ratings.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "bleu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_CSV = OUTPUT_DIR / "bleu_by_label.csv"
DIST_CSV = OUTPUT_DIR / "bleu_pairwise_distribution.csv"
PARTICIPANT_CSV = OUTPUT_DIR / "participant_bleu.csv"
PARTICIPANT_GROUP_CSV = OUTPUT_DIR / "participant_bleu_by_group.csv"
META_JSON = OUTPUT_DIR / "bleu_cache_meta.json"

# Single switch:
#   False -> use cached results when they cover the current entry set,
#            re-run only when the cache is stale or missing.
#   True  -> always recompute pairwise BLEU from scratch (slow).
RECOMPUTE: bool = False

MIN_PARTICIPANT_ENTRIES = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ── Loading ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load journals + final labels, keep only valid AI/Human entries."""
    journals = pd.read_csv(INPUT_DIR / "journals.csv", dtype=str)
    journals = journals.drop_duplicates(subset=["JournalUniqueID"])

    ratings = pd.read_csv(RATINGS_CSV)

    # Drop entries flagged invalid by majority vote.
    if "excluded_invalid_majority" in ratings.columns:
        excluded = (ratings["excluded_invalid_majority"]
                      .astype(str).str.lower()
                      .isin(["true", "1", "1.0"]))
        ratings = ratings[~excluded].copy()

    # Keep only AI / human (drop "invalid" labels).
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


# ── Cache: tie outputs to the exact set of entries scored ─────────────────

def cache_key(df: pd.DataFrame) -> dict:
    """Manifest of the input set BLEU was/will be computed on.

    Coverage is checked at the entry level: per-label counts AND a hash of
    the sorted JournalUniqueID list per label. Two runs collide only if
    they scored the same group memberships exactly; adding even a single
    row invalidates the cache and triggers a re-run.
    """
    def _hash_ids(ids: list[str]) -> str:
        h = hashlib.sha256("\n".join(ids).encode("utf-8"))
        return h.hexdigest()

    ai_ids = sorted(df.loc[df["final_label"] == "ai", "JournalUniqueID"]
                       .astype(str).tolist())
    hu_ids = sorted(df.loc[df["final_label"] == "human", "JournalUniqueID"]
                       .astype(str).tolist())
    return {
        "n_ai": len(ai_ids),
        "n_human": len(hu_ids),
        "ai_ids_hash": _hash_ids(ai_ids),
        "human_ids_hash": _hash_ids(hu_ids),
        "min_participant_entries": MIN_PARTICIPANT_ENTRIES,
    }


def cache_status(current_key: dict) -> tuple[bool, str]:
    """Decide whether the on-disk cache matches `current_key`.

    Returns (is_valid, human_readable_reason).
    """
    required = [LABEL_CSV, DIST_CSV, META_JSON]  # participant CSV is optional
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return False, f"missing cache files: {', '.join(missing)}"

    try:
        cached = json.loads(META_JSON.read_text())
    except json.JSONDecodeError:
        return False, "cache manifest is unreadable"

    diffs = [k for k in current_key if cached.get(k) != current_key[k]]
    if diffs:
        deltas = []
        for k in diffs[:3]:
            cur = current_key[k] if "hash" not in k else "<new>"
            old = cached.get(k) if "hash" not in k else "<old>"
            deltas.append(f"{k}: {old} -> {cur}")
        suffix = f" (+{len(diffs)-3} more)" if len(diffs) > 3 else ""
        return False, "cache stale (" + "; ".join(deltas) + suffix + ")"

    return True, (f"cache covers AI n={cached['n_ai']} and "
                  f"Human n={cached['n_human']} for the current entry set")


def write_cache_meta(current_key: dict) -> None:
    META_JSON.write_text(json.dumps(current_key, indent=2))


# ── BLEU computation (single-pass, no duplication) ────────────────────────

def compute_pairwise_bleu(texts: list[str]) -> list[float]:
    """Pairwise sentence-BLEU for every unordered pair of texts.

    Returns one normalised score in [0, 1] per pair, where 1.0 means the
    two texts share all observed n-grams. Uses sacrebleu's effective_order
    so that very short texts do not crash on missing n-gram orders.
    """
    if len(texts) < 2:
        return []

    n = len(texts)
    total_pairs = n * (n - 1) // 2
    print(f"    Computing {total_pairs:,} pairwise comparisons for {n} texts...")

    bleu = BLEU(effective_order=True)
    scores: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                s = bleu.sentence_score(texts[i], [texts[j]])
                scores.append(s.score / 100.0)
            except Exception:
                continue
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{n} texts...")
    return scores


def _summary_stats(scores: list[float]) -> dict:
    """Distribution summary of a list of pairwise scores."""
    if not scores:
        return {"mean": np.nan, "std": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan, "n_pairs": 0}
    return {
        "mean":   float(np.mean(scores)),
        "std":    float(np.std(scores)),
        "median": float(np.median(scores)),
        "min":    float(np.min(scores)),
        "max":    float(np.max(scores)),
        "n_pairs": len(scores),
    }


def analyze_by_label(df: pd.DataFrame) -> dict[str, dict]:
    """One pairwise sweep per label; returns mean (self-BLEU) + dist stats."""
    out: dict[str, dict] = {}
    for label in ("ai", "human"):
        texts = df.loc[df["final_label"] == label, "text"].tolist()
        print(f"  {label.upper()} group ({len(texts)} entries):")
        scores = compute_pairwise_bleu(texts)
        out[label] = {"n_entries": len(texts), **_summary_stats(scores)}
    return out


def analyze_within_participant(df: pd.DataFrame,
                               min_entries: int = MIN_PARTICIPANT_ENTRIES
                               ) -> pd.DataFrame:
    """Self-BLEU within each participant who has >= `min_entries` entries.

    A high score means the participant's entries copy from each other --
    a heuristic for paste-from-LLM behaviour or low effort.
    """
    rows: list[dict] = []
    eligible = sum(1 for _, g in df.groupby("ParticipantID") if len(g) >= min_entries)
    print(f"\n  Analysing {eligible} participants with >= {min_entries} entries...")

    for pid, group in df.groupby("ParticipantID"):
        texts = group["text"].tolist()
        if len(texts) < min_entries:
            continue
        scores = compute_pairwise_bleu(texts)
        if not scores:
            continue
        n_ai = int((group["final_label"] == "ai").sum())
        n_human = int((group["final_label"] == "human").sum())
        rows.append({
            "ParticipantID": pid,
            "bleu": float(np.mean(scores)),
            "n_entries": len(texts),
            "n_ai": n_ai,
            "n_human": n_human,
            "pct_ai": n_ai / len(texts) * 100,
        })

    return (pd.DataFrame(rows)
              .sort_values("bleu", ascending=False)
              .reset_index(drop=True))


# ── Reporting ──────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_groups(group_results: dict) -> None:
    """Print headline per-label BLEU + full distribution stats."""
    _print_header("BLEU by label")
    for label in ("ai", "human"):
        s = group_results[label]
        print(f"\n  {label.upper()} (n={s['n_entries']}):")
        print(f"    Mean BLEU (self-BLEU): {s['mean']:.4f}")
        print(f"    SD                   : {s['std']:.4f}")
        print(f"    Median               : {s['median']:.4f}")
        print(f"    Range                : [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"    Pairs scored         : {s['n_pairs']:,}")

    direction = ("LESS diverse"
                 if group_results["ai"]["mean"] > group_results["human"]["mean"]
                 else "MORE diverse")
    print(f"\n  -> AI entries are {direction} than human entries.")


def _bucket_by_ai_pct(pct_ai: float) -> str:
    """human_only -> 0% AI; ai_only -> 100% AI; mixed otherwise."""
    if pct_ai == 0: return "human_only"
    if pct_ai == 100: return "ai_only"
    return "mixed"


def summarize_participant_by_group(participant_df: pd.DataFrame) -> pd.DataFrame:
    """Per-group within-participant BLEU stats.

    Buckets participants by their AI usage (the same scheme used in
    07_participant_demographics) and reports n / mean / median / std / range
    of within-participant self-BLEU per group. Saved to participant_bleu_by_group.csv.
    """
    df = participant_df.copy()
    df["group"] = df["pct_ai"].apply(_bucket_by_ai_pct)
    summary = (df.groupby("group")["bleu"]
                  .agg(n="count", mean="mean", median="median",
                       std="std", min="min", max="max")
                  .reindex(["human_only", "mixed", "ai_only"])
                  .round(4))
    return summary


def participant_group_test(participant_df: pd.DataFrame) -> dict:
    """Welch's t + Mann-Whitney U comparing Human-only vs Has-AI participants.

    "Has AI" pools mixed + AI-only, mirroring the binary contrast in the
    paper's demographic comparison. Effect size is Cohen's d on the raw
    BLEU scores (positive d means Human-only is MORE repetitive).
    """
    from scipy import stats
    df = participant_df.copy()
    df["group"] = df["pct_ai"].apply(_bucket_by_ai_pct)
    human = df.loc[df["group"] == "human_only", "bleu"]
    has_ai = df.loc[df["group"] != "human_only", "bleu"]
    if len(human) < 2 or len(has_ai) < 2:
        return {}
    res = stats.ttest_ind(human, has_ai, equal_var=False)
    u, pu = stats.mannwhitneyu(human, has_ai, alternative="two-sided")
    pooled_sd = np.sqrt((human.std()**2 + has_ai.std()**2) / 2)
    d = (human.mean() - has_ai.mean()) / pooled_sd if pooled_sd else float("nan")
    return {
        "human_only_n": len(human), "has_ai_n": len(has_ai),
        "human_only_mean": float(human.mean()),
        "has_ai_mean": float(has_ai.mean()),
        "welch_t": float(res.statistic),
        "welch_df": float(res.df),
        "welch_p": float(res.pvalue),
        "mw_u": float(u), "mw_p": float(pu),
        "cohen_d": float(d),
    }


def report_participant(participant_df: pd.DataFrame) -> None:
    """Print:
       1. Per-group within-participant BLEU summary (human-only / mixed / AI-only)
       2. Top-10 most repetitive participants
       3. Welch's t-test on Human-only vs Has-AI participants
       4. AI% / BLEU correlation across all participants
    """
    if participant_df.empty:
        print("  (no participants met the minimum-entries threshold)")
        return
    _print_header("Within-participant BLEU (repetition detection)")

    summary = summarize_participant_by_group(participant_df)
    summary.to_csv(PARTICIPANT_GROUP_CSV)
    print("\n  Per-group within-participant BLEU "
          "(higher = more self-repetition within one person's entries):")
    # Indent by 2 so it nests under the section banner.
    print("\n".join("  " + line for line in summary.to_string().splitlines()))

    test = participant_group_test(participant_df)
    if test:
        print(f"\n  Human-only vs Has-AI (Welch's t):  "
              f"t({test['welch_df']:.0f})={test['welch_t']:.3f}, "
              f"p={test['welch_p']:.4f}, d={test['cohen_d']:.3f}  "
              f"(positive d \u2192 Human-only MORE repetitive)")
        print(f"  Human-only vs Has-AI (Mann-Whitney): "
              f"U={test['mw_u']:.0f}, p={test['mw_p']:.4f}")

    print("\n  Top 10 most repetitive participants (highest BLEU):")
    df = participant_df.copy()
    df["group"] = df["pct_ai"].apply(_bucket_by_ai_pct)
    cols = ["ParticipantID", "bleu", "n_entries", "n_ai", "n_human",
            "pct_ai", "group"]
    print(df.sort_values("bleu", ascending=False).head(10)[cols]
            .to_string(index=False))

    if len(participant_df) > 5:
        r = participant_df["pct_ai"].corr(participant_df["bleu"])
        trend = ("higher AI% participants MORE repetitive within their own entries"
                 if r > 0
                 else "higher AI% participants LESS repetitive within their own entries")
        print(f"\n  Correlation (AI% vs BLEU across all participants): "
              f"r = {r:.3f}  ({trend})")


def report_paper_paragraph(group_results: dict,
                           participant_df: Optional[pd.DataFrame] = None
                           ) -> None:
    """Print every number used in §BLEU analysis next to its sentence role,
    plus the within-participant breakdown so you can see whether the
    cross-entry headline claim ("AI is less diverse") is contradicted by
    the within-person picture.
    """
    ai = group_results["ai"]["mean"]
    hu = group_results["human"]["mean"]
    pct_more = (ai - hu) / hu * 100 if hu else float("nan")

    _print_header("Paper paragraph cross-check")
    print("Cross-entry numbers used in '\u00a7BLEU analysis indicated lower "
          "lexical diversity...':\n")
    print(f"  AI BLEU                = {ai:.4f}  "
          "(\u2192 paragraph: '0.018')")
    print(f"  Human BLEU             = {hu:.4f}  "
          "(\u2192 paragraph: '0.011')")
    print(f"  Ratio (AI / Human)     = {ai/hu:.2f}x")
    print(f"  Approx. % more overlap = {pct_more:.1f}%  "
          "(\u2192 paragraph: 'approximately 74%')")
    print("  Interpretation: a random pair of AI entries (across "
          "participants) is more lexically similar")
    print("                  than a random pair of human entries.\n")

    if participant_df is None or participant_df.empty:
        return

    # Within-participant contrast (why the paper claim needs care).
    test = participant_group_test(participant_df)
    if not test:
        return

    print("Within-participant numbers (NOT in the current paper paragraph):\n")
    print(f"  Mean within-participant BLEU, Human-only "
          f"(n={test['human_only_n']}): {test['human_only_mean']:.4f}")
    print(f"  Mean within-participant BLEU, Has AI     "
          f"(n={test['has_ai_n']}): {test['has_ai_mean']:.4f}")
    direction = ("Human-only" if test["human_only_mean"] > test["has_ai_mean"]
                 else "Has AI")
    print(f"  -> {direction} participants are MORE self-repetitive "
          f"(Welch t({test['welch_df']:.0f})={test['welch_t']:.2f}, "
          f"p={test['welch_p']:.4f}, d={test['cohen_d']:.2f}).\n")
    print("  Caveat: the headline claim is correct AT THE ENTRY LEVEL "
          "(AI entries from\n"
          "  different people resemble each other more than human entries "
          "do), but at the\n"
          "  PARTICIPANT level the most self-repetitive writers are "
          "human-only. These\n"
          "  measure different things: cross-entry BLEU captures shared "
          "LLM 'voice'\n"
          "  across users, while within-participant BLEU captures one "
          "person repeating\n"
          "  themselves across their own entries.")


# ── Driver ─────────────────────────────────────────────────────────────────

def _run_full_analysis(df: pd.DataFrame, current_key: dict) -> dict:
    """Recompute everything, save outputs, refresh manifest."""
    group_results = analyze_by_label(df)

    # Headline self-BLEU per label.
    label_df = pd.DataFrame([
        {"label": "ai",    "bleu": group_results["ai"]["mean"],
         "n_entries": group_results["ai"]["n_entries"]},
        {"label": "human", "bleu": group_results["human"]["mean"],
         "n_entries": group_results["human"]["n_entries"]},
    ])
    label_df.to_csv(LABEL_CSV, index=False)
    print(f"\n\u2713 Saved {LABEL_CSV.name}")

    # Full distribution stats.
    dist_df = pd.DataFrame([
        {"label": "ai",    **group_results["ai"]},
        {"label": "human", **group_results["human"]},
    ])
    dist_df.to_csv(DIST_CSV, index=False)
    print(f"\u2713 Saved {DIST_CSV.name}")

    # Within-participant analysis (separate pass; not double work).
    participant_df = analyze_within_participant(df)
    if not participant_df.empty:
        participant_df.to_csv(PARTICIPANT_CSV, index=False)
        print(f"\u2713 Saved {PARTICIPANT_CSV.name}")

    write_cache_meta(current_key)
    print(f"\u2713 Wrote cache manifest {META_JSON.name}")
    return group_results


def _load_cached_group_results() -> dict[str, dict]:
    """Reconstruct `group_results` from the on-disk CSVs.

    `bleu_by_label.csv` has the per-label entry counts; `bleu_pairwise_distribution.csv`
    has the full distribution stats. We merge them so the structure matches
    what `_run_full_analysis` returns.
    """
    label = pd.read_csv(LABEL_CSV).set_index("label")
    dist = pd.read_csv(DIST_CSV).set_index("label")
    out: dict[str, dict] = {}
    for lbl in dist.index:
        d = dist.loc[lbl].to_dict()
        d["n_entries"] = int(label.loc[lbl, "n_entries"]) if lbl in label.index else None
        d["n_pairs"] = int(d.get("n_pairs", 0))
        out[lbl] = d
    return out


def main(recompute: bool = RECOMPUTE) -> None:
    """Pipeline driver.

    Reads the current AI/Human entry set, decides whether the on-disk cache
    covers it, and either recomputes pairwise BLEU or just reads results
    from disk. The paper-paragraph cross-check is always printed at the
    end so you can verify the numbers in '§BLEU analysis' without scrolling
    back up.

    Pass `recompute=True` (or set the module-level constant `RECOMPUTE` to
    True) to force a re-run.
    """
    df = load_data()
    current_key = cache_key(df)
    is_valid, reason = cache_status(current_key)

    if recompute:
        print("[cache] recompute=True requested; ignoring cache.")
        group_results = _run_full_analysis(df, current_key)
    elif is_valid:
        print(f"[cache] valid: {reason}")
        print("[cache] skipping recomputation; reading from disk.")
        group_results = _load_cached_group_results()
        report_groups(group_results)
    else:
        print(f"[cache] invalid: {reason}")
        print("[cache] recomputing.")
        group_results = _run_full_analysis(df, current_key)
        report_groups(group_results)

    # Within-participant block runs from the on-disk CSV in all cases so
    # both `cache hit` and `recompute` produce the same console layout.
    participant_df = (pd.read_csv(PARTICIPANT_CSV)
                      if PARTICIPANT_CSV.exists() else pd.DataFrame())
    if not participant_df.empty:
        report_participant(participant_df)

    report_paper_paragraph(group_results, participant_df)


if __name__ == "__main__":
    main()
