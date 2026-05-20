#!/usr/bin/env python3
"""Compare GPTZero predictions against human majority-vote labels.

Reads:
  output/export/final_journals.csv   (one row per entry; column `label` ∈ {ai, human})
  output/gptzero/gptzero_results.csv (per-entry GPTZero predictions)

Writes:
  output/compare_gptzero/gptzero_comparison.csv   (per-entry agreement table)
  output/compare_gptzero/confusion_heatmap.png    (figure used in 5results.tex)
  overleaf_paper/figures/confusion_heatmap.png    (mirror copy)

Prints:
  • Headline accuracy and per-class accuracy
  • Confusion matrix (counts and per-true-class percentages)
  • Detection metrics (precision, recall, specificity, F1) with AI as positive
  • GPTZero AI-probability mean / SD by human label
  • A "Paper paragraph cross-check" block that lists the exact numbers used
    in §Results / Automated detection performance
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Paths ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
JOURNALS_CSV = ROOT / "output" / "export" / "final_journals.csv"
GPTZERO_CSV = ROOT / "output" / "gptzero" / "gptzero_results.csv"
OUTPUT_DIR = ROOT / "output" / "compare_gptzero"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS = ROOT / "overleaf_paper" / "figures"

# Class order for tables / plots: AI first, human second (treated as the
# positive class in detection-metric language).
LABEL_ORDER = ["ai", "human"]
DISPLAY_LABELS = ["AI", "Human"]


# ── Load + align labels ────────────────────────────────────────────────────

def load_merged() -> pd.DataFrame:
    """Join human labels with GPTZero predictions and add a `match` column.

    GPTZero's three-way `predicted_class` (`ai` / `human` / `mixed`) is
    collapsed to two-way by mapping `mixed` → `ai`, matching how the paper
    treats borderline AI-leaning predictions.
    """
    journals = pd.read_csv(JOURNALS_CSV)
    gptzero = pd.read_csv(GPTZERO_CSV)
    df = journals.merge(gptzero, on="JournalUniqueID")
    df = df.dropna(subset=["gptzero_class"])
    df["human_label"] = df["label"]
    df["gptzero_label"] = df["gptzero_class"].replace({"mixed": "ai"})
    df["match"] = df["human_label"] == df["gptzero_label"]
    return df


# ── Stats / tables ─────────────────────────────────────────────────────────

def confusion_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts crosstab with rows = human, cols = GPTZero, in LABEL_ORDER."""
    return (pd.crosstab(df["human_label"], df["gptzero_label"])
              .reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0))


def detection_metrics(counts: pd.DataFrame) -> dict:
    """Treat AI as positive class. Returns tp/fp/tn/fn + precision/recall/F1."""
    tp = int(counts.loc["ai", "ai"])
    fn = int(counts.loc["ai", "human"])
    fp = int(counts.loc["human", "ai"])
    tn = int(counts.loc["human", "human"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
    }


def probability_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean / median / SD of `gptzero_ai_prob` for each human label."""
    out = {}
    for label in LABEL_ORDER:
        probs = df.loc[df["human_label"] == label, "gptzero_ai_prob"]
        out[label] = {"mean": probs.mean(),
                      "median": probs.median(),
                      "std": probs.std()}
    return out


# ── Reporting blocks ───────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def report_overall(df: pd.DataFrame) -> None:
    accuracy = df["match"].mean()
    print(f"\nOverall accuracy: {accuracy:.1%} "
          f"({int(df['match'].sum())}/{len(df)})")


def report_confusion(counts: pd.DataFrame) -> None:
    print("\n--- Confusion Matrix (rows=human, cols=GPTZero) ---")
    print(counts.assign(All=counts.sum(axis=1))
                .pipe(lambda t: pd.concat([t, t.sum(axis=0).to_frame("All").T])))


def report_per_class_accuracy(df: pd.DataFrame) -> None:
    print("\n--- Accuracy by Human Label ---")
    for label in LABEL_ORDER:
        subset = df[df["human_label"] == label]
        print(f"  {label}: {subset['match'].mean():.1%} "
              f"({int(subset['match'].sum())}/{len(subset)})")


def report_metrics(m: dict) -> None:
    print("\n--- Detection Metrics (AI as positive class) ---")
    print(f"  True Positives  (AI    -> AI):    {m['tp']}")
    print(f"  True Negatives  (Human -> Human): {m['tn']}")
    print(f"  False Positives (Human -> AI):    {m['fp']}")
    print(f"  False Negatives (AI    -> Human): {m['fn']}")
    print(f"\n  Precision:   {m['precision']:.3f}  "
          "(of predicted AI, how many are actually AI)")
    print(f"  Recall:      {m['recall']:.3f}  "
          "(of actual AI, how many detected)")
    print(f"  Specificity: {m['specificity']:.3f}  "
          "(of actual human, how many correctly identified)")
    print(f"  F1 Score:    {m['f1']:.3f}")


def report_probabilities(stats: dict) -> None:
    print("\n--- GPTZero AI Probability by Human Label ---")
    for label in LABEL_ORDER:
        s = stats[label]
        print(f"  {label}: mean={s['mean']:.3f}, "
              f"median={s['median']:.3f}, std={s['std']:.3f}")


def report_mixed(df: pd.DataFrame) -> None:
    """How many entries GPTZero classified as `mixed` (collapsed to AI above)."""
    mixed_count = int((df["gptzero_class"] == "mixed").sum())
    if not mixed_count:
        return
    mixed = df[df["gptzero_class"] == "mixed"]
    print(f"\n--- Mixed Classifications: {mixed_count} ---")
    print(f"  Of which human-labeled: "
          f"ai={int((mixed['human_label']=='ai').sum())}, "
          f"human={int((mixed['human_label']=='human').sum())}")


def report_paper_paragraph(df: pd.DataFrame, counts: pd.DataFrame,
                           m: dict, stats: dict) -> None:
    """Print every number that appears in the paper's "Automated detection
    performance" paragraph next to its sentence-level role.

    Intentionally verbose: the goal is one-glance verification of the
    prose, not pretty formatting.
    """
    n_total = int(counts.values.sum())
    n_ai = int(counts.loc["ai"].sum())
    n_human = int(counts.loc["human"].sum())

    accuracy = df["match"].mean()
    fp_rate = m["fp"] / n_human if n_human else 0      # human → AI / # human
    fn_rate = m["fn"] / n_ai if n_ai else 0            # AI → human / # AI

    _print_header("Paper paragraph cross-check")
    print("Sentence-by-sentence numbers used in §Automated detection performance:\n")
    print(f"  total entries scored             = {n_total:,}")
    print(f"  agreement with majority vote     = {accuracy:.1%}  "
          f"(\u2192 paragraph: \"94.3% of entries\")")
    print(f"  AI \u2192 AI    (correct on AI)        = {m['tp']:,} of {n_ai:,}  "
          f"(\u2192 \"712 of 739 AI-labelled entries\")")
    print(f"  human \u2192 human (correct on human)  = {m['tn']:,} of {n_human:,}  "
          f"(\u2192 \"1,614 of 1,727 human-labelled entries\")")
    print(f"  human \u2192 AI    (false positives)   = {m['fp']:,}  ({fp_rate:.1%} "
          f"of human-labelled)  (\u2192 \"113 entries, 6.5% of human-labelled\")")
    print(f"  AI    \u2192 human (false negatives)   = {m['fn']:,}  ({fn_rate:.1%} "
          f"of AI-labelled)     (\u2192 \"27 entries, 3.7% of AI-labelled\")")
    print(f"  AI-labelled mean p(AI)           = {stats['ai']['mean']:.3f} "
          f"(SD {stats['ai']['std']:.3f})  "
          f"(\u2192 \"M=0.93, SD=0.20\")")
    print(f"  human-labelled mean p(AI)        = {stats['human']['mean']:.3f} "
          f"(SD {stats['human']['std']:.3f})  "
          f"(\u2192 \"M=0.06, SD=0.22\")")


# ── Confusion-matrix figure ────────────────────────────────────────────────

def plot_confusion_matrix(df: pd.DataFrame) -> None:
    """Save the heatmap figure used in the paper.

    Each cell shows the count and the per-true-class (row-normalised)
    percentage. This means diagonal cells = recall on each class, and
    off-diagonal cells = error rate on each class — matching the
    denominators the paper text uses ("3.7% of AI-labelled entries").
    """
    counts = confusion_counts(df)
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    row_pct = counts.div(row_totals, axis=0) * 100

    annot = np.empty_like(counts.values, dtype=object)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            n, p = int(counts.values[i, j]), row_pct.values[i, j]
            annot[i, j] = f"{n}\n({p:.1f}%)" if not np.isnan(p) else f"{n}"

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    sns.heatmap(
        counts, annot=annot, fmt="", cmap="Blues",
        cbar=True, cbar_kws={"label": "Count"},
        annot_kws={"size": 17, "weight": "bold"},
        linewidths=2.0, linecolor="white",
        xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
        ax=ax,
    )
    # White-on-dark contrast for the dominant cell.
    vmax = counts.values.max()
    for text, val in zip(ax.texts, counts.values.ravel()):
        text.set_color("white" if val > 0.55 * vmax else "#1a1a1f")

    n_total = int(counts.values.sum())
    ax.set_title(f"GPTZero vs Human Label Confusion Matrix\n"
                 f"(n={n_total:,} entries; cells show count and "
                 f"% of true-class row)",
                 fontsize=12, color="#1a1a1f", pad=12)
    ax.set_xlabel("GPTZero Predicted Label", fontsize=12,
                  fontweight="bold", color="#1a1a1f")
    ax.set_ylabel("Human Annotator Label", fontsize=12,
                  fontweight="bold", color="#1a1a1f")
    ax.tick_params(axis="both", labelsize=12, colors="#1a1a1f")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    out = OUTPUT_DIR / "confusion_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\u2713 Saved confusion matrix \u2192 {out}")

    if PAPER_FIGS.exists():
        paper_out = PAPER_FIGS / "confusion_heatmap.png"
        shutil.copyfile(out, paper_out)
        print(f"\u2713 Copied \u2192 {paper_out}")
    else:
        print(f"WARNING: {PAPER_FIGS} not found, skipping copy")


# ── Driver ─────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_merged()
    counts = confusion_counts(df)
    metrics = detection_metrics(counts)
    prob_stats = probability_stats(df)

    _print_header("GPTZero vs Human Labels Comparison")
    report_overall(df)
    report_confusion(counts)
    report_per_class_accuracy(df)
    report_metrics(metrics)
    report_probabilities(prob_stats)
    report_mixed(df)

    plot_confusion_matrix(df)

    out_path = OUTPUT_DIR / "gptzero_comparison.csv"
    df[["JournalUniqueID", "human_label", "gptzero_label", "gptzero_class",
        "gptzero_ai_prob", "match"]].to_csv(out_path, index=False)
    print(f"\u2713 Saved detailed comparison \u2192 {out_path}")

    # Final block, placed after the CSV save so it's the last thing the
    # user sees in the terminal.
    report_paper_paragraph(df, counts, metrics, prob_stats)


if __name__ == "__main__":
    main()
