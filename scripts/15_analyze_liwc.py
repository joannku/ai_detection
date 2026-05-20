#!/usr/bin/env python3
"""LASSO regression on LIWC features to classify perceived human vs AI entries."""

import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output" / "analyze_liwc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RATINGS_PATH = ROOT_DIR / "ratings" / "all_ratings.csv"
JOURNALS_PATH = ROOT_DIR / "input" / "journals.csv"
LIWC_PATH = ROOT_DIR / "output" / "liwc" / "liwc.csv"
TOPWORDS_PATH = ROOT_DIR / "output" / "get_top_words" / "liwc_topwords.json"

# CV configuration: nested (outer for unbiased performance, inner for tuning)
# and group-aware (StratifiedGroupKFold on ParticipantID), matching the
# meta-classifier setup in scripts/21_meta_classifiers.py exactly.
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 42
C_GRID = np.logspace(-3, 2, 10)

# LIWC-22's four proprietary "summary" variables are computed from the underlying
# word categories using internal algorithms that are not transparent and cannot be
# replicated from the lexicon alone. We always drop them so the LASSO sees only
# interpretable lexical features.
LIWC_PROPRIETARY = {"Analytic", "Clout", "Authentic", "Tone"}

# Length / dictionary-coverage stats that overlap with our other length features
# (word_count, readability metrics, etc.). Excluded only in the word-only run.
# `BigWords` is kept here because it is a content-of-text statistic (% of long
# words) that the readability and word-count features do not capture directly.
LIWC_LENGTH_PROXIES = {"WC", "WPS", "Dic"}

# Backwards-compat alias used by external callers / paper text formatting.
LIWC_SUMMARY_VARS = LIWC_PROPRIETARY | LIWC_LENGTH_PROXIES


def load_data() -> pd.DataFrame:
    """Load LIWC features merged with AI/Human labels and ParticipantID.

    ParticipantID is required for group-aware cross-validation: ~91% of
    participants only ever have one label, so non-grouped folds leak label
    information into the test set.
    """
    liwc = pd.read_csv(LIWC_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    journals = pd.read_csv(JOURNALS_PATH)

    # Keep only human/AI labels (excludes all invalid entries)
    ratings = ratings[ratings["final_label"].isin(["ai", "human"])].copy()
    ratings["label"] = ratings["final_label"].map({"ai": "AI", "human": "Human"})

    # journals.csv contains ~59 duplicate JournalUniqueIDs with identical
    # text — dedup so we don't inflate group sizes.
    journals = journals.drop_duplicates(subset=["JournalUniqueID"], keep="first")

    # Remove duplicate IDs in LIWC (keep first)
    liwc = liwc.drop_duplicates(subset=["JournalUniqueID"], keep="first")

    # Inner merge to get only entries with both LIWC and valid labels;
    # left-join ParticipantID (warn if any are missing).
    df = liwc.merge(ratings[["JournalUniqueID", "label"]],
                    on="JournalUniqueID", how="inner")
    df = df.merge(journals[["JournalUniqueID", "ParticipantID"]],
                  on="JournalUniqueID", how="left")
    n_missing_pid = int(df["ParticipantID"].isna().sum())
    if n_missing_pid:
        print(f"WARNING: {n_missing_pid} entries are missing ParticipantID; "
              "they will be dropped from group-aware CV.")
        df = df.dropna(subset=["ParticipantID"]).copy()
    return df


def get_liwc_columns(df: pd.DataFrame, exclude_summary: bool = False) -> List[str]:
    """Get numeric LIWC columns, excluding metadata.

    The four LIWC-22 proprietary summary variables (Analytic, Clout, Authentic,
    Tone) are *always* dropped. When ``exclude_summary`` is True the length /
    coverage proxies (WC, WPS, Dic) are dropped as well, leaving only
    word-category percentages.
    """
    banned = {"journaluniqueid", "label", "rowid", "row", "rownumber", "segment", "filename", "source"}
    cols = [c for c in df.columns 
            if pd.api.types.is_numeric_dtype(df[c]) 
            and re.sub(r"[^a-z0-9]", "", c.lower()) not in banned]
    cols = [c for c in cols if c not in LIWC_PROPRIETARY]
    if exclude_summary:
        cols = [c for c in cols if c not in LIWC_LENGTH_PROXIES]
    return cols


def load_topwords() -> Dict[str, List[str]]:
    """Load top words for each LIWC category."""
    if not TOPWORDS_PATH.exists():
        return {}
    with open(TOPWORDS_PATH) as f:
        data = json.load(f)
    # Return top 4 words for each category (keys are words, values are counts)
    return {cat: list(words.keys())[:4] for cat, words in data.items() if words}


def get_liwc_stats(words_only: bool = False) -> dict:
    """Get LIWC LASSO stats from saved CSV files."""
    suffix = "_words_only" if words_only else ""
    cv_path = OUTPUT_DIR / f"liwc_lasso_cv{suffix}.csv"
    coef_path = OUTPUT_DIR / f"liwc_lasso_coeffs{suffix}.csv"
    
    cv_df = pd.read_csv(cv_path)
    coef_df = pd.read_csv(coef_path)
    topwords = load_topwords()
    
    row = cv_df.iloc[0]
    
    # Get top features by direction
    selected = coef_df[coef_df["selected"] == True].copy()
    top_ai = selected[selected["direction"] == "AI"].nlargest(10, "abs_coeff")["metric"].tolist()
    top_human = selected[selected["direction"] == "Human"].nlargest(10, "abs_coeff")["metric"].tolist()
    
    return {
        "auc_mean": row["cv_auc_mean"],
        "auc_std": row["cv_auc_std"],
        "accuracy_mean": row["accuracy_mean"],
        "accuracy_std": row["accuracy_std"],
        "f1_mean": row["f1_mean"],
        "f1_std": row["f1_std"],
        "num_selected": int(row["num_selected"]),
        "num_features": int(row["num_features"]),
        "top_ai_features": top_ai,
        "top_human_features": top_human,
        "topwords": topwords,
    }


def _build_pipe() -> Pipeline:
    """LASSO pipeline: standardise then L1 logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000,
            class_weight="balanced",
        )),
    ])


def run_lasso(df: pd.DataFrame, liwc_cols: List[str], suffix: str = "") -> dict:
    """Nested + group-aware LASSO analysis.

    * Outer  : 5-fold StratifiedGroupKFold on ParticipantID -> unbiased perf.
    * Inner  : 3-fold StratifiedGroupKFold inside each outer training fold,
               grid-searches C ∈ logspace(-3, 2, 10) using ROC-AUC.
    * Final  : a separate inner-CV grid-search on ALL data refits the model
               whose coefficients we save.

    This matches the canonical pattern used by `scripts/21_meta_classifiers.py`
    so the LIWC-only LASSO is comparable with the full-feature meta-classifier
    table (no participant leakage; no hyperparameter leakage into the
    reported scores).
    """
    X = df[liwc_cols]
    y = (df["label"] == "AI").astype(int).to_numpy()
    groups = df["ParticipantID"].astype(str).to_numpy()

    if len(np.unique(y)) < 2:
        print("Need both classes to run LASSO")
        return {}

    pipe = _build_pipe()
    param_grid = {"clf__C": C_GRID}
    outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS)
    inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS)

    print(f"  CV scheme: nested StratifiedGroupKFold "
          f"(outer={OUTER_FOLDS}, inner={INNER_FOLDS}) on ParticipantID "
          f"({len(np.unique(groups))} unique participants)")
    print(f"  Inner grid : C \u2208 {{{C_GRID[0]:.0e} \u2026 "
          f"{C_GRID[-1]:.0e}}} ({len(C_GRID)} values), scoring=roc_auc")

    nested_estimator = GridSearchCV(
        pipe, param_grid, scoring="roc_auc",
        cv=inner_cv, refit=True, n_jobs=1,
    )
    cv_scores = cross_validate(
        nested_estimator, X, y,
        cv=outer_cv, groups=groups,
        scoring={"roc_auc": "roc_auc",
                 "avg_prec": "average_precision",
                 "accuracy": "accuracy",
                 "f1": "f1"},
        n_jobs=1, return_estimator=True,
        params={"groups": groups},
    )
    per_fold_best_C = [est.best_params_["clf__C"] for est in cv_scores["estimator"]]

    # Final refit on ALL data with a fresh inner-CV grid search; these are
    # the coefficients we save and inspect (clearly separated from the
    # unbiased outer-CV scores above).
    gs_full = GridSearchCV(
        _build_pipe(), param_grid, scoring="roc_auc",
        cv=StratifiedGroupKFold(n_splits=INNER_FOLDS),
        refit=True, n_jobs=1,
    )
    gs_full.fit(X, y, groups=groups)
    best_C_final = gs_full.best_params_["clf__C"]
    final_clf = gs_full.best_estimator_.named_steps["clf"]

    coefs = final_clf.coef_.ravel()
    coef_df = pd.DataFrame({
        "metric": liwc_cols,
        "coefficient": coefs,
        "abs_coeff": np.abs(coefs),
        "direction": np.where(coefs > 0, "AI", np.where(coefs < 0, "Human", "Neutral")),
        "selected": coefs != 0,
    }).sort_values("abs_coeff", ascending=False)

    num_selected = int(coef_df["selected"].sum())
    num_features = len(coef_df)

    coef_df.to_csv(OUTPUT_DIR / f"liwc_lasso_coeffs{suffix}.csv", index=False)

    cv_results = pd.DataFrame([{
        "cv_scheme": "StratifiedGroupKFold (nested, group-aware on ParticipantID)",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "n_groups": int(len(np.unique(groups))),
        "C_grid": ", ".join(f"{c:.4g}" for c in C_GRID),
        "per_fold_best_C": ", ".join(f"{c:.4g}" for c in per_fold_best_C),
        "best_C": best_C_final,
        "cv_auc_mean":   cv_scores["test_roc_auc"].mean(),
        "cv_auc_std":    cv_scores["test_roc_auc"].std(),
        "accuracy_mean": cv_scores["test_accuracy"].mean(),
        "accuracy_std":  cv_scores["test_accuracy"].std(),
        "f1_mean":       cv_scores["test_f1"].mean(),
        "f1_std":        cv_scores["test_f1"].std(),
        "pr_auc_mean":   cv_scores["test_avg_prec"].mean(),
        "pr_auc_std":    cv_scores["test_avg_prec"].std(),
        "num_selected":  num_selected,
        "num_features":  num_features,
    }])
    cv_results.to_csv(OUTPUT_DIR / f"liwc_lasso_cv{suffix}.csv", index=False)

    fmt = lambda m, s: f"{m:.3f} \u00b1 {s:.3f}"
    print(f"\n  AUC           : {fmt(cv_scores['test_roc_auc'].mean(), cv_scores['test_roc_auc'].std())}")
    print(f"  Accuracy      : {fmt(cv_scores['test_accuracy'].mean(), cv_scores['test_accuracy'].std())}")
    print(f"  F1            : {fmt(cv_scores['test_f1'].mean(), cv_scores['test_f1'].std())}")
    print(f"  PR-AUC        : {fmt(cv_scores['test_avg_prec'].mean(), cv_scores['test_avg_prec'].std())}")
    print(f"  Selected      : {num_selected}/{num_features} features")
    print(f"  Per-fold best C: {per_fold_best_C}")
    print(f"  Final refit C  : {best_C_final:.4g}")

    print(f"\n  Top 10 selected features:")
    for _, row in coef_df[coef_df["selected"]].head(10).iterrows():
        print(f"    {row['metric']:20s} {row['coefficient']:+.4f} ({row['direction']})")

    try:
        top_df = coef_df[coef_df["selected"]].nlargest(25, "abs_coeff").iloc[::-1]
        fig = px.bar(top_df, x="coefficient", y="metric", color="direction", orientation="h",
                     title=f"Top LASSO coefficients{' (word-based only)' if suffix else ''}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=600)
        fig.write_html((OUTPUT_DIR / f"liwc_lasso_coeffs_plot{suffix}.html").as_posix(), include_plotlyjs="cdn")
        fig.write_image((OUTPUT_DIR / f"liwc_lasso_coeffs_plot{suffix}.png").as_posix(), width=1800, height=1200)
    except Exception as e:
        print(f"  Note: Could not save plot ({e})")

    return {
        "cv_scores": cv_scores,
        "coef_df": coef_df,
        "best_C": best_C_final,
        "per_fold_best_C": per_fold_best_C,
    }


def report_paper_paragraph() -> None:
    """Cross-check vs §Psycholinguistic features in 5results.tex.

    Reads back the WORDS-ONLY LASSO results (the run cited in the paper) and
    prints every number, feature name, and example word from the paragraph
    alongside its computed value, so a glance verifies the paragraph.
    """
    cv_path = OUTPUT_DIR / "liwc_lasso_cv_words_only.csv"
    coef_path = OUTPUT_DIR / "liwc_lasso_coeffs_words_only.csv"
    if not cv_path.exists() or not coef_path.exists():
        print("\n(Paper cross-check skipped: words-only CSVs not yet on disk.)")
        return

    cv = pd.read_csv(cv_path).iloc[0]
    coef = pd.read_csv(coef_path)
    selected_ai = (coef[(coef["selected"] == True) & (coef["direction"] == "AI")]
                     .nlargest(10, "abs_coeff"))
    topwords = load_topwords()

    print(f"\n{'='*60}")
    print("Paper paragraph cross-check (§Psycholinguistic features)")
    print(f"{'='*60}")

    # CV setup details (for the methods section / paragraph caveats).
    print("\nCross-validation setup:\n")
    scheme = cv.get("cv_scheme", "(legacy: non-nested StratifiedKFold)")
    outer = int(cv["outer_folds"]) if "outer_folds" in cv else 5
    inner = int(cv["inner_folds"]) if "inner_folds" in cv else 0
    n_groups = int(cv["n_groups"]) if "n_groups" in cv else 0
    print(f"  Scheme           : {scheme}")
    if inner:
        print(f"  Outer folds      : {outer}  (unbiased performance)")
        print(f"  Inner folds      : {inner}  (per-fold C tuning)")
    else:
        print(f"  Folds            : {outer}")
    if n_groups:
        print(f"  Unique groups    : {n_groups} participants")
    if "C_grid" in cv:
        print(f"  Inner C grid     : {cv['C_grid']}")
    if "per_fold_best_C" in cv:
        print(f"  Per-fold best C  : [{cv['per_fold_best_C']}]")
    print(f"  Final refit C    : {cv['best_C']}")

    print("\nWords-only LASSO numbers (the run cited in the paragraph):\n")
    print(f"  AUC                = {cv['cv_auc_mean']:.3f} \u00b1 "
          f"{cv['cv_auc_std']:.3f}   (\u2192 paragraph: '0.987 \u00b1 0.005')")
    print(f"  Accuracy           = {cv['accuracy_mean']:.3f} \u00b1 "
          f"{cv['accuracy_std']:.3f}   (\u2192 paragraph: '0.942 \u00b1 0.019')")
    print(f"  F1                 = {cv['f1_mean']:.3f} \u00b1 "
          f"{cv['f1_std']:.3f}   (\u2192 paragraph: '0.910 \u00b1 0.029')")
    print(f"  Selected features  = {int(cv['num_selected'])} of "
          f"{int(cv['num_features'])}   (\u2192 paragraph: '54 of 111')")

    # Top 10 AI-direction features cited verbatim in the paragraph.
    print("\nTop 10 features predictive of perceived AI authorship "
          "(by |coefficient|):\n")
    print("  rank  metric        coef     (paragraph mentions, in order:")
    print("                                BigWords, article, prep, det, AllPunc,")
    print("                                Comma, Perception, memory, curiosity,")
    print("                                fulfill)\n")
    for rank, (_, row) in enumerate(selected_ai.iterrows(), start=1):
        print(f"  {rank:>2}.   {row['metric']:<13s} {row['coefficient']:+.3f}")

    # Example words quoted in the paragraph.
    print("\nExample words for the categories quoted in the paragraph:\n")
    quoted = {
        "article":   ["the", "a", "an"],
        "prep":      ["to", "of", "in", "with"],
        "memory":    ["memories", "reminder", "reminded", "remember"],
        "curiosity": ["creating", "challenging", "creative", "captivating"],
    }
    for cat, paper_words in quoted.items():
        actual = topwords.get(cat, [])
        match = "\u2713" if actual[:len(paper_words)] == paper_words else "\u26a0"
        print(f"  {match} {cat:<10s} paragraph: {paper_words}")
        print(f"     {'':>10s} computed top: {actual[:max(4, len(paper_words))]}")
    if not topwords:
        print("  (No liwc_topwords.json on disk; run scripts/14_get_top_words.py "
              "to populate.)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    
    print(f"Loaded {len(df)} entries")
    print(f"Always excluding LIWC proprietary summary vars: "
          f"{', '.join(sorted(LIWC_PROPRIETARY))}")

    # Run with all interpretable LIWC features (proprietary summary vars dropped)
    liwc_cols_all = get_liwc_columns(df, exclude_summary=False)
    print(f"\n{'='*60}")
    print(f"LASSO (all {len(liwc_cols_all)} LIWC features, no proprietary summaries)")
    print(f"{'='*60}")
    run_lasso(df, liwc_cols_all, suffix="")

    # Run with word-category features only (also dropping length / coverage proxies)
    liwc_cols_words = get_liwc_columns(df, exclude_summary=True)
    print(f"\n{'='*60}")
    print(f"LASSO (word-based only: {len(liwc_cols_words)} features)")
    print(f"Also excluded length proxies: {', '.join(sorted(LIWC_LENGTH_PROXIES))}")
    print(f"{'='*60}")
    run_lasso(df, liwc_cols_words, suffix="_words_only")

    # Paper paragraph cross-check using the freshly-saved words-only results.
    report_paper_paragraph()

    print(f"\n\u2713 Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
