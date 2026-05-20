#!/usr/bin/env python3
"""LASSO regression on LIWC features to classify perceived human vs AI entries."""

import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output" / "analyze_liwc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RATINGS_PATH = ROOT_DIR / "ratings" / "all_ratings.csv"
LIWC_PATH = ROOT_DIR / "output" / "liwc" / "liwc.csv"
TOPWORDS_PATH = ROOT_DIR / "output" / "get_top_words" / "liwc_topwords.json"

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
    """Load LIWC data merged with labels from ratings."""
    liwc = pd.read_csv(LIWC_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    
    # Keep only human/AI labels (excludes all invalid entries)
    ratings = ratings[ratings["final_label"].isin(["ai", "human"])].copy()
    ratings["label"] = ratings["final_label"].map({"ai": "AI", "human": "Human"})
    
    # Remove duplicate IDs in LIWC (keep first)
    liwc = liwc.drop_duplicates(subset=["JournalUniqueID"], keep="first")
    
    # Inner merge to get only entries with both LIWC and valid labels
    df = liwc.merge(ratings[["JournalUniqueID", "label"]], on="JournalUniqueID", how="inner")
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


def run_lasso(df: pd.DataFrame, liwc_cols: List[str], suffix: str = "") -> dict:
    """Run LASSO analysis and return results."""
    X = df[liwc_cols]
    y = (df["label"] == "AI").astype(int).to_numpy()
    
    if len(np.unique(y)) < 2:
        print("Need both classes to run LASSO")
        return {}
    
    # LASSO with grid search over C
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, class_weight="balanced")),
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, {"clf__C": np.logspace(-3, 2, 10)}, scoring="roc_auc", cv=cv, refit=True)
    grid.fit(X, y)
    
    best_C = grid.best_params_["clf__C"]
    best_clf = grid.best_estimator_.named_steps["clf"]
    
    # Cross-validate with best C for all metrics
    cv_scores = cross_validate(
        Pipeline([("scaler", StandardScaler()), 
                  ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, 
                                            class_weight="balanced", C=best_C))]),
        X, y, cv=cv, 
        scoring={"roc_auc": "roc_auc", "avg_prec": "average_precision", "accuracy": "accuracy", "f1": "f1"})
    
    # Extract coefficients
    coefs = best_clf.coef_.ravel()
    coef_df = pd.DataFrame({
        "metric": liwc_cols,
        "coefficient": coefs,
        "abs_coeff": np.abs(coefs),
        "direction": np.where(coefs > 0, "AI", np.where(coefs < 0, "Human", "Neutral")),
        "selected": coefs != 0,
    }).sort_values("abs_coeff", ascending=False)
    
    num_selected = int(coef_df["selected"].sum())
    num_features = len(coef_df)
    
    # Save results
    coef_df.to_csv(OUTPUT_DIR / f"liwc_lasso_coeffs{suffix}.csv", index=False)
    
    cv_results = pd.DataFrame([{
        "best_C": best_C,
        "cv_auc_mean": cv_scores["test_roc_auc"].mean(),
        "cv_auc_std": cv_scores["test_roc_auc"].std(),
        "accuracy_mean": cv_scores["test_accuracy"].mean(),
        "accuracy_std": cv_scores["test_accuracy"].std(),
        "f1_mean": cv_scores["test_f1"].mean(),
        "f1_std": cv_scores["test_f1"].std(),
        "pr_auc_mean": cv_scores["test_avg_prec"].mean(),
        "pr_auc_std": cv_scores["test_avg_prec"].std(),
        "num_selected": num_selected,
        "num_features": num_features,
    }])
    cv_results.to_csv(OUTPUT_DIR / f"liwc_lasso_cv{suffix}.csv", index=False)
    
    # Print summary
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    print(f"\n  Best C:    {best_C:.4g}")
    print(f"  AUC:       {fmt(cv_scores['test_roc_auc'].mean(), cv_scores['test_roc_auc'].std())}")
    print(f"  Accuracy:  {fmt(cv_scores['test_accuracy'].mean(), cv_scores['test_accuracy'].std())}")
    print(f"  F1:        {fmt(cv_scores['test_f1'].mean(), cv_scores['test_f1'].std())}")
    print(f"  PR AUC:    {fmt(cv_scores['test_avg_prec'].mean(), cv_scores['test_avg_prec'].std())}")
    print(f"  Selected:  {num_selected}/{num_features} features")
    
    # Top 10 features
    print(f"\n  Top 10 selected features:")
    for _, row in coef_df[coef_df["selected"]].head(10).iterrows():
        print(f"    {row['metric']:20s} {row['coefficient']:+.4f} ({row['direction']})")
    
    # Save plot
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
        "best_C": best_C,
    }


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

    print(f"\n✓ Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
