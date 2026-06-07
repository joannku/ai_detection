#!/usr/bin/env python3
# %% [markdown]
# # Interpretable meta-classifiers on the unified feature set
#
# Trains five interpretable models on the high-confidence subset and reports
# fully group-aware nested cross-validated performance. The five models are:
#
# | Model              | Why                                            |
# | ------------------ | ---------------------------------------------- |
# | LASSO logistic     | sparse linear coefficients, easy to interpret  |
# | Decision Tree      | explicit if/else rules                         |
# | Random Forest      | non-linear ensemble baseline                   |
# | XGBoost            | gradient-boosted ensemble baseline             |
# | (Linear SVM lives  | margin-based triage; see scripts/24_meta_svm.py)|
#
# **CV scheme.** Outer 5-fold and inner 3-fold *StratifiedGroupKFold* on
# `ParticipantID`. Outer folds give an *unbiased* performance estimate;
# inner folds tune hyperparameters within each outer training fold.
# Group-aware folds are essential here because ~91% of participants have
# only one label, so non-grouped folds would leak label info into the test
# set.
#
# **Final fit.** After the outer-CV evaluation we refit each model on all
# data with a fresh inner-CV grid search; that fit is what we save SHAP
# values, LASSO coefficients, and the decision-tree rules from. We never
# read performance numbers from this final fit.
#
# **Outputs (all under `output/meta_classifiers/`):**
#
# * `meta_cv_results.csv`         per-model AUC / Acc / F1 / Precision / Recall
# * `meta_lasso_coeffs.csv`       LASSO coefficients
# * `meta_tree_rules.txt`         Decision-Tree rules in text form
# * `decision_tree.png`           Decision-Tree diagram
# * `shap_values_*.csv`           SHAP values per model
# * `shap_summary_*.png`          SHAP summary plots per model
# * `meta_feature_importance.csv` mean |SHAP| per feature, all models merged
#
# **Cell-runnable:** the file is annotated with `# %%` cell markers, so each
# section (config -> helpers -> load -> per-model -> aggregate) can be run
# independently in a Jupyter / VSCode interactive window. Per-model results
# are accumulated in `all_cv_results` / `all_importance` / `final_pipelines`
# dicts keyed by model name; re-running any single model cell updates only
# its entry.

# %% Imports and global config
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed - skipping XGBoost model")

# Feature-set variant ------------------------------------------------------
# Selects which merged meta-feature matrix is read and the suffix added to
# every output file. Set this once at the top of the file (no CLI args, so the
# script stays runnable cell-by-cell).
#   "orig"      -> meta_features.csv           (original metrics)
#   "sw"        -> meta_features_sw.csv         (sliding-window metrics)
#   "drop"      -> meta_features_drop.csv       (short entries dropped)
#   "drop_nowc" -> meta_features_drop_nowc.csv  (drop + word_count removed)
VARIANT = "drop"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"


def variant_name(filename: str) -> str:
    """Insert the variant SUFFIX before a filename's extension.

    e.g. with VARIANT="sw": "meta_cv_results.csv" -> "meta_cv_results_sw.csv".
    With VARIANT="orig" the filename is returned unchanged.
    """
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}" if dot else f"{filename}{SUFFIX}"


# Project paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_CSV = REPO_ROOT / "output" / "merge_features" / f"meta_features{SUFFIX}.csv"
OUTPUT_DIR = REPO_ROOT / "output" / "meta_classifiers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CV configuration ----------------------------------------------------------
SEED = 42
OUTER_CV_FOLDS = 5     # outer loop -> unbiased performance estimate
INNER_CV_FOLDS = 3     # inner loop -> hyperparameter tuning per outer fold

# Metrics that get reported in tab:meta_cv (5results.tex).
SCORING = {
    "roc_auc":   "roc_auc",
    "accuracy":  "accuracy",
    "f1":        "f1",
    "precision": "precision",
    "recall":    "recall",
}

# Features we never train on (they are bookkeeping / metadata):
#   nli_n_sentences / nli_n_pairs - correlate r > 0.88 with sentence_count
#   bleu_participant              - participant-level, not entry-level signal
DROP_FEATURES = ["nli_n_sentences", "nli_n_pairs", "nli_segmentation", "bleu_participant"]

# Containers populated by the per-model cells. Keyed by model display name
# (for cv results) or short suffix (for SHAP / pipelines), so individual
# cells can be re-run without invalidating the others.
all_cv_results: dict[str, dict] = {}
all_importance: dict[str, pd.DataFrame] = {}
final_pipelines: dict[str, Pipeline] = {}


# %% Helper functions
def load_data(path: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load the unified meta-features CSV and split into X, y, groups, columns.

    * X        : feature matrix (entries x features)
    * y        : 1 if the entry is labelled AI, else 0
    * groups   : participant id per entry (used for group-aware CV folds)
    * feature_cols : list of column names corresponding to columns of X
    """
    if path is None:
        path = META_FEATURES_CSV
    df = pd.read_csv(path)

    # Identifier columns that are never features.
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = [c for c in df.columns if c not in id_cols]

    # Drop the bookkeeping / leaky features defined above.
    bookkeeping = [c for c in DROP_FEATURES if c in feature_cols]
    if bookkeeping:
        feature_cols = [c for c in feature_cols if c not in bookkeeping]
        print(f"  Dropped non-feature columns: {bookkeeping}")

    X = df[feature_cols].copy()
    y = (df["final_label"] == "ai").astype(int).to_numpy()
    groups = df["ParticipantID"].to_numpy()

    print(f"Loaded {len(df)} entries, {len(feature_cols)} features")
    print(f"  Class distribution           : ai={int(y.sum())}, "
          f"human={len(y) - int(y.sum())}")
    print(f"  Unique participants (groups) : {len(np.unique(groups))}")
    return X, y, groups, feature_cols


def build_pipeline(estimator, needs_scaling: bool = True) -> Pipeline:
    """Wrap `estimator` in a sklearn Pipeline with median imputation
    (always) and standardisation (only when the estimator needs it).

    LASSO and SVM-style models need scaling; tree-based models don't.
    """
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def _format_params(params: dict) -> str:
    """Pretty-print a sklearn best_params_ dict without the np.float64 noise."""
    items = ", ".join(f"{k}: {float(v):.6g}" for k, v in params.items())
    return "{" + items + "}"


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    outer_cv: StratifiedGroupKFold,
    param_grid: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Run nested CV and return (final_refit_pipeline, summary_dict).

    Steps:
      1. Outer 5-fold CV. If a `param_grid` is given, each outer training
         fold is given to a fresh GridSearchCV that tunes by ROC-AUC over
         a 3-fold inner StratifiedGroupKFold.
      2. The mean and SD of every metric in SCORING are reported across
         the 5 outer folds (these are the table 5 numbers).
      3. The model is then refit on ALL data with one more inner-CV grid
         search; this refit is what we hand to SHAP and to the LASSO /
         tree extractors. We never read performance numbers from it.
    """
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    # 1. Build the per-fold estimator. With a grid, that's a GridSearchCV.
    if param_grid:
        inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
        per_fold_estimator = GridSearchCV(
            pipeline, param_grid,
            scoring="roc_auc",
            cv=inner_cv, refit=True, n_jobs=1,
        )
    else:
        per_fold_estimator = pipeline

    # 2. Outer-CV scoring (the unbiased numbers).
    cv_extra = {"groups": groups} if param_grid else {}
    scores = cross_validate(
        per_fold_estimator, X, y,
        cv=outer_cv, groups=groups,
        scoring=SCORING,
        n_jobs=1,
        return_estimator=True,
        params=cv_extra,
    )

    # Print the metrics: 4 dp matches the precision in tab:meta_cv;
    # per-fold values use 6 dp so small inter-fold differences are visible.
    summary = {}
    for metric in SCORING:
        per_fold = scores[f"test_{metric}"]
        m, s = float(per_fold.mean()), float(per_fold.std())
        summary[f"{metric}_mean"] = m
        summary[f"{metric}_std"] = s
        per_fold_str = ", ".join(f"{v:.6f}" for v in per_fold)
        print(f"  {metric:12s}: {m:.4f} \u00b1 {s:.4f}    "
              f"per-fold: [{per_fold_str}]")

    # If we tuned, show the C / depth / etc. picked in each outer fold.
    if param_grid:
        per_fold_best = [_format_params(est.best_params_) for est in scores["estimator"]]
        print(f"  Inner-CV best params (per fold): [{', '.join(per_fold_best)}]")

    # 3. Final refit on all data (clearly separated from outer-CV scores).
    if param_grid:
        gs_full = GridSearchCV(
            pipeline, param_grid,
            scoring="roc_auc",
            cv=StratifiedGroupKFold(n_splits=INNER_CV_FOLDS),
            refit=True, n_jobs=1,
        )
        gs_full.fit(X, y, groups=groups)
        final_pipeline = gs_full.best_estimator_
        print(f"  Final refit best params: {_format_params(gs_full.best_params_)}")
    else:
        final_pipeline = pipeline
        final_pipeline.fit(X, y)

    return final_pipeline, summary


# ------------------------------------------------------------------
# Per-model export helpers (run after evaluate_model)
# ------------------------------------------------------------------

def extract_lasso_coeffs(pipeline: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """LASSO coefficients with sign + abs-magnitude, sorted by importance."""
    coefs = pipeline.named_steps["clf"].coef_.ravel()
    direction = np.where(coefs > 0, "AI", np.where(coefs < 0, "Human", "Zero"))
    df = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": coefs,
        "abs_coeff":   np.abs(coefs),
        "direction":   direction,
        "selected":    coefs != 0,
    }).sort_values("abs_coeff", ascending=False)

    n_selected = int(df["selected"].sum())
    print(f"  Selected {n_selected}/{len(df)} features")
    print(f"\n  Top 15 features by |coef|:")
    for _, row in df[df["selected"]].head(15).iterrows():
        print(f"    {row['feature']:35s} {row['coefficient']:+.4f} ({row['direction']})")
    return df


def extract_tree_rules(pipeline: Pipeline, feature_cols: list[str]) -> str:
    """Human-readable if/else rules for the fitted DecisionTreeClassifier."""
    return export_text(pipeline.named_steps["clf"],
                       feature_names=feature_cols, max_depth=5)


def extract_ensemble_importances(
    pipeline: Pipeline, feature_cols: list[str], display_name: str,
) -> pd.DataFrame:
    """Top features for a tree ensemble (RF / XGBoost / DT) by built-in
    impurity-based / gain-based importance.

    Returns a DataFrame with columns (feature, importance), sorted desc.
    Also prints a top-15 list and forest/tree-depth statistics so we
    have something quantitative to write about beyond just AUC.
    """
    clf = pipeline.named_steps["clf"]
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Top 15 features by importance ({display_name}):")
    for _, row in importances.head(15).iterrows():
        print(f"    {row['feature']:35s} {row['importance']:.4f}")

    # Ensemble structure stats - useful sanity checks for the paper.
    if hasattr(clf, "estimators_") and isinstance(clf.estimators_, (list, np.ndarray)):
        depths = [t.get_depth() for t in clf.estimators_]
        leaves = [t.get_n_leaves() for t in clf.estimators_]
        print(f"  Forest size: {len(clf.estimators_)} trees, "
              f"mean depth = {np.mean(depths):.1f} "
              f"(min {np.min(depths)}, max {np.max(depths)}), "
              f"mean leaves = {np.mean(leaves):.0f}")
    elif hasattr(clf, "n_estimators"):                  # XGBoost path
        try:
            booster = clf.get_booster()
            n_trees = len(booster.get_dump())
            depth = clf.max_depth if hasattr(clf, "max_depth") else None
            print(f"  Booster: {n_trees} trees, max_depth = {depth}, "
                  f"learning_rate = {clf.learning_rate}")
        except Exception:
            pass

    return importances


def report_cross_model_overlap(
    importance_tables: dict[str, pd.DataFrame], top_k: int = 10,
) -> None:
    """Print how much the top-k features agree across the listed models.

    `importance_tables` maps model name -> a feature/importance frame
    sorted by importance desc (e.g. the LASSO |coef| frame and the
    `feature_importances_` frames from RF/XGBoost). The agreement count
    is the obvious story for the Results section: "model X's top 10
    features overlap N with the LASSO's top 10".
    """
    if not importance_tables:
        return
    top_sets: dict[str, set[str]] = {}
    print("\n" + "=" * 60)
    print(f"  Top-{top_k} feature agreement across models")
    print("=" * 60)
    for name, df in importance_tables.items():
        feat_col = "feature" if "feature" in df.columns else df.columns[0]
        top = df[feat_col].astype(str).head(top_k).tolist()
        top_sets[name] = set(top)
        print(f"\n  {name} top {top_k}:")
        for f in top:
            print(f"    - {f}")

    # Pairwise overlap table.
    names = list(top_sets)
    print(f"\n  Pairwise overlap (|A \u2229 B|, out of {top_k}):")
    header = "  " + " " * 18 + "  ".join(f"{n:>14s}" for n in names)
    print(header)
    for a in names:
        row = [f"{a:18s}"]
        for b in names:
            row.append(f"{len(top_sets[a] & top_sets[b]):>14d}")
        print("  " + "  ".join(row))

    # Features common to ALL models in the dict.
    common_all = set.intersection(*top_sets.values())
    print(f"\n  Common to all {len(top_sets)} models (top {top_k}): "
          f"{len(common_all)} features -> "
          f"{sorted(common_all) if common_all else '(none)'}")


def plot_decision_tree(pipeline: Pipeline, feature_cols: list[str]) -> None:
    """Save a depth-3 visualisation of the decision tree to PNG."""
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        pipeline.named_steps["clf"],
        feature_names=feature_cols, class_names=["Human", "AI"],
        filled=True, rounded=True, max_depth=3, fontsize=9,
        ax=ax, proportion=True, impurity=False,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / variant_name("decision_tree.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved tree diagram -> {out}")


def _transform_X_for_shap(pipeline: Pipeline, X: pd.DataFrame,
                          feature_cols: list[str]) -> pd.DataFrame:
    """Push X through every pipeline step EXCEPT the classifier itself,
    so SHAP sees the same numeric matrix the classifier saw at fit time.
    """
    X_t = X.copy()
    for _, step in pipeline.steps[:-1]:
        X_t = pd.DataFrame(step.transform(X_t), columns=feature_cols)
    return X_t


def _normalise_shap(shap_values, feature_count: int) -> np.ndarray:
    """SHAP returns different shapes for binary classifiers depending on
    the explainer type. Always return the (n_entries, n_features) matrix
    of contributions toward the positive class.
    """
    # LinearExplainer returns a list [neg, pos]; take the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # TreeExplainer for binary models returns (n_entries, n_features, 2).
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    assert shap_values.shape[1] == feature_count
    return shap_values


def compute_shap(name: str, pipeline: Pipeline, X: pd.DataFrame,
                 feature_cols: list[str]) -> pd.DataFrame:
    """Compute SHAP values, save per-entry CSV + summary PNG, and return
    a 2-column DataFrame of (feature, shap_<name>) for cross-model merge.
    """
    clf = pipeline.named_steps["clf"]
    X_transformed = _transform_X_for_shap(pipeline, X, feature_cols)

    # Tree models -> TreeExplainer (exact); linear models -> LinearExplainer.
    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, X_transformed)

    shap_values = _normalise_shap(explainer.shap_values(X_transformed),
                                  feature_count=len(feature_cols))

    # Save per-entry SHAP table.
    pd.DataFrame(shap_values, columns=feature_cols).to_csv(
        OUTPUT_DIR / variant_name(f"shap_values_{name}.csv"), index=False,
    )

    # Mean |SHAP| per feature -> single-column importance frame.
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    importance_df = (mean_abs.sort_values(ascending=False)
                            .rename(f"shap_{name}").reset_index()
                            .rename(columns={"index": "feature"}))

    # Save the SHAP summary beeswarm.
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_cols,
        show=False, max_display=20,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / variant_name(f"shap_summary_{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary -> {out}")

    return importance_df


# %% Load data + define outer CV
# Run this cell once, then any of the per-model cells below.
print(f"Variant: {VARIANT}  ->  reading {META_FEATURES_CSV.name}")
X, y, groups, feature_cols = load_data()
outer_cv = StratifiedGroupKFold(n_splits=OUTER_CV_FOLDS)


# %% Model 1 - LASSO logistic regression
# Coefficient-based interpretability (sign + magnitude per feature).
# class_weight="balanced" because the high-confidence subset is ~75% Human.
lasso_pipe = build_pipeline(
    LogisticRegression(
        penalty="l1", solver="liblinear",
        max_iter=2000, class_weight="balanced",
    ),
    needs_scaling=True,
)
lasso_pipe, lasso_res = evaluate_model(
    "LASSO Logistic Regression",
    lasso_pipe, X, y, groups, outer_cv,
    param_grid={"clf__C": np.logspace(-3, 2, 10)},
)
all_cv_results["LASSO"] = lasso_res
final_pipelines["lasso"] = lasso_pipe

# Save the coefficient table that 5results.tex (line ~242) cites.
coeffs_df = extract_lasso_coeffs(lasso_pipe, feature_cols)
coeffs_df.to_csv(OUTPUT_DIR / variant_name("meta_lasso_coeffs.csv"), index=False)

# SHAP for cross-model importance comparison.
all_importance["lasso"] = compute_shap("lasso", lasso_pipe, X, feature_cols)


# %% Model 2 - Decision Tree
# Explicit if/else rules; tuned over depth and minimum leaf size.
# No scaling: tree splits are scale-invariant.
tree_pipe = build_pipeline(
    DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    needs_scaling=False,
)
tree_pipe, tree_res = evaluate_model(
    "Decision Tree",
    tree_pipe, X, y, groups, outer_cv,
    param_grid={
        "clf__max_depth":        [3, 4, 5, 6, 7],
        "clf__min_samples_leaf": [10, 20, 30, 50],
    },
)
all_cv_results["Decision Tree"] = tree_res
final_pipelines["tree"] = tree_pipe

# Save the rules and a depth-3 diagram.
rules = extract_tree_rules(tree_pipe, feature_cols)
(OUTPUT_DIR / variant_name("meta_tree_rules.txt")).write_text(rules)
print(f"  Saved tree rules -> {variant_name('meta_tree_rules.txt')}")
print(f"\n  Tree rules (depth <= 5):\n{rules[:2000]}")
plot_decision_tree(tree_pipe, feature_cols)

all_importance["tree"] = compute_shap("tree", tree_pipe, X, feature_cols)


# %% Model 3 - Random Forest
# Non-linear ensemble baseline; tuned over depth and minimum leaf size.
rf_pipe = build_pipeline(
    RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=SEED, n_jobs=1,
    ),
    needs_scaling=False,
)
rf_pipe, rf_res = evaluate_model(
    "Random Forest",
    rf_pipe, X, y, groups, outer_cv,
    param_grid={
        "clf__max_depth":        [5, 10, 15, None],
        "clf__min_samples_leaf": [5, 10, 20],
    },
)
all_cv_results["Random Forest"] = rf_res
final_pipelines["rf"] = rf_pipe

# Built-in feature_importances_ (Gini): saved separately so we can write
# the Results paragraph about RF without having to re-derive importances.
rf_importances = extract_ensemble_importances(rf_pipe, feature_cols, "Random Forest")
rf_importances.to_csv(OUTPUT_DIR / variant_name("meta_rf_feature_importance.csv"), index=False)

all_importance["rf"] = compute_shap("rf", rf_pipe, X, feature_cols)


# %% Model 4 - XGBoost
# Gradient-boosted ensemble baseline. Skipped if xgboost is not installed.
# scale_pos_weight = (#human / #ai) is XGBoost's equivalent of "balanced".
if HAS_XGB:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    xgb_pipe = build_pipeline(
        XGBClassifier(
            n_estimators=200,
            scale_pos_weight=n_neg / n_pos,
            random_state=SEED,
            eval_metric="logloss",
        ),
        needs_scaling=False,
    )
    xgb_pipe, xgb_res = evaluate_model(
        "XGBoost",
        xgb_pipe, X, y, groups, outer_cv,
        param_grid={
            "clf__max_depth":     [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2],
        },
    )
    all_cv_results["XGBoost"] = xgb_res
    final_pipelines["xgb"] = xgb_pipe

    # Built-in feature_importances_ (gain): saved separately for the
    # XGBoost paragraph in the Results section.
    xgb_importances = extract_ensemble_importances(xgb_pipe, feature_cols, "XGBoost")
    xgb_importances.to_csv(OUTPUT_DIR / variant_name("meta_xgb_feature_importance.csv"), index=False)

    all_importance["xgb"] = compute_shap("xgb", xgb_pipe, X, feature_cols)


# %% Aggregate + save combined results
# This cell takes whatever is currently in `all_cv_results` and
# `all_importance`, so it works after a partial run too (e.g. you re-ran
# only the Decision Tree cell).

cv_df = pd.DataFrame(
    [{"model": name, **summary} for name, summary in all_cv_results.items()]
)
cv_df.to_csv(OUTPUT_DIR / variant_name("meta_cv_results.csv"), index=False)

# Pretty-print the summary table with the precision the paper uses in
# tab:meta_cv: AUC at 4 dp, the rest at 3 dp, all with mean +/- SD.
def _mean_sd(row, key, dp):
    return f"{row[f'{key}_mean']:.{dp}f} \u00b1 {row[f'{key}_std']:.{dp}f}"

display_df = pd.DataFrame({
    "model":     cv_df["model"],
    "AUC":       cv_df.apply(lambda r: _mean_sd(r, "roc_auc",   4), axis=1),
    "Accuracy":  cv_df.apply(lambda r: _mean_sd(r, "accuracy",  3), axis=1),
    "F1":        cv_df.apply(lambda r: _mean_sd(r, "f1",        3), axis=1),
    "Precision": cv_df.apply(lambda r: _mean_sd(r, "precision", 3), axis=1),
    "Recall":    cv_df.apply(lambda r: _mean_sd(r, "recall",    3), axis=1),
})

print(f"\n{'='*72}")
print("Group-aware nested cross-validation results (all models)")
print(f"{'='*72}")
print(display_df.to_string(index=False))

# Combine SHAP importance from every model into a single feature-keyed
# table. Outer-merge so a feature missing in one model still appears.
if all_importance:
    importance_df = None
    for imp in all_importance.values():
        importance_df = imp if importance_df is None \
                            else importance_df.merge(imp, on="feature", how="outer")
    importance_df = importance_df.sort_values(
        by=[c for c in importance_df.columns if c.startswith("shap_")],
        ascending=False,
    )
    importance_df.to_csv(OUTPUT_DIR / variant_name("meta_feature_importance.csv"), index=False)

# Cross-model top-feature agreement: this is the actual reason we run
# multiple models alongside the LASSO. If LASSO, RF, and XGBoost (with
# very different inductive biases) agree on a small set of top features,
# the convergence is evidence that those features are doing real work
# rather than artefacts of any single model class.
overlap_tables: dict[str, pd.DataFrame] = {}
lasso_coef_path = OUTPUT_DIR / variant_name("meta_lasso_coeffs.csv")
if lasso_coef_path.exists():
    lasso_coef = pd.read_csv(lasso_coef_path)
    # Match the column convention used elsewhere: feature, importance.
    overlap_tables["LASSO (|coef|)"] = lasso_coef[["feature", "abs_coeff"]].rename(
        columns={"abs_coeff": "importance"}
    )
rf_imp_path = OUTPUT_DIR / variant_name("meta_rf_feature_importance.csv")
if rf_imp_path.exists():
    overlap_tables["Random Forest (Gini)"] = pd.read_csv(rf_imp_path)
xgb_imp_path = OUTPUT_DIR / variant_name("meta_xgb_feature_importance.csv")
if xgb_imp_path.exists():
    overlap_tables["XGBoost (gain)"] = pd.read_csv(xgb_imp_path)
if len(overlap_tables) >= 2:
    report_cross_model_overlap(overlap_tables, top_k=10)

print(f"\n\u2713 All outputs saved to {OUTPUT_DIR}")


# %% Script entry point (no-op when run cell-by-cell)
if __name__ == "__main__" and "__file__" in globals():
    # If you run this file from the command line, all the cells above
    # will already have executed top-to-bottom; nothing left to do.
    pass
