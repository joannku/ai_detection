#!/usr/bin/env python3
"""
Interpretable meta-classifiers on the unified feature set.

Models:
  1. LASSO logistic regression (coefficient-based interpretability)
  2. Decision Tree (explicit rule-based interpretability)
  3. Random Forest (ensemble comparison + feature importance)
  4. XGBoost (ensemble comparison + feature importance)

Uses proper nested cross-validation: hyperparameters are tuned in an
inner loop (3-fold) within each outer evaluation fold (5-fold), so
reported scores are unbiased estimates of generalisation performance.

A final model is then refit on all data for SHAP / interpretability
exports (clearly separated from the unbiased CV scores).

Outputs (output/):
  - meta_cv_results.csv         (cross-validated performance for all models)
  - meta_lasso_coeffs.csv       (LASSO coefficients)
  - meta_tree_rules.txt         (Decision Tree rules in text form)
  - meta_feature_importance.csv (feature importances across models)
  - shap_values_*.csv           (SHAP values per model)
  - shap_summary_*.png          (SHAP summary plots per model)
  - decision_tree.png           (tree diagram)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, roc_auc_score
import shap

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost model")

_REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_CSV = _REPO_ROOT / "output" / "merge_features" / "meta_features.csv"
REVIEWS_OUTPUT = _REPO_ROOT / "output" / "meta_classifiers"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
SEED = 42
CV_FOLDS = 5
INNER_CV_FOLDS = 3
SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}

DROP_FEATURES = ["nli_n_sentences", "nli_n_pairs", "bleu_participant"]


def load_data(path=None):
    if path is None:
        path = META_FEATURES_CSV
    df = pd.read_csv(path)

    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = [c for c in df.columns if c not in id_cols]

    drop = [c for c in DROP_FEATURES if c in feature_cols]
    if drop:
        feature_cols = [c for c in feature_cols if c not in drop]
        print(f"  Dropped features: {drop}")

    X = df[feature_cols].copy()
    y = (df["final_label"] == "ai").astype(int).values
    groups = df["ParticipantID"].values

    print(f"Loaded {len(df)} entries, {len(feature_cols)} features")
    print(f"  Class distribution: ai={y.sum()}, human={len(y)-y.sum()}")
    print(f"  Unique participants (groups): {len(np.unique(groups))}")
    return X, y, groups, feature_cols


def build_pipeline(estimator, needs_scaling=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def evaluate_model(name, pipeline, X, y, groups, outer_cv, param_grid=None):
    """Nested CV: inner loop tunes hyperparameters, outer loop evaluates."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if param_grid:
        inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
        estimator = GridSearchCV(pipeline, param_grid, scoring="roc_auc",
                                 cv=inner_cv, refit=True, n_jobs=1)
    else:
        estimator = pipeline

    cv_params = {"groups": groups} if param_grid else {}
    scores = cross_validate(estimator, X, y, cv=outer_cv, scoring=SCORING,
                            n_jobs=1, return_estimator=True, groups=groups,
                            params=cv_params)

    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    results = {}
    for metric in SCORING:
        key = f"test_{metric}"
        m, s = scores[key].mean(), scores[key].std()
        results[f"{metric}_mean"] = m
        results[f"{metric}_std"] = s
        print(f"  {metric:12s}: {fmt(m, s)}")

    if param_grid:
        best_params = [e.best_params_ for e in scores["estimator"]]
        print(f"  Inner-CV best params (per fold): {best_params}")

    # Refit final model on ALL data for interpretability exports
    if param_grid:
        inner_cv_full = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
        gs_full = GridSearchCV(pipeline, param_grid, scoring="roc_auc",
                               cv=inner_cv_full, refit=True, n_jobs=1)
        gs_full.fit(X, y, groups=groups)
        final_pipeline = gs_full.best_estimator_
        print(f"  Final refit best params: {gs_full.best_params_}")
    else:
        final_pipeline = pipeline
        final_pipeline.fit(X, y)

    return final_pipeline, results


def extract_lasso_coeffs(pipeline, feature_cols):
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_.ravel()
    df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coeff": np.abs(coefs),
        "direction": np.where(coefs > 0, "AI", np.where(coefs < 0, "Human", "Zero")),
        "selected": coefs != 0,
    }).sort_values("abs_coeff", ascending=False)

    n_sel = df["selected"].sum()
    print(f"  Selected {n_sel}/{len(df)} features")
    print(f"\n  Top 15 features:")
    for _, row in df[df["selected"]].head(15).iterrows():
        print(f"    {row['feature']:35s} {row['coefficient']:+.4f} ({row['direction']})")

    return df


def extract_tree_rules(pipeline, feature_cols):
    clf = pipeline.named_steps["clf"]
    rules = export_text(clf, feature_names=feature_cols, max_depth=5)
    return rules


def plot_decision_tree(pipeline, feature_cols):
    clf = pipeline.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(clf, feature_names=feature_cols, class_names=["Human", "AI"],
              filled=True, rounded=True, max_depth=3, fontsize=9, ax=ax,
              proportion=True, impurity=False)
    plt.tight_layout()
    out = REVIEWS_OUTPUT / "decision_tree.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved tree diagram → {out}")


def compute_shap(name, pipeline, X, feature_cols):
    """Compute SHAP values and save summary plot + values."""
    clf = pipeline.named_steps["clf"]

    # Transform X through preprocessing steps
    X_transformed = X.copy()
    for step_name, step in pipeline.steps[:-1]:
        X_transformed = pd.DataFrame(
            step.transform(X_transformed), columns=feature_cols
        )

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, X_transformed)

    shap_values = explainer.shap_values(X_transformed)

    # For binary classifiers that return 2D array, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_csv(REVIEWS_OUTPUT / f"shap_values_{name}.csv", index=False)

    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    importance_df = pd.DataFrame({
        "feature": mean_abs.index,
        f"shap_{name}": mean_abs.values,
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    out = REVIEWS_OUTPUT / f"shap_summary_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary → {out}")

    return importance_df


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
    X, y, groups, feature_cols = load_data()
    outer_cv = StratifiedGroupKFold(n_splits=CV_FOLDS)

    all_cv_results = []
    all_importance = None

    # ── 1. LASSO Logistic Regression ────────────────────────────────
    lasso_pipe = build_pipeline(
        LogisticRegression(penalty="l1", solver="liblinear",
                           max_iter=2000, class_weight="balanced"),
        needs_scaling=True,
    )
    lasso_pipe, lasso_res = evaluate_model(
        "LASSO Logistic Regression", lasso_pipe, X, y, groups, outer_cv,
        param_grid={"clf__C": np.logspace(-3, 2, 10)},
    )
    all_cv_results.append({"model": "LASSO", **lasso_res})

    coeffs_df = extract_lasso_coeffs(lasso_pipe, feature_cols)
    coeffs_df.to_csv(REVIEWS_OUTPUT / "meta_lasso_coeffs.csv", index=False)

    lasso_imp = compute_shap("lasso", lasso_pipe, X, feature_cols)
    all_importance = lasso_imp

    # ── 2. Decision Tree ────────────────────────────────────────────
    tree_pipe = build_pipeline(
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        needs_scaling=False,
    )
    tree_pipe, tree_res = evaluate_model(
        "Decision Tree", tree_pipe, X, y, groups, outer_cv,
        param_grid={
            "clf__max_depth": [3, 4, 5, 6, 7],
            "clf__min_samples_leaf": [10, 20, 30, 50],
        },
    )
    all_cv_results.append({"model": "Decision Tree", **tree_res})

    rules = extract_tree_rules(tree_pipe, feature_cols)
    (REVIEWS_OUTPUT / "meta_tree_rules.txt").write_text(rules)
    print(f"  Saved tree rules → meta_tree_rules.txt")
    print(f"\n  Tree rules (depth ≤ 5):\n{rules[:2000]}")

    plot_decision_tree(tree_pipe, feature_cols)

    tree_imp = compute_shap("tree", tree_pipe, X, feature_cols)
    all_importance = all_importance.merge(tree_imp, on="feature", how="outer")

    # ── 3. Random Forest ────────────────────────────────────────────
    rf_pipe = build_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=SEED, n_jobs=1),
        needs_scaling=False,
    )
    rf_pipe, rf_res = evaluate_model(
        "Random Forest", rf_pipe, X, y, groups, outer_cv,
        param_grid={
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_leaf": [5, 10, 20],
        },
    )
    all_cv_results.append({"model": "Random Forest", **rf_res})

    rf_imp = compute_shap("rf", rf_pipe, X, feature_cols)
    all_importance = all_importance.merge(rf_imp, on="feature", how="outer")

    # ── 4. XGBoost ──────────────────────────────────────────────────
    if HAS_XGB:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        xgb_pipe = build_pipeline(
            XGBClassifier(n_estimators=200, scale_pos_weight=n_neg / n_pos,
                          random_state=SEED, eval_metric="logloss",
                          use_label_encoder=False),
            needs_scaling=False,
        )
        xgb_pipe, xgb_res = evaluate_model(
            "XGBoost", xgb_pipe, X, y, groups, outer_cv,
            param_grid={
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.1, 0.2],
            },
        )
        all_cv_results.append({"model": "XGBoost", **xgb_res})

        xgb_imp = compute_shap("xgb", xgb_pipe, X, feature_cols)
        all_importance = all_importance.merge(xgb_imp, on="feature", how="outer")

    # ── Save combined results ───────────────────────────────────────
    cv_df = pd.DataFrame(all_cv_results)
    cv_df.to_csv(REVIEWS_OUTPUT / "meta_cv_results.csv", index=False)
    print(f"\n{'='*60}")
    print("Group-aware nested cross-validation results (all models)")
    print(f"{'='*60}")
    print(cv_df[["model", "roc_auc_mean", "accuracy_mean", "f1_mean"]].to_string(index=False))

    all_importance = all_importance.sort_values(
        by=[c for c in all_importance.columns if c.startswith("shap_")],
        ascending=False,
    )
    all_importance.to_csv(REVIEWS_OUTPUT / "meta_feature_importance.csv", index=False)
    print(f"\n✓ All outputs saved to {REVIEWS_OUTPUT}")


if __name__ == "__main__":
    main()
