#!/usr/bin/env python3
"""
Run nested-CV classifiers on the FULL dataset (2,466 entries) for comparison
against the high-confidence subset results.

Outputs go to output/ with a "full_" prefix.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Feature-set variant: "orig" | "sw" | "drop" | "drop_nowc". Selects the full-dataset merged
# matrix and suffixes the CV results output.
VARIANT = "drop"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"


def variant_name(filename: str) -> str:
    """Insert the variant SUFFIX before a filename's extension."""
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}" if dot else f"{filename}{SUFFIX}"


_REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_FULL_CSV = _REPO_ROOT / "output" / "merge_features_full" / f"meta_features_full{SUFFIX}.csv"
REVIEWS_OUTPUT = _REPO_ROOT / "output" / "run_full_dataset"
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

DROP_FEATURES = ["nli_n_sentences", "nli_n_pairs", "nli_segmentation", "bleu_participant"]


def load_data():
    path = META_FEATURES_FULL_CSV
    df = pd.read_csv(path)
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    drop = [c for c in DROP_FEATURES if c in feature_cols]
    if drop:
        feature_cols = [c for c in feature_cols if c not in drop]
        print(f"  Dropped redundant features: {drop}")
    X = df[feature_cols].copy()
    y = (df["final_label"] == "ai").astype(int).values
    groups = df["ParticipantID"].values
    print(f"Loaded {len(df)} entries, {len(feature_cols)} features")
    print(f"  Unique participants: {len(np.unique(groups))}")
    print(f"  Class distribution: ai={y.sum()}, human={len(y)-y.sum()}")
    return X, y, groups, feature_cols


def build_pipeline(estimator, needs_scaling=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def evaluate(name, pipeline, X, y, groups, outer_cv, param_grid):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
    estimator = GridSearchCV(pipeline, param_grid, scoring="roc_auc",
                             cv=inner_cv, refit=True, n_jobs=1)

    scores = cross_validate(estimator, X, y, cv=outer_cv, scoring=SCORING,
                            groups=groups, n_jobs=1, return_estimator=True,
                            params={"groups": groups})

    results = {"model": name}
    for metric in SCORING:
        key = f"test_{metric}"
        m, s = scores[key].mean(), scores[key].std()
        results[f"{metric}_mean"] = m
        results[f"{metric}_std"] = s
        print(f"  {metric:12s}: {m:.3f} ± {s:.3f}")

    best_params = [e.best_params_ for e in scores["estimator"]]
    print(f"  Inner-CV best params (per fold): {best_params}")
    return results


def main():
    print(f"Variant: {VARIANT}  ->  reading {META_FEATURES_FULL_CSV.name}")
    X, y, groups, feature_cols = load_data()
    outer_cv = StratifiedGroupKFold(n_splits=CV_FOLDS)

    all_results = []

    all_results.append(evaluate(
        "LASSO", build_pipeline(
            LogisticRegression(penalty="l1", solver="liblinear",
                               max_iter=2000, class_weight="balanced"),
            needs_scaling=True),
        X, y, groups, outer_cv,
        param_grid={"clf__C": np.logspace(-3, 2, 10)},
    ))

    all_results.append(evaluate(
        "Decision Tree", build_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            needs_scaling=False),
        X, y, groups, outer_cv,
        param_grid={
            "clf__max_depth": [3, 4, 5, 6, 7],
            "clf__min_samples_leaf": [10, 20, 30, 50],
        },
    ))

    all_results.append(evaluate(
        "Random Forest", build_pipeline(
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=SEED, n_jobs=1),
            needs_scaling=False),
        X, y, groups, outer_cv,
        param_grid={
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_leaf": [5, 10, 20],
        },
    ))

    if HAS_XGB:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        all_results.append(evaluate(
            "XGBoost", build_pipeline(
                XGBClassifier(n_estimators=200, scale_pos_weight=n_neg / n_pos,
                              random_state=SEED, eval_metric="logloss"),
                needs_scaling=False),
            X, y, groups, outer_cv,
            param_grid={
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.1, 0.2],
            },
        ))

    all_results.append(evaluate(
        "Linear SVM", build_pipeline(
            SVC(kernel="linear", class_weight="balanced", random_state=SEED,
                probability=True),
            needs_scaling=True),
        X, y, groups, outer_cv,
        param_grid={"clf__C": np.logspace(-3, 2, 10)},
    ))

    cv_df = pd.DataFrame(all_results)
    out_path = REVIEWS_OUTPUT / variant_name("full_cv_results.csv")
    cv_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("Full-dataset group-aware nested CV results (all models)")
    print(f"{'='*60}")
    print(cv_df[["model", "roc_auc_mean", "accuracy_mean", "f1_mean"]].to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
