#!/usr/bin/env python3
"""
Linear SVM meta-classifier with margin-based flagging.

Uses proper nested cross-validation: hyperparameters are tuned in an
inner loop (3-fold) within each outer evaluation fold (5-fold), so
reported scores are unbiased estimates of generalisation performance.

A final model is then refit on all data for interpretability exports
(coefficients, margin flags, SHAP), clearly separated from CV scores.

Outputs (output/):
  - svm_cv_results.csv           (cross-validated performance)
  - svm_coefficients.csv         (feature weights from linear SVM)
  - svm_margin_flags.csv         (all entries with distance-to-boundary)
  - svm_margin_histogram.png     (distribution of distances)
  - svm_boundary_pca.png         (decision boundary in PCA space)
  - svm_boundary_pairs.png       (decision boundaries in 2D feature pairs)
  - shap_values_svm.csv          (SHAP values)
  - shap_summary_svm.png         (SHAP summary plot)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
import shap

_REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_CSV = _REPO_ROOT / "output" / "merge_features" / "meta_features.csv"
META_CLASSIFIERS_DIR = _REPO_ROOT / "output" / "meta_classifiers"
REVIEWS_OUTPUT = _REPO_ROOT / "output" / "meta_svm"
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

FEATURE_DISPLAY_NAMES = {
    "coherence": "Coherence",
    "style_similarity": "Style Similarity",
    "read_flesch_reading_ease": "Flesch Reading Ease",
    "read_flesch_kincaid_grade": "FK Grade Level",
    "read_coleman_liau_index": "Coleman-Liau Index",
    "read_gunning_fog": "Gunning Fog",
    "read_syllable_count": "Syllable Count",
    "nli_consistency_score": "NLI Consistency",
    "nli_entailment_rate": "Entailment Rate",
    "nli_neutral_rate": "Neutral Rate",
    "word_count": "Word Count",
    "liwc_Dic": "LIWC Dictionary",
    "liwc_article": "LIWC Articles",
    "liwc_ppron": "LIWC Personal Pronouns",
    "liwc_focuspresent": "LIWC Focus Present",
}

BOUNDARY_PAIRS = [
    ("coherence", "style_similarity"),
    ("coherence", "liwc_Dic"),
    ("read_coleman_liau_index", "liwc_article"),
    ("word_count", "liwc_Dic"),
]


def display_name(feat):
    return FEATURE_DISPLAY_NAMES.get(feat, feat)


def load_data(path=None):
    if path is None:
        path = META_FEATURES_CSV
    df = pd.read_csv(path)
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = [c for c in df.columns if c not in id_cols]

    drop = [c for c in DROP_FEATURES if c in feature_cols]
    if drop:
        feature_cols = [c for c in feature_cols if c not in drop]
        print(f"  Dropped redundant features: {drop}")

    X = df[feature_cols].copy()
    y = (df["final_label"] == "ai").astype(int).values
    print(f"Loaded {len(df)} entries, {len(feature_cols)} features")
    print(f"  Class distribution: ai={y.sum()}, human={len(y)-y.sum()}")
    groups = df["ParticipantID"].values
    print(f"  Unique participants: {len(np.unique(groups))}")
    return df, X, y, groups, feature_cols


def build_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


def train_and_evaluate(X, y, groups, outer_cv):
    """Nested CV: inner loop tunes C, outer loop evaluates unbiasedly."""
    print(f"\n{'='*60}")
    print("  Linear SVM (nested CV)")
    print(f"{'='*60}")

    svm = SVC(kernel="linear", class_weight="balanced", random_state=SEED,
              probability=True)
    pipe = build_pipeline(svm)

    param_grid = {"clf__C": np.logspace(-3, 2, 10)}
    inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
    estimator = GridSearchCV(pipe, param_grid, scoring="roc_auc",
                             cv=inner_cv, refit=True, n_jobs=1)

    scores = cross_validate(estimator, X, y, cv=outer_cv, scoring=SCORING,
                            n_jobs=1, return_estimator=True, groups=groups,
                            params={"groups": groups})

    results = {}
    for metric in SCORING:
        key = f"test_{metric}"
        m, s = scores[key].mean(), scores[key].std()
        results[f"{metric}_mean"] = m
        results[f"{metric}_std"] = s
        print(f"  {metric:12s}: {m:.3f} ± {s:.3f}")

    best_params = [e.best_params_ for e in scores["estimator"]]
    print(f"  Inner-CV best params (per fold): {best_params}")

    # Refit final model on ALL data for interpretability exports
    inner_cv_full = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
    gs_full = GridSearchCV(pipe, param_grid, scoring="roc_auc",
                           cv=inner_cv_full, refit=True, n_jobs=1)
    gs_full.fit(X, y, groups=groups)
    final_pipe = gs_full.best_estimator_
    print(f"  Final refit best C: {gs_full.best_params_['clf__C']:.4f}")

    return final_pipe, results


def extract_coefficients(pipeline, feature_cols):
    """Extract and rank linear SVM coefficients (analogous to LASSO)."""
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_.ravel()
    df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coeff": np.abs(coefs),
        "direction": np.where(coefs > 0, "AI", np.where(coefs < 0, "Human", "Zero")),
    }).sort_values("abs_coeff", ascending=False)

    print(f"\n  Top 15 SVM coefficients:")
    for _, row in df.head(15).iterrows():
        print(f"    {row['feature']:35s} {row['coefficient']:+.4f} ({row['direction']})")

    return df


def compute_margin_flags(pipeline, X, y, df_full):
    """
    Compute signed distance to the decision boundary for every entry.
    Entries within ±1 (inside the margin) are flagged for human review.
    """
    X_transformed = X.copy()
    for step_name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)

    clf = pipeline.named_steps["clf"]
    distances = clf.decision_function(X_transformed)

    flags = pd.DataFrame({
        "JournalUniqueID": df_full["JournalUniqueID"].values,
        "final_label": df_full["final_label"].values,
        "svm_distance": distances,
        "svm_prediction": np.where(distances >= 0, "ai", "human"),
        "in_margin": np.abs(distances) <= 1.0,
        "flag_for_review": np.abs(distances) <= 1.0,
    })

    n_margin = flags["in_margin"].sum()
    n_correct = (flags["svm_prediction"] == flags["final_label"]).sum()

    print(f"\n  Margin analysis:")
    print(f"    Total entries:        {len(flags)}")
    print(f"    In margin (|d| ≤ 1):  {n_margin} ({n_margin/len(flags):.1%})")
    print(f"    Outside margin:       {len(flags) - n_margin}")
    print(f"    Overall accuracy:     {n_correct/len(flags):.3f}")

    margin_entries = flags[flags["in_margin"]]
    if len(margin_entries) > 0:
        label_dist = margin_entries["final_label"].value_counts()
        print(f"    Margin breakdown:     {dict(label_dist)}")

    return flags


def plot_margin_histogram(flags):
    """Histogram of decision-function distances with margin zone highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))

    human = flags[flags["final_label"] == "human"]["svm_distance"]
    ai = flags[flags["final_label"] == "ai"]["svm_distance"]

    ax.hist(human, bins=60, alpha=0.6, color="#4C9BE8", label="Human-labelled",
            density=True)
    ax.hist(ai, bins=60, alpha=0.6, color="#E85C5C", label="AI-labelled",
            density=True)

    ax.axvline(0, color="black", linewidth=2, linestyle="--", label="Decision boundary")
    ax.axvspan(-1, 1, alpha=0.12, color="orange", label="Margin zone (flag for review)")

    ax.set_xlabel("Signed distance to decision boundary", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("SVM Decision Function: Distance Distribution", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = REVIEWS_OUTPUT / "svm_margin_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved margin histogram → {out}")

    paper_figs = _REPO_ROOT / "overleaf_paper" / "figures"
    if paper_figs.exists():
        import shutil
        dst = paper_figs / "svm_margin_histogram.png"
        shutil.copyfile(out, dst)
        print(f"  Copied → {dst}")


def plot_boundary_pca(pipeline, X, y, feature_cols):
    """Decision boundary in PCA space, coloured by SVM distance."""
    X_transformed = X.copy()
    for step_name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_transformed)

    clf = pipeline.named_steps["clf"]

    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 300),
        np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 300),
    )
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_orig = pca.inverse_transform(grid_pca)
    Z = clf.decision_function(grid_orig).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50),
                cmap="RdBu_r", alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2,
               linestyles="--")
    ax.contour(xx, yy, Z, levels=[-1, 1], colors="grey", linewidths=1,
               linestyles=":")

    human_mask = y == 0
    ax.scatter(X_pca[human_mask, 0], X_pca[human_mask, 1],
               c="#4C9BE8", s=12, alpha=0.4, label="Human", edgecolors="none")
    ax.scatter(X_pca[~human_mask, 0], X_pca[~human_mask, 1],
               c="#E85C5C", s=12, alpha=0.4, label="AI", edgecolors="none")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=11)
    ax.set_title("SVM Decision Boundary in PCA Space\n"
                 "(solid = boundary, dotted = margin)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out = REVIEWS_OUTPUT / "svm_boundary_pca.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SVM PCA boundary → {out}")


def plot_boundary_pairs(X, y):
    """Decision boundaries for 2D feature pairs using a local linear SVM."""
    pairs = [(a, b) for a, b in BOUNDARY_PAIRS if a in X.columns and b in X.columns]
    if not pairs:
        print("  No valid feature pairs found, skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (f1, f2) in enumerate(pairs):
        ax = axes[idx]
        X_pair = X[[f1, f2]].copy()
        mask = X_pair.notna().all(axis=1)
        X_pair = X_pair[mask].values
        y_pair = y[mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pair)

        clf = SVC(kernel="linear", C=1.0, class_weight="balanced")
        clf.fit(X_scaled, y_pair)

        x_min, x_max = X_pair[:, 0].min() - 0.5, X_pair[:, 0].max() + 0.5
        y_min, y_max = X_pair[:, 1].min() - 0.5, X_pair[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                              np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)
        Z = clf.decision_function(grid_scaled).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50),
                     cmap="RdBu_r", alpha=0.25)
        ax.contour(xx, yy, Z, levels=[0], colors="black",
                    linewidths=1.5, linestyles="--")
        ax.contour(xx, yy, Z, levels=[-1, 1], colors="grey",
                    linewidths=1, linestyles=":")

        human_mask = y_pair == 0
        ax.scatter(X_pair[human_mask, 0], X_pair[human_mask, 1],
                   c="#4C9BE8", s=8, alpha=0.35, label="Human", edgecolors="none")
        ax.scatter(X_pair[~human_mask, 0], X_pair[~human_mask, 1],
                   c="#E85C5C", s=8, alpha=0.35, label="AI", edgecolors="none")

        ax.set_xlabel(display_name(f1), fontsize=10)
        ax.set_ylabel(display_name(f2), fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=8)

    for j in range(len(pairs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("SVM Decision Boundaries (2D Feature Pairs)\n"
                 "Solid = boundary, dotted = margin",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = REVIEWS_OUTPUT / "svm_boundary_pairs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SVM pair boundaries → {out}")


def compute_shap(pipeline, X, feature_cols):
    """Compute SHAP values using LinearExplainer."""
    clf = pipeline.named_steps["clf"]

    X_transformed = X.copy()
    for step_name, step in pipeline.steps[:-1]:
        X_transformed = pd.DataFrame(
            step.transform(X_transformed), columns=feature_cols
        )

    explainer = shap.LinearExplainer(clf, X_transformed)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_csv(REVIEWS_OUTPUT / "shap_values_svm.csv", index=False)

    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    importance_df = pd.DataFrame({
        "feature": mean_abs.index,
        "shap_svm": mean_abs.values,
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    out = REVIEWS_OUTPUT / "shap_summary_svm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary → {out}")

    print(f"\n  Top 15 features by SHAP:")
    for _, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:35s} {row['shap_svm']:.4f}")

    return importance_df


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
    df_full, X, y, groups, feature_cols = load_data()
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS)

    # Train and evaluate
    pipeline, cv_results = train_and_evaluate(X, y, groups, cv)

    cv_df = pd.DataFrame([{"model": "Linear SVM", **cv_results}])
    cv_df.to_csv(REVIEWS_OUTPUT / "svm_cv_results.csv", index=False)
    print(f"\n  Saved CV results → svm_cv_results.csv")

    # Feature coefficients
    coeff_df = extract_coefficients(pipeline, feature_cols)
    coeff_df.to_csv(REVIEWS_OUTPUT / "svm_coefficients.csv", index=False)
    print(f"  Saved coefficients → svm_coefficients.csv")

    # Margin-based flagging
    flags = compute_margin_flags(pipeline, X, y, df_full)
    flags.to_csv(REVIEWS_OUTPUT / "svm_margin_flags.csv", index=False)
    print(f"  Saved margin flags → svm_margin_flags.csv")

    # Visualizations
    print(f"\n{'='*60}")
    print("  Visualizations")
    print(f"{'='*60}")
    plot_margin_histogram(flags)
    plot_boundary_pca(pipeline, X, y, feature_cols)
    plot_boundary_pairs(X, y)

    # SHAP
    print(f"\n{'='*60}")
    print("  SHAP Analysis")
    print(f"{'='*60}")
    shap_imp = compute_shap(pipeline, X, feature_cols)
    shap_imp.to_csv(REVIEWS_OUTPUT / "svm_shap_importance.csv", index=False)

    # Append SVM SHAP to the combined feature importance file (lives next to
    # 21_meta_classifiers' outputs, since that script created the original file).
    combined_path = META_CLASSIFIERS_DIR / "meta_feature_importance.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        if "shap_svm" in combined.columns:
            combined = combined.drop(columns=["shap_svm"])
        combined = combined.merge(shap_imp, on="feature", how="outer")
        combined.to_csv(combined_path, index=False)
        print(f"  Updated combined feature importance with SVM SHAP")

    # Append to combined CV results (lives next to 21_meta_classifiers' outputs).
    combined_cv_path = META_CLASSIFIERS_DIR / "meta_cv_results.csv"
    if combined_cv_path.exists():
        existing = pd.read_csv(combined_cv_path)
        existing = existing[existing["model"] != "Linear SVM"]
        combined_cv = pd.concat([existing, cv_df], ignore_index=True)
        combined_cv.to_csv(combined_cv_path, index=False)
        print(f"  Updated combined CV results with SVM")

    # Pretty-print the SVM row in the same precision as tab:meta_cv in
    # 5results.tex (AUC at 4 dp, the rest at 3 dp). Copy-paste straight
    # into the paper table.
    r = cv_df.iloc[0]
    def _ms(key, dp):
        return f"{r[f'{key}_mean']:.{dp}f} \u00b1 {r[f'{key}_std']:.{dp}f}"
    row_label = "Linear SVM"
    cells = [
        row_label,
        _ms("roc_auc",   4),
        _ms("accuracy",  3),
        _ms("f1",        3),
        _ms("precision", 3),
        _ms("recall",    3),
    ]
    print(f"\n{'='*72}")
    print("Linear SVM row (paste into Table 5)")
    print(f"{'='*72}")
    headers = ["Model", "AUC", "Accuracy", "F1", "Precision", "Recall"]
    widths = [max(len(h), len(c)) for h, c in zip(headers, cells)]
    print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("  " + "  ".join(f"{c:<{w}}" for c, w in zip(cells, widths)))
    # Markdown table row, ready to drop into a Google Doc / Markdown paper.
    print("\n  Markdown:")
    print("  | " + " | ".join(cells) + " |")
    # LaTeX table row, ready to drop into 5results.tex tab:meta_cv (.4f / .3f
    # styling matches the existing rows in the file).
    latex_cells = [
        row_label,
        f"{r['roc_auc_mean']:.4f} $\\pm$ {r['roc_auc_std']:.4f}",
        f"{r['accuracy_mean']:.3f} $\\pm$ {r['accuracy_std']:.3f}",
        f"{r['f1_mean']:.3f} $\\pm$ {r['f1_std']:.3f}",
        f"{r['precision_mean']:.3f} $\\pm$ {r['precision_std']:.3f}",
        f"{r['recall_mean']:.3f} $\\pm$ {r['recall_std']:.3f}",
    ]
    print("\n  LaTeX:")
    print("  " + " & ".join(latex_cells) + r" \\")

    print(f"\n{'='*72}")
    print(f"\u2713 All SVM outputs saved to {REVIEWS_OUTPUT}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
