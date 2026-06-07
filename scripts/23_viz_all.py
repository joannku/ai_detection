#!/usr/bin/env python3
"""
Visualizations for the meta-classifier framework.

Outputs (output/):
  1. violin_distributions.png    — violin plots of key features, human vs AI
  2. decision_boundary_pca.png   — PCA 2D with decision boundary
  3. decision_boundary_pairs.png — feature-pair scatter plots with boundaries
  4. shap_comparison.png         — cross-model SHAP feature importance
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Feature-set variant: "orig" | "sw" | "drop" | "drop_nowc". Selects the merged matrix and
# the matching meta-classifier importances, and suffixes every output. Paper
# figure copies happen only for the canonical "orig" variant.
VARIANT = "drop"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"


def variant_name(filename: str) -> str:
    """Insert the variant SUFFIX before a filename's extension."""
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}" if dot else f"{filename}{SUFFIX}"


_REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_CSV = _REPO_ROOT / "output" / "merge_features" / f"meta_features{SUFFIX}.csv"
META_IMPORTANCE_CSV = _REPO_ROOT / "output" / "meta_classifiers" / f"meta_feature_importance{SUFFIX}.csv"
REVIEWS_OUTPUT = _REPO_ROOT / "output" / "viz_all"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
SEED = 42

FEATURE_DISPLAY_NAMES = {
    "coherence": "Coherence",
    "style_similarity": "Style Similarity",
    "read_flesch_reading_ease": "Flesch Reading Ease",
    "read_flesch_kincaid_grade": "FK Grade Level",
    "read_coleman_liau_index": "Coleman-Liau Index",
    "read_gunning_fog": "Gunning Fog",
    "read_syllable_count": "Syllable Count",
    "nli_consistency_score": "NLI Consistency",
    "nli_contradiction_rate": "Contradiction Rate",
    "nli_entailment_rate": "Entailment Rate",
    "nli_neutral_rate": "Neutral Rate",
    "nli_n_sentences": "Sentence Count (NLI)",
    "word_count": "Word Count",
    "sentence_count": "Sentence Count",
    "character_count": "Character Count",
    "liwc_Dic": "LIWC Dictionary",
    "liwc_article": "LIWC Articles",
    "liwc_Comma": "LIWC Commas",
    "liwc_Period": "LIWC Periods",
    "liwc_AllPunc": "LIWC All Punct.",
    "liwc_OtherP": "LIWC Other Punct.",
    "liwc_Perception": "LIWC Perception",
    "liwc_Cognition": "LIWC Cognition",
    "liwc_adverb": "LIWC Adverbs",
    "liwc_verb": "LIWC Verbs",
    "liwc_auxverb": "LIWC Aux. Verbs",
    "liwc_prep": "LIWC Prepositions",
    "liwc_number": "LIWC Numbers",
    "liwc_det": "LIWC Determiners",
    "liwc_conj": "LIWC Conjunctions",
    "liwc_pronoun": "LIWC Pronouns",
    "liwc_ppron": "LIWC Pers. Pronouns",
    "liwc_focuspresent": "LIWC Focus Present",
    "liwc_allure": "LIWC Allure",
    "liwc_memory": "LIWC Memory",
    "liwc_motion": "LIWC Motion",
    "liwc_feeling": "LIWC Feeling",
    "liwc_fulfill": "LIWC Fulfill",
    "liwc_BigWords": "LIWC Big Words",
    "bleu_participant": "BLEU (Participant)",
}

def top_features_by_mean_rank(n: int = 15) -> list[str]:
    """Return the top-n features by mean SHAP rank across the five
    meta-classifiers, in the same vertical order used by Figure 5.

    This couples the violin grid (Figure 7) to the cross-model heatmap
    (Figure 5) so every feature shown as a distribution is also visible
    in the heatmap, in the same row order. Avoids the previous problem
    where the violin grid had a hand-curated list that drifted from the
    actual top features (and silently lost panels when a feature was
    excluded upstream, e.g. liwc_Dic).

    Falls back to a curated list if META_IMPORTANCE_CSV is missing or
    has no shap_ columns yet (e.g. before 21_meta_classifiers.py has
    been run).
    """
    fallback = [
        "liwc_article", "read_coleman_liau_index", "liwc_Comma",
        "coherence", "sentence_count", "character_count",
        "liwc_BigWords", "liwc_Cognition", "liwc_verb",
        "liwc_adverb", "read_syllable_count", "style_similarity",
        "liwc_auxverb", "liwc_allure", "liwc_det",
    ]
    if not META_IMPORTANCE_CSV.exists():
        print("  [warning] meta_feature_importance.csv missing; "
              "using fallback violin list")
        return fallback[:n]
    imp = pd.read_csv(META_IMPORTANCE_CSV)
    shap_cols = [c for c in imp.columns if c.startswith("shap_")]
    if not shap_cols:
        print("  [warning] no shap_ columns in meta_feature_importance.csv; "
              "using fallback violin list")
        return fallback[:n]
    ranks = pd.DataFrame({"feature": imp["feature"]})
    for c in shap_cols:
        ranks[c] = imp[c].rank(method="min", ascending=False).astype(int)
    ranks["mean_rank"] = ranks[shap_cols].mean(axis=1)
    return ranks.nsmallest(n, "mean_rank")["feature"].tolist()

BOUNDARY_PAIRS = [
    ("coherence", "style_similarity"),
    ("coherence", "liwc_Dic"),
    ("read_coleman_liau_index", "liwc_article"),
    ("word_count", "liwc_Dic"),
]


def load_data():
    df = pd.read_csv(META_FEATURES_CSV)
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    X = df[feature_cols].copy()
    y = (df["final_label"] == "ai").astype(int).values
    labels = df["final_label"].values
    return X, y, labels, feature_cols


def display_name(feat):
    return FEATURE_DISPLAY_NAMES.get(feat, feat)


# ── 1. Violin plots ────────────────────────────────────────────────

def plot_violins(X, labels, feats=None):
    """Violin grid of the top-N meta-features, ordered by mean SHAP rank
    across the five meta-classifiers (same order as Figure 5).

    `feats` may be passed explicitly to override the default top-15.
    Anything not present in `X.columns` is silently dropped.
    """
    if feats is None:
        feats = top_features_by_mean_rank(15)
    feats = [f for f in feats if f in X.columns]
    n = len(feats)
    # 5 columns × 3 rows for n=15. Wider, landscape-oriented layout that
    # mirrors the row order of Figure 5 left-to-right, top-to-bottom and
    # is well suited to a full-text-width figure in the paper.
    cols = 5
    rows = (n + cols - 1) // cols

    HUMAN = "#4C9BE8"
    AI = "#E85C5C"

    # Font sizes tuned for the figure being rendered at single ACL
    # column width (~3.3 in); the figure is downscaled by roughly a
    # factor of 2.5 from its source size.
    TITLE_FS = 17
    TICK_FS = 14

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3f3f44",
        "axes.linewidth": 0.9,
    }):
        # Width scales with column count to keep each panel ~2.9 in wide,
        # matching the per-panel size of the previous 3-col layout.
        fig, axes = plt.subplots(rows, cols, figsize=(2.93 * cols, 2.55 * rows))
        axes = axes.flatten()

        legend_handles = None
        for i, feat in enumerate(feats):
            ax = axes[i]
            human_vals = X.loc[labels == "human", feat].dropna().values
            ai_vals = X.loc[labels == "ai", feat].dropna().values

            # Clip y-axis to the central 99% across both groups so a
            # handful of outliers don't squash the visible distribution.
            combined = np.concatenate([human_vals, ai_vals])
            lo = np.nanpercentile(combined, 0.5)
            hi = np.nanpercentile(combined, 99.5)
            pad = 0.04 * (hi - lo) if hi > lo else 1.0

            for vals, pos, col in [(human_vals, 0, HUMAN), (ai_vals, 1, AI)]:
                parts = ax.violinplot([vals], positions=[pos],
                                      showmeans=False, showextrema=False,
                                      showmedians=False, widths=0.78)
                for pc in parts["bodies"]:
                    pc.set_facecolor(col)
                    pc.set_edgecolor(col)
                    pc.set_alpha(0.55)
                    pc.set_linewidth(0.0)

                # Overlay a thin boxplot for median + IQR + whiskers.
                bp = ax.boxplot(vals, positions=[pos], widths=0.16,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="white",
                                                 linewidth=1.6),
                                boxprops=dict(facecolor=col,
                                              edgecolor=col,
                                              linewidth=0),
                                whiskerprops=dict(color=col, linewidth=1.0),
                                capprops=dict(color=col, linewidth=1.0))
                if legend_handles is None and pos == 0:
                    h_human = bp["boxes"][0]
                if i == 0 and pos == 1:
                    legend_handles = [h_human, bp["boxes"][0]]

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Human", "AI"], fontsize=TICK_FS)
            for tick, col in zip(ax.get_xticklabels(), [HUMAN, AI]):
                tick.set_color(col)
                tick.set_fontweight("semibold")
            ax.set_title(display_name(feat), fontsize=TITLE_FS,
                         color="#1a1a1f", pad=8)
            ax.tick_params(axis="y", labelsize=TICK_FS - 2,
                           colors="#3f3f44", length=3, width=0.8)
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(axis="y", color="#dddde1", linewidth=0.7,
                    alpha=0.7, zorder=-1)
            ax.set_axisbelow(True)

        for j in range(len(feats), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.4,
                         rect=(0, 0, 1, 0.965))

        out = REVIEWS_OUTPUT / variant_name("violin_distributions.png")
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white",
                    pad_inches=0.1)
        plt.close()
        print(f"  Saved violin plots → {out}")


# ── 2. Decision boundary: PCA ──────────────────────────────────────

def plot_boundary_pca(X, y, feature_cols):
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imp.fit_transform(X))

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_clean)

    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.6,
                             class_weight="balanced", max_iter=2000)
    clf.fit(X_clean, y)

    # Project decision boundary into PCA space
    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 300),
        np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 300),
    )
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_orig = pca.inverse_transform(grid_pca)
    Z = clf.predict_proba(grid_orig)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 50), cmap="RdBu_r", alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2, linestyles="--")

    human_mask = y == 0
    ax.scatter(X_pca[human_mask, 0], X_pca[human_mask, 1],
               c="#4C9BE8", s=12, alpha=0.4, label="Human", edgecolors="none")
    ax.scatter(X_pca[~human_mask, 0], X_pca[~human_mask, 1],
               c="#E85C5C", s=12, alpha=0.4, label="AI", edgecolors="none")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=11)
    ax.set_title("Decision Boundary (LASSO) in PCA Space", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out = REVIEWS_OUTPUT / variant_name("decision_boundary_pca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA boundary → {out}")


# ── 3. Decision boundary: feature pairs ────────────────────────────

def plot_boundary_pairs(X, y):
    pairs = [(a, b) for a, b in BOUNDARY_PAIRS if a in X.columns and b in X.columns]
    if not pairs:
        print("  No valid feature pairs found, skipping")
        return

    # Font sizes are tuned for the figure being rendered at single
    # ACL column width (~3.3 in), where the figure is downscaled by
    # roughly a factor of 2.5 from its source size.
    LABEL_FS = 19
    TICK_FS = 16
    LEGEND_FS = 16

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3f3f44",
        "axes.linewidth": 0.9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.4))
        axes = axes.flatten()

        legend_handles = None
        for idx, (f1, f2) in enumerate(pairs):
            ax = axes[idx]
            X_pair = X[[f1, f2]].copy()
            mask = X_pair.notna().all(axis=1)
            X_pair = X_pair[mask].values
            y_pair = y[mask]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pair)

            clf = LogisticRegression(penalty="l2", C=1.0,
                                     class_weight="balanced", max_iter=2000)
            clf.fit(X_scaled, y_pair)

            x_min, x_max = X_pair[:, 0].min() - 0.5, X_pair[:, 0].max() + 0.5
            y_min, y_max = X_pair[:, 1].min() - 0.5, X_pair[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid_scaled = scaler.transform(grid)
            Z = clf.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

            ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 50),
                        cmap="RdBu_r", alpha=0.25)
            ax.contour(xx, yy, Z, levels=[0.5], colors="black",
                       linewidths=2.0, linestyles="--")

            human_mask = y_pair == 0
            h_human = ax.scatter(X_pair[human_mask, 0], X_pair[human_mask, 1],
                                 c="#4C9BE8", s=18, alpha=0.45,
                                 label="Human", edgecolors="none")
            h_ai = ax.scatter(X_pair[~human_mask, 0], X_pair[~human_mask, 1],
                              c="#E85C5C", s=18, alpha=0.45,
                              label="AI", edgecolors="none")
            if idx == 0:
                legend_handles = [h_human, h_ai]

            ax.set_xlabel(display_name(f1), fontsize=LABEL_FS,
                          color="#1a1a1f")
            ax.set_ylabel(display_name(f2), fontsize=LABEL_FS,
                          color="#1a1a1f")
            ax.tick_params(labelsize=TICK_FS, colors="#3f3f44",
                           length=4, width=0.9)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        for j in range(len(pairs), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=0.6, w_pad=1.0, h_pad=1.4,
                         rect=(0, 0, 1, 0.94))
        if legend_handles is not None:
            leg = fig.legend(legend_handles, ["Human", "AI"],
                             fontsize=LEGEND_FS,
                             loc="upper center",
                             ncol=2,
                             frameon=False,
                             bbox_to_anchor=(0.5, 0.99),
                             handletextpad=0.4,
                             columnspacing=2.0)
            for handle in leg.legend_handles:
                handle.set_alpha(0.95)
                handle.set_sizes([90])

        out = REVIEWS_OUTPUT / variant_name("decision_boundary_pairs.png")
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white",
                    pad_inches=0.1)
        plt.close()
        print(f"  Saved pair boundaries → {out}")


# ── 4. SHAP comparison heatmap ─────────────────────────────────────

def plot_shap_comparison(top_k: int = 15):
    """Column-normalized SHAP importance heatmap with rank annotations.

    Why a heatmap (not grouped bars)?  The five models have wildly different
    raw |SHAP| scales (LASSO max ≈ 2.0, Random Forest max ≈ 0.04 — a ~50×
    spread), so grouped bars are dominated by one or two models and the rest
    appear as noise.  Within-column normalization makes the rankings directly
    comparable; the per-cell rank label preserves exact ordering.
    """
    imp = pd.read_csv(META_IMPORTANCE_CSV)
    shap_cols = [c for c in imp.columns if c.startswith("shap_")]

    # Display order on the columns: LASSO, Tree, RF, XGB, SVM
    col_order = [c for c in
                 ["shap_lasso", "shap_tree", "shap_rf", "shap_xgb", "shap_svm"]
                 if c in shap_cols]
    model_labels = {
        "shap_lasso": "LASSO",
        "shap_tree": "Dec. Tree",
        "shap_rf":   "Rand. Forest",
        "shap_xgb":  "XGBoost",
        "shap_svm":  "Linear SVM",
    }

    # Rank within each model (1 = most important).  Use min-rank so ties
    # (typically zero-importance features) share the same number.
    ranks = pd.DataFrame({"feature": imp["feature"]})
    for c in col_order:
        ranks[c] = imp[c].rank(method="min", ascending=False).astype(int)
    ranks["mean_rank"] = ranks[col_order].mean(axis=1)

    # Pick top features by mean rank across models, then resort by mean rank.
    top = ranks.nsmallest(top_k, "mean_rank").reset_index(drop=True)
    feats = top["feature"].tolist()

    # Column-normalised |SHAP| (each model self-scaled to [0, 1] over the
    # selected feature subset) drives the colour intensity.
    norm = pd.DataFrame(index=feats, columns=col_order, dtype=float)
    for c in col_order:
        col_vals = imp.set_index("feature").loc[feats, c].astype(float)
        denom = col_vals.max() if col_vals.max() > 0 else 1.0
        norm[c] = (col_vals / denom).values

    rank_grid = top.set_index("feature")[col_order]

    # ── Plot ───────────────────────────────────────────────────────
    LABEL_FS = 12
    TICK_FS = 11
    ANNOT_FS = 10

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }):
        fig, (ax, ax_mean) = plt.subplots(
            1, 2, figsize=(8.0, 0.42 * top_k + 1.6),
            gridspec_kw={"width_ratios": [len(col_order), 1.05], "wspace": 0.06},
        )

        # Main heatmap
        im = ax.imshow(norm.values, aspect="auto", cmap="Blues",
                       vmin=0.0, vmax=1.0)

        ax.set_xticks(np.arange(len(col_order)))
        ax.set_xticklabels([model_labels.get(c, c) for c in col_order],
                           fontsize=TICK_FS, rotation=20, ha="right",
                           color="#1a1a1f")
        ax.set_yticks(np.arange(len(feats)))
        ax.set_yticklabels([display_name(f) for f in feats],
                           fontsize=TICK_FS, color="#1a1a1f")

        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticks(np.arange(len(col_order) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(feats) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

        # Per-cell rank annotation — white text on dark cells, dark on light.
        for i, feat in enumerate(feats):
            for j, col in enumerate(col_order):
                v = norm.iloc[i, j]
                rank = int(rank_grid.iloc[i, j])
                txt_color = "white" if v > 0.55 else "#1a1a1f"
                ax.text(j, i, f"{rank}", ha="center", va="center",
                        color=txt_color, fontsize=ANNOT_FS,
                        fontweight="semibold")

        ax.set_title("Within-model rank (1 = most important)",
                     fontsize=LABEL_FS, color="#1a1a1f", pad=10)

        # Right-hand "mean rank" strip
        mean_rank = top["mean_rank"].values.reshape(-1, 1)
        # Reverse colour direction so lower mean rank (better) is darker.
        ax_mean.imshow(-mean_rank, aspect="auto", cmap="Greys",
                       vmin=-mean_rank.max(), vmax=-mean_rank.min())
        ax_mean.set_xticks([0])
        ax_mean.set_xticklabels(["Mean rank"], fontsize=TICK_FS,
                                rotation=20, ha="right", color="#1a1a1f")
        ax_mean.set_yticks([])
        ax_mean.tick_params(axis="both", which="both", length=0)
        for i, mr in enumerate(top["mean_rank"].values):
            shade = (mr - top["mean_rank"].min()) / max(
                top["mean_rank"].max() - top["mean_rank"].min(), 1e-9)
            txt_color = "white" if shade < 0.45 else "#1a1a1f"
            ax_mean.text(0, i, f"{mr:.1f}", ha="center", va="center",
                         color=txt_color, fontsize=ANNOT_FS,
                         fontweight="semibold")

        # Shared colour bar below the main heatmap
        cbar = fig.colorbar(im, ax=[ax, ax_mean], orientation="horizontal",
                            fraction=0.04, pad=0.12, aspect=40)
        cbar.set_label("Within-model normalised |SHAP|  (0 = unused, 1 = top)",
                       fontsize=TICK_FS, color="#1a1a1f")
        cbar.ax.tick_params(labelsize=TICK_FS - 1, colors="#3f3f44")

        out = REVIEWS_OUTPUT / variant_name("shap_comparison.png")
        plt.savefig(out, dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.15)
        plt.close()
        print(f"  Saved SHAP comparison heatmap → {out}")

        # Mirror to overleaf_paper/figures/ only for the canonical "orig"
        # variant, so sw/drop runs never overwrite the published figures.
        paper_figs = _REPO_ROOT / "overleaf_paper" / "figures"
        if SUFFIX:
            print(f"  Variant '{VARIANT}': skipping paper-figure copies")
        elif paper_figs.exists():
            import shutil
            paper_out = paper_figs / "shap_comparison.png"
            shutil.copyfile(out, paper_out)
            print(f"  Copied → {paper_out}")
            # Also mirror the violin and decision-boundary figures, which are
            # produced earlier in this script and otherwise drift out of sync
            # with the paper.
            for fname in ("violin_distributions.png", "decision_boundary_pairs.png"):
                src = REVIEWS_OUTPUT / fname
                if src.exists():
                    dst = paper_figs / fname
                    shutil.copyfile(src, dst)
                    print(f"  Copied → {dst}")


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Variant: {VARIANT}  ->  reading {META_FEATURES_CSV.name}")
    X, y, labels, feature_cols = load_data()

    print("Generating visualizations...")
    plot_violins(X, labels)
    plot_boundary_pca(X, y, feature_cols)
    plot_boundary_pairs(X, y)
    plot_shap_comparison()
    print(f"\n✓ All visualizations saved to {REVIEWS_OUTPUT}")


if __name__ == "__main__":
    main()
