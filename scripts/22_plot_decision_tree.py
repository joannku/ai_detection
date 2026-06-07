#!/usr/bin/env python3
"""
Re-render the decision tree figure used in the paper with:
  - humanised feature labels (e.g. liwc_verb -> "LIWC verb %"),
  - larger, paper-friendly fonts,
  - only the top 3 levels shown so it remains readable in a single column,
  - a clean colour map (orange = perceived AI, blue = perceived human),
  - explicit class proportions in each shown node.

The full deeper tree is preserved in `meta_tree_rules.txt` (text form).

Outputs:
  - output/plot_decision_tree/decision_tree.png
  - overleaf_paper/figures/decision_tree.png   (copied for the paper)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _export as _sk_tree_export

# Feature-set variant: "orig" | "sw" | "drop" | "drop_nowc". Selects the merged matrix and
# the suffix added to every output. The paper figure copy only happens for the
# canonical "orig" variant so sw/drop runs never clobber the published figure.
VARIANT = "drop"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"


def variant_name(filename: str) -> str:
    """Insert the variant SUFFIX before a filename's extension."""
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}" if dot else f"{filename}{SUFFIX}"


REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_CSV = REPO_ROOT / "output" / "merge_features" / f"meta_features{SUFFIX}.csv"
REVIEWS_OUTPUT = REPO_ROOT / "output" / "plot_decision_tree"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
PAPER_FIGS = REPO_ROOT / "overleaf_paper" / "figures"

SEED = 42
INNER_CV_FOLDS = 3
DROP_FEATURES = ["nli_n_sentences", "nli_n_pairs", "nli_segmentation", "bleu_participant"]
# `DISPLAY_DEPTH = None` shows every (post-pruned) node of the fitted
# tree.  Same-class sibling-leaf pairs are collapsed by
# `collapse_redundant_splits` after fitting, leaving only splits that
# actually change the discrete prediction.  The figure is rendered
# text-width and used with `\begin{figure*}` in the paper.
DISPLAY_DEPTH = None
NODE_FONTSIZE = 13

# Project-wide colour scheme (matches viz_all.py and meta_svm.py):
# blue for "Human", coral red for "AI". Saturation in the rendered
# tree scales with class purity at each node.
HUMAN_COLOR_RGB = (76, 155, 232)   # #4C9BE8
AI_COLOR_RGB = (232, 92, 92)       # #E85C5C


def _patch_sklearn_tree_colors():
    """Override sklearn's internal `_color_brew` so `plot_tree(filled=True)`
    uses the project palette (Human=blue, AI=coral) instead of its
    default orange/blue."""
    def _brew(n):
        if n == 2:
            return [list(HUMAN_COLOR_RGB), list(AI_COLOR_RGB)]
        return _sk_tree_export._color_brew_original(n)
    if not hasattr(_sk_tree_export, "_color_brew_original"):
        _sk_tree_export._color_brew_original = _sk_tree_export._color_brew
    _sk_tree_export._color_brew = _brew

# Map raw feature names → short, paper-friendly labels.
FEATURE_DISPLAY = {
    "word_count": "Word count",
    "sentence_count": "Sentence count",
    "character_count": "Character count",
    "coherence": "Coherence",
    "style_similarity": "Style similarity",
    "read_flesch_reading_ease": "Flesch reading ease",
    "read_flesch_kincaid_grade": "FK grade level",
    "read_gunning_fog": "Gunning fog",
    "read_smog_index": "SMOG index",
    "read_automated_readability_index": "ARI",
    "read_coleman_liau_index": "Coleman-Liau",
    "read_syllable_count": "Syllable count",
    "nli_consistency_score": "NLI consistency",
    "nli_entailment_rate": "NLI entailment rate",
    "nli_neutral_rate": "NLI neutral rate",
    "nli_contradiction_rate": "NLI contradiction rate",
}


def humanise(name: str) -> str:
    if name in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[name]
    if name.startswith("liwc_"):
        return f"LIWC {name[len('liwc_'):]}"
    if name.startswith("read_"):
        return f"Readability: {name[len('read_'):].replace('_', ' ')}"
    if name.startswith("nli_"):
        return f"NLI {name[len('nli_'):].replace('_', ' ')}"
    return name.replace("_", " ")


def load():
    df = pd.read_csv(META_FEATURES_CSV)
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feat_cols = [c for c in df.columns
                 if c not in id_cols and c not in DROP_FEATURES]
    X = df[feat_cols].copy()
    y = (df["final_label"] == "ai").astype(int).values
    groups = df["ParticipantID"].values
    print(f"Loaded {len(df)} entries × {len(feat_cols)} features "
          f"({len(np.unique(groups))} participants)")
    return X, y, groups, feat_cols


def fit_tree(X, y, groups):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(class_weight="balanced",
                                       random_state=SEED)),
    ])
    grid = {
        "clf__max_depth": [3, 4, 5, 6, 7],
        "clf__min_samples_leaf": [10, 20, 30, 50],
    }
    inner_cv = StratifiedGroupKFold(n_splits=INNER_CV_FOLDS)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv,
                      refit=True, n_jobs=1)
    gs.fit(X, y, groups=groups)
    print(f"Best params: {gs.best_params_}  (mean inner-CV AUC = "
          f"{gs.best_score_:.4f})")
    return gs.best_estimator_


def collapse_redundant_splits(clf):
    """Post-prune the decision tree by collapsing every internal node
    whose two child leaves predict the same class.

    Such splits are produced by the Gini criterion when they improve
    weighted impurity / probability ranking but do not change the
    discrete prediction. Removing them does not change `predict`
    output, simplifies the tree visually, and only marginally affects
    `predict_proba` (and therefore AUC).
    """
    tree = clf.tree_
    LEAF = -1
    n_collapsed = 0
    changed = True
    while changed:
        changed = False
        for i in range(tree.node_count):
            left = tree.children_left[i]
            right = tree.children_right[i]
            if left == LEAF or right == LEAF:
                continue
            if (tree.children_left[left] != LEAF
                    or tree.children_left[right] != LEAF):
                continue  # at least one child is not itself a leaf
            l_class = tree.value[left][0].argmax()
            r_class = tree.value[right][0].argmax()
            if l_class == r_class:
                tree.children_left[i] = LEAF
                tree.children_right[i] = LEAF
                tree.feature[i] = -2          # _tree.TREE_UNDEFINED
                tree.threshold[i] = -2.0
                n_collapsed += 1
                changed = True
    print(f"Collapsed {n_collapsed} redundant same-class splits "
          f"({tree.node_count} → {sum(tree.children_left == LEAF)} leaves)")
    return clf


def report_metrics(pipe, X, y, label):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    proba = pipe.predict_proba(X)[:, 1]
    pred = pipe.predict(X)
    print(f"  {label:18s}  "
          f"AUC = {roc_auc_score(y, proba):.4f} | "
          f"Acc = {accuracy_score(y, pred):.4f} | "
          f"F1 = {f1_score(y, pred):.4f}")


_NODE_BOX_STYLE = "round,pad=0.45,rounding_size=0.30"


def _restyle_nodes(ax, clf):
    """Modernise the auto-generated `plot_tree` output:
       - rewrite each node's text into a structured 2-3 line layout,
       - swap sklearn's heavy rounded-square bbox for a softer card
         with a thin border,
       - leave the True / False edge labels intact,
       - hide the grey `(...)` placeholders matplotlib draws for
         subtrees pruned by `max_depth`."""
    placeholders = []
    for txt in ax.texts:
        s = txt.get_text()
        s_stripped = s.strip()
        if s_stripped == "(...)":
            placeholders.append(txt)
            continue
        if s_stripped in ("True", "False"):
            txt.set_fontsize(NODE_FONTSIZE - 2)
            txt.set_color((0.25, 0.25, 0.28, 1.0))
            txt.set_fontstyle("italic")
            continue

        lines = [l.strip() for l in s.split("\n") if l.strip()]
        feat_thresh = next((l for l in lines if "<=" in l), "")
        n_line = next((l for l in lines if l.startswith("samples")), "")
        cls_line = next((l for l in lines if l.startswith("class")), "")

        n_pretty = n_line.replace("samples = ", "n = ")
        cls_pretty = (
            cls_line.replace("class = Perceived human", "→ Human")
                    .replace("class = Perceived AI", "→ AI")
        )
        new_text = "\n".join(p for p in (feat_thresh, n_pretty, cls_pretty) if p)
        txt.set_text(new_text)
        txt.set_fontsize(NODE_FONTSIZE)
        txt.set_fontweight("normal")
        txt.set_linespacing(1.3)
        txt.set_color((0.10, 0.10, 0.12))

        bbox = txt.get_bbox_patch()
        if bbox is not None:
            face = bbox.get_facecolor()
            bbox.set_boxstyle(_NODE_BOX_STYLE)
            bbox.set_edgecolor((0.25, 0.25, 0.28, 0.85))
            bbox.set_linewidth(0.8)
            bbox.set_facecolor(face)

    for txt in placeholders:
        bbox = txt.get_bbox_patch()
        if bbox is not None:
            bbox.set_visible(False)
        txt.set_visible(False)


def _restyle_edges(ax):
    """Soften the connector lines so they don't compete visually with
    the larger node labels."""
    for line in ax.lines:
        line.set_color((0.35, 0.35, 0.38, 0.9))
        line.set_linewidth(1.0)


def render(pipe, feature_cols):
    _patch_sklearn_tree_colors()
    clf = pipe.named_steps["clf"]
    display_names = [humanise(c) for c in feature_cols]

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.weight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }):
        fig, ax = plt.subplots(figsize=(13.0, 4.6), dpi=200)
        plot_tree(
            clf,
            feature_names=display_names,
            class_names=["Perceived human", "Perceived AI"],
            filled=True,
            rounded=True,
            max_depth=DISPLAY_DEPTH,
            fontsize=NODE_FONTSIZE,
            ax=ax,
            proportion=True,
            impurity=False,
            precision=2,
        )
        _restyle_nodes(ax, clf)
        _restyle_edges(ax)
        ax.set_axis_off()
        if DISPLAY_DEPTH is not None:
            full_depth = clf.get_depth()
            if full_depth > DISPLAY_DEPTH:
                bottom = (full_depth - DISPLAY_DEPTH - 0.4) / (full_depth + 1)
                ax.set_ylim(bottom, 1.02)

        out = REVIEWS_OUTPUT / variant_name("decision_tree.png")
        fig.subplots_adjust(left=0.005, right=0.995, top=1.0, bottom=0.0)
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight",
                    pad_inches=0.15)
        plt.close(fig)
        print(f"Saved → {out}")

    # Only the canonical "orig" variant is mirrored into the paper figures.
    if SUFFIX:
        print(f"Variant '{VARIANT}': skipping paper-figure copy")
    elif PAPER_FIGS.exists():
        paper_out = PAPER_FIGS / "decision_tree.png"
        shutil.copyfile(out, paper_out)
        print(f"Copied → {paper_out}")
    else:
        print(f"WARNING: {PAPER_FIGS} not found, skipping copy")


def main():
    print(f"Variant: {VARIANT}  ->  reading {META_FEATURES_CSV.name}")
    X, y, groups, feat_cols = load()
    pipe = fit_tree(X, y, groups)

    print("\nIn-sample metrics (refit on full HC subset):")
    report_metrics(pipe, X, y, "Before pruning")

    collapse_redundant_splits(pipe.named_steps["clf"])
    report_metrics(pipe, X, y, "After pruning")

    # Save the pruned rules text for the supplementary material.
    from sklearn.tree import export_text
    rules = export_text(
        pipe.named_steps["clf"],
        feature_names=[humanise(c) for c in feat_cols],
        max_depth=10,
    )
    rules_path = REVIEWS_OUTPUT / variant_name("meta_tree_rules.txt")
    rules_path.write_text(
        "Post-pruned decision tree (same-class sibling-leaf pairs collapsed)\n"
        "===================================================================\n\n"
        + rules
    )
    print(f"Saved pruned rules → {rules_path}")

    render(pipe, feat_cols)


if __name__ == "__main__":
    main()
