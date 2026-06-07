#!/usr/bin/env python3
"""Predict ai/human from all features, generate SHAP importance.

Outputs
-------
- output/automl_shap/all_features.csv   - merged feature matrix
- output/automl_shap/shap_summary.png   - SHAP beeswarm (top 20 features)
- output/automl_shap/shap_importance.csv- mean |SHAP| per feature, sorted
- overleaf_paper/figures/shap_beeswarm.png  (mirrored copy for the paper)
"""

import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "automl_shap"
OUT.mkdir(parents=True, exist_ok=True)
PAPER_FIGS = ROOT / "overleaf_paper" / "figures"

# Feature-set variant: "orig" | "sw" | "drop" | "drop_nowc". Only the NLI source
# changes between variants here (this script does not use coherence/style).
# "drop_nowc" reuses the "drop" NLI data and additionally drops the LIWC word
# count (WC) feature. Suffixes every output; the paper figure copy happens only
# for the canonical "orig".
VARIANT = "drop"

_DATA_VARIANT = "drop" if VARIANT == "drop_nowc" else VARIANT
DATA_SUFFIX = "" if _DATA_VARIANT == "orig" else f"_{_DATA_VARIANT}"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"
EXCLUDE_WC = VARIANT == "drop_nowc"


def variant_name(filename: str) -> str:
    """Insert the variant SUFFIX before a filename's extension."""
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}" if dot else f"{filename}{SUFFIX}"

# Map raw feature column names (no `liwc_` prefix here because we read
# liwc.csv directly) to publication-friendly display names.
FEATURE_DISPLAY_NAMES = {
    "WC": "Word Count",
    "WPS": "Words / Sentence",
    "BigWords": "Big Words",
    "Dic": "LIWC Dictionary",
    "Comma": "LIWC Commas",
    "Period": "LIWC Periods",
    "AllPunc": "LIWC All Punct.",
    "OtherP": "LIWC Other Punct.",
    "QMark": "LIWC ?-marks",
    "Exclam": "LIWC !-marks",
    "Apostro": "LIWC Apostrophes",
    "article": "LIWC Articles",
    "verb": "LIWC Verbs",
    "auxverb": "LIWC Aux. Verbs",
    "adverb": "LIWC Adverbs",
    "prep": "LIWC Prepositions",
    "conj": "LIWC Conjunctions",
    "det": "LIWC Determiners",
    "number": "LIWC Numbers",
    "ppron": "LIWC Pers. Pronouns",
    "i": "LIWC `I` (1st-person)",
    "we": "LIWC `we`",
    "you": "LIWC `you`",
    "shehe": "LIWC `she/he`",
    "they": "LIWC `they`",
    "Cognition": "LIWC Cognition",
    "Perception": "LIWC Perception",
    "Affect": "LIWC Affect",
    "motion": "LIWC Motion",
    "memory": "LIWC Memory",
    "allure": "LIWC Allure",
    "feeling": "LIWC Feeling",
    "fulfill": "LIWC Fulfill",
    "focuspresent": "LIWC Focus Present",
    "focuspast": "LIWC Focus Past",
    "focusfuture": "LIWC Focus Future",
    # readability
    "flesch_reading_ease": "Flesch Reading Ease",
    "flesch_kincaid_grade": "FK Grade",
    "gunning_fog": "Gunning Fog",
    "smog_index": "SMOG Index",
    "coleman_liau_index": "Coleman-Liau",
    # NLI
    "consistency_score": "NLI Consistency",
    "contradiction_rate": "NLI Contradiction",
    "entailment_rate": "NLI Entailment",
    "neutral_rate": "NLI Neutral",
}


def display_name(feat: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feat, feat)


def load_features():
    """Merge all feature sources with labels."""
    df = pd.read_csv(ROOT / "output" / "export" / "final_journals.csv")[["JournalUniqueID", "label"]]
    df["y"] = (df["label"] == "ai").astype(int)

    # Merge each feature file (path is relative to output/<producer>/).
    # We drop the four LIWC-22 proprietary summary variables (Analytic, Clout,
    # Authentic, Tone) so the SHAP model only attributes importance to
    # interpretable lexical categories.
    # The NLI source (and its kept column names) depends on the variant; the
    # variant CSV uses suffixed names that we normalise back to canonical.
    nli_keep = ["JournalUniqueID", "consistency_score", "contradiction_rate",
                "entailment_rate", "neutral_rate"]
    sources = {
        "liwc/liwc.csv": {"drop": ["Row ID", "Segment", "Analytic", "Clout", "Authentic", "Tone"]},
        "readability/readability.csv": {"keep": ["JournalUniqueID", "flesch_reading_ease", "flesch_kincaid_grade",
                                                 "gunning_fog", "smog_index", "coleman_liau_index"]},
        f"nli_consistency{DATA_SUFFIX}/nli_consistency{DATA_SUFFIX}.csv": {
            "rename": {
                f"consistency_score{DATA_SUFFIX}": "consistency_score",
                f"contradiction_rate{DATA_SUFFIX}": "contradiction_rate",
                f"entailment_rate{DATA_SUFFIX}": "entailment_rate",
                f"neutral_rate{DATA_SUFFIX}": "neutral_rate",
            },
            "keep": nli_keep,
        },
    }
    for fname, opts in sources.items():
        path = ROOT / "output" / fname
        if not path.exists():
            continue
        feat = pd.read_csv(path)
        if "drop" in opts:
            feat = feat.drop(columns=[c for c in opts["drop"] if c in feat.columns])
        if "rename" in opts:
            feat = feat.rename(columns={k: v for k, v in opts["rename"].items() if k in feat.columns})
        if "keep" in opts:
            feat = feat[[c for c in opts["keep"] if c in feat.columns]]
        df = df.merge(feat, on="JournalUniqueID", how="left")
        print(f"✓ {fname}: {len(feat.columns)-1} features")
    
    return df


def main():
    print("=" * 50)
    print("AutoML + SHAP")
    print("=" * 50)
    
    print(f"Variant: {VARIANT} (output suffix: '{SUFFIX}')")
    df = load_features()
    df.to_csv(OUT / variant_name("all_features.csv"), index=False)
    print(f"✓ Saved {variant_name('all_features.csv')} ({len(df)} rows)")
    
    exclude = {"JournalUniqueID", "label", "y", "final_label", "ParticipantID"}
    if EXCLUDE_WC:
        exclude.add("WC")  # drop_nowc ablation: remove LIWC word count
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    print(f"\nFeatures: {len(features)}")
    
    X = StandardScaler().fit_transform(df[features].fillna(0))
    y = df["y"]
    
    # Cross-validated predictions
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                               random_state=42, eval_metric="logloss")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba > 0.5).astype(int)
    
    print(f"\nROC-AUC: {roc_auc_score(y, proba):.3f}")
    print(classification_report(y, pred, target_names=["human", "ai"]))
    
    # SHAP
    model.fit(X, y)
    shap_vals = shap.TreeExplainer(model).shap_values(X)

    # ── Paper-ready beeswarm ────────────────────────────────────────
    # We render with publication-friendly typography and humanised feature
    # labels.  Top-20 features keeps the figure readable at single-column
    # width while still surfacing the long tail beyond the obvious top-5.
    display_features = [display_name(f) for f in features]

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3f3f44",
        "axes.linewidth": 0.9,
    }):
        plt.figure(figsize=(7.5, 8.0))
        shap.summary_plot(
            shap_vals, X,
            feature_names=display_features,
            show=False,
            max_display=20,
            plot_size=None,
            color_bar_label="Feature value",
        )
        ax = plt.gca()
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=11,
                      color="#1a1a1f")
        ax.tick_params(axis="both", labelsize=10, colors="#3f3f44")
        plt.tight_layout(pad=0.5)
        out_png = OUT / variant_name("shap_summary.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        plt.close()

    # Only the canonical "orig" variant is mirrored into the paper figures.
    if SUFFIX:
        print(f"Variant '{VARIANT}': skipping paper-figure copy")
    elif PAPER_FIGS.exists():
        paper_out = PAPER_FIGS / "shap_beeswarm.png"
        shutil.copyfile(out_png, paper_out)
        print(f"✓ Copied → {paper_out}")
    else:
        print(f"WARNING: {PAPER_FIGS} not found, skipping copy")

    # Save importance (using raw feature names so downstream code keeps working)
    importance = pd.DataFrame({"feature": features, "importance": np.abs(shap_vals).mean(0)})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(OUT / variant_name("shap_importance.csv"), index=False)

    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))
    print(f"\n✓ Saved {variant_name('shap_summary.png')}, {variant_name('shap_importance.csv')}")


if __name__ == "__main__":
    main()
