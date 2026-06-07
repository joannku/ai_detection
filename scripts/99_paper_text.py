#!/usr/bin/env python3
"""Generate all paper text by calling functions from processing scripts."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Feature-set variant: "orig" | "sw" | "drop" | "drop_nowc". Controls which
# variant-specific outputs the paper text is assembled from (NLI by-label means,
# meta-classifier tables/coeffs, SVM, permutation test, decision-tree rules).
# Variant-invariant sources (sample, IRR, demographics, BLEU, readability, LIWC)
# are unaffected. Note: the coherence/style means come from
# 12_analyze_features.py via import; set VARIANT there to match for a fully
# consistent variant run.
#
# "drop_nowc" reuses the "drop" descriptive data (NLI / coherence / style means
# are identical to "drop") but reads the distinct "_drop_nowc" model outputs.
VARIANT = "drop"

_DATA_VARIANT = "drop" if VARIANT == "drop_nowc" else VARIANT
DATA_SUFFIX = "" if _DATA_VARIANT == "orig" else f"_{_DATA_VARIANT}"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"


def variant_name(filename: str) -> str:
    """Insert the variant SUFFIX before a filename's extension."""
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}" if dot else f"{filename}{SUFFIX}"

liwc_labels = {
    "Bigwords": "Percentage of total words that are 7 letters or longer",
    "Dic": "Percent words captured by LIWC",
    "Analytic": "Metric of logical, formal thinking",
    "Clout": "Language of leadership, status",
    "Authentic": "Perceived honesty, genuineness",
    "Tone": "Degree of positive (negative) tone",
    "WC": "Total word count",
    "WPS": "Average words per sentence",
}


def format_sample_text(stats: dict) -> str:
    """Format sample description for paper."""
    s = stats
    entries_per_participant = s["total_entries"] / s["participants_with_entries"]
    human_pct_labelled = 100 * s["human_count"] / s["total_labelled"]
    ai_pct_labelled = 100 * s["ai_count"] / s["total_labelled"]
    invalid_pct_labelled = 100 * s["invalid_count"] / s["total_labelled"]
    retained = s["human_count"] + s["ai_count"]
    human_pct_retained = 100 * s["human_count"] / retained
    ai_pct_retained = 100 * s["ai_count"] / retained
    no_majority = s["invalid_count"] - s["invalid_majority"]
    
    lines = [
        f"In total, n={s['total_participants']} participants enrolled in the study,",
        f"of whom {s['participants_with_entries']} contributed {s['total_entries']:,} entries",
        f"({s['text_entries']:,} text, {s['voice_entries']} voice notes), after excluding",
        f"content duplicates and empty entries. On average, each",
        f"participant submitted {entries_per_participant:.1f} entries. Entries contained",
        f"a mean of {s['mean_words']:.1f} words ({s['mean_sentences']:.1f} sentences,",
        f"{s['mean_chars']:.1f} characters). For the current analysis, we only",
        f"consider text entries.",
        "",
        f"Of the {s['total_labelled']:,} labelled entries, {s['human_count']:,} were judged",
        f"as human-written ({human_pct_labelled:.1f}\\%) and {s['ai_count']:,} as AI-generated",
        f"({ai_pct_labelled:.1f}\\%) by majority vote. The remaining {s['invalid_count']:,} entries ({invalid_pct_labelled:.1f}\\%) were",
        f"classified as invalid, comprising {s['invalid_majority']:,} with invalid-",
        f"majority labels and {no_majority:,} with no clear majority.",
        f"We excluded all {s['invalid_count']:,} invalid entries, retaining {retained:,}",
        f"entries for analysis ({s['human_count']:,} perceived human, {human_pct_retained:.1f}\\%;",
        f"{s['ai_count']:,} perceived AI, {ai_pct_retained:.1f}\\%).",
    ]
    return "\n".join(lines)


def format_irr_text(stats: dict) -> str:
    """Format IRR section for paper."""
    s = stats
    alpha_str = "NA" if s["alpha"] is None else f"{s['alpha']:.3f}"
    mean_kappa_str = "NA" if s["mean_kappa"] is None else f"{s['mean_kappa']:.3f}"
    weighted_kappa_str = "NA" if s["weighted_kappa"] is None else f"{s['weighted_kappa']:.3f}"
    
    lines = ["Inter-rater reliability was assessed on all entries with at least"]
    
    if s["entries_single"] > 0:
        lines.append(f"two independent labels (n={s['entries_2plus']:,} of {s['total_entries']:,} total;")
        lines.append(f"{s['entries_single']:,} entries had only single ratings). Agreement across all")
    else:
        lines.append(f"two independent labels (n={s['entries_2plus']:,}). Agreement across all")
    
    lines.extend([
        "three categories (human, AI, invalid) was substantial:",
        f"Krippendorff's $\\alpha$ (nominal) = {alpha_str}, and pairwise",
        f"Cohen's $\\kappa$ averaged {mean_kappa_str} (weighted mean $\\kappa$ = {weighted_kappa_str})",
        f"across {s['num_pairs']} rater pairs with overlapping entries.",
    ])
    return "\n".join(lines)


def _latex_num(val: float, decimals: int = 1) -> str:
    """Format number with LaTeX thousands separator {,}."""
    formatted = f"{val:,.{decimals}f}"
    return formatted.replace(",", "{,}")


def _latex_diff(val: float, decimals: int = 1) -> str:
    """Format difference with + sign (and a TeX-friendly minus) plus thousands separator."""
    if val >= 0:
        formatted = f"+{val:,.{decimals}f}"
    else:
        formatted = f"$-${abs(val):,.{decimals}f}"
    return formatted.replace(",", "{,}")


def _signed_diff_3dp(val: float) -> str:
    """Like _latex_diff but with 3 decimal places and a TeX-friendly minus sign."""
    if val >= 0:
        return f"+{val:.3f}"
    return f"$-${abs(val):.3f}"


def _signed_diff_1dp(val: float) -> str:
    """Like _latex_diff but with 1 decimal place and no thousands separator (used inside table cells)."""
    if val >= 0:
        return f"+{val:.1f}"
    return f"$-${abs(val):.1f}"


def format_bleu_methods() -> str:
    """Generate methods paragraph for BLEU analysis."""
    lines = [
        r"\paragraph{Lexical diversity (BLEU).}",
        r"To quantify lexical diversity within each label group, we computed mean pairwise",
        r"BLEU scores using SacreBLEU\footnote{\url{https://github.com/mjpost/sacrebleu}.}.",
        r"For each group (perceived AI, perceived human), we calculated BLEU scores between",
        r"all $\binom{n}{2}$ entry pairs. BLEU measures n-gram overlap between two texts;",
        r"higher scores indicate greater similarity. The group-level score is the mean of",
        r"all pairwise comparisons---higher values indicate lower diversity (more formulaic",
        r"language). We used \texttt{effective\_order=True} to handle short texts and",
        r"normalised scores to 0--1.",
    ]
    return "\n".join(lines)


def get_bleu_stats() -> dict:
    """Load BLEU stats from CSV if available."""
    bleu_path = PROJECT_DIR / "output" / "bleu" / "bleu_by_label.csv"
    if not bleu_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(bleu_path)
    ai_row = df[df["label"] == "ai"].iloc[0] if len(df[df["label"] == "ai"]) > 0 else None
    human_row = df[df["label"] == "human"].iloc[0] if len(df[df["label"] == "human"]) > 0 else None
    if ai_row is None or human_row is None:
        return {}
    # Support both column names
    bleu_col = "bleu" if "bleu" in df.columns else "self_bleu"
    return {
        "bleu_ai": float(ai_row[bleu_col]),
        "bleu_human": float(human_row[bleu_col]),
    }


def format_nli_methods() -> str:
    """Generate methods paragraph for NLI consistency analysis."""
    lines = [
        r"\paragraph{Internal consistency (NLI).}",
        r"We assessed internal consistency using natural language inference (NLI).",
        r"Each journal entry was split into sentences, and for all $\binom{n}{2}$ sentence pairs",
        r"within an entry, we classified their relationship as entailment,",
        r"contradiction, or neutral using BART-large-MNLI\footnote{\url{https://huggingface.co/facebook/bart-large-mnli}.}.",
        r"The entry-level consistency score is defined as $1 - r_{\text{contradiction}}$,",
        r"where $r_{\text{contradiction}}$ is the fraction of sentence pairs classified",
        r"as contradictions. Higher scores indicate greater internal coherence;",
        r"entries with fewer than two qualifying sentences were excluded from "
        r"rate summaries (undefined consistency).",
    ]
    return "\n".join(lines)


def format_readability_methods() -> str:
    """Generate methods paragraph for readability analysis."""
    lines = [
        r"\paragraph{Readability.}",
        r"We computed readability metrics using the \texttt{textstat} library.",
        r"Flesch-Kincaid Grade Level estimates the US school grade required to",
        r"understand the text, based on average sentence length and syllables per word;",
        r"higher values indicate more complex text.",
        r"Flesch Reading Ease scores range from 0 to 100, with higher values",
        r"indicating easier readability (70--80 = fairly easy, 30--50 = difficult).",
    ]
    return "\n".join(lines)


def get_nli_stats() -> dict:
    """Load NLI consistency stats from CSV if available."""
    nli_path = (PROJECT_DIR / "output" / f"nli_consistency{DATA_SUFFIX}"
                / f"nli_consistency{DATA_SUFFIX}_by_label.csv")
    if not nli_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(nli_path)
    ai_row = df[df["label"] == "ai"].iloc[0] if len(df[df["label"] == "ai"]) > 0 else None
    human_row = df[df["label"] == "human"].iloc[0] if len(df[df["label"] == "human"]) > 0 else None
    if ai_row is None or human_row is None:
        return {}
    return {
        "consistency_ai": float(ai_row["consistency_mean"]),
        "consistency_ai_std": float(ai_row["consistency_std"]),
        "consistency_human": float(human_row["consistency_mean"]),
        "consistency_human_std": float(human_row["consistency_std"]),
        "contradiction_ai": float(ai_row["contradiction_rate"]),
        "contradiction_human": float(human_row["contradiction_rate"]),
        "entailment_ai": float(ai_row["entailment_rate"]),
        "entailment_human": float(human_row["entailment_rate"]),
        "neutral_ai": float(ai_row["neutral_rate"]),
        "neutral_human": float(human_row["neutral_rate"]),
    }


def get_readability_stats() -> dict:
    """Load readability stats from CSV if available."""
    read_path = PROJECT_DIR / "output" / "readability" / "readability_by_label.csv"
    if not read_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(read_path)
    ai_row = df[df["label"] == "ai"].iloc[0] if len(df[df["label"] == "ai"]) > 0 else None
    human_row = df[df["label"] == "human"].iloc[0] if len(df[df["label"] == "human"]) > 0 else None
    if ai_row is None or human_row is None:
        return {}
    return {
        "fk_grade_ai": float(ai_row["flesch_kincaid_grade_mean"]),
        "fk_grade_ai_std": float(ai_row["flesch_kincaid_grade_std"]),
        "fk_grade_human": float(human_row["flesch_kincaid_grade_mean"]),
        "fk_grade_human_std": float(human_row["flesch_kincaid_grade_std"]),
        "fre_ai": float(ai_row["flesch_reading_ease_mean"]),
        "fre_ai_std": float(ai_row["flesch_reading_ease_std"]),
        "fre_human": float(human_row["flesch_reading_ease_mean"]),
        "fre_human_std": float(human_row["flesch_reading_ease_std"]),
    }


def format_nli_text(stats: dict) -> str:
    """Format NLI consistency results for paper."""
    if not stats:
        return "NLI consistency data not available."
    s = stats
    lines = [
        f"NLI-based consistency analysis revealed that entries perceived as AI-generated exhibited",
        f"higher internal consistency ({s['consistency_ai']:.3f} $\\pm$ {s['consistency_ai_std']:.3f})",
        f"compared to entries judged as human ({s['consistency_human']:.3f} $\\pm$ {s['consistency_human_std']:.3f}).",
        f"Notably, {s['entailment_ai']*100:.0f}\\% of sentence pairs in AI-labelled entries showed entailment",
        f"relationships, compared to only {s['entailment_human']*100:.0f}\\% in human-labelled entries,",
        f"while human-labelled entries contained substantially more neutral (tangential) sentence pairs",
        f"({s['neutral_human']*100:.0f}\\% vs.\\ {s['neutral_ai']*100:.1f}\\%).",
    ]
    return "\n".join(lines)


def format_readability_text(stats: dict) -> str:
    """Format readability results for paper."""
    if not stats:
        return "Readability data not available."
    s = stats
    lines = [
        f"Entries judged as AI-generated exhibited higher Flesch-Kincaid Grade Level",
        f"({s['fk_grade_ai']:.1f} $\\pm$ {s['fk_grade_ai_std']:.1f} vs.\\ {s['fk_grade_human']:.1f} $\\pm$ {s['fk_grade_human_std']:.1f})",
        f"and lower Flesch Reading Ease ({s['fre_ai']:.1f} vs.\\ {s['fre_human']:.1f}).",
    ]
    return "\n".join(lines)


def format_features_table(stats: dict, bleu_stats: dict = None, nli_stats: dict = None, read_stats: dict = None) -> str:
    """Generate LaTeX table for feature means."""
    s = stats
    b = bleu_stats or {}
    n = nli_stats or {}
    r = read_stats or {}
    
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Semantic and structural features: human-written vs.\ AI-generated entries.}",
        r"\label{tab:feature_means_semantic}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Feature} & \textbf{Human} & \textbf{AI} & \textbf{Diff} \\",
        r"\midrule",
        f"Coherence & {s['coherence_human']:.3f} & {s['coherence_ai']:.3f} & {_signed_diff_3dp(s['coherence_ai'] - s['coherence_human'])} \\\\",
        f"Style similarity & {s['style_human']:.3f} & {s['style_ai']:.3f} & {_signed_diff_3dp(s['style_ai'] - s['style_human'])} \\\\",
    ]

    # Add BLEU if available
    if b.get("bleu_ai") is not None and b.get("bleu_human") is not None:
        diff = b["bleu_ai"] - b["bleu_human"]
        lines.append(f"BLEU & {b['bleu_human']:.3f} & {b['bleu_ai']:.3f} & {_signed_diff_3dp(diff)} \\\\")

    # Add NLI metrics section if available
    if n.get("consistency_ai") is not None and n.get("consistency_human") is not None:
        lines.append(r"\midrule")
        diff = n["consistency_ai"] - n["consistency_human"]
        lines.append(f"NLI consistency & {n['consistency_human']:.3f} & {n['consistency_ai']:.3f} & {_signed_diff_3dp(diff)} \\\\")
        diff = n["entailment_ai"] - n["entailment_human"]
        lines.append(f"Entailment rate & {n['entailment_human']:.3f} & {n['entailment_ai']:.3f} & {_signed_diff_3dp(diff)} \\\\")
        diff = n["neutral_ai"] - n["neutral_human"]
        lines.append(f"Neutral rate & {n['neutral_human']:.3f} & {n['neutral_ai']:.3f} & {_signed_diff_3dp(diff)} \\\\")

    # Add readability metrics if available
    if r.get("fk_grade_ai") is not None and r.get("fk_grade_human") is not None:
        lines.append(r"\midrule")
        diff = r["fk_grade_ai"] - r["fk_grade_human"]
        lines.append(f"FK grade level & {r['fk_grade_human']:.1f} & {r['fk_grade_ai']:.1f} & {_signed_diff_1dp(diff)} \\\\")
        diff = r["fre_ai"] - r["fre_human"]
        lines.append(f"Reading ease & {r['fre_human']:.1f} & {r['fre_ai']:.1f} & {_signed_diff_1dp(diff)} \\\\")

    lines.extend([
        r"\midrule",
        f"Word count & {_latex_num(s['words_human'])} & {_latex_num(s['words_ai'])} & {_latex_diff(s['words_ai'] - s['words_human'])} \\\\",
        f"Sentence count & {_latex_num(s['sentences_human'])} & {_latex_num(s['sentences_ai'])} & {_latex_diff(s['sentences_ai'] - s['sentences_human'])} \\\\",
        f"Char.\\ count & {_latex_num(s['chars_human'])} & {_latex_num(s['chars_ai'])} & {_latex_diff(s['chars_ai'] - s['chars_human'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_features_text(stats: dict, bleu_stats: dict = None) -> str:
    """Format feature analysis section for paper."""
    s = stats
    b = bleu_stats or {}

    lines = [
        f"Within the high-confidence subset, entries perceived as AI-generated were substantially",
        f"longer, averaging {_latex_num(s['words_ai'])} words (vs.\\ {_latex_num(s['words_human'])} for entries judged as human),",
        f"{_latex_num(s['sentences_ai'])} sentences (vs.\\ {_latex_num(s['sentences_human'])}), and",
        f"{_latex_num(s['chars_ai'])} characters (vs.\\ {_latex_num(s['chars_human'])}).",
        "",
        f"Entries labelled as AI also exhibited higher semantic coherence",
        f"({s['coherence_ai']:.3f} vs.\\ {s['coherence_human']:.3f}) and style similarity",
        f"({s['style_ai']:.3f} vs.\\ {s['style_human']:.3f}) compared to those judged as human-written",
        f"(Table~\\ref{{tab:feature_means_semantic}}).",
    ]

    # Add BLEU finding if available, in the paper's wording.
    if b.get("bleu_ai") is not None and b.get("bleu_human") is not None:
        pct_increase = ((b["bleu_ai"] / b["bleu_human"]) - 1) * 100 if b["bleu_human"] > 0 else 0
        lines.append("")
        lines.append(
            f"BLEU analysis indicated lower lexical diversity among AI-labelled entries"
            f" ({b['bleu_ai']:.3f} vs.\\ {b['bleu_human']:.3f}), with approximately"
            f" {pct_increase:.0f}\\% more n-gram overlap."
        )

    return "\n".join(lines)


def format_liwc_figure() -> str:
    """Generate LaTeX figure code for LIWC coefficients plot."""
    lines = [
        r"\begin{figure}[tbp]",
        r"    \centering",
        r"    \includegraphics[width=\columnwidth]{figures/liwc_lasso_coeffs_multiplot_words_only.png}",
        r"    \caption{LASSO coefficients for word-based LIWC-22 features predicting perceived AI vs.\ human authorship. Left panel shows features associated with entries judged as human-written (negative coefficients); right panel shows features associated with entries judged as AI-generated (positive coefficients). Only features with non-zero coefficients after L1 regularization are shown.}",
        r"    \label{fig:liwc_coeffs}",
        r"\end{figure}",
    ]
    return "\n".join(lines)


def format_wordcloud_figure() -> str:
    """Generate LaTeX figure code for LIWC wordclouds multipanel."""
    lines = [
        r"\begin{figure}[tbp]",
        r"    \centering",
        r"    \includegraphics[width=\columnwidth]{figures/wordclouds_multipanel.png}",
        r"    \caption{Wordclouds for selected LIWC categories distinguishing perceived human vs.\ AI authorship. Left column shows categories associated with entries judged as human-written (\textsc{allure}, \textsc{focuspresent}, \textsc{ppron}); right column shows categories associated with entries judged as AI-generated (\textsc{Perception}, \textsc{curiosity}, \textsc{memory}). Word size reflects frequency within each category.}",
        r"    \label{fig:liwc_wordclouds}",
        r"\end{figure}",
    ]
    return "\n".join(lines)


def format_liwc_text(stats: dict) -> str:
    """Format LIWC LASSO results section for paper."""
    s = stats
    topwords = s.get("topwords", {})
    
    def fmt_category_with_words(cat: str) -> str:
        """Format category with top 4 example words."""
        cat_tex = cat.replace("_", "\\_")
        words = topwords.get(cat, [])
        if words:
            word_list = ", ".join(f"\\emph{{{w}}}" for w in words[:4])
            return f"\\textsc{{{cat_tex}}} (e.g., {word_list})"
        return f"\\textsc{{{cat_tex}}}"
    
    def fmt_features(features):
        return ", ".join(fmt_category_with_words(f) for f in features) if features else "none"
    
    top_ai = fmt_features(s["top_ai_features"][:10])
    top_human = fmt_features(s["top_human_features"][:10])
    
    lines = [
        f"The LASSO logistic regression trained on LIWC-22 features achieved strong",
        f"cross-validated performance (AUC = {s['auc_mean']:.3f} $\\pm$ {s['auc_std']:.3f},",
        f"accuracy = {s['accuracy_mean']:.3f} $\\pm$ {s['accuracy_std']:.3f},",
        f"F1 = {s['f1_mean']:.3f} $\\pm$ {s['f1_std']:.3f}),",
        f"with L1 regularization selecting {s['num_selected']} of {s['num_features']} features",
        r"(Figure~\ref{fig:liwc_coeffs}).",
        f"Features most predictive of perceived AI included:",
        f"{top_ai}.",
        f"Features most associated with perceived human authorship included:",
        f"{top_human}.",
        r"Example words for selected categories are presented in Figure~\ref{fig:liwc_wordclouds}.",
        (
            r"These LIWC-only results motivate the extended meta-classifier analysis below, "
            r"which incorporates all feature groups into a unified model to test whether "
            r"structural and semantic features provide additional discriminative signal beyond "
            r"psycholinguistic categories alone."
        ),
    ]
    return "\n".join(lines)


def _read_output_file(rel: str) -> str | None:
    """Read a text file under output/<rel>; return None if missing.

    `rel` should include the producer subfolder, e.g.
    "rater_demographics/rater_demographics_paragraph.tex".
    """
    p = PROJECT_DIR / "output" / rel
    return p.read_text() if p.exists() else None


def format_participant_demographics() -> str:
    """Source-participant demographics summary written by 07_participant_demographics.py."""
    txt = _read_output_file("participant_demographics/participant_demographics.txt")
    if txt is None:
        return "Participant demographics not available — run scripts/07_participant_demographics.py."
    return txt


def format_rater_demographics() -> str:
    """LaTeX paragraph for annotator demographics written by 03_rater_demographics.py."""
    tex = _read_output_file("rater_demographics/rater_demographics_paragraph.tex")
    if tex is None:
        return "Rater demographics not available — run scripts/03_rater_demographics.py."
    return tex


def format_high_confidence_subset() -> str:
    """High-confidence subset summary written by 06_high_confidence_subset.py."""
    txt = _read_output_file("high_confidence_subset/high_confidence_summary.txt")
    if txt is None:
        return "High-confidence subset not available — run scripts/06_high_confidence_subset.py."
    return txt


def format_meta_classifier_table() -> str:
    """LaTeX table of cross-validated meta-classifier performance from 21_meta_classifiers.py."""
    p = PROJECT_DIR / "output" / "meta_classifiers" / variant_name("meta_cv_results.csv")
    if not p.exists():
        return "Meta-classifier results not available — run scripts/21_meta_classifiers.py."
    import pandas as pd
    df = pd.read_csv(p)
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Cross-validated performance of meta-classifiers on the high-confidence subset (5-fold nested CV; mean $\pm$ SD).}",
        r"\label{tab:meta_cv}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{AUC} & \textbf{Accuracy} & \textbf{F1} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['model']} & "
            f"{row['roc_auc_mean']:.3f} $\\pm$ {row['roc_auc_std']:.3f} & "
            f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f} & "
            f"{row['f1_mean']:.3f} $\\pm$ {row['f1_std']:.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# LIWC categories that are usually rendered as proper nouns in the paper
# (most are lowercase function-word codes, but the punctuation / summary
# style ones are written in title-case). We just pass through whatever
# casing LIWC uses inside \textsc{...}.
def _meta_feature_display_name(feat: str) -> str:
    """Map a meta-feature column name to its paper-ready LaTeX form."""
    if feat.startswith("liwc_"):
        return f"LIWC \\textsc{{{feat[len('liwc_'):]}}}"
    readable = {
        "coherence": "semantic coherence",
        "style_similarity": "style similarity",
        "sentence_count": "sentence count",
        "character_count": "character count",
        "word_count": "word count",
        "read_flesch_reading_ease": "Flesch Reading Ease",
        "read_flesch_kincaid_grade": "Flesch-Kincaid grade level",
        "read_coleman_liau_index": "Coleman-Liau readability index",
        "read_gunning_fog": "Gunning Fog",
        "read_smog_index": "SMOG index",
        "read_automated_readability_index": "ARI",
        "read_syllable_count": "syllable count",
        "nli_consistency_score": "NLI consistency",
        "nli_contradiction_rate": "NLI contradiction rate",
        "nli_entailment_rate": "NLI entailment rate",
        "nli_neutral_rate": "NLI neutral rate",
        "bleu_participant": "self-BLEU (participant-level)",
    }
    return readable.get(feat, feat.replace("_", " "))


def _meta_format_coef(c: float) -> str:
    """LaTeX-format a coefficient with explicit sign (+1.23 or $-$1.23)."""
    if c >= 0:
        return f"+{c:.2f}"
    return f"$-${abs(c):.2f}"


def format_meta_classifier_text() -> str:
    """Auto-generate the §5 extended-LASSO paragraph.

    Pulls counts and coefficients from `meta_lasso_coeffs.csv`, the meta-LASSO
    AUC from `meta_cv_results.csv`, and the LIWC-only LASSO AUC from
    `liwc_lasso_cv_words_only.csv` so the paper text always matches the
    currently-fitted models.  The 6+4 split (top-6 AI features, top-4 human
    features) follows the exact prose layout used in `5results.tex`.
    """
    cv_p = PROJECT_DIR / "output" / "meta_classifiers" / variant_name("meta_cv_results.csv")
    coef_p = PROJECT_DIR / "output" / "meta_classifiers" / variant_name("meta_lasso_coeffs.csv")
    liwc_cv_p = (PROJECT_DIR / "output" / "analyze_liwc"
                 / "liwc_lasso_cv_words_only.csv")
    if not cv_p.exists() or not coef_p.exists():
        return ("Meta-classifier text not available — "
                "run scripts/21_meta_classifiers.py.")

    import pandas as pd
    cv = pd.read_csv(cv_p).set_index("model")
    coefs = pd.read_csv(coef_p)
    selected = coefs[coefs["selected"]].sort_values("abs_coeff", ascending=False)
    n_selected = len(selected)
    n_total = len(coefs)

    pos = selected[selected["coefficient"] > 0].head(6)
    neg = selected[selected["coefficient"] < 0].head(4)

    def _list(rows) -> str:
        items = [
            f"{_meta_feature_display_name(r['feature'])} "
            f"({_meta_format_coef(r['coefficient'])})"
            for _, r in rows.iterrows()
        ]
        if len(items) <= 1:
            return items[0] if items else "none"
        return ", ".join(items[:-1]) + ", and " + items[-1]

    pos_str = _list(pos)
    neg_str = _list(neg)

    auc_meta = cv.loc["LASSO"]["roc_auc_mean"]
    comparison_sentence = ""
    if liwc_cv_p.exists():
        auc_liwc = pd.read_csv(liwc_cv_p).iloc[0]["cv_auc_mean"]
        comparison_sentence = (
            f" The extended model incorporating all feature types "
            f"outperformed the LIWC-only LASSO "
            f"(AUC = {auc_meta:.4f} vs.\\ {auc_liwc:.3f} for the "
            f"LIWC-only model), confirming that structural and semantic "
            f"features provide complementary signal beyond "
            f"psycholinguistic categories alone."
        )

    paragraph = (
        f"The extended LASSO selected {n_selected} of {n_total} features, "
        f"spanning multiple feature groups. "
        f"The strongest positive coefficients (associated with perceived AI) "
        f"were {pos_str}. "
        f"The strongest negative coefficients (associated with perceived "
        f"human authorship) were {neg_str}."
        f"{comparison_sentence}"
    )
    return paragraph


def format_decision_tree_rules() -> str:
    """Post-pruned decision tree rules dump from 21_meta_classifiers.py."""
    txt = _read_output_file(f"meta_classifiers/{variant_name('meta_tree_rules.txt')}")
    if txt is None:
        return "Decision tree rules not available — run scripts/21_meta_classifiers.py."
    return txt


def format_svm_text() -> str:
    """Linear SVM CV scores + margin-flagging counts from 24_meta_svm.py (slot unchanged)."""
    cv_p = PROJECT_DIR / "output" / "meta_svm" / variant_name("svm_cv_results.csv")
    flags_p = PROJECT_DIR / "output" / "meta_svm" / variant_name("svm_margin_flags.csv")
    if not cv_p.exists() or not flags_p.exists():
        return "SVM results not available — run scripts/24_meta_svm.py."
    import pandas as pd
    cv = pd.read_csv(cv_p).iloc[0]
    flags = pd.read_csv(flags_p)
    n_total = len(flags)
    n_flag = int(flags["flag_for_review"].sum())
    pct = 100.0 * n_flag / n_total if n_total else 0.0
    return (
        f"The linear SVM (probability-calibrated) achieved AUC = {cv['roc_auc_mean']:.3f} "
        f"$\\pm$ {cv['roc_auc_std']:.3f}, accuracy = {cv['accuracy_mean']:.3f} "
        f"$\\pm$ {cv['accuracy_std']:.3f}, and F1 = {cv['f1_mean']:.3f} "
        f"$\\pm$ {cv['f1_std']:.3f} on the high-confidence subset. Of the {n_total:,} "
        f"entries scored, {n_flag} ({pct:.1f}\\%) fell within the SVM decision margin "
        f"and were flagged for human review under the margin-based audit protocol."
    )


def format_permutation_test() -> str:
    """Permutation-test summary from 26_permutation_test.py."""
    txt = _read_output_file(f"permutation_test/{variant_name('permutation_test_summary.txt')}")
    if txt is None:
        return "Permutation test results not available — run scripts/26_permutation_test.py."
    return txt


def format_decision_tree_figure() -> str:
    """LaTeX figure for the post-pruned decision tree."""
    lines = [
        r"\begin{figure}[tbp]",
        r"    \centering",
        r"    \includegraphics[width=\columnwidth]{figures/decision_tree.png}",
        r"    \caption{Post-pruned decision tree on the unified meta-feature set, restricted to the top three levels for readability. Class proportions are shown in each node; orange = perceived AI, blue = perceived human.}",
        r"    \label{fig:decision_tree}",
        r"\end{figure}",
    ]
    return "\n".join(lines)


def format_decision_boundary_figure() -> str:
    """LaTeX figure for the meta-classifier decision boundaries."""
    lines = [
        r"\begin{figure}[tbp]",
        r"    \centering",
        r"    \includegraphics[width=\columnwidth]{figures/decision_boundary_pairs.png}",
        r"    \caption{Decision boundaries of the LASSO meta-classifier projected onto four pairs of strongly-weighted meta-features. Points show high-confidence-subset entries; shaded regions indicate the model's class predictions.}",
        r"    \label{fig:decision_boundary_pairs}",
        r"\end{figure}",
    ]
    return "\n".join(lines)


def format_violin_figure() -> str:
    """LaTeX figure for the violin plots of meta-features."""
    lines = [
        r"\begin{figure}[tbp]",
        r"    \centering",
        r"    \includegraphics[width=\columnwidth]{figures/violin_distributions.png}",
        r"    \caption{Distributions of the most discriminative meta-features for entries judged as AI- vs.\ human-written.}",
        r"    \label{fig:violins}",
        r"\end{figure}",
    ]
    return "\n".join(lines)


def format_shap_comparison_figure() -> str:
    """LaTeX figure for cross-model SHAP comparison."""
    lines = [
        r"\begin{figure}[tbp]",
        r"    \centering",
        r"    \includegraphics[width=\columnwidth]{figures/shap_comparison.png}",
        r"    \caption{Top-15 SHAP feature importances across LASSO, Decision Tree, Random Forest, XGBoost, and Linear SVM, computed on the high-confidence subset.}",
        r"    \label{fig:shap_comparison}",
        r"\end{figure}",
    ]
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("PAPER TEXT - Generated from processing pipeline outputs")
    print(f"Feature-set variant: {VARIANT}")
    print("=" * 70)
    
    # Section 1: Sample Description
    print("\n[SAMPLE DESCRIPTION]")
    print("-" * 70)
    try:
        import importlib
        combine_ratings = importlib.import_module("00_combine_ratings")
        sample_stats = combine_ratings.get_sample_stats()
        print(format_sample_text(sample_stats))
    except Exception as e:
        print(f"Error: {e}")
    
    # Section 2: Inter-Rater Reliability
    print("\n[INTER-RATER RELIABILITY]")
    print("-" * 70)
    try:
        irr = importlib.import_module("02_irr")
        irr_stats = irr.get_irr_stats()
        print(format_irr_text(irr_stats))
    except Exception as e:
        print(f"Error: {e}")

    # Section 2a: Source-Participant Demographics (from 07_participant_demographics.py)
    print("\n[PARTICIPANT DEMOGRAPHICS]")
    print("-" * 70)
    try:
        print(format_participant_demographics())
    except Exception as e:
        print(f"Error: {e}")

    # Section 2b: Annotator/Rater Demographics (from 03_rater_demographics.py)
    print("\n[RATER DEMOGRAPHICS]")
    print("-" * 70)
    try:
        print(format_rater_demographics())
    except Exception as e:
        print(f"Error: {e}")

    # Section 2c: High-Confidence Subset (from 06_high_confidence_subset.py)
    print("\n[HIGH-CONFIDENCE SUBSET]")
    print("-" * 70)
    try:
        print(format_high_confidence_subset())
    except Exception as e:
        print(f"Error: {e}")

    # Section 3: BLEU Methods
    print("\n[BLEU METHODS]")
    print("-" * 70)
    print(format_bleu_methods())
    
    # Section 4: NLI Methods
    print("\n[NLI METHODS]")
    print("-" * 70)
    print(format_nli_methods())
    
    # Section: Readability Methods
    print("\n[READABILITY METHODS]")
    print("-" * 70)
    print(format_readability_methods())
    
    # Section 5: Feature Analysis (including BLEU)
    print("\n[FEATURE ANALYSIS]")
    print("-" * 70)
    try:
        analyze_features = importlib.import_module("12_analyze_features")
        # Point the imported module at this run's variant p-values file so the
        # coherence/style means match the rest of the variant outputs.
        analyze_features.PVALS_CSV = (PROJECT_DIR / "output" / "analyze_features"
                                      / variant_name("pvals.csv"))
        feature_stats = analyze_features.get_feature_stats()
        bleu_stats = get_bleu_stats()
        print(format_features_text(feature_stats, bleu_stats))
    except Exception as e:
        print(f"Error: {e}")
    
    # Section 4: Feature Table (with NLI and readability)
    print("\n[FEATURE TABLE]")
    print("-" * 70)
    try:
        nli_stats = get_nli_stats()
        read_stats = get_readability_stats()
        print(format_features_table(feature_stats, bleu_stats, nli_stats, read_stats))
    except Exception as e:
        print(f"Error: {e}")
    
    # Section: Readability Results
    print("\n[READABILITY RESULTS]")
    print("-" * 70)
    try:
        read_stats = get_readability_stats()
        print(format_readability_text(read_stats))
    except Exception as e:
        print(f"Error: {e}")
    
    # Section 5: LIWC Analysis (word-based features only, excluding summary vars)
    print("\n[LIWC ANALYSIS]")
    print("-" * 70)
    try:
        import importlib
        analyze_liwc = importlib.import_module("15_analyze_liwc")
        importlib.reload(analyze_liwc)  # Force reload to get latest version
        liwc_stats = analyze_liwc.get_liwc_stats(words_only=True)
        print(format_liwc_text(liwc_stats))
    except Exception as e:
        print(f"Error: {e}")
    
    # Section 6: LIWC Figure
    print("\n[LIWC FIGURE]")
    print("-" * 70)
    print(format_liwc_figure())
    
    # Section 7: Wordcloud Figure
    print("\n[WORDCLOUD FIGURE]")
    print("-" * 70)
    print(format_wordcloud_figure())
    
    # Section 8: NLI Consistency Results
    print("\n[NLI CONSISTENCY RESULTS]")
    print("-" * 70)
    try:
        nli_stats = get_nli_stats()
        print(format_nli_text(nli_stats))
    except Exception as e:
        print(f"Error: {e}")

    # Section 9: Meta-classifier table + paragraph (from 21_meta_classifiers.py)
    print("\n[META-CLASSIFIER TABLE]")
    print("-" * 70)
    try:
        print(format_meta_classifier_table())
    except Exception as e:
        print(f"Error: {e}")

    print("\n[META-CLASSIFIER TEXT]")
    print("-" * 70)
    try:
        print(format_meta_classifier_text())
    except Exception as e:
        print(f"Error: {e}")

    # Section 10: Decision tree rules + figure (from 21_meta_classifiers.py / 22_plot_decision_tree.py)
    print("\n[DECISION TREE RULES]")
    print("-" * 70)
    try:
        print(format_decision_tree_rules())
    except Exception as e:
        print(f"Error: {e}")

    print("\n[DECISION TREE FIGURE]")
    print("-" * 70)
    print(format_decision_tree_figure())

    # Section 11: Violin / decision-boundary / SHAP-comparison figures (from 23_viz_all.py)
    print("\n[META-FEATURE VIOLIN FIGURE]")
    print("-" * 70)
    print(format_violin_figure())

    print("\n[DECISION BOUNDARY FIGURE]")
    print("-" * 70)
    print(format_decision_boundary_figure())

    print("\n[SHAP COMPARISON FIGURE]")
    print("-" * 70)
    print(format_shap_comparison_figure())

    # Section 12: Linear SVM + margin flagging (from 24_meta_svm.py)
    print("\n[LINEAR SVM RESULTS]")
    print("-" * 70)
    try:
        print(format_svm_text())
    except Exception as e:
        print(f"Error: {e}")

    # Section 13: Permutation-test summary (from 26_permutation_test.py)
    print("\n[PERMUTATION TEST SUMMARY]")
    print("-" * 70)
    try:
        print(format_permutation_test())
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("END OF PAPER TEXT")
    print("=" * 70)


if __name__ == "__main__":
    main()
