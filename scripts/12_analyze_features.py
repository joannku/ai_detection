import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / 'output' / 'analyze_features'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature-set variant: "orig" | "sw" | "drop" | "drop_nowc". Selects which
# per-entry features CSV is analysed and suffixes the p-values output. The
# variant CSVs use suffixed metric names (coherence_sw, ...) and a lowercase
# `final_label` column, both of which are normalised in main() so the rest of
# the logic and get_feature_stats() are unchanged. "drop_nowc" reuses the
# "drop" inputs but excludes word_count from the analysed metrics.
VARIANT = "drop"

_DATA_VARIANT = "drop" if VARIANT == "drop_nowc" else VARIANT
DATA_SUFFIX = "" if _DATA_VARIANT == "orig" else f"_{_DATA_VARIANT}"
SUFFIX = "" if VARIANT == "orig" else f"_{VARIANT}"
EXCLUDE_WC = VARIANT == "drop_nowc"

FEATURES_CSV = ROOT_DIR / 'output' / f'generate_features{DATA_SUFFIX}' / f'features{DATA_SUFFIX}.csv'
PVALS_CSV = OUTPUT_DIR / f'pvals{SUFFIX}.csv'


def get_feature_stats() -> dict:
    """Compute feature comparison statistics for paper text."""
    pvals_df = pd.read_csv(PVALS_CSV)
    
    def get_val(metric: str, col: str) -> float:
        row = pvals_df[pvals_df["metric"] == metric]
        return float(row.iloc[0][col]) if not row.empty else float("nan")
    
    return {
        "coherence_ai": get_val("coherence", "mean_ai"),
        "coherence_human": get_val("coherence", "mean_human"),
        "style_ai": get_val("style_similarity", "mean_ai"),
        "style_human": get_val("style_similarity", "mean_human"),
        "words_ai": get_val("word_count", "mean_ai"),
        "words_human": get_val("word_count", "mean_human"),
        "sentences_ai": get_val("sentence_count", "mean_ai"),
        "sentences_human": get_val("sentence_count", "mean_human"),
        "chars_ai": get_val("character_count", "mean_ai"),
        "chars_human": get_val("character_count", "mean_human"),
    }


def permutation_p_value(group_a: np.ndarray, group_b: np.ndarray, seed: int = 42) -> float:
    group_a = group_a.astype(float)
    group_b = group_b.astype(float)
    obs_diff = float(np.mean(group_a) - np.mean(group_b))
    all_vals = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    rng = np.random.default_rng(seed)
    n_perm = 10000 if all_vals.size <= 2000 else 5000

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(all_vals)
        diff = perm[:n_a].mean() - perm[n_a:].mean()
        if abs(diff) >= abs(obs_diff):
            count += 1
    return (count + 1) / (n_perm + 1)


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)

    # Normalise the sliding-window / drop variants back to the original schema:
    #   - suffixed metric names (coherence_sw, ...) -> canonical names
    #   - lowercase `final_label` (ai/human) -> `label` (AI/Human)
    df = df.rename(columns={
        f"coherence{DATA_SUFFIX}": "coherence",
        f"style_similarity{DATA_SUFFIX}": "style_similarity",
    })
    if 'label' not in df.columns and 'final_label' in df.columns:
        df['label'] = df['final_label'].map({'ai': 'AI', 'human': 'Human'})

    # Expect a 'label' column with values 'Human'/'AI'
    if 'label' not in df.columns:
        raise ValueError("features.csv must include a 'label' column for comparisons")

    # Identify numeric columns to test (exclude id/label-like columns)
    exclude_cols = {"JournalUniqueID", "label"}
    if EXCLUDE_WC:
        exclude_cols.add("word_count")  # drop_nowc ablation
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    results = []
    data = df[df['label'].isin(['Human', 'AI'])]

    for metric in numeric_cols:
        series = data[["label", metric]].dropna()
        a = series[series["label"] == "Human"][metric].to_numpy()
        b = series[series["label"] == "AI"][metric].to_numpy()

        mean_a = float(np.mean(a)) if len(a) else float("nan")
        mean_b = float(np.mean(b)) if len(b) else float("nan")

        if len(a) > 1 and len(b) > 1:
            pval = permutation_p_value(a, b)
            obs_diff = mean_a - mean_b
        else:
            pval = float("nan")
            obs_diff = float("nan")

        results.append({
            "metric": metric,
            "n_human": int(len(a)),
            "n_ai": int(len(b)),
            "mean_human": mean_a,
            "mean_ai": mean_b,
            "obs_diff": float(obs_diff),
            "p_value": float(pval),
            "significant_0.05": bool(pval < 0.05) if not np.isnan(pval) else False,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    out_path = PVALS_CSV
    res_df.to_csv(out_path, index=False)
    # Print semantic feature means for quick preview on slides
    def _fmt(x: float) -> str:
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "nan"
    print(f"✓ Wrote {out_path}")
    for metric in ["coherence", "style_similarity"]:
        row = res_df[res_df["metric"] == metric]
        if not row.empty:
            r = row.iloc[0]
            print(
                f"{metric}\t"
                f"n_human={int(r['n_human'])}\t"
                f"n_ai={int(r['n_ai'])}\t"
                f"mean_human={_fmt(r['mean_human'])}\t"
                f"mean_ai={_fmt(r['mean_ai'])}\t"
                f"diff(H-A)={_fmt(r['obs_diff'])}\t"
                f"p={_fmt(r['p_value'])}"
            )


if __name__ == '__main__':
    main()
