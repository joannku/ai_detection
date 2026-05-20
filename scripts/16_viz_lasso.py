#!/usr/bin/env python3
"""
Build interactive figures for LASSO coefficient analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT_DIR = Path(__file__).resolve().parent.parent
COEFFS_DIR = ROOT_DIR / "output" / "analyze_liwc"
OUTPUT_DIR = ROOT_DIR / "output" / "viz_lasso"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === CONFIGURATION ===
# Set to "all" for all 118 LIWC features, or "words" for word-based only (110, excludes summary vars)
FEATURE_SET = "words"

# CSV file options
COEFF_FILES = {
    "all": "liwc_lasso_coeffs.csv",
    "words": "liwc_lasso_coeffs_words_only.csv",
}


def make_lasso_coeff_figure(coef_df: pd.DataFrame, top_k: int = 25):
    """
    Create a Plotly bar chart for top-|coefficient| LIWC features.
    Expects columns: metric, coefficient, abs_coeff, direction.
    Returns a Plotly Figure.
    """
    if coef_df.empty:
        raise ValueError("Empty coefficient DataFrame.")
    cols_needed = {"metric", "coefficient", "abs_coeff"}
    missing = cols_needed - set(coef_df.columns)
    if missing:
        raise ValueError(f"Missing columns in coef_df: {missing}")
    if "direction" not in coef_df.columns:
        # derive direction if not present
        coef_df = coef_df.copy()
        coef_df["direction"] = coef_df["coefficient"].apply(
            lambda v: "AI" if v > 0 else ("Human" if v < 0 else "Neutral")
        )
    # Order by direction first (AI, then Human, then Neutral), each by |coef| desc
    ai_df = coef_df[coef_df["coefficient"] > 0].sort_values("abs_coeff", ascending=False)
    human_df = coef_df[coef_df["coefficient"] < 0].sort_values("abs_coeff", ascending=False)
    neutral_df = coef_df[coef_df["coefficient"] == 0].copy()
    # Human first (negative), then AI (positive), then Neutral
    ordered_df = pd.concat([human_df, ai_df, neutral_df], axis=0)
    # take top-k overall while preserving the AI-then-Human order
    top_k = max(1, min(top_k, len(ordered_df)))
    top_df = ordered_df.head(top_k).copy()
    fig = px.bar(
        top_df,
        x="coefficient",
        y="metric",
        color="direction",
        orientation="h",
        title=f"Top {top_k} LASSO coefficients (AI vs Human)",
        labels={"coefficient": "Coefficient (L1 logistic)", "metric": "LIWC metric"},
    )
    # Scale height with number of bars; widen left margin for long labels
    height_px = max(600, min(2200, int(top_k * 24)))
    fig.update_layout(margin=dict(l=220, r=10, t=40, b=10), height=height_px)
    # Preserve bar order (descending) and put largest at the TOP
    fig.update_yaxes(categoryorder="array", categoryarray=top_df["metric"].tolist(), automargin=True, autorange="reversed")
    return fig


def make_lasso_coeff_figure_from_csv(csv_path: Path | str, top_k: int = 53):
    """
    Convenience wrapper: read CSV and build the same figure.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return make_lasso_coeff_figure(df, top_k=top_k)


def make_lasso_coeff_multiplot(coef_df: pd.DataFrame, top_k_per_panel: int = 25):
    """
    Side-by-side bar charts: Human-directed (negative) on left, AI-directed (positive) on right.
    Returns a Plotly Figure.
    """
    if coef_df.empty:
        raise ValueError("Empty coefficient DataFrame.")
    cols_needed = {"metric", "coefficient", "abs_coeff"}
    missing = cols_needed - set(coef_df.columns)
    if missing:
        raise ValueError(f"Missing columns in coef_df: {missing}")
    # Ensure direction column exists
    if "direction" not in coef_df.columns:
        coef_df = coef_df.copy()
        coef_df["direction"] = coef_df["coefficient"].apply(
            lambda v: "AI" if v > 0 else ("Human" if v < 0 else "Neutral")
        )
    human_df = coef_df[coef_df["coefficient"] < 0].sort_values("abs_coeff", ascending=False).head(top_k_per_panel).copy()
    ai_df = coef_df[coef_df["coefficient"] > 0].sort_values("abs_coeff", ascending=False).head(top_k_per_panel).copy()

    max_rows = max(len(human_df), len(ai_df))
    height_px = max(600, min(2400, int(max_rows * 24)))
    # Keep export width proportional to height so saved image matches on-screen ratio
    width_px = max(900, min(1400, int(height_px * 0.9)))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Human", "AI"),
        horizontal_spacing=0.12,
    )
    # Human panel (left)
    fig.add_trace(
        go.Bar(
            x=human_df["coefficient"],
            y=human_df["metric"],
            orientation="h",
            marker_color="#1f77b4",  # blue
            name="Human",
            hovertemplate="metric=%{y}<br>coef=%{x:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # AI panel (right)
    fig.add_trace(
        go.Bar(
            x=ai_df["coefficient"],
            y=ai_df["metric"],
            orientation="h",
            marker_color="#d62728",  # red
            name="AI",
            hovertemplate="metric=%{y}<br>coef=%{x:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_layout(showlegend=False)
    # Ensure all tick labels are shown and not auto-thinned
    hlabels = human_df["metric"].tolist()
    alabels = ai_df["metric"].tolist()
    
    # Font sizes
    TICK_FONT_SIZE = 18
    AXIS_TITLE_SIZE = 22
    SUBPLOT_TITLE_SIZE = 26
    
    # Left y-axis
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=hlabels,
        automargin=True,
        autorange="reversed",
        tickmode="array",
        tickvals=hlabels,
        ticktext=hlabels,
        title_text="LIWC metric",
        title_font_size=AXIS_TITLE_SIZE,
        tickfont_size=TICK_FONT_SIZE,
        row=1, col=1,
    )
    # Right y-axis
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=alabels,
        automargin=True,
        autorange="reversed",
        showticklabels=True,
        side="right",
        tickmode="array",
        tickvals=alabels,
        ticktext=alabels,
        tickfont_size=TICK_FONT_SIZE,
        row=1, col=2,
    )
    # Reasonable margins and height so all labels fit
    fig.update_layout(
        margin=dict(l=250, r=250, t=60, b=60),
        height=height_px,
        width=width_px,
        font=dict(size=TICK_FONT_SIZE),
    )
    # Update subplot titles font size
    for annotation in fig.layout.annotations:
        annotation.font.size = SUBPLOT_TITLE_SIZE
    
    # Fixed scale -1 to 1 for both panels
    fig.update_xaxes(title_text="Coefficient", range=[-1, 0], title_font_size=AXIS_TITLE_SIZE, tickfont_size=TICK_FONT_SIZE, row=1, col=1)
    fig.update_xaxes(title_text="Coefficient", range=[0, 1], title_font_size=AXIS_TITLE_SIZE, tickfont_size=TICK_FONT_SIZE, row=1, col=2)
    return fig


def main(feature_set: str = "words"):
    """
    Default runner: loads LASSO coefficients CSV, shows a side-by-side multiplot,
    and saves PNG/HTML copies to output/.
    
    Args:
        feature_set: "all" for all LIWC features, "words" for word-based only (excludes summary vars)
    """
    csv_name = COEFF_FILES.get(feature_set, COEFF_FILES["words"])
    csv = COEFFS_DIR / csv_name
    
    if not csv.exists():
        print(f"Error: {csv} not found. Run 15_analyze_liwc.py first.")
        return
    
    print(f"Loading: {csv_name} ({feature_set} features)")
    df = pd.read_csv(csv)
    
    # Count selected features
    n_selected = df[df["selected"] == True].shape[0] if "selected" in df.columns else len(df)
    print(f"Visualizing {n_selected} selected features")
    
    fig = make_lasso_coeff_multiplot(df, top_k_per_panel=n_selected)
    fig.show()  # interactive
    
    # Save PNG and HTML with suffix
    suffix = "_words_only" if feature_set == "words" else ""
    out = OUTPUT_DIR / f"liwc_lasso_coeffs_multiplot{suffix}.png"
    try:
        save_w = int(fig.layout.width) if fig.layout.width else 1000
        save_h = int(fig.layout.height) if fig.layout.height else 900
        # scale=3 for high resolution (300 DPI equivalent)
        fig.write_image(out.as_posix(), width=save_w, height=save_h, scale=3)
        print(f"✓ Saved {out} (high resolution, scale=3)")
    except Exception as e:
        print(f"Could not save PNG (is kaleido installed/compatible?): {e}")
    
    html_out = OUTPUT_DIR / f"liwc_lasso_coeffs_multiplot{suffix}.html"
    try:
        fig.write_html(html_out.as_posix(), include_plotlyjs="cdn")
        print(f"✓ Saved {html_out}")
    except Exception:
        pass

    paper_figs = ROOT_DIR / "overleaf_paper" / "figures"
    if paper_figs.exists() and out.exists():
        import shutil
        dst = paper_figs / out.name
        shutil.copyfile(out, dst)
        print(f"  Copied → {dst}")


if __name__ == "__main__":
    main(feature_set=FEATURE_SET)


