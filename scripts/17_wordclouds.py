#!/usr/bin/env python3
"""
Generate wordclouds for selected LIWC categories from output/get_top_words/liwc_topwords.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
TOPWORDS_PATH = OUTPUT_DIR / "get_top_words" / "liwc_topwords.json"
WC_OUT_DIR = OUTPUT_DIR / "wordclouds"
WC_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_category(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower())


def _to_freqs(value: Any) -> Dict[str, float]:
    """
    Convert various structures into a {word: weight} mapping.
    Supports:
      - dict: {word: weight}
      - list of [word, weight]
      - list of {"word": w, "weight": v}
      - list of strings (treated as weight=1.0)
    """
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    freqs: Dict[str, float] = {}
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                w = item.get("word") or item.get("term") or item.get("token")
                v = item.get("weight") or item.get("score") or 1.0
                if w is not None:
                    freqs[str(w)] = float(v)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                freqs[str(item[0])] = float(item[1])
            else:
                freqs[str(item)] = freqs.get(str(item), 0.0) + 1.0
    return freqs


def load_topwords(path: Path = TOPWORDS_PATH) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with path.open() as f:
        data = json.load(f)
    # Expect a mapping: {category: list/dict}
    if isinstance(data, dict):
        return {str(cat): _to_freqs(vals) for cat, vals in data.items()}
    # Or a list of records with category + words
    out: Dict[str, Dict[str, float]] = {}
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            cat = rec.get("category") or rec.get("metric") or rec.get("name")
            vals = rec.get("words") or rec.get("topwords") or rec.get("tokens") or rec.get("items")
            if cat is None or vals is None:
                continue
            out[str(cat)] = _to_freqs(vals)
    return out


def generate_wordclouds(
    categories: Union[List[str], Dict[str, Union[str, Dict[str, Any]]]],
    topwords: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    width: int = 1600,
    height: int = 1000,
    max_words: int = 50,
    background_color: str = "white",
    colormap: str = "viridis",
    seed: int = 42,
    seeds_by_category: Optional[Dict[str, int]] = None,
    show: bool = True,
) -> List[Tuple[str, Path]]:
    """
    Build wordcloud PNGs for the given categories. Returns [(category, path)].

    categories:
      - If list[str]: all categories use the same 'colormap' argument
      - If dict[str, str]: maps category -> matplotlib colormap name (e.g., 'Blues_r', 'gist_heat')
      - If dict[str, dict]: per-category config, keys accepted:
          { "cmap" | "colormap": <name>, "seed": <int> }

    seed:
      - Base seed to make layouts reproducible. We combine it with a stable hash
        of the category name so different categories look different but are deterministic.
    seeds_by_category:
      - Optional mapping overriding the seed per category (ignored if categories[cat] provides "seed").
    show:
      - If True, display each wordcloud interactively using matplotlib.
    """
    if topwords is None:
        topwords = load_topwords()
    WC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[str, Path]] = []

    # Normalize categories to a mapping: category -> config dict
    if isinstance(categories, dict):
        norm: Dict[str, Dict[str, Any]] = {}
        for cat, cfg in categories.items():
            if isinstance(cfg, dict):
                cm = cfg.get("cmap") or cfg.get("colormap") or colormap
                sd = cfg.get("seed")
                norm[cat] = {"cmap": cm, "seed": sd}
            else:
                norm[cat] = {"cmap": str(cfg), "seed": None}
    else:
        norm = {cat: {"cmap": colormap, "seed": None} for cat in categories}

    for cat, cfg in norm.items():
        if cat not in topwords:
            print(f"Warning: category not found in topwords: {cat}")
            continue
        freqs = topwords[cat]
        # Resolve colormap and seed
        cmap = cfg.get("cmap") or colormap
        explicit_seed = cfg.get("seed")
        if explicit_seed is None and seeds_by_category and cat in seeds_by_category:
            explicit_seed = seeds_by_category[cat]
        # Derive deterministic but distinct seed per category (fallback)
        derived_seed = int(explicit_seed) if explicit_seed is not None else int((seed + (abs(hash(cat)) % 10_000_000)) % (2**31 - 1))
        
        print(f"Generating: {cat} (cmap={cmap}, seed={derived_seed})")
        
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=cmap or colormap,
            prefer_horizontal=1.0,
            random_state=derived_seed,
        )
        wc.generate_from_frequencies(freqs)
        
        # Display interactively
        if show:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{cat} (cmap={cmap}, seed={derived_seed})", fontsize=16)
            plt.tight_layout()
            plt.show()
        
        out_path = WC_OUT_DIR / f"{_normalize_category(cat)}.png"
        wc.to_file(out_path.as_posix())
        results.append((cat, out_path))
        print(f"✓ Saved {out_path}")
    return results

def generate_multipanel_figure(
    human_categories: Dict[str, Dict[str, Any]],
    ai_categories: Dict[str, Dict[str, Any]],
    topwords: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    width: int = 800,
    height: int = 500,
    max_words: int = 25,
    background_color: str = "white",
    figsize: Tuple[float, float] = (12, 12),
    dpi: int = 300,
) -> None:
    """
    Create a 2-column x 3-row multipanel figure:
    - Left column: Human-associated categories (blue)
    - Right column: AI-associated categories (red/warm)
    """
    if topwords is None:
        topwords = load_topwords()
    
    human_cats = list(human_categories.keys())
    ai_cats = list(ai_categories.keys())
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Column titles (increased pad to avoid overlap with category labels)
    axes[0, 0].set_title("Human", fontsize=20, fontweight="bold", pad=40)
    axes[0, 1].set_title("AI", fontsize=20, fontweight="bold", pad=40)
    
    # Generate Human wordclouds (left column)
    for i, cat in enumerate(human_cats):
        cfg = human_categories[cat]
        cmap = cfg.get("cmap", "Blues_r")
        seed = cfg.get("seed", 42)
        
        if cat not in topwords:
            print(f"Warning: {cat} not found")
            continue
            
        wc = WordCloud(
            width=width, height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=cmap,
            prefer_horizontal=1.0,
            random_state=seed,
        )
        wc.generate_from_frequencies(topwords[cat])
        
        ax = axes[i, 0]
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.text(0.5, 1.08, cat, transform=ax.transAxes, fontsize=16, 
                ha="center", va="bottom", style="italic")
    
    # Generate AI wordclouds (right column)
    for i, cat in enumerate(ai_cats):
        cfg = ai_categories[cat]
        cmap = cfg.get("cmap", "gist_heat")
        seed = cfg.get("seed", 42)
        
        if cat not in topwords:
            print(f"Warning: {cat} not found")
            continue
            
        wc = WordCloud(
            width=width, height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=cmap,
            prefer_horizontal=1.0,
            random_state=seed,
        )
        wc.generate_from_frequencies(topwords[cat])
        
        ax = axes[i, 1]
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.text(0.5, 1.08, cat, transform=ax.transAxes, fontsize=16,
                ha="center", va="bottom", style="italic")
    
    # Adjust spacing: no gap between columns, some vertical space for labels
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, hspace=0.22, wspace=0)
    
    # Save high-resolution figure
    out_path = WC_OUT_DIR / "wordclouds_multipanel.png"
    WC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    print(f"✓ Saved {out_path} (dpi={dpi})")

    paper_figs = WC_OUT_DIR.parent.parent / "overleaf_paper" / "figures"
    if paper_figs.exists():
        import shutil
        dst = paper_figs / "wordclouds_multipanel.png"
        shutil.copyfile(out_path, dst)
        print(f"  Copied → {dst}")
    
    plt.show()


def load_liwc_categories() -> List[str]:
    """Iterate over all liwc categories from liwc_topwords.json, load keys as list.

    Skips the four LIWC-22 proprietary summary variables (Analytic, Clout,
    Authentic, Tone) - these are composite scores, not lexical categories, so
    they have no meaningful wordcloud.
    """
    proprietary = {"Analytic", "Clout", "Authentic", "Tone"}
    with open(TOPWORDS_PATH, "r") as f:
        data = json.load(f)
    categories = [c for c in data.keys() if c not in proprietary]
    print(f"Loaded {len(categories)} LIWC categories")
    for category in list(categories):
        if len(data[category]) == 0 or data[category] == dict():
            categories.remove(category)
            print(f"Removed category {category} because it has no words")
    print(f"Loaded {len(categories)} LIWC categories with words")
    return categories


def main() -> None:
    """
    Generate wordclouds for selected LIWC categories:
    - Human-associated (blue): allure, focuspresent, ppron
    - AI-associated (red/warm): Perception, curiosity, memory
    
    Change seeds to get different color arrangements within the colormap.
    """
    # Human-associated categories (blue colormaps)
    human_categories = {
        "allure": {"cmap": "Blues_r", "seed": 42},
        "focuspresent": {"cmap": "Blues_r", "seed": 7},
        "ppron": {"cmap": "Blues_r", "seed": 23},
    }
    
    # AI-associated categories (warm/red colormaps - gist_heat as previously used)
    ai_categories = {
        "Perception": {"cmap": "gist_heat", "seed": 1},
        "curiosity": {"cmap": "gist_heat", "seed": 300},
        "memory": {"cmap": "gist_heat", "seed": 1},
    }
    
    CATEGORIES = {**human_categories, **ai_categories}
    SEED = 42  # base fallback if a category-specific seed is not provided

    # === CONFIG ===
    GENERATE_INDIVIDUAL = False  # Generate individual wordcloud PNGs
    GENERATE_MULTIPANEL = True   # Generate combined 2x3 multipanel figure
    SHOW_INTERACTIVE = True      # Show figures interactively

    print(f"Generating wordclouds for {len(CATEGORIES)} categories:")
    print(f"  Human (Blues_r): {list(human_categories.keys())}")
    print(f"  AI (gist_heat): {list(ai_categories.keys())}")
    print()
    
    if GENERATE_INDIVIDUAL:
        generate_wordclouds(CATEGORIES, seed=SEED, show=SHOW_INTERACTIVE)
    
    if GENERATE_MULTIPANEL:
        generate_multipanel_figure(human_categories, ai_categories)


if __name__ == "__main__":
    main()


# CATEGORIES: Dict[str, Dict[str, Any]] = {
#     "emo_anger": {"cmap": "Blues_r", "seed": 42},
#     # "Perception": {"cmap": "gist_heat", "seed": 1},
#     # "ipron": {"cmap": "gist_heat", "seed": 10},
#     "focuspresent": {"cmap": "Blues_r", "seed": 206},
#     # "quantity": {"cmap": "gist_heat", "seed": 202},
#     # "Cognition": {"cmap": "Blues_r", "seed": 505},
# }