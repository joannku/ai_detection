#!/usr/bin/env python3
"""Extract top words per LIWC category from project journals.

Reads input/journals.csv, runs LIWC-22 CLI in context mode for each LIWC
category, and writes output/get_top_words/liwc_topwords.json as
{category: {word: freq, ...}, ...}. Skips the run if the output already exists.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import start_liwc_server  # type: ignore[import-not-found]  # noqa: E402  (from ../tools)

ROOT = Path(__file__).resolve().parent.parent
JOURNALS_CSV = ROOT / "input" / "journals.csv"
LIWC_CSV = ROOT / "output" / "liwc" / "liwc.csv"
OUT_PATH = ROOT / "output" / "get_top_words" / "liwc_topwords.json"

# Analytic/Clout/Authentic/Tone are LIWC composite scores, not lexical
# categories, so the per-category CLI mode returns nothing useful for them.
META_COLS = {"JournalUniqueID", "Row ID", "Segment", "WC"}
COMPOSITE_SCORES = {"Analytic", "Clout", "Authentic", "Tone"}

LIWC_CLI = (
    shutil.which("LIWC-22-cli")
    or "/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-cli"
)


def load_journals() -> pd.DataFrame:
    """Return a (JournalUniqueID, Text) frame, preferring the anonymised text."""
    df = pd.read_csv(JOURNALS_CSV)
    text_col = "JournalAnonymised" if "JournalAnonymised" in df.columns else "JournalContent"
    out = df[["JournalUniqueID", text_col]].rename(columns={text_col: "Text"})
    out["Text"] = out["Text"].fillna("").astype(str)
    return out


def liwc_category_cols(liwc_df: pd.DataFrame) -> list[str]:
    """Numeric LIWC category columns, with metadata and composite scores stripped."""
    skip = META_COLS | COMPOSITE_SCORES
    return [
        c for c in liwc_df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(liwc_df[c])
    ]


def liwc_matched_words(text: str, category: str) -> list[str]:
    """Return every word in `text` that LIWC tags as belonging to `category`."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name

    try:
        cmd = [
            LIWC_CLI,
            "--mode", "context",
            "--input", tmp,
            "--output", "console",
            "--dictionary", "LIWC22",
            "--category", category,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  LIWC CLI error for {category}: {result.stderr or result.stdout}")
            return []

        # The CLI prints one JSON object per matched token, e.g. {"Match": "happy", ...}.
        words: list[str] = []
        for line in result.stdout.splitlines():
            if not line.startswith("{"):
                continue
            try:
                match = json.loads(line).get("Match")
            except json.JSONDecodeError:
                continue
            if isinstance(match, str) and match.strip():
                words.append(match.strip().lower())
        return words
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def top_words_for_category(merged: pd.DataFrame, category: str) -> dict[str, int]:
    """Aggregate texts that scored >0 for `category` and return word -> count."""
    texts = [t for t in merged.loc[merged[category] > 0, "Text"] if t.strip()]
    if not texts:
        return {}
    words = liwc_matched_words("\n\n".join(texts), category)
    return dict(Counter(words).most_common())


def main() -> None:
    if OUT_PATH.exists():
        print(f"{OUT_PATH} already exists - skipping.")
        return
    if not JOURNALS_CSV.exists():
        sys.exit(f"Missing {JOURNALS_CSV}")
    if not LIWC_CSV.exists():
        sys.exit(f"Missing {LIWC_CSV} (run 13_liwc.py first)")

    # Best-effort: only relevant on macOS where the LIWC app is installed.
    try:
        start_liwc_server.start_liwc_server()
    except Exception as e:
        print(f"Could not start LIWC server ({e}); assuming it's already running.")

    liwc_df = pd.read_csv(LIWC_CSV)
    merged = liwc_df.merge(load_journals(), on="JournalUniqueID", how="left")
    categories = liwc_category_cols(liwc_df)
    print(f"Analyzing {len(categories)} LIWC categories...")

    results: dict[str, dict[str, int]] = {}
    for cat in categories:
        counts = top_words_for_category(merged, cat)
        results[cat] = counts
        top5 = ", ".join(f"{w}({c})" for w, c in list(counts.items())[:5]) or "(no matches)"
        print(f"  {cat}: {len(counts)} unique | top: {top5}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
