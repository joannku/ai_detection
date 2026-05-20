#!/usr/bin/env python3
"""
Extract top words per LIWC category for this project.
Reads input/journals.csv for text, uses LIWC CLI context mode, and writes
output/get_top_words/liwc_topwords.json with {category: {word: freq, ...}, ...}.
Only runs if file doesn't exist yet.
"""

import pandas as pd
import json
import subprocess
import tempfile
import os
from pathlib import Path
from collections import Counter
import re
import shutil
import start_liwc_server

ROOT = Path(__file__).resolve().parent.parent


def load_project_texts() -> pd.DataFrame:
    """Load project journals (id + text)."""
    journals_path = ROOT / "input" / "journals.csv"
    if not journals_path.exists():
        print(f"❌ {journals_path} not found")
        return pd.DataFrame()
    df = pd.read_csv(journals_path)
    text_col = None
    if "JournalAnonymised" in df.columns:
        text_col = "JournalAnonymised"
    elif "JournalContent" in df.columns:
        text_col = "JournalContent"
    else:
        print("❌ No text column found (expected JournalAnonymised or JournalContent)")
        return pd.DataFrame()
    out = df[["JournalUniqueID", text_col]].copy()
    out.rename(columns={text_col: "Text"}, inplace=True)
    out["Text"] = out["Text"].fillna("").astype(str)
    return out


def load_liwc_results() -> pd.DataFrame:
    """Load LIWC results to get category list and select rows per category."""
    liwc_path = ROOT / "output" / "liwc" / "liwc.csv"
    if not liwc_path.exists():
        print(f"❌ {liwc_path} not found (run liwc.py first)")
        return pd.DataFrame()
    return pd.read_csv(liwc_path)

def get_liwc_columns(liwc_df: pd.DataFrame) -> list[str]:
    """Get LIWC category columns from liwc.csv, excluding metadata and global indices.

    The four LIWC-22 proprietary summary variables (Analytic, Clout, Authentic,
    Tone) are not lexical categories - they are composite scores computed by
    LIWC's internal algorithms - so the CLI's per-category mode does not return
    meaningful word lists for them. We skip them outright.
    """
    excluded = {
        'JournalUniqueID', 'Row ID', 'Segment', 'WC',
        'Analytic', 'Clout', 'Authentic', 'Tone',
    }
    cols = [c for c in liwc_df.columns if c not in excluded]
    # Keep only numeric columns
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(liwc_df[c])]
    print(f"📊 Using {len(cols)} LIWC categories")
    return cols

def get_liwc_category_name(liwc_col: str) -> str:
    """The liwc.csv column names match LIWC category names."""
    return liwc_col

def get_category_words_with_liwc_cli(content_text: str, liwc_category: str) -> list[str]:
    """Use LIWC CLI (context mode) to list words matching a category from text."""
    # Resolve CLI path
    cli = shutil.which("LIWC-22-cli") or \
          "/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-cli"
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content_text)
            temp_file = f.name

        # Primary call with long flags
        cmd = [
            cli,
            '--mode', 'context',
            '--input', temp_file,
            '--output', 'console',
            '--dictionary', 'LIWC22',
            '--category', liwc_category,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Fallback to short flags if needed
        if result.returncode != 0:
            cmd = [cli, '-m', 'context', '-i', temp_file, '-o', 'console', '-d', 'LIWC22', '-categ', liwc_category]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"   ⚠️  LIWC CLI error for {liwc_category}: {result.stderr or result.stdout}")
                return []

        matched_words = []
        for line in result.stdout.strip().split('\n'):
            if line.strip() and line.startswith('{'):
                try:
                    data = json.loads(line)
                    if 'Match' in data and isinstance(data['Match'], str):
                        w = data['Match'].strip().lower()
                        if w:
                            matched_words.append(w)
                except json.JSONDecodeError:
                    continue

        return matched_words
    except Exception as e:
        print(f"   ⚠️  Error processing {liwc_category}: {e}")
        return []
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass

def analyze_category_words(merged_df: pd.DataFrame, liwc_col: str) -> dict[str, int]:
    """Aggregate texts where category>0, run LIWC context, return word->count."""
    rows = merged_df[merged_df[liwc_col] > 0]
    if rows.empty:
        return {}
    texts = [t for t in rows['Text'] if isinstance(t, str) and t.strip()]
    if not texts:
        return {}
    combined_text = "\n\n".join(texts)
    words = get_category_words_with_liwc_cli(combined_text, liwc_col)
    return dict(Counter(words).most_common())

def main():
    
    print("🔍 Analyzing LIWC categories for word frequencies (project journals)...")

    # Ensure LIWC license server is running
    try:
        start_liwc_server.start_liwc_server()
    except Exception:
        pass

    liwc_df = load_liwc_results()
    texts_df = load_project_texts()
    if liwc_df.empty or texts_df.empty:
        return

    # Merge text onto LIWC results to gather content per category
    merged = liwc_df.merge(texts_df, on="JournalUniqueID", how="left")
    liwc_cols = get_liwc_columns(liwc_df)

    results: dict[str, dict[str, int]] = {}
    for col in liwc_cols:
        print(f"📝 Analyzing {col}...")
        word_counts = analyze_category_words(merged, col)
        results[col] = word_counts
        if word_counts:
            top5 = list(word_counts.items())[:5]
            print(f"   ✅ {len(word_counts)} unique words | top: {', '.join([f'{w}({c})' for w,c in top5])}")
        else:
            print(f"   ⚠️  No words identified for {col}")

    # Save
    out_dir = ROOT / 'output' / 'get_top_words'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'liwc_topwords.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n🎯 Results saved to {out_path}")
    print(f"📊 Analyzed {len(liwc_cols)} LIWC categories")

if __name__ == "__main__":
    main()
