#!/usr/bin/env python3
"""Run GPTZero AI detection on every journal entry.

Reads `output/export/final_journals.csv`, calls the GPTZero `/v2/predict/text`
endpoint once per entry, and writes per-entry predictions to
`output/gptzero/gptzero_results.csv`.

The script is **resumable**: results are flushed to disk every 50 calls and
on completion, and the next run skips any `JournalUniqueID` that is already
present. Running again therefore costs only the (rate-limited) API calls
for new or previously-failed rows.

Output columns:
  JournalUniqueID, gptzero_ai_prob, gptzero_mixed_prob, gptzero_class,
  gptzero_burstiness, gptzero_error (only set on failures)

The agreement / accuracy analysis used by the paper is in
`scripts/05_compare_gptzero.py`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ── Configuration ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "output" / "export" / "final_journals.csv"
OUTPUT_DIR = ROOT / "output" / "gptzero"
OUTPUT_CSV = OUTPUT_DIR / "gptzero_results.csv"

API_URL = "https://api.gptzero.me/v2/predict/text"
SAVE_EVERY = 50            # flush partial results every N successful calls
RATE_LIMIT_S = 0.5         # seconds to sleep between calls (be a polite client)
REQUEST_TIMEOUT_S = 30

load_dotenv()
API_KEY = os.getenv("gpt_zero_api")
if not API_KEY:
    raise RuntimeError("Set gpt_zero_api in .env")


# ── API call ───────────────────────────────────────────────────────────────

def detect(text: str) -> dict:
    """Call GPTZero on a single text and return the first document object.

    GPTZero returns a list under the `documents` key; we always send one
    document per call so we read element 0. Network / HTTP errors are
    propagated to the caller, which records them per-row.
    """
    resp = requests.post(
        API_URL,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
        json={"document": text},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    documents = resp.json().get("documents", [{}])
    return documents[0] if documents else {}


def _row_from_response(journal_id: str, doc: dict) -> dict:
    """Pull the four fields we care about out of a GPTZero `documents[0]` dict."""
    return {
        "JournalUniqueID": journal_id,
        "gptzero_ai_prob": doc.get("completely_generated_prob"),
        "gptzero_mixed_prob": doc.get("average_generated_prob"),
        "gptzero_class": doc.get("predicted_class"),
        "gptzero_burstiness": doc.get("overall_burstiness"),
    }


# ── Disk I/O (resumable accumulator) ──────────────────────────────────────

def _load_done(out_path: Path) -> tuple[pd.DataFrame, set[str]]:
    """Return any previously-saved results plus the set of completed IDs."""
    if not out_path.exists():
        return pd.DataFrame(), set()
    done = pd.read_csv(out_path)
    return done, set(done["JournalUniqueID"].astype(str))


def _flush(new_rows: list[dict], out_path: Path) -> None:
    """Append `new_rows` to the on-disk results file, deduping by entry ID."""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if out_path.exists():
        prior = pd.read_csv(out_path)
        new_df = (pd.concat([prior, new_df], ignore_index=True)
                    .drop_duplicates("JournalUniqueID", keep="last"))
    new_df.to_csv(out_path, index=False)


# ── Driver ────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    journals = pd.read_csv(INPUT_CSV)
    _, done_ids = _load_done(OUTPUT_CSV)
    if done_ids:
        print(f"Resuming: {len(done_ids)} entries already processed")

    pending = journals[~journals["JournalUniqueID"].astype(str).isin(done_ids)]
    n_pending = len(pending)
    if n_pending == 0:
        print("Nothing to do — all entries already have GPTZero predictions.")
        return

    new_rows: list[dict] = []
    for i, (_, row) in enumerate(pending.iterrows(), start=1):
        journal_id = row["JournalUniqueID"]
        try:
            doc = detect(row["text"])
            new_rows.append(_row_from_response(journal_id, doc))
            print(f"[{i}/{n_pending}] {journal_id}: "
                  f"{doc.get('predicted_class')}")
        except Exception as e:
            new_rows.append({"JournalUniqueID": journal_id, "gptzero_error": str(e)})
            print(f"[{i}/{n_pending}] [ERROR] {journal_id}: {e}")

        time.sleep(RATE_LIMIT_S)

        # Periodically flush so a crash mid-run does not throw away work.
        if i % SAVE_EVERY == 0:
            _flush(new_rows, OUTPUT_CSV)
            new_rows = []  # already on disk

    _flush(new_rows, OUTPUT_CSV)
    print(f"\u2713 Done \u2192 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
