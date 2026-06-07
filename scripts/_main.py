#!/usr/bin/env python3
"""Pipeline runner: executes every scripts/NN_*.py in order, then 99_paper_text.py.

Each step runs in a fresh subprocess using the same Python interpreter that
launched this script. To use the project venv, invoke with:

    .venv/bin/python scripts/_main.py

By default, steps whose declared output files already exist (and, for
journal-keyed CSVs, whose outputs cover every JournalUniqueID in
``output/export/final_journals.csv``) are treated as "cached" and skipped.
Pass ``--force`` to re-run everything regardless of cache.

The runner halts on the first non-zero exit code.

Examples:
  python scripts/_main.py                       # run all (cached steps skipped)
  python scripts/_main.py --force               # re-run everything
  python scripts/_main.py --from 19             # resume from step 19
  python scripts/_main.py --only 21,22          # run just those steps
  python scripts/_main.py --skip 04,13          # skip extra steps
  python scripts/_main.py --dry-run             # preview without executing
  python scripts/_main.py --include-default-skipped   # also run 04_gptzero (needs API key)
  python scripts/_main.py --no-paper-text       # don't run 99_paper_text at the end
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
STEP_PATTERN = re.compile(r"^(\d{2})_[A-Za-z0-9_]+\.py$")
ORCHESTRATOR = "99_paper_text.py"

# Steps skipped by default — they need external credentials or are very slow.
# Override with --include-default-skipped (or list explicitly via --only / --from).
DEFAULT_SKIP: set[str] = {"04"}  # 04_gptzero — needs GPTZERO_API_KEY and is rate-limited

# Reference set used for journal-coverage cache checks.
REFERENCE_FILE = "output/export/final_journals.csv"
REFERENCE_COL = "JournalUniqueID"


@dataclass
class StepCheck:
    """Defines completion criteria for one pipeline step.

    ``files``: paths (relative to repo root) that must all exist & be non-empty.
        If empty, the step is treated as having no persisted output and is
        always re-run (e.g. `02_irr` only prints).
    ``coverage``: optional ``(file, column)`` tuple — column values must include
        every JournalUniqueID in ``REFERENCE_FILE``. Skipped if the reference
        is missing or empty (e.g. on a cold first run).
    """
    files: list[str] = field(default_factory=list)
    coverage: tuple[str, str] | None = None


# Per-step completion checks. Subfolder names match script basenames so this
# table doesn't churn if scripts are renumbered.
CHECKS: dict[str, StepCheck] = {
    "00_combine_ratings": StepCheck(
        files=["ratings/all_ratings.csv"],
    ),
    "01_export": StepCheck(
        files=["output/export/final_journals.csv"],
    ),
    "02_irr": StepCheck(
        files=[],  # only prints; always re-run (fast)
    ),
    "03_rater_demographics": StepCheck(
        files=[
            "output/rater_demographics/rater_demographics_clean.csv",
            "output/rater_demographics/rater_demographics.txt",
            "output/rater_demographics/rater_demographics.tex",
            "output/rater_demographics/rater_demographics_paragraph.tex",
        ],
    ),
    "04_gptzero": StepCheck(
        files=["output/gptzero/gptzero_results.csv"],
        coverage=("output/gptzero/gptzero_results.csv", "JournalUniqueID"),
    ),
    "05_compare_gptzero": StepCheck(
        files=[
            "output/compare_gptzero/gptzero_comparison.csv",
            "output/compare_gptzero/confusion_heatmap.png",
        ],
        coverage=("output/compare_gptzero/gptzero_comparison.csv", "JournalUniqueID"),
    ),
    "06_high_confidence_subset": StepCheck(
        files=[
            "output/high_confidence_subset/high_confidence_subset.csv",
            "output/high_confidence_subset/high_confidence_summary.txt",
        ],
        # No coverage check — this is a *subset* by design.
    ),
    "07_participant_demographics": StepCheck(
        files=[
            "output/participant_demographics/participant_demographics.csv",
            "output/participant_demographics/participant_demographics.txt",
        ],
    ),
    "08_bleu": StepCheck(
        files=[
            "output/bleu/bleu_by_label.csv",
            "output/bleu/bleu_pairwise_distribution.csv",
            "output/bleu/participant_bleu.csv",
        ],
        # participant_bleu is participant-keyed; bleu_by_label is aggregate.
        # No journal-coverage check — bleu is computed at participant level.
    ),
    # Canonical pipeline drops entries with <2 sentences, so outputs carry the
    # _drop suffix and are a *subset* (no journal-coverage check).
    "09_nli_consistency": StepCheck(
        files=[
            "output/nli_consistency_drop/nli_consistency_drop.csv",
            "output/nli_consistency_drop/nli_consistency_drop_by_label.csv",
        ],
    ),
    "10_readability": StepCheck(
        files=[
            "output/readability/readability.csv",
            "output/readability/readability_by_label.csv",
        ],
        coverage=("output/readability/readability.csv", "JournalUniqueID"),
    ),
    "11_generate_features": StepCheck(
        files=["output/generate_features_drop/features_drop.csv"],
        # Subset by design (drops <2-sentence entries) — no coverage check.
    ),
    "12_analyze_features": StepCheck(
        files=["output/analyze_features/pvals_drop.csv"],
    ),
    "13_liwc": StepCheck(
        files=["output/liwc/liwc.csv"],
        coverage=("output/liwc/liwc.csv", "JournalUniqueID"),
    ),
    "14_get_top_words": StepCheck(
        files=["output/get_top_words/liwc_topwords.json"],
    ),
    "15_analyze_liwc": StepCheck(
        files=[
            "output/analyze_liwc/liwc_lasso_coeffs.csv",
            "output/analyze_liwc/liwc_lasso_coeffs_words_only.csv",
            "output/analyze_liwc/liwc_lasso_cv.csv",
            "output/analyze_liwc/liwc_lasso_cv_words_only.csv",
        ],
    ),
    "16_viz_lasso": StepCheck(
        files=[
            "output/viz_lasso/liwc_lasso_coeffs_multiplot_words_only.png",
            "output/viz_lasso/liwc_lasso_coeffs_multiplot_words_only.html",
        ],
    ),
    "17_wordclouds": StepCheck(
        files=["output/wordclouds/wordclouds_multipanel.png"],
    ),
    "18_automl_shap": StepCheck(
        files=[
            "output/automl_shap/all_features_drop.csv",
            "output/automl_shap/shap_summary_drop.png",
            "output/automl_shap/shap_importance_drop.csv",
        ],
    ),
    "19_merge_features": StepCheck(
        files=["output/merge_features/meta_features_drop.csv"],
        # meta_features is the high-confidence subset — not all journals.
    ),
    "20_merge_features_full": StepCheck(
        files=["output/merge_features_full/meta_features_full_drop.csv"],
        # All valid (non-excluded) entries — also a subset of all journals.
    ),
    "21_meta_classifiers": StepCheck(
        files=[
            "output/meta_classifiers/meta_lasso_coeffs_drop.csv",
            "output/meta_classifiers/meta_tree_rules_drop.txt",
            "output/meta_classifiers/meta_cv_results_drop.csv",
            "output/meta_classifiers/meta_feature_importance_drop.csv",
            "output/meta_classifiers/decision_tree_drop.png",
        ],
    ),
    "22_plot_decision_tree": StepCheck(
        files=["output/plot_decision_tree/decision_tree_drop.png"],
    ),
    "23_viz_all": StepCheck(
        files=[
            "output/viz_all/violin_distributions_drop.png",
            "output/viz_all/decision_boundary_pca_drop.png",
            "output/viz_all/decision_boundary_pairs_drop.png",
            "output/viz_all/shap_comparison_drop.png",
        ],
    ),
    "24_meta_svm": StepCheck(
        files=[
            "output/meta_svm/svm_cv_results_drop.csv",
            "output/meta_svm/svm_coefficients_drop.csv",
            "output/meta_svm/svm_margin_flags_drop.csv",
            "output/meta_svm/svm_margin_histogram_drop.png",
            "output/meta_svm/svm_boundary_pca_drop.png",
            "output/meta_svm/svm_boundary_pairs_drop.png",
            "output/meta_svm/svm_shap_importance_drop.csv",
            "output/meta_svm/shap_summary_svm_drop.png",
            "output/meta_svm/shap_values_svm_drop.csv",
        ],
    ),
    "25_run_full_dataset": StepCheck(
        files=["output/run_full_dataset/full_cv_results_drop.csv"],
    ),
    "26_permutation_test": StepCheck(
        files=[
            "output/permutation_test/permutation_test_results_drop.csv",
            "output/permutation_test/permutation_test_summary_drop.txt",
        ],
    ),
}


def step_num(p: Path) -> str:
    m = STEP_PATTERN.match(p.name)
    assert m, p.name
    return m.group(1)


def discover_steps() -> list[Path]:
    return sorted(
        p for p in SCRIPTS_DIR.iterdir()
        if STEP_PATTERN.match(p.name) and p.name != ORCHESTRATOR
    )


def parse_numbers(spec: str) -> set[str]:
    """Parse '04,13,25-27' -> {'04','13','25','26','27'}."""
    out: set[str] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(f"{i:02d}" for i in range(int(lo), int(hi) + 1))
        else:
            out.add(f"{int(part):02d}")
    return out


def _file_exists_and_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _load_reference_ids() -> set[str] | None:
    """Load JournalUniqueIDs from the canonical reference set, or None if unavailable."""
    p = ROOT_DIR / REFERENCE_FILE
    if not _file_exists_and_nonempty(p):
        return None
    try:
        import pandas as pd  # lazy: only needed when caching is active
        df = pd.read_csv(p, usecols=[REFERENCE_COL], dtype=str)
    except Exception:
        return None
    ids = set(df[REFERENCE_COL].dropna())
    return ids if ids else None


def is_step_complete(name: str, ref_ids: set[str] | None) -> tuple[bool, str]:
    """Return ``(done, reason)``. ``done=True`` means the step's outputs are ready."""
    spec = CHECKS.get(name)
    if spec is None:
        return False, "no completion spec defined"
    if not spec.files:
        return False, "no persisted output (always re-run)"

    for rel in spec.files:
        p = ROOT_DIR / rel
        if not _file_exists_and_nonempty(p):
            return False, f"missing {rel}"

    if spec.coverage and ref_ids is not None:
        cov_file, cov_col = spec.coverage
        try:
            import pandas as pd
            df = pd.read_csv(ROOT_DIR / cov_file, usecols=[cov_col], dtype=str)
        except Exception as e:
            return False, f"could not read {cov_file}: {e}"
        present = set(df[cov_col].dropna())
        missing = ref_ids - present
        if missing:
            return False, f"{cov_file} missing {len(missing)}/{len(ref_ids)} journals"
        return True, f"outputs present + all {len(ref_ids)} journals covered"

    return True, "outputs present"


def run_subprocess(label: str, cmd: list[str], stdout_path: Path | None = None) -> int:
    """Run a single subprocess; return its exit code. Streams stdout unless redirected."""
    print(f"\n{'=' * 70}\n▶ {label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w") as f:
                subprocess.run(cmd, check=True, cwd=ROOT_DIR, stdout=f)
            print(f"  (stdout → {stdout_path})")
        else:
            subprocess.run(cmd, check=True, cwd=ROOT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {label} failed after {time.time() - t0:.1f}s (exit {e.returncode})",
              file=sys.stderr)
        return e.returncode
    print(f"✓ {label} done in {time.time() - t0:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--from", dest="start", metavar="NN",
                    help="resume from this step number (inclusive)")
    ap.add_argument("--only", metavar="NN[,NN...]",
                    help="run only these step numbers; ranges allowed (e.g. 21-23)")
    ap.add_argument("--skip", metavar="NN[,NN...]",
                    help="skip these step numbers in addition to defaults; ranges allowed")
    ap.add_argument("--include-default-skipped", action="store_true",
                    help=f"include steps skipped by default ({sorted(DEFAULT_SKIP)})")
    ap.add_argument("--force", action="store_true",
                    help="re-run steps even if their cached outputs already exist")
    ap.add_argument("--no-paper-text", action="store_true",
                    help=f"do not run {ORCHESTRATOR} at the end")
    ap.add_argument("--paper-text-out", default="output/paper_text.txt",
                    metavar="PATH",
                    help=f"path to write {ORCHESTRATOR} output (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, don't execute")
    args = ap.parse_args()

    all_steps = discover_steps()
    if not all_steps:
        print("No NN_*.py scripts found in scripts/.", file=sys.stderr)
        return 1

    skip: set[str] = set() if args.include_default_skipped else set(DEFAULT_SKIP)
    if args.skip:
        skip |= parse_numbers(args.skip)

    if args.only:
        wanted = parse_numbers(args.only)
        candidate_steps = [s for s in all_steps if step_num(s) in wanted]
        skipped_in_filter: set[str] = set()
    else:
        candidate_steps = list(all_steps)
        if args.start:
            start = f"{int(args.start):02d}"
            candidate_steps = [s for s in candidate_steps if step_num(s) >= start]
        skipped_in_filter = {step_num(s) for s in candidate_steps if step_num(s) in skip}
        candidate_steps = [s for s in candidate_steps if step_num(s) not in skip]

    # Cache check (unless --force).
    cached: list[tuple[Path, str]] = []
    to_run: list[Path] = []
    if args.force:
        to_run = candidate_steps
    else:
        ref_ids = _load_reference_ids()
        for s in candidate_steps:
            done, reason = is_step_complete(s.stem, ref_ids)
            if done:
                cached.append((s, reason))
            else:
                to_run.append(s)

    will_run_orch = not args.no_paper_text and not args.only

    print(f"Interpreter: {sys.executable}")
    if skipped_in_filter:
        print(f"Filtered:    {', '.join(sorted(skipped_in_filter))} (via --skip)")
    if cached:
        print(f"Cached ({len(cached)}; pass --force to re-run):")
        for s, reason in cached:
            print(f"  ⊘ {s.name}  ({reason})")
    print(f"To run ({len(to_run)} step{'s' if len(to_run) != 1 else ''}):")
    for s in to_run:
        print(f"  ▶ {s.name}")
    if will_run_orch:
        print(f"  ▷ {ORCHESTRATOR}  →  {args.paper_text_out}")

    if args.dry_run:
        return 0
    if not to_run and not will_run_orch:
        print("Nothing to run.", file=sys.stderr)
        return 0

    overall_t0 = time.time()
    for s in to_run:
        rc = run_subprocess(s.name, [sys.executable, str(s)])
        if rc != 0:
            return rc

    if will_run_orch:
        out_path = (ROOT_DIR / args.paper_text_out).resolve()
        rc = run_subprocess(
            ORCHESTRATOR,
            [sys.executable, str(SCRIPTS_DIR / ORCHESTRATOR)],
            stdout_path=out_path,
        )
        if rc != 0:
            return rc

    print(f"\nAll done in {time.time() - overall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
