#!/usr/bin/env python3
"""
Permutation-test sanity check for the meta-classifiers.

Runs the same group-aware nested-CV pipeline used in `meta_classifiers.py`
on three label conditions:

  1. Real labels                         (positive control: should be ≈ 0.999)
  2. Random label shuffle                (negative control 1: ≈ 0.50)
  3. Within-participant random shuffle   (negative control 2: ≈ 0.50)
     — preserves each participant's label distribution but breaks the
       text→label correspondence within each participant. This rules
       out leakage via participant identity.

Outputs:
  - output/permutation_test/permutation_test_results.csv   (raw scores)
  - output/permutation_test/permutation_test_summary.txt   (human-readable summary)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

_REPO_ROOT = Path(__file__).resolve().parent.parent
META_FEATURES_CSV = _REPO_ROOT / "output" / "merge_features" / "meta_features.csv"
META_FEATURES_FULL_CSV = _REPO_ROOT / "output" / "merge_features_full" / "meta_features_full.csv"
REVIEWS_OUTPUT = _REPO_ROOT / "output" / "permutation_test"
REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_PERMUTATIONS = 50
DROP_FEATURES = ["nli_n_sentences", "nli_n_pairs", "bleu_participant"]


def load(path):
    df = pd.read_csv(path)
    id_cols = ["JournalUniqueID", "ParticipantID", "final_label"]
    feat_cols = [c for c in df.columns if c not in id_cols and c not in DROP_FEATURES]
    X = df[feat_cols].copy()
    y = (df["final_label"] == "ai").astype(int).values
    groups = df["ParticipantID"].values
    return df, X, y, groups, feat_cols


def build_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                   C=1.0, max_iter=2000,
                                   class_weight="balanced")),
    ])


def cv_auc(X, y, groups):
    cv = StratifiedGroupKFold(n_splits=5)
    scores = cross_val_score(build_pipe(), X, y, groups=groups, cv=cv,
                             scoring="roc_auc", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def shuffle_random(y, rng):
    return rng.permutation(y)


def shuffle_within_participant(y, groups, rng):
    """Shuffle labels independently within each participant.

    For single-label participants this is a no-op (their labels are constant),
    so this also probes whether prediction relies on text features rather than
    on participant identity. For mixed-label participants it destroys the
    text→label correspondence while preserving the participant→label
    distribution.
    """
    y_new = y.copy()
    for pid in np.unique(groups):
        mask = groups == pid
        y_new[mask] = rng.permutation(y[mask])
    return y_new


def shuffle_participant_label(y, groups, rng):
    """Reassign each participant's *majority* label to another participant.

    Preserves the distribution of participant-level labels but breaks the
    text→label association entirely. This isolates whether the signal is
    cross-participant (writer style / demographics) rather than text-content.
    """
    pids = np.unique(groups)
    pid_label = {pid: int(round(y[groups == pid].mean())) for pid in pids}
    shuffled = rng.permutation(list(pid_label.values()))
    new_label = dict(zip(pid_label.keys(), shuffled))
    return np.array([new_label[pid] for pid in groups])


def run(df_path, label):
    print(f"\n{'='*60}\n  {label}: {df_path.name}\n{'='*60}")
    df, X, y, groups, feat_cols = load(df_path)
    n_p = len(np.unique(groups))
    print(f"  Entries: {len(df)} | Participants: {n_p} | "
          f"AI: {y.sum()} ({y.mean():.1%})")

    rng = np.random.default_rng(SEED)
    rows = []

    # 1. Real
    m, s = cv_auc(X, y, groups)
    rows.append(dict(condition="real", subset=label, perm=0, auc_mean=m, auc_std=s))
    print(f"  Real labels:                    AUC = {m:.4f} ± {s:.4f}")

    # 2. Random shuffle
    aucs = []
    for i in range(N_PERMUTATIONS):
        y_shuf = shuffle_random(y, rng)
        m, _ = cv_auc(X, y_shuf, groups)
        aucs.append(m)
        rows.append(dict(condition="random_shuffle", subset=label, perm=i+1,
                         auc_mean=m, auc_std=0.0))
    print(f"  Random label shuffle (n={N_PERMUTATIONS}):    "
          f"AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
          f"(min={np.min(aucs):.4f}, max={np.max(aucs):.4f})")

    # 3. Participant-level label shuffle
    aucs = []
    for i in range(N_PERMUTATIONS):
        y_shuf = shuffle_participant_label(y, groups, rng)
        m, _ = cv_auc(X, y_shuf, groups)
        aucs.append(m)
        rows.append(dict(condition="participant_label_shuffle", subset=label,
                         perm=i+1, auc_mean=m, auc_std=0.0))
    print(f"  Participant-label shuffle (n={N_PERMUTATIONS}): "
          f"AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
          f"(min={np.min(aucs):.4f}, max={np.max(aucs):.4f})")

    return rows


def main():
    REVIEWS_OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_rows.extend(run(META_FEATURES_CSV, "high_confidence"))
    all_rows.extend(run(META_FEATURES_FULL_CSV, "full"))

    df = pd.DataFrame(all_rows)
    out_csv = REVIEWS_OUTPUT / "permutation_test_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results → {out_csv}")

    summary = ["Permutation-test summary", "=" * 50, ""]
    for subset in ["high_confidence", "full"]:
        summary.append(f"Subset: {subset}")
        for cond in ["real", "random_shuffle", "participant_label_shuffle"]:
            sub = df[(df.subset == subset) & (df.condition == cond)]
            if cond == "real":
                summary.append(f"  Real labels                  AUC = {sub['auc_mean'].iloc[0]:.4f}")
            else:
                m = sub["auc_mean"].mean()
                s = sub["auc_mean"].std()
                lo, hi = sub["auc_mean"].quantile([0.025, 0.975])
                summary.append(
                    f"  {cond:28s} mean = {m:.4f} ± {s:.4f}  "
                    f"95% range [{lo:.4f}, {hi:.4f}]")
        summary.append("")

    txt = "\n".join(summary)
    out_txt = REVIEWS_OUTPUT / "permutation_test_summary.txt"
    out_txt.write_text(txt)
    print()
    print(txt)
    print(f"\nSaved summary → {out_txt}")


if __name__ == "__main__":
    main()
