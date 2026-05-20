#!/usr/bin/env python3
"""Summarise participant demographics for the paper.

Two reports are produced:

1. **Aggregate descriptive stats** for all study participants and for the
   high-confidence subset of participants. Same fields as before
   (age, gender, country, ethnicity, education, employment, language).

2. **Demographic comparison by AI usage** (Table 3 in §Results). Each
   participant is bucketed by the labels of their entries:
       human-only  -> 0 AI-labelled entries
       mixed       -> at least one of each
       AI-only     -> every entry AI-labelled
   The paper compares "human only" (= human-only) vs "Has AI" (= mixed +
   AI-only) on age, gender, language, ethnicity, education, country, and
   per-participant entry / AI volume.

   Denominator policy: every percentage uses the full group n (159 / 92)
   as denominator, including participants with missing demographic data.
   This keeps the table internally consistent (rows are sub-counts of the
   full group), at the cost of slightly lower per-category percentages
   when there are missing values. The table prints missingness explicitly
   so the reader can audit it.

Inputs:
  input/journals_master.csv                            (Prolific pre-screening fields)
  output/export/final_journals.csv                     (one row per valid entry)
  output/high_confidence_subset/high_confidence_subset.csv

Outputs (under output/participant_demographics/):
  participant_demographics.txt   - aggregate descriptive report
  participant_demographics.csv   - per-participant demographics table
  ai_usage_comparison.txt        - readable Table-3 numbers + paper cross-check
  ai_usage_comparison.csv        - per-participant AI-usage grouping table
  ai_usage_table.tex             - LaTeX table matching tab:demo_by_group
  ai_usage_paragraph.tex         - LaTeX paragraph snippet for §demo_comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "participant_demographics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOURNALS_MASTER = ROOT / "input" / "journals_master.csv"
FINAL_JOURNALS = ROOT / "output" / "export" / "final_journals.csv"
HC_SUBSET = ROOT / "output" / "high_confidence_subset" / "high_confidence_subset.csv"


# ── Pre-screening column map ───────────────────────────────────────────────
# These are the Prolific pre-screening fields we surface; renaming makes the
# downstream code readable.

DEMO_COLS = {
    "Prescreening_ps-gender":               "gender",
    "Prescreening_ps-age":                  "age",
    "Prescreening_se-country":              "country",
    "Prescreening_se-ethnicity":            "ethnicity",
    "Prescreening_se-education":            "education",
    "Prescreening_se-language#3_1_1":       "native_language",
    "Prescreening_se-language#4_1":         "english_proficiency",
    "Prescreening_se-language#2_1":         "is_native_lang1",
    "ParticipantID":                        "ParticipantID",
}

# Employment was administered as a multi-column "select all" question; we
# collapse it to a single string per participant (first non-null wins).
EMPLOYMENT_COLS = [f"Prescreening_se-employment_{i}" for i in (1, 4, 5, 6, 7, 8, 9, 11)]


# ── Loading + cleaning ─────────────────────────────────────────────────────

def load_demographics() -> pd.DataFrame:
    """Return a per-participant DataFrame with demographics + study membership."""
    jm = pd.read_csv(JOURNALS_MASTER, low_memory=False)
    fj = pd.read_csv(FINAL_JOURNALS)
    hc = pd.read_csv(HC_SUBSET)

    study_pids = set(fj["ParticipantID"].unique())
    hc_pids = set(fj.loc[fj["JournalUniqueID"].isin(hc["JournalUniqueID"]),
                         "ParticipantID"].unique())

    # Keep only the columns we know about, rename to friendly names.
    keep = [c for c in DEMO_COLS if c in jm.columns]
    emp_cols = [c for c in EMPLOYMENT_COLS if c in jm.columns]
    df = (jm[keep + emp_cols]
            .rename(columns={k: v for k, v in DEMO_COLS.items() if k in jm.columns})
            .drop_duplicates(subset=["ParticipantID"], keep="first"))

    # Collapse the eight employment columns into a single value per participant.
    df["employment"] = df[emp_cols].bfill(axis=1).iloc[:, 0] if emp_cols else np.nan
    df = df.drop(columns=emp_cols, errors="ignore")

    df["in_study"] = df["ParticipantID"].isin(study_pids)
    df["in_high_confidence"] = df["ParticipantID"].isin(hc_pids)
    return df


def per_participant_labels(fj: pd.DataFrame) -> pd.DataFrame:
    """Per-participant entry counts + AI/human split + group label.

    group ∈ {human_only, mixed, ai_only}; has_ai = mixed | ai_only.
    """
    g = (fj.groupby("ParticipantID")["label"]
            .agg(n_entries="count",
                 n_ai=lambda s: int((s == "ai").sum()),
                 n_human=lambda s: int((s == "human").sum()))
            .reset_index())
    g["pct_ai"] = g["n_ai"] / g["n_entries"] * 100
    g["group"] = np.where(g["n_ai"] == 0, "human_only",
                  np.where(g["n_ai"] == g["n_entries"], "ai_only", "mixed"))
    g["has_ai"] = g["group"].isin({"mixed", "ai_only"})
    return g


# ── Helpers: percentages over a fixed denominator ─────────────────────────

def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator else float("nan")


def pct_eq(series: pd.Series, value: str, n_total: int) -> float:
    """Percentage of `n_total` whose value equals `value` (case-sensitive)."""
    return _pct(int((series == value).sum()), n_total)


def pct_contains(series: pd.Series, substring: str, n_total: int) -> float:
    """Percentage of `n_total` whose value contains `substring` (case-insensitive)."""
    s = series.dropna().astype(str).str.lower()
    return _pct(int(s.str.contains(substring).sum()), n_total)


def pct_in(series: pd.Series, values: Iterable[str], n_total: int) -> float:
    """Percentage of `n_total` whose value is in the set `values`."""
    return _pct(int(series.isin(set(values)).sum()), n_total)


# ── Output 1: aggregate descriptive report (existing format) ──────────────

def _distribution(series: pd.Series, label: str) -> str:
    """One section of the descriptive report (e.g., Gender (n=251): ...)."""
    counts = series.value_counts(dropna=False)
    total = len(series)
    lines = [f"  {label} (n={total}):"]
    for val, n in counts.items():
        val_str = str(val) if pd.notna(val) else "Missing"
        lines.append(f"    {val_str}: {n} ({n/total:.1%})")
    return "\n".join(lines)


def aggregate_report(df: pd.DataFrame, subset_name: str) -> str:
    """Free-text descriptive stats for the given participant subset."""
    lines = ["=" * 60, f"  {subset_name} (n={len(df)})", "=" * 60]

    age = df["age"].dropna()
    if len(age):
        lines.append(f"\n  Age: M={age.mean():.1f}, SD={age.std():.1f}, "
                     f"range={age.min():.0f}\u2013{age.max():.0f}")

    section_specs = [
        ("gender", "Gender"), ("country", "Country"),
        ("ethnicity", "Ethnicity"), ("education", "Education"),
        ("employment", "Employment"),
        ("native_language", "Native language (L1)"),
        ("english_proficiency", "English proficiency (L1)"),
    ]
    for col, label in section_specs:
        if col in df.columns:
            lines.append("\n" + _distribution(df[col], label))

    if "native_language" in df.columns:
        lang = df["native_language"].dropna().str.strip().str.lower()
        n_eng = int(lang.str.contains("english|british english").sum())
        n_other = len(lang) - n_eng
        lines.append(f"\n  English as L1: {n_eng} ({n_eng/len(lang):.1%}), "
                     f"Other L1: {n_other} ({n_other/len(lang):.1%})")
    return "\n".join(lines)


def write_aggregate_outputs(df: pd.DataFrame) -> None:
    """Reproduce the existing two-subset descriptive report + per-participant CSV."""
    study = df[df["in_study"]]
    hc = df[df["in_high_confidence"]]

    report = "\n".join([
        "Participant Demographics Report",
        "=" * 60,
        "\n" + aggregate_report(study, "All study participants"),
        "\n" + aggregate_report(hc, "High-confidence subset participants"),
    ])
    (OUTPUT_DIR / "participant_demographics.txt").write_text(report)
    print(report)

    study.to_csv(OUTPUT_DIR / "participant_demographics.csv", index=False)


# ── Output 2: AI-usage comparison (Table 3) ───────────────────────────────
#
# Three-column comparison: Human only / Mixed / AI only, plus an additional
# "Has AI" column (= Mixed + AI only) for paragraph-level claims that combine
# the latter two. Categorical values are aggregated where the raw response
# space is too sparse to read row-by-row (e.g., 4 different "Asian" tags
# collapsed to one Asian row).

GROUP_KEYS = ["human_only", "mixed", "ai_only"]
GROUP_LABELS = {
    "human_only": "Human only",
    "mixed":      "Mixed",
    "ai_only":    "AI only",
    "has_ai":     "Has AI",
}

# Aggregated category buckets for table rows.
ETHNICITY_BUCKETS = {
    "white":  ["White/Caucasian"],
    "black":  ["Black/African/Caribbean/Black British"],
    "asian":  ["South Asian", "Asian British", "East Asian", "Southeast Asian"],
    "other":  ["Mixed/Multiracial", "Arab/Middle Eastern",
               "Latin American/Hispanic",
               "Other Ethnic Group (please specify)",
               "Indigenous/Native/Aboriginal"],
}
EDUCATION_BUCKETS = {
    "high_school": ["High School"],
    "bachelor":    ["Bachelor degree"],
    "master":      ["Master\u2019s degree"],
    "phd":         ["PhD"],
    "other":       ["Postgraduate diploma (e.g., PGCE)",
                    "Other (please specify)"],
}
KNOWN_COUNTRIES_TOPLEVEL = {"United Kingdom", "United States"}


def _group_subsets(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the four subset DataFrames keyed by GROUP_KEYS + 'has_ai'."""
    subs = {key: merged[merged["group"] == key].copy() for key in GROUP_KEYS}
    subs["has_ai"] = merged[merged["has_ai"]].copy()
    return subs


def _row_pct(subs: dict[str, pd.DataFrame], col: str, values: Iterable[str]
             ) -> dict[str, float]:
    """Per-group `% of n` for participants whose `col` is in `values`."""
    return {k: pct_in(subs[k][col], values, len(subs[k])) for k in subs}


def _row_pct_contains(subs: dict[str, pd.DataFrame], col: str, sub: str
                      ) -> dict[str, float]:
    return {k: pct_contains(subs[k][col], sub, len(subs[k])) for k in subs}


def _row_country_other(subs: dict[str, pd.DataFrame]) -> dict[str, float]:
    """% of group whose country is reported AND not UK/US (i.e., 'Other')."""
    out = {}
    for k, s in subs.items():
        c = s["country"].dropna()
        n_other = int((~c.isin(KNOWN_COUNTRIES_TOPLEVEL)).sum())
        out[k] = _pct(n_other, len(s))
    return out


def _row_native_other(subs: dict[str, pd.DataFrame]) -> dict[str, float]:
    """% of group whose native language is reported AND not English."""
    out = {}
    for k, s in subs.items():
        lang = s["native_language"].dropna().astype(str).str.lower()
        n_other = int((~lang.str.contains("english")).sum())
        out[k] = _pct(n_other, len(s))
    return out


def _row_missing(subs: dict[str, pd.DataFrame], col: str) -> dict[str, float]:
    """% of group whose value for `col` is missing (NaN)."""
    return {k: _pct(int(s[col].isna().sum()), len(s)) for k, s in subs.items()}


def _row_continuous(subs: dict[str, pd.DataFrame], col: str, fn: str
                    ) -> dict[str, float]:
    """Per-group continuous summary stat (e.g., mean, std, median, min, max)."""
    return {k: getattr(subs[k][col], fn)() for k in subs}


def compute_ai_usage_table(df_demo: pd.DataFrame,
                           per_pid: pd.DataFrame) -> dict:
    """Return a dict of headline numbers used by the paper text + LaTeX table.

    The dict has one entry per measured row of Table 3. Each row maps to a
    sub-dict keyed by group name (`human_only`, `mixed`, `ai_only`,
    `has_ai`), holding the percentage / mean / count for that group.
    """
    m = df_demo.merge(per_pid, on="ParticipantID", how="inner")
    subs = _group_subsets(m)
    ns = {k: len(v) for k, v in subs.items()}

    # Continuous: age + per-participant entry volume.
    age_mean   = _row_continuous(subs, "age", "mean")
    age_sd     = _row_continuous(subs, "age", "std")
    age_min    = _row_continuous(subs, "age", "min")
    age_max    = _row_continuous(subs, "age", "max")
    n_entries_total  = _row_continuous(subs, "n_entries", "sum")
    n_entries_mean   = _row_continuous(subs, "n_entries", "mean")
    n_entries_median = _row_continuous(subs, "n_entries", "median")
    pct_ai_mean      = _row_continuous(subs, "pct_ai", "mean")

    table = {
        "ns": ns,                              # {human_only:159, mixed:62, ai_only:30, has_ai:92}
        "n_total": len(m),

        # Volume
        "n_entries_total":  n_entries_total,
        "n_entries_mean":   n_entries_mean,
        "n_entries_median": n_entries_median,
        "pct_ai_mean":      pct_ai_mean,

        # Age
        "age_mean": age_mean, "age_sd": age_sd,
        "age_min":  age_min,  "age_max": age_max,

        # Gender (denominator = full group n, including missing)
        "male":   _row_pct(subs, "gender", ["Male"]),
        "female": _row_pct(subs, "gender", ["Female"]),

        # Native language: % whose response contains "english", + "Other" + missing
        "english_l1":   _row_pct_contains(subs, "native_language", "english"),
        "native_other": _row_native_other(subs),
        "native_missing": _row_missing(subs, "native_language"),

        # Gender missing (Male + Female + missing should sum to 100%)
        "gender_missing": _row_missing(subs, "gender"),

        # Ethnicity (aggregated) + missing
        "ethnicity_white":   _row_pct(subs, "ethnicity", ETHNICITY_BUCKETS["white"]),
        "ethnicity_black":   _row_pct(subs, "ethnicity", ETHNICITY_BUCKETS["black"]),
        "ethnicity_asian":   _row_pct(subs, "ethnicity", ETHNICITY_BUCKETS["asian"]),
        "ethnicity_other":   _row_pct(subs, "ethnicity", ETHNICITY_BUCKETS["other"]),
        "ethnicity_missing": _row_missing(subs, "ethnicity"),

        # Education (aggregated) + missing
        "edu_high_school": _row_pct(subs, "education", EDUCATION_BUCKETS["high_school"]),
        "edu_bachelor":    _row_pct(subs, "education", EDUCATION_BUCKETS["bachelor"]),
        "edu_master":      _row_pct(subs, "education", EDUCATION_BUCKETS["master"]),
        "edu_phd":         _row_pct(subs, "education", EDUCATION_BUCKETS["phd"]),
        "edu_other":       _row_pct(subs, "education", EDUCATION_BUCKETS["other"]),
        "edu_missing":     _row_missing(subs, "education"),

        # Country (top two + Other + missing)
        "country_uk":      _row_pct(subs, "country", ["United Kingdom"]),
        "country_us":      _row_pct(subs, "country", ["United States"]),
        "country_other":   _row_country_other(subs),
        "country_missing": _row_missing(subs, "country"),

        # Age missingness
        "age_missing": _row_missing(subs, "age"),

        # Missingness audit (counted in the denominators above)
        "missing": {k: {c: int(subs[k][c].isna().sum())
                        for c in ["gender", "ethnicity", "education",
                                  "country", "native_language", "age"]}
                    for k in subs},

        "merged_table": m,
    }
    return table


def _fmt_pct(p: float) -> str:
    return f"{p:.1f}%" if not np.isnan(p) else "--"


def _row_pct_strings(t: dict, key: str) -> list[str]:
    """Return [human_only, mixed, ai_only, has_ai] formatted % strings."""
    return [_fmt_pct(t[key][g]) for g in GROUP_KEYS + ["has_ai"]]


def report_ai_usage(t: dict) -> str:
    """Human-readable 3-column Table 3 + paper paragraph cross-check."""
    ns = t["ns"]
    am, asd = t["age_mean"], t["age_sd"]
    amin, amax = t["age_min"], t["age_max"]
    nem, nemed, net = t["n_entries_mean"], t["n_entries_median"], t["n_entries_total"]
    pam = t["pct_ai_mean"]

    # Section: header rows are tuples of (label, [human_only, mixed, ai_only, has_ai]).
    sections: list[tuple[str, list[tuple[str, list[str]]]]] = [
        ("Per-participant volume", [
            ("Total entries (sum)",
             [f"{int(net[g])}" for g in GROUP_KEYS + ["has_ai"]]),
            ("Mean entries",
             [f"{nem[g]:.1f}" for g in GROUP_KEYS + ["has_ai"]]),
            ("Median entries",
             [f"{nemed[g]:.1f}" for g in GROUP_KEYS + ["has_ai"]]),
            ("Mean % AI entries",
             [_fmt_pct(pam[g]) for g in GROUP_KEYS + ["has_ai"]]),
        ]),
        ("Age", [
            ("M (SD)",
             [f"{am[g]:.1f} ({asd[g]:.1f})" for g in GROUP_KEYS + ["has_ai"]]),
            ("Range",
             [f"{int(amin[g])}\u2013{int(amax[g])}" for g in GROUP_KEYS + ["has_ai"]]),
        ]),
        ("Gender", [
            ("Male",   _row_pct_strings(t, "male")),
            ("Female", _row_pct_strings(t, "female")),
        ]),
        ("Native language", [
            ("English L1", _row_pct_strings(t, "english_l1")),
        ]),
        ("Ethnicity", [
            ("White/Caucasian",         _row_pct_strings(t, "ethnicity_white")),
            ("Black/Afr./Caribbean",    _row_pct_strings(t, "ethnicity_black")),
            ("Asian (incl. British)",   _row_pct_strings(t, "ethnicity_asian")),
            ("Mixed/Other",             _row_pct_strings(t, "ethnicity_other")),
        ]),
        ("Education", [
            ("High School", _row_pct_strings(t, "edu_high_school")),
            ("Bachelor's",  _row_pct_strings(t, "edu_bachelor")),
            ("Master's",    _row_pct_strings(t, "edu_master")),
            ("PhD",         _row_pct_strings(t, "edu_phd")),
            ("Other / postgrad. dipl.", _row_pct_strings(t, "edu_other")),
        ]),
        ("Country of residence", [
            ("United Kingdom", _row_pct_strings(t, "country_uk")),
            ("United States",  _row_pct_strings(t, "country_us")),
            ("Other",          _row_pct_strings(t, "country_other")),
        ]),
    ]

    lines = []
    lines.append("=" * 76)
    lines.append("  Demographic comparison by AI usage")
    lines.append("=" * 76)
    lines.append(f"\n  Total participants in the analysis: {t['n_total']}")
    lines.append(f"    human-only (no AI entries)        : {ns['human_only']}")
    lines.append(f"    mixed      (\u2265 1 of each)         : {ns['mixed']}")
    lines.append(f"    AI-only    (every entry AI)       : {ns['ai_only']}")
    lines.append(f"    Has AI     (mixed + AI-only)      : {ns['has_ai']}")
    lines.append("")
    lines.append("  Denominator for all percentages = full group n "
                 "(missing values counted in the denominator).")
    lines.append("")

    label_w, col_w = 28, 12
    header = (f"  {'Measure':<{label_w}}  "
              f"{'Human only':>{col_w}}  {'Mixed':>{col_w}}  "
              f"{'AI only':>{col_w}}  {'Has AI':>{col_w}}")
    sub = (f"  {'(n=)':<{label_w}}  "
           f"{ns['human_only']:>{col_w}}  {ns['mixed']:>{col_w}}  "
           f"{ns['ai_only']:>{col_w}}  {ns['has_ai']:>{col_w}}")
    rule = "  " + "-" * (label_w + 4 * col_w + 6)
    lines.extend([header, sub, rule])

    for section_name, rows in sections:
        lines.append(f"  {section_name}:")
        for label, vals in rows:
            row_str = (f"    {label:<{label_w-2}}  "
                       + "  ".join(f"{v:>{col_w}}" for v in vals))
            lines.append(row_str)
        lines.append("")

    lines.append("  Missing demographic data (counted in the denominator):")
    for k in GROUP_KEYS + ["has_ai"]:
        miss = t["missing"][k]
        bits = ", ".join(f"{c}={v}" for c, v in miss.items() if v)
        lines.append(f"    {GROUP_LABELS[k]:<11}: {bits or 'none'}")

    # Paper paragraph cross-check (the paragraph compares Human-only vs. Has-AI)
    lines.append("")
    lines.append("=" * 76)
    lines.append("  Paper paragraph cross-check (\u00a7Demographic comparison)")
    lines.append("=" * 76)
    lines.append("Sentence-by-sentence numbers used in the paragraph "
                 "(Human only vs Has AI):\n")
    lines.append(f"  '159 participants submitted only human-labelled entries'   "
                 f"\u2192 {ns['human_only']}")
    lines.append(f"  '62 submitted a mix'                                       "
                 f"\u2192 {ns['mixed']}")
    lines.append(f"  '30 submitted exclusively AI-labelled entries'             "
                 f"\u2192 {ns['ai_only']}")
    lines.append(f"  'more predominantly male, 85.9% vs 58.4%'                  "
                 f"\u2192 {t['male']['has_ai']:.1f}% vs "
                 f"{t['male']['human_only']:.1f}%")
    lines.append(f"  'native English speakers, 95.2% vs 80.7%'                  "
                 f"\u2192 {t['english_l1']['has_ai']:.1f}% vs "
                 f"{t['english_l1']['human_only']:.1f}%")
    lines.append(f"  'White/Caucasian, 62.0% vs 27.7%'                          "
                 f"\u2192 {t['ethnicity_white']['has_ai']:.1f}% vs "
                 f"{t['ethnicity_white']['human_only']:.1f}%")
    lines.append(f"  'submitted more entries on average, 11.7 vs 8.8'           "
                 f"\u2192 {t['n_entries_mean']['has_ai']:.2f} vs "
                 f"{t['n_entries_mean']['human_only']:.2f}")
    lines.append(f"  'mean % AI entries: 0.0% vs 68.8%'                          "
                 f"\u2192 {t['pct_ai_mean']['human_only']:.1f}% vs "
                 f"{t['pct_ai_mean']['has_ai']:.1f}%")
    return "\n".join(lines)


# ── LaTeX outputs (drop-in replacements for the paper) ────────────────────

def _latex_pct(value: float) -> str:
    return f"{value:.1f}\\%" if not np.isnan(value) else "--"


def _three_cells(t: dict, key: str, fmt=_latex_pct) -> str:
    """Return ' & h & m & a' for the three group cells of one row."""
    return " & ".join(fmt(t[key][g]) for g in GROUP_KEYS)


def _quad_row(label: str, cells: str) -> str:
    """A `\\quad`-indented entry row with three data cells."""
    return f"\\quad {label} & {cells}\\\\"


def _section(title: str) -> str:
    """Italic section heading row, rater-demographics style."""
    return f"\\addlinespace[2pt]\\textit{{{title}}} & & & \\\\"


def build_latex_table(t: dict) -> str:
    """LaTeX table styled like `tab:rater_demo` (italic section headings,
    `\\quad`-indented entries, right-aligned numerical columns, "Not
    reported" rows when missingness is non-zero).

    Three group columns: Human only / Mixed / AI only.
    """
    ns = t["ns"]
    am, asd = t["age_mean"], t["age_sd"]
    amin, amax = t["age_min"], t["age_max"]

    def _maybe_not_reported(missing_key: str) -> list[str]:
        """Add a 'Not reported' row only when at least one group has missing data."""
        miss = t[missing_key]
        if max(miss[g] for g in GROUP_KEYS) <= 0:
            return []
        return [_quad_row("Not reported", _three_cells(t, missing_key))]

    rows: list[str] = []

    # Entries per participant
    rows.append(_section("Entries per participant"))
    rows.append(_quad_row("Total (sum)",
        " & ".join(f"{int(t['n_entries_total'][g])}" for g in GROUP_KEYS)))
    rows.append(_quad_row("$M$",
        " & ".join(f"{t['n_entries_mean'][g]:.1f}" for g in GROUP_KEYS)))
    rows.append(_quad_row("Median",
        " & ".join(f"{t['n_entries_median'][g]:.1f}" for g in GROUP_KEYS)))
    rows.append(_quad_row("Mean \\% AI-labelled",
        _three_cells(t, "pct_ai_mean")))

    # Age
    rows.append(_section("Age (years)"))
    rows.append(_quad_row("$M$ ($SD$)",
        " & ".join(f"{am[g]:.1f} ({asd[g]:.1f})" for g in GROUP_KEYS)))
    rows.append(_quad_row("Range",
        " & ".join(f"{int(amin[g])}--{int(amax[g])}" for g in GROUP_KEYS)))
    rows.extend(_maybe_not_reported("age_missing"))

    # Gender
    rows.append(_section("Gender"))
    rows.append(_quad_row("Male",   _three_cells(t, "male")))
    rows.append(_quad_row("Female", _three_cells(t, "female")))
    rows.extend(_maybe_not_reported("gender_missing"))

    # Native language
    rows.append(_section("Native language"))
    rows.append(_quad_row("English",          _three_cells(t, "english_l1")))
    rows.append(_quad_row("Other",            _three_cells(t, "native_other")))
    rows.extend(_maybe_not_reported("native_missing"))

    # Ethnicity
    rows.append(_section("Ethnicity"))
    rows.append(_quad_row("White/Caucasian",        _three_cells(t, "ethnicity_white")))
    rows.append(_quad_row("Black/African/Caribbean", _three_cells(t, "ethnicity_black")))
    rows.append(_quad_row("Asian (incl.\\ British)", _three_cells(t, "ethnicity_asian")))
    rows.append(_quad_row("Mixed/Other",             _three_cells(t, "ethnicity_other")))
    rows.extend(_maybe_not_reported("ethnicity_missing"))

    # Education
    rows.append(_section("Education"))
    rows.append(_quad_row("High School",        _three_cells(t, "edu_high_school")))
    rows.append(_quad_row("Bachelor's degree",  _three_cells(t, "edu_bachelor")))
    rows.append(_quad_row("Master's degree",    _three_cells(t, "edu_master")))
    rows.append(_quad_row("PhD",                _three_cells(t, "edu_phd")))
    rows.append(_quad_row("Other / postgrad.\\ diploma",
                          _three_cells(t, "edu_other")))
    rows.extend(_maybe_not_reported("edu_missing"))

    # Country
    rows.append(_section("Country of residence"))
    rows.append(_quad_row("United Kingdom", _three_cells(t, "country_uk")))
    rows.append(_quad_row("United States",  _three_cells(t, "country_us")))
    rows.append(_quad_row("Other",          _three_cells(t, "country_other")))
    rows.extend(_maybe_not_reported("country_missing"))

    body = "\n".join(rows)
    return (
        "% Auto-generated by scripts/07_participant_demographics.py\n"
        "\\begin{table}[tbp]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Demographic comparison of the 251 study participants by "
        "labelling outcome of their entries. ``Human only'' = every entry by "
        "the participant was rated as human-written; ``Mixed'' = at least "
        "one entry of each label; ``AI only'' = every entry was rated as "
        "AI-generated. Percentages use the full group $n$ as denominator and "
        "may not sum to 100\\% in groups with missing values, which are "
        "shown explicitly as ``Not reported'' wherever they occur.}\n"
        "\\label{tab:demo_by_group}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrrr@{}}\n"
        "\\toprule\n"
        " & \\textbf{Human only} & \\textbf{Mixed} & \\textbf{AI only}\\\\\n"
        f" & ($n$={ns['human_only']}) & ($n$={ns['mixed']}) "
        f"& ($n$={ns['ai_only']})\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def build_latex_paragraph(t: dict) -> str:
    """Drop-in replacement for the body paragraph in §demo_comparison.

    The paragraph still contrasts Human-only vs Has-AI (= Mixed + AI-only)
    because that's the headline claim; readers can drill into the 3-column
    table for the Mixed-vs-AI-only nuance.
    """
    ns = t["ns"]
    male_ai, male_ho = t["male"]["has_ai"],       t["male"]["human_only"]
    eng_ai, eng_ho   = t["english_l1"]["has_ai"], t["english_l1"]["human_only"]
    white_ai, white_ho = (t["ethnicity_white"]["has_ai"],
                          t["ethnicity_white"]["human_only"])
    ent_ai, ent_ho   = (t["n_entries_mean"]["has_ai"],
                        t["n_entries_mean"]["human_only"])

    return (
        "% Auto-generated by scripts/07_participant_demographics.py\n"
        f"At the participant level, {ns['human_only']} participants submitted "
        f"only human-labelled entries, {ns['mixed']} submitted a mix of human "
        f"and AI-labelled entries, and {ns['ai_only']} submitted exclusively "
        f"AI-labelled entries (Table~\\ref{{tab:demo_by_group}}). Participants "
        f"who submitted any AI-labelled entries differed from the human-only "
        f"group in several respects: they were more predominantly male "
        f"({male_ai:.1f}\\% vs.\\ {male_ho:.1f}\\%), more likely to be native "
        f"English speakers ({eng_ai:.1f}\\% vs.\\ {eng_ho:.1f}\\%), more likely "
        f"to identify as White/Caucasian ({white_ai:.1f}\\% vs.\\ {white_ho:.1f}\\%), "
        f"and submitted more entries on average ({ent_ai:.1f} vs.\\ {ent_ho:.1f}). "
        f"Age and educational attainment were comparable between groups. "
        f"These demographic differences represent potential confounds, as some "
        f"of the linguistic patterns attributed to AI usage may partly reflect "
        f"characteristics of the writers themselves.\n"
    )


# ── Driver ─────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_demographics()
    fj = pd.read_csv(FINAL_JOURNALS)
    per_pid = per_participant_labels(fj)

    # 1) Existing aggregate descriptive report (unchanged behaviour).
    write_aggregate_outputs(df)
    print(f"\n\u2713 Saved \u2192 {OUTPUT_DIR / 'participant_demographics.txt'}")
    print(f"\u2713 Saved \u2192 {OUTPUT_DIR / 'participant_demographics.csv'}")

    # 2) AI-usage split + paper cross-check.
    study = df[df["in_study"]]
    table = compute_ai_usage_table(study, per_pid)

    text_report = report_ai_usage(table)
    print("\n" + text_report)

    (OUTPUT_DIR / "ai_usage_comparison.txt").write_text(text_report + "\n")
    table["merged_table"][[
        "ParticipantID", "group", "has_ai", "n_entries", "n_ai", "n_human",
        "pct_ai", "age", "gender", "country", "ethnicity", "education",
        "native_language",
    ]].to_csv(OUTPUT_DIR / "ai_usage_comparison.csv", index=False)
    (OUTPUT_DIR / "ai_usage_table.tex").write_text(build_latex_table(table))
    (OUTPUT_DIR / "ai_usage_paragraph.tex").write_text(build_latex_paragraph(table))

    print(f"\n\u2713 Saved \u2192 {OUTPUT_DIR / 'ai_usage_comparison.txt'}")
    print(f"\u2713 Saved \u2192 {OUTPUT_DIR / 'ai_usage_comparison.csv'}")
    print(f"\u2713 Saved \u2192 {OUTPUT_DIR / 'ai_usage_table.tex'}")
    print(f"\u2713 Saved \u2192 {OUTPUT_DIR / 'ai_usage_paragraph.tex'}")


if __name__ == "__main__":
    main()
