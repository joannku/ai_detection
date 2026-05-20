#!/usr/bin/env python3
"""Aggregate rater (annotator) demographics for the paper.

Cleans the Qualtrics export of the retrospective rater-demographics survey
(20 UCL students were invited; the lead author JK is the 21st annotator and
is reported separately in the paper) and emits four artefacts:

  output/rater_demographics/
    rater_demographics_clean.csv          one row per valid response
    rater_demographics.txt                aggregate distributions (debug)
    rater_demographics.tex                LaTeX table for 5results.tex
    rater_demographics_paragraph.tex      LaTeX paragraph for 99_paper_text.py

Reported in aggregate (per the format suggested by collaborators):
  age (mean / SD / range, computed from DOB at the labelling-task midpoint),
  gender, native language (English vs other), English proficiency,
  programme of study, year of study, AI familiarity, prior AI-detection
  experience (multi-select), personal journalling experience, and self-rated
  labelling confidence.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "overleaf_paper"
DATA_DIR = ROOT / "input" / "demographics"
OUTPUT_DIR = ROOT / "output" / "rater_demographics"

# Reference date for age calculation: approximate midpoint of the labelling
# task. The rating files are dated Dec 2025 – Jan 2026.
LABELLING_DATE = datetime(2026, 1, 1)

# How many students were invited (the lead-author rater_0/JK is excluded from
# this survey; she labelled the entire dataset and is reported separately).
N_INVITED = 20


# ── Discovering the latest Qualtrics export ────────────────────────────────
# Qualtrics filenames look like:  "Demographics_May 11, 2026_19.07.csv".
# We pick the one with the most recent timestamp encoded in its name; ties
# are broken by filesystem mtime so re-saved files still resolve.

_DEMOGRAPHICS_FILENAME_RE = re.compile(
    r"^Demographics_"
    r"(?P<month>[A-Za-z]+) (?P<day>\d{1,2}), (?P<year>\d{4})"
    r"_(?P<hour>\d{1,2})\.(?P<minute>\d{2})\.csv$"
)


def _filename_timestamp(path: Path) -> datetime:
    m = _DEMOGRAPHICS_FILENAME_RE.match(path.name)
    if m is None:
        return datetime.min
    try:
        return datetime.strptime(
            f"{m['month']} {m['day']} {m['year']} {m['hour']}:{m['minute']}",
            "%B %d %Y %H:%M",
        )
    except ValueError:
        return datetime.min


def _latest_demographics_csv() -> Path:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Demographics directory not found: {DATA_DIR}. "
            "Drop the latest Qualtrics export into input/demographics/."
        )
    candidates = list(DATA_DIR.glob("Demographics_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No Demographics_*.csv found in {DATA_DIR}. "
            "Drop the latest Qualtrics export into input/demographics/."
        )
    return max(candidates, key=lambda p: (_filename_timestamp(p), p.stat().st_mtime))


INPUT_CSV = _latest_demographics_csv()


# ── Loading & cleaning ─────────────────────────────────────────────────────

# In the raw CSV the column header `q_ai` appears twice; pandas auto-renames
# the second to `q_ai.1`. The second one is actually the journalling
# question, which is why the mapping below distinguishes them.
COLUMN_RENAME = {
    "q_ucl": "ucl_id",
    "q_name": "name",
    "q_dob": "dob_raw",
    "q_gender": "gender",
    "q_degree": "programme",
    "q_study": "year_of_study",
    "q_native": "native_language_choice",
    "q_native_2_TEXT": "native_language_other",
    "q_english": "english_proficiency",
    "q_ai": "ai_familiarity",
    "q_labelling": "ai_detection_experience",
    "q_ai.1": "journalling_experience",
    "q_confidence": "confidence",
}


def load_raw() -> pd.DataFrame:
    """Read the export, skipping Qualtrics' two non-data header rows."""
    df = pd.read_csv(INPUT_CSV, skiprows=[1, 2], dtype=str)
    return df.rename(columns=COLUMN_RENAME)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop blank submissions, the JK test row, and duplicate UCL IDs.

    On duplicates we keep the most recently submitted response (sorted by
    Qualtrics' RecordedDate column).
    """
    df = df.copy()
    df["ucl_id"] = df["ucl_id"].astype(str).str.strip()
    df = df[df["ucl_id"].str.lower().ne("nan")]
    df = df[df["ucl_id"].str.len() > 0]

    # Drop the lead-author test submission (UCL field starts with "Test").
    df = df[~df["ucl_id"].str.lower().str.startswith("test")]

    df["ucl_id_norm"] = df["ucl_id"].str.lower().str.replace(r"\s+", "", regex=True)
    df = df.sort_values("RecordedDate").drop_duplicates("ucl_id_norm", keep="last")
    return df.drop(columns=["ucl_id_norm"]).reset_index(drop=True)


# ── Field normalisation ────────────────────────────────────────────────────

def parse_dob(raw) -> datetime | None:
    """Parse a DOB entered in any of several free-text formats."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Compact 8-digit form (e.g. 14032007 → 14-03-2007).
    if re.fullmatch(r"\d{8}", text):
        text = f"{text[0:2]}-{text[2:4]}-{text[4:8]}"
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dob"] = df["dob_raw"].apply(parse_dob)
    df["age"] = df["dob"].apply(
        lambda d: None if d is None else (LABELLING_DATE - d).days / 365.25
    )
    return df


# Free-text language entries: lightly canonicalise but preserve self-id.
# "Chinese" without further qualification is read as Mandarin.
_LANGUAGE_ALIASES = {
    "chinese": "Mandarin Chinese",
    "mandarin": "Mandarin Chinese",
    "mandarin chinese": "Mandarin Chinese",
    "putonghua": "Mandarin Chinese",
}


def _canonical_language(text) -> str:
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return _LANGUAGE_ALIASES.get(cleaned.lower(), cleaned.title())


def normalise_native_language(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve the dropdown + free-text native-language pair into one column."""
    df = df.copy()
    choice = df["native_language_choice"].fillna("").str.strip()
    other = df["native_language_other"].fillna("").str.strip()
    df["native_language"] = np.where(
        choice.str.lower().str.startswith("other"),
        other.replace("", "Other (unspecified)"),
        choice,
    ).tolist()
    df["native_language"] = df["native_language"].apply(_canonical_language)
    df["english_l1"] = df["native_language"].str.lower().eq("english")
    return df


def _bucket_programme(text) -> str:
    """Fine-grained programme bucket (BSc/MSci Psychology, BA Linguistics, ...)."""
    if not isinstance(text, str) or not text.strip():
        return "Missing"
    lower = text.lower()
    is_psych = "psycholog" in lower
    is_ling = "linguistic" in lower or "language" in lower
    if is_psych and ("msci" in lower or "m.sci" in lower):
        return "MSci Psychology"
    if is_psych and ("bsc" in lower or "bcs" in lower or "b.sc" in lower):
        return "BSc Psychology"
    if is_psych and "ba " in lower:
        return "BA Psychology"
    if is_psych:
        return "Psychology (degree unspecified)"
    if is_ling and ("ba " in lower or "bsc" not in lower):
        return "BA Linguistics"
    if is_ling:
        return "Linguistics"
    return text.strip().title()


def _field_of_study(bucket: str) -> str:
    if bucket == "Missing":
        return "Missing"
    if "Psychology" in bucket:
        return "Psychology"
    if "Linguistics" in bucket:
        return "Linguistics"
    return bucket


def _degree_level(text) -> str:
    if not isinstance(text, str):
        return "Missing"
    lower = text.lower()
    if "phd" in lower or "doctoral" in lower:
        return "PhD"
    if any(t in lower for t in ("msci", "bsc", "bcs", "ba ", "ba\xa0", "ba_",
                                "psycholog", "linguistic")):
        return "Undergraduate"
    return text.strip()


def add_programme_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add `programme_bucket`, `field_of_study`, `degree_level` columns.

    We treat an unspecified-degree Psychology answer as BSc when the
    rater is an undergraduate (UCL's BSc is the default; if they were on
    the MSci track they would have written it).
    """
    df = df.copy()
    df["programme_bucket"] = df["programme"].apply(_bucket_programme)

    is_unspecified = df["programme_bucket"].eq("Psychology (degree unspecified)")
    is_undergrad = (df["year_of_study"].fillna("").str.lower()
                    .str.startswith("undergraduate"))
    df.loc[is_unspecified & is_undergrad, "programme_bucket"] = "BSc Psychology"

    df["field_of_study"] = df["programme_bucket"].apply(_field_of_study)
    df["degree_level"] = df["programme"].apply(_degree_level)
    return df


def short_familiarity(label) -> str:
    """Strip the explanatory clause from "Slightly familiar — ..." options."""
    if not isinstance(label, str):
        return "Missing"
    return label.split("\u2014")[0].strip()


# ── Survey scales & display labels ─────────────────────────────────────────
# Scales run from lowest to highest level of the construct so that
# `_ordered_counts()` renders them in survey-defined (not frequency) order.

AI_FAMILIARITY_SCALE = [
    "Not at all familiar", "Slightly familiar", "Moderately familiar",
    "Very familiar", "Extremely familiar",
]

CONFIDENCE_SCALE = [
    "Not at all confident", "Slightly confident", "Moderately confident",
    "Very confident", "Extremely confident",
]

JOURNALLING_SCALE = [
    "No, I have never journalled",
    "I have tried journalling but do not do it regularly",
    "I journal occasionally (e.g., a few times a month)",
    "I journal regularly (e.g., weekly or more)",
]

# Note: Qualtrics encodes the dashes inside option labels as en-dashes
# (U+2013), not ASCII hyphens. Keep these strings byte-identical to the
# survey export so they match exactly.
ENGLISH_PROFICIENCY_SCALE = [
    "Basic (A1\u2013A2)",
    "Intermediate (B1\u2013B2)",
    "Advanced (C1\u2013C2: comfortable reading and writing academic-level English)",
    "Near-native / bilingual",
    "Native speaker",
]

YEAR_OF_STUDY_SCALE = [
    "Undergraduate Year 1", "Undergraduate Year 2", "Undergraduate Year 3",
    "Undergraduate Year 4+", "Master's (taught)", "Master's (research / MRes)",
    "PhD / Doctoral",
]

PROGRAMME_ORDER = [
    "BSc Psychology", "MSci Psychology", "BA Psychology",
    "Psychology (degree unspecified)",
    "BA Linguistics", "Linguistics",
]

# Human-friendly short forms used in tables / paragraph prose.
JOURNALLING_SHORT = {
    "No, I have never journalled": "never journalled",
    "I have tried journalling but do not do it regularly": "tried journalling but not regularly",
    "I journal occasionally (e.g., a few times a month)": "journalled occasionally",
    "I journal regularly (e.g., weekly or more)": "journalled regularly",
}

ENGLISH_PROFICIENCY_SHORT = {
    "Basic (A1\u2013A2)": "basic (A1\u2013A2)",
    "Intermediate (B1\u2013B2)": "intermediate (B1\u2013B2)",
    "Advanced (C1\u2013C2: comfortable reading and writing academic-level English)":
        "advanced (C1\u2013C2)",
    "Near-native / bilingual": "near-native / bilingual",
    "Native speaker": "native speaker",
}

# Q9 (multi-select). Qualtrics joins selected options with ", " but some option
# labels contain commas, so we cannot just split — we substring-match instead.
AI_EXPERIENCE_OPTIONS = [
    "I have compared AI-generated and human-written text out of personal curiosity",
    "I have checked student or peer submissions for potential AI use (e.g., as a teaching assistant, tutor, or marker)",
    "I have used AI detection tools (e.g., GPTZero, Turnitin AI detection)",
    "I have read articles or research about identifying AI-generated text",
    "I have discussed how to spot AI writing with friends, colleagues, or in class",
    "I have participated in other AI detection or annotation tasks",
    "None of the above",
]

AI_EXPERIENCE_SHORT = {
    AI_EXPERIENCE_OPTIONS[0]: "compared AI vs human text out of curiosity",
    AI_EXPERIENCE_OPTIONS[1]: "checked submissions as TA/tutor/marker",
    AI_EXPERIENCE_OPTIONS[2]: "used AI detection tools",
    AI_EXPERIENCE_OPTIONS[3]: "read articles/research on AI-text detection",
    AI_EXPERIENCE_OPTIONS[4]: "discussed AI writing socially or in class",
    AI_EXPERIENCE_OPTIONS[5]: "participated in other detection tasks",
    AI_EXPERIENCE_OPTIONS[6]: "none of the above",
}

# Friendly noun-phrase forms used by the LaTeX paragraph (not the table).
AI_EXPERIENCE_PHRASE = {
    AI_EXPERIENCE_OPTIONS[0]: "comparing AI- and human-written text out of curiosity",
    AI_EXPERIENCE_OPTIONS[1]: "checking submissions as a TA/tutor/marker",
    AI_EXPERIENCE_OPTIONS[2]: "having used AI-detection tools",
    AI_EXPERIENCE_OPTIONS[3]: "reading articles or research on AI-text detection",
    AI_EXPERIENCE_OPTIONS[4]: "discussing AI writing socially or in class",
    AI_EXPERIENCE_OPTIONS[5]: "participating in other detection tasks",
}


def parse_ai_experience(value) -> list[str]:
    """Recover the selected options by matching against the known labels."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [opt for opt in AI_EXPERIENCE_OPTIONS if opt in value]


# ── Aggregation helpers ────────────────────────────────────────────────────

def _ordered_counts(
    series: pd.Series,
    scale: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    *,
    show_zero: bool = False,
) -> list[tuple[str, int]]:
    """Return [(display_label, count)] for a categorical series.

    If ``scale`` is given, rows follow the survey scale order (with any
    unexpected values appended at the end), and zero-count rows are
    included only when ``show_zero=True``. ``label_map`` rewrites any
    label before display. Without ``scale``, rows are returned in
    descending-frequency order.
    """
    counts = series.value_counts()
    if scale is None:
        return [(value, int(n)) for value, n in counts.items()]

    out: list[tuple[str, int]] = []
    seen: set = set()
    for level in scale:
        n = int(counts.get(level, 0))
        if n == 0 and not show_zero:
            continue
        label = label_map.get(level, level) if label_map else level
        out.append((label, n))
        seen.add(level)
    for value, n in counts.items():
        if value in seen or int(n) == 0:
            continue
        label = label_map.get(value, value) if label_map else value
        out.append((label, int(n)))
    return out


def _count_with_pct(n: int, total: int) -> str:
    """Plain-text "<n> (<pct>%)" used by the .txt summary."""
    if total == 0:
        return "0 (0.0%)"
    return f"{n} ({n / total:.1%})"


def _latex_pct(n: int, total: int) -> str:
    """LaTeX-safe percentage like "85.0\\%"."""
    return f"{n / total:.1%}".replace("%", "\\%") if total else "0.0\\%"


def _join_oxford(parts: list[str]) -> str:
    """Comma-separate a list with an Oxford 'and' before the final item."""
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _ai_experience_counts(df: pd.DataFrame) -> pd.Series:
    """Frequency of each Q9 option across all responses (multi-select)."""
    flat: list[str] = []
    for v in df["ai_detection_experience"].fillna(""):
        flat.extend(parse_ai_experience(v))
    return pd.Series(flat).value_counts() if flat else pd.Series(dtype=int)


# ── Output 1: text summary (rater_demographics.txt + stdout) ───────────────

def _format_distribution(
    series: pd.Series,
    label: str,
    *,
    scale: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    show_zero: bool = False,
) -> str:
    """Render a column as a `_count_with_pct` block for the .txt output."""
    series = series.fillna("Missing")
    pairs = _ordered_counts(series, scale=scale, label_map=label_map,
                            show_zero=show_zero)
    total = len(series)
    lines = [f"  {label} (n={total}):"]
    for value, n in pairs:
        lines.append(f"    {value}: {_count_with_pct(n, total)}")
    return "\n".join(lines)


def build_summary(df: pd.DataFrame) -> str:
    n = len(df)
    age = df["age"].dropna()

    lines = [
        "Rater (annotator) demographics summary",
        "=" * 60,
        f"  Invited:    {N_INVITED} UCL students",
        f"  Responded:  {n} ({n / N_INVITED:.1%})",
        f"  Outstanding: {N_INVITED - n}",
        "",
        "Note: the lead author (rater_0 / JK) labelled the entire dataset",
        "and is reported separately. This summary covers the student annotators.",
    ]
    if len(age):
        lines += ["", f"  Age (n={len(age)}): M={age.mean():.1f}, "
                      f"SD={age.std():.1f}, "
                      f"range={age.min():.0f}\u2013{age.max():.0f}"]

    lines += [
        "",
        _format_distribution(df["gender"], "Gender"),
        "",
        _format_distribution(df["native_language"], "Native language"),
        "",
        f"  English as L1: {_count_with_pct(int(df['english_l1'].sum()), n)}; "
        f"Other L1: {_count_with_pct(int((~df['english_l1']).sum()), n)}",
        "",
        _format_distribution(df["english_proficiency"], "English proficiency",
                             scale=ENGLISH_PROFICIENCY_SCALE, show_zero=True),
        "",
        _format_distribution(df["programme_bucket"], "Programme of study (detailed)"),
        "",
        _format_distribution(df["field_of_study"], "Field of study"),
        "",
        _format_distribution(df["year_of_study"], "Year of study",
                             scale=YEAR_OF_STUDY_SCALE),
        "",
        _format_distribution(df["ai_familiarity"].apply(short_familiarity),
                             "AI familiarity",
                             scale=AI_FAMILIARITY_SCALE, show_zero=True),
        "",
        _format_distribution(df["journalling_experience"],
                             "Personal journalling experience",
                             scale=JOURNALLING_SCALE, show_zero=True),
        "",
        _format_distribution(df["confidence"], "Self-rated labelling confidence",
                             scale=CONFIDENCE_SCALE, show_zero=True),
        "",
        f"  Prior AI-detection experience (n raters = {n}, multi-select):",
    ]
    counts = _ai_experience_counts(df)
    for option in AI_EXPERIENCE_OPTIONS:
        c = int(counts.get(option, 0))
        short = AI_EXPERIENCE_SHORT.get(option, option)
        lines.append(f"    {short}: {c} raters ({c / n:.1%})")

    return "\n".join(lines)


# ── Output 2: LaTeX table (rater_demographics.tex) ─────────────────────────

def _latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _titlecase_first(s: str) -> str:
    """Upper-case only the first character (preserves acronyms / scale codes)."""
    return s[0].upper() + s[1:] if s else s


def _table_row(label: str, n: int, total: int, *, indent: bool = False) -> str:
    prefix = "\\quad " if indent else ""
    return f"{prefix}{_latex_escape(label)} & {n} & {n / total * 100:.1f}\\\\"


def _section_header(title: str) -> str:
    return f"\\addlinespace[2pt]\\textit{{{_latex_escape(title)}}} & & \\\\"


def build_latex_table(df: pd.DataFrame) -> str:
    """Compact (n, %) table mirroring the participant-demographics style."""
    n = len(df)
    age = df["age"].dropna()
    age_str = (
        f"$M$={age.mean():.1f} ($SD$={age.std():.1f}), "
        f"range {age.min():.0f}--{age.max():.0f}"
        if len(age) else "not reported"
    )

    rows: list[str] = [
        f"Age (years) & \\multicolumn{{2}}{{r}}{{{age_str}}}\\\\",
    ]

    # ── Gender ──
    rows.append(_section_header("Gender"))
    for value, c in _ordered_counts(df["gender"].fillna("Missing")):
        if c:
            rows.append(_table_row(value, c, n, indent=True))

    # ── Year of study ──
    rows.append(_section_header("Year of study"))
    for label, c in _ordered_counts(df["year_of_study"].fillna("Missing"),
                                    YEAR_OF_STUDY_SCALE):
        rows.append(_table_row(label, c, n, indent=True))

    # ── Programme of study (Missing rendered as "Not reported") ──
    rows.append(_section_header("Programme"))
    for label, c in _ordered_counts(df["programme_bucket"].fillna("Missing"),
                                    PROGRAMME_ORDER):
        if label != "Missing":
            rows.append(_table_row(label, c, n, indent=True))
    n_missing_prog = int(df["programme_bucket"].fillna("Missing")
                         .eq("Missing").sum())
    if n_missing_prog:
        rows.append(_table_row("Not reported", n_missing_prog, n, indent=True))

    # ── Native language: collapse "other" into a single row + footnote ──
    rows.append(_section_header("Native language"))
    n_english = int(df["english_l1"].sum())
    rows.append(_table_row("English", n_english, n, indent=True))
    others = sorted({l for l in df.loc[~df["english_l1"], "native_language"]
                     .dropna() if l})
    if others:
        # The full list is too long to fit a single column without overflow,
        # so we show a short marker on the row and put the languages in a
        # parbox-wrapped footnote below \bottomrule.
        rows.append(_table_row("Other$^\\dagger$", n - n_english, n, indent=True))
    table_footnote = (
        "\\multicolumn{3}{@{}l@{}}{"
        "\\parbox[t]{\\columnwidth}{\\footnotesize "
        "$^\\dagger$Other native languages: "
        + ", ".join(others) + ".}}\\\\"
        if others else ""
    )

    # ── English proficiency ──
    rows.append(_section_header("English proficiency"))
    for label, c in _ordered_counts(df["english_proficiency"].fillna("Missing"),
                                    ENGLISH_PROFICIENCY_SCALE,
                                    ENGLISH_PROFICIENCY_SHORT):
        rows.append(_table_row(_titlecase_first(label), c, n, indent=True))

    # ── AI familiarity ──
    rows.append(_section_header("AI familiarity"))
    for label, c in _ordered_counts(
            df["ai_familiarity"].apply(short_familiarity), AI_FAMILIARITY_SCALE):
        rows.append(_table_row(label, c, n, indent=True))

    # ── Journalling experience ──
    rows.append(_section_header("Journalling experience"))
    for label, c in _ordered_counts(df["journalling_experience"].fillna("Missing"),
                                    JOURNALLING_SCALE, JOURNALLING_SHORT):
        rows.append(_table_row(_titlecase_first(label), c, n, indent=True))

    # ── Labelling confidence ──
    rows.append(_section_header("Labelling confidence"))
    for label, c in _ordered_counts(df["confidence"].fillna("Missing"),
                                    CONFIDENCE_SCALE):
        rows.append(_table_row(label, c, n, indent=True))

    # ── Prior AI-detection experience (multi-select, "None" pinned last) ──
    rows.append(_section_header("Prior AI-detection experience (multi-select)"))
    counts = _ai_experience_counts(df)
    for option in AI_EXPERIENCE_OPTIONS:
        if option == "None of the above":
            continue
        c = int(counts.get(option, 0))
        if c:
            rows.append(_table_row(_titlecase_first(AI_EXPERIENCE_SHORT[option]),
                                   c, n, indent=True))
    none_count = int(counts.get("None of the above", 0))
    if none_count:
        rows.append(_table_row("None of the above", none_count, n, indent=True))

    body = "\n".join(rows)
    footnote_block = f"\\addlinespace[2pt]\n{table_footnote}\n" if table_footnote else ""
    return (
        "\\begin{table}[tbp]\n"
        "\\centering\n\\small\n"
        "\\caption{Demographics, study background and prior AI exposure of "
        f"the {n} UCL student annotators (the lead author, who labelled the "
        "entire dataset, is reported separately in the text).}\n"
        "\\label{tab:rater_demo}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrr@{}}\n"
        "\\toprule\n"
        "& \\textbf{$n$} & \\textbf{\\%}\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        f"{footnote_block}"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ── Output 3: LaTeX paragraph (rater_demographics_paragraph.tex) ───────────

def _phrase_with_shared_suffix(series: pd.Series, scale: list[str],
                               shared_suffix: str) -> str:
    """Render an ordinal series as e.g. "10% slightly, 20% moderately".

    Each scale level is lowercased and stripped of the shared trailing
    word (so "Slightly familiar" → "slightly"); the shared word is
    re-attached once by the calling sentence template.
    """
    counts = series.value_counts()
    total = len(series)
    parts = []
    for level in scale:
        c = int(counts.get(level, 0))
        if c == 0:
            continue
        label = level.lower()
        if label.endswith(" " + shared_suffix):
            label = label[: -(len(shared_suffix) + 1)]
        parts.append(f"{_latex_pct(c, total)} {label}")
    return _join_oxford(parts)


def build_latex_paragraph(df: pd.DataFrame) -> str:
    """A single dense paragraph mirroring the participant-demographics style."""
    n = len(df)
    age = df["age"].dropna()

    # Gender — frequency order, lowercase ("90.0\\% female, 10.0\\% male").
    gender_phrase = ", ".join(
        f"{_latex_pct(int(c), n)} {v.lower()}"
        for v, c in df["gender"].value_counts().items()
    )

    # Year of study — first item keeps the "Undergraduate" prefix.
    year_parts = []
    for label, c in _ordered_counts(df["year_of_study"].fillna("Missing"),
                                    YEAR_OF_STUDY_SCALE):
        short = label if not year_parts else label.replace("Undergraduate ", "")
        year_parts.append(f"{_latex_pct(c, n)} in {short}")
    year_clause = _join_oxford(year_parts)

    # Programme — "BSc Psychology (60.0\\%), MSci Psychology (15.0\\%), …".
    # We deliberately exclude "Missing" from the listing and surface it via
    # the trailing "X annotators did not specify a programme" clause.
    programme_no_missing = df["programme_bucket"][
        df["programme_bucket"] != "Missing"]
    prog_pairs = [f"{label} ({_latex_pct(c, n)})"
                  for label, c in _ordered_counts(programme_no_missing,
                                                  PROGRAMME_ORDER)]
    programme_phrase = _join_oxford(prog_pairs)
    n_missing_prog = int(df["programme_bucket"].eq("Missing").sum())
    programme_missing_tail = (
        f"; {n_missing_prog} annotator{'s' if n_missing_prog != 1 else ''} "
        "did not specify a programme" if n_missing_prog else ""
    )

    # Native language — English first; non-English listed with "or".
    n_english = int(df["english_l1"].sum())
    other_lang_phrase = ""
    if n - n_english > 0:
        sorted_langs = sorted({
            l for l in df.loc[~df["english_l1"], "native_language"].dropna() if l
        })
        if len(sorted_langs) >= 2:
            others_listed = ", ".join(sorted_langs[:-1]) + ", or " + sorted_langs[-1]
        else:
            others_listed = sorted_langs[0]
        other_lang_phrase = f"; the remaining {n - n_english} reported {others_listed}"

    # English proficiency — render in highest→lowest order (mirrors paper style).
    eng_prof_pairs = [
        f"{_latex_pct(c, n)} {label}"
        for label, c in _ordered_counts(
            df["english_proficiency"].dropna(),
            list(reversed(ENGLISH_PROFICIENCY_SCALE)),
            ENGLISH_PROFICIENCY_SHORT,
        )
    ]
    eng_prof_phrase = _join_oxford(eng_prof_pairs)

    # AI familiarity / labelling confidence — share a trailing adjective.
    fam_phrase = _phrase_with_shared_suffix(
        df["ai_familiarity"].apply(short_familiarity)
                            .pipe(lambda s: s[s != "Missing"]),
        AI_FAMILIARITY_SCALE, shared_suffix="familiar",
    )
    conf_phrase = _phrase_with_shared_suffix(
        df["confidence"].dropna(), CONFIDENCE_SCALE, shared_suffix="confident",
    )

    # Prior AI-detection experience — descending frequency, "none" split out.
    counts = _ai_experience_counts(df)
    exp_pairs = []
    for option, c in counts.items():
        if option == "None of the above" or c == 0:
            continue
        exp_pairs.append(
            f"{AI_EXPERIENCE_PHRASE.get(option, option)} ({_latex_pct(int(c), n)})"
        )
    exp_phrase = _join_oxford(exp_pairs) if exp_pairs else "no prior experience reported"
    none_count = int(counts.get("None of the above", 0))
    none_tail = (f"; a further {_latex_pct(none_count, n)} reported none of these"
                 if none_count else "")

    # Journalling — "20.0\\% had never journalled, 60.0\\% had tried…".
    # We iterate the scale directly so the verb tense is consistent across
    # all four levels.
    jou_counts = df["journalling_experience"].dropna().value_counts()
    jou_pairs = [f"{_latex_pct(int(jou_counts.get(level, 0)), n)} had "
                 f"{JOURNALLING_SHORT[level]}"
                 for level in JOURNALLING_SCALE
                 if int(jou_counts.get(level, 0)) > 0]
    jou_phrase = _join_oxford(jou_pairs)

    return (
        f"\\paragraph{{Annotator demographics.}}\n"
        f"Of the {N_INVITED} UCL student annotators, the mean age was "
        f"{age.mean():.1f} years ($SD$={age.std():.1f}, "
        f"range {age.min():.0f}--{age.max():.0f}), with {year_clause}. "
        f"They were drawn from {programme_phrase} programmes"
        f"{programme_missing_tail}. "
        f"The pool was {gender_phrase}. "
        f"English was the first language for {_latex_pct(n_english, n)} of annotators"
        f"{other_lang_phrase}. "
        f"Self-rated English proficiency was {eng_prof_phrase}. "
        f"At the time of labelling, annotators described themselves as "
        f"{fam_phrase} familiar with AI-generated text. "
        f"The most common prior AI-detection experiences were "
        f"{exp_phrase}{none_tail}. "
        f"Personal journalling experience was that {jou_phrase}. "
        f"Self-rated confidence in distinguishing AI-generated from human-written "
        f"entries was {conf_phrase} confident.\n"
    )


# ── Driver ─────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = (load_raw()
          .pipe(clean)
          .pipe(add_age)
          .pipe(normalise_native_language)
          .pipe(add_programme_fields))

    # 1. Cleaned per-row CSV (subset of columns, in a stable order).
    keep_cols = [
        "ucl_id", "name", "dob_raw", "dob", "age", "gender", "programme",
        "programme_bucket", "field_of_study", "degree_level", "year_of_study",
        "native_language", "english_l1", "english_proficiency",
        "ai_familiarity", "ai_detection_experience", "journalling_experience",
        "confidence", "RecordedDate",
    ]
    df[[c for c in keep_cols if c in df.columns]].to_csv(
        OUTPUT_DIR / "rater_demographics_clean.csv", index=False
    )

    # 2. Plain-text aggregate summary (also printed to stdout).
    summary = build_summary(df)
    (OUTPUT_DIR / "rater_demographics.txt").write_text(summary + "\n")

    # 3. LaTeX paragraph (consumed by 99_paper_text.py).
    (OUTPUT_DIR / "rater_demographics_paragraph.tex").write_text(
        build_latex_paragraph(df)
    )

    # 4. LaTeX table (used directly by 5results.tex).
    table_path = OUTPUT_DIR / "rater_demographics.tex"
    table_path.write_text(build_latex_table(df))
    paper_table_path = PAPER_DIR / "rater_demographics.tex"
    if PAPER_DIR.exists():
        shutil.copyfile(table_path, paper_table_path)

    # Echo summary + output locations.
    print(summary)
    print()
    print("=" * 60)
    print(f"Cleaned table   → {OUTPUT_DIR / 'rater_demographics_clean.csv'}")
    print(f"Text summary    → {OUTPUT_DIR / 'rater_demographics.txt'}")
    print(f"LaTeX table     → {table_path}")
    print(f"LaTeX paragraph → {OUTPUT_DIR / 'rater_demographics_paragraph.tex'}")
    if PAPER_DIR.exists():
        print(f"Copied to       → {paper_table_path}")


if __name__ == "__main__":
    main()
