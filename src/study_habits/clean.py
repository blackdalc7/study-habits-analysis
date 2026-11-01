"""Turn the messy survey export into a tidy DataFrame.

Every step records what it did in a ``CleaningLog`` so the report can show it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from study_habits.simulate import MAJORS

MAJOR_ALIASES = {
    "cs": "Computer Science",
    "comp sci": "Computer Science",
    "compsci": "Computer Science",
    "computer science": "Computer Science",
    "bio": "Biology",
    "biology": "Biology",
    "bus.": "Business",
    "bus": "Business",
    "business": "Business",
    "eng": "Engineering",
    "engg": "Engineering",
    "engineering": "Engineering",
    "art": "Arts",
    "arts": "Arts",
}

YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}

# Plausible ranges. Anything outside becomes NaN instead of silently skewing averages.
VALID_RANGES = {
    "sleep_hours": (2, 14),
    "study_hours_week": (0, 80),
    "social_media_hours_day": (0, 16),
    "coffee_cups_day": (0, 12),
    "attendance_pct": (0, 100),
    "exam_score": (0, 100),
}

NUMERIC = list(VALID_RANGES)


@dataclass
class CleaningLog:
    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, what: str, count: int) -> None:
        self.steps.append((what, int(count)))

    def as_markdown(self) -> str:
        rows = "\n".join(f"| {what} | {count} |" for what, count in self.steps)
        return f"| Step | Rows / values affected |\n|---|---|\n{rows}"


def normalise_major(value: object) -> str | float:
    if not isinstance(value, str):
        return np.nan
    key = value.strip().lower()
    if key in MAJOR_ALIASES:
        return MAJOR_ALIASES[key]
    for major in MAJORS:
        if key == major.lower():
            return major
    return np.nan


def parse_yes_no(value: object) -> bool | float:
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in YES:
        return True
    if key in NO:
        return False
    return np.nan


def to_number(series: pd.Series) -> pd.Series:
    """'2 cups' -> 2.0, '7.5h' -> 7.5, 'n/a' -> NaN"""
    extracted = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    df = raw.copy()
    log.add("rows in raw export", len(df))

    before = len(df)
    df = df.drop_duplicates()
    log.add("exact duplicate rows removed", before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first")
    log.add("repeat submissions (same student_id) removed", before - len(df))

    original = df["major"].copy()
    df["major"] = df["major"].map(normalise_major)
    log.add("major spellings standardised", (original != df["major"]).sum())
    df["major"] = pd.Categorical(df["major"], categories=MAJORS)

    original = df["part_time_job"].copy()
    df["part_time_job"] = df["part_time_job"].map(parse_yes_no).astype("boolean")
    log.add("yes/no answers standardised", (~original.astype(str).isin(["yes", "no"])).sum())

    for col in NUMERIC:
        had_text = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        df[col] = to_number(df[col])
        if had_text.any():
            log.add(f"{col}: units/text stripped", had_text.sum())

    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        df.loc[bad, col] = np.nan
        if bad.any():
            log.add(f"{col}: impossible values set to missing", bad.sum())

    missing = df[NUMERIC].isna().sum().sum()
    log.add("missing numeric values (filled with the major's median)", missing)
    for col in NUMERIC:
        df[col] = df[col].fillna(df.groupby("major", observed=True)[col].transform("median"))

    df["year"] = df["year"].astype(int)
    df["coffee_cups_day"] = df["coffee_cups_day"].round().astype(int)
    df = df.sort_values("student_id").reset_index(drop=True)
    log.add("rows in clean dataset", len(df))
    return df, log
