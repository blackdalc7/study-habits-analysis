"""The actual questions, answered with pandas and NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "study_hours_week",
    "sleep_hours",
    "social_media_hours_day",
    "coffee_cups_day",
    "attendance_pct",
]

SLEEP_BINS = [0, 5, 6, 7, 8, 9, 24]
SLEEP_LABELS = ["<5", "5-6", "6-7", "7-8", "8-9", "9+"]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [*FEATURES, "exam_score"]
    return df[cols].describe().T[["mean", "std", "min", "50%", "max"]].round(1)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df[[*FEATURES, "exam_score"]].corr().round(2)


def score_by_sleep(df: pd.DataFrame) -> pd.DataFrame:
    bucket = pd.cut(df["sleep_hours"], SLEEP_BINS, labels=SLEEP_LABELS, right=False)
    out = df.groupby(bucket, observed=False)["exam_score"].agg(["mean", "count"])
    return out.round(1)


def score_by_coffee(df: pd.DataFrame) -> pd.DataFrame:
    cups = df["coffee_cups_day"].clip(upper=5)
    out = df.groupby(cups)["exam_score"].agg(["mean", "sem", "count"])
    out.index = [*map(str, out.index[:-1]), "5+"] if out.index.max() == 5 else out.index.map(str)
    out.index.name = "coffee_cups_day"
    return out.round(2)


def score_by_major(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("major", observed=True)
        .agg(
            students=("exam_score", "size"),
            avg_score=("exam_score", "mean"),
            avg_study=("study_hours_week", "mean"),
            avg_coffee=("coffee_cups_day", "mean"),
        )
        .sort_values("avg_score", ascending=False)
        .round(1)
    )


def job_vs_no_job(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(
        index="part_time_job",
        values=["exam_score", "sleep_hours", "study_hours_week"],
        aggfunc="mean",
    ).round(1)
    table.index = table.index.map({True: "has a job", False: "no job"})
    return table


@dataclass
class LinearFit:
    names: list[str]
    coefficients: np.ndarray
    intercept: float
    r_squared: float

    def as_series(self) -> pd.Series:
        return pd.Series(self.coefficients, index=self.names).round(3)


def linear_model(df: pd.DataFrame, features: list[str] = FEATURES) -> LinearFit:
    """Ordinary least squares with plain NumPy (no scikit-learn)."""
    x = df[features].to_numpy(dtype=float)
    y = df["exam_score"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return LinearFit(features, beta[1:], float(beta[0]), 1 - ss_res / ss_tot)


def best_sleep_bucket(df: pd.DataFrame) -> str:
    table = score_by_sleep(df)
    table = table[table["count"] >= 10]
    return str(table["mean"].idxmax())
