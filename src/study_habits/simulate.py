"""Generate a realistic *simulated* survey dataset.

There is no real student data in this project. The numbers come from a seeded
random generator with relationships I chose on purpose, so the analysis has
something to find. Then the data is deliberately made messy (typos, missing
values, impossible values, duplicate rows, units typed into number columns)
so the cleaning step has real work to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAJORS = ["Computer Science", "Biology", "Business", "Engineering", "Arts"]

# Ways people might type each major into a free-text survey box.
MAJOR_TYPOS = {
    "Computer Science": ["computer science", "CS", "Comp Sci", "compsci "],
    "Biology": ["biology", "Bio", "BIOLOGY"],
    "Business": ["business", "Bus.", "Business "],
    "Engineering": ["engineering", "Eng", "Engg"],
    "Arts": ["arts", "Art", "ARTS"],
}


def simulate(n: int = 400, seed: int = 2026) -> pd.DataFrame:
    """Return a clean simulated dataset with ``n`` students."""
    rng = np.random.default_rng(seed)
    major = rng.choice(MAJORS, size=n, p=[0.28, 0.2, 0.2, 0.2, 0.12])
    year = rng.integers(1, 5, size=n)
    job = rng.random(n) < 0.35

    sleep = np.clip(rng.normal(6.8, 1.2, n) - job * 0.4, 3.5, 10)
    study = np.clip(rng.gamma(4, 3.2, n) - job * 2 + (major == "Engineering") * 2, 0, 40)
    social = np.clip(rng.normal(3.2, 1.4, n) - study * 0.03, 0.2, 9)
    coffee = rng.poisson(1.6 + (major == "Computer Science") * 0.8 + (sleep < 6) * 0.7, n)
    coffee = np.clip(coffee, 0, 8)
    attendance = np.clip(rng.normal(82, 11, n) - social * 1.5 + 5, 30, 100)

    score = (
        42
        + 1.05 * np.minimum(study, 25)          # study helps, with diminishing returns
        + 0.2 * np.maximum(study - 25, 0)
        - 1.4 * (sleep - 7.8) ** 2 + 2           # best around 7-8 hours of sleep
        - 1.6 * social                            # phone / social media time hurts a bit
        + 0.18 * attendance
        + 1.5 * np.minimum(coffee, 2)             # a little coffee helps...
        - 2.0 * np.maximum(coffee - 3, 0)         # ...too much does not
        + rng.normal(0, 6, n)
    )
    score = np.clip(score, 0, 100)

    return pd.DataFrame(
        {
            "student_id": [f"S{1000 + i}" for i in range(n)],
            "major": major,
            "year": year,
            "part_time_job": np.where(job, "yes", "no"),
            "sleep_hours": sleep.round(1),
            "study_hours_week": study.round(1),
            "social_media_hours_day": social.round(1),
            "coffee_cups_day": coffee,
            "attendance_pct": attendance.round(0),
            "exam_score": score.round(1),
        }
    )


def make_messy(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Add the kinds of problems you see in real survey exports."""
    rng = np.random.default_rng(seed)
    messy = df.copy()
    n = len(messy)

    # 1. free-text majors with inconsistent spelling
    for i in rng.choice(n, size=n // 3, replace=False):
        messy.loc[i, "major"] = rng.choice(MAJOR_TYPOS[messy.loc[i, "major"]])

    # 2. units typed into number columns ("2 cups", "7.5h")
    messy["coffee_cups_day"] = messy["coffee_cups_day"].astype(object)
    for i in rng.choice(n, size=25, replace=False):
        messy.loc[i, "coffee_cups_day"] = f"{df.loc[i, 'coffee_cups_day']} cups"
    messy["sleep_hours"] = messy["sleep_hours"].astype(object)
    for i in rng.choice(n, size=15, replace=False):
        messy.loc[i, "sleep_hours"] = f"{df.loc[i, 'sleep_hours']}h"

    # 3. missing answers
    for col in ["sleep_hours", "social_media_hours_day", "attendance_pct"]:
        messy.loc[rng.choice(n, size=12, replace=False), col] = np.nan

    # 4. impossible values (typos like 75 instead of 7.5)
    messy.loc[rng.choice(n, size=4, replace=False), "sleep_hours"] = 75
    messy.loc[rng.choice(n, size=3, replace=False), "attendance_pct"] = 180

    # 5. yes/no in different styles
    messy["part_time_job"] = messy["part_time_job"].replace(
        {"yes": rng.choice(["yes", "Yes", "Y", "TRUE"]), "no": rng.choice(["no", "No", "N"])}
    )
    for i in rng.choice(n, size=20, replace=False):
        messy.loc[i, "part_time_job"] = rng.choice(["Y", "Yes", "N", "no", "FALSE"])

    # 6. a few people submitted the form twice
    dupes = messy.sample(8, random_state=int(seed))
    return pd.concat([messy, dupes]).sample(frac=1, random_state=int(seed)).reset_index(drop=True)
