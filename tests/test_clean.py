import numpy as np
import pandas as pd
import pytest

from study_habits.clean import clean, normalise_major, parse_yes_no, to_number
from study_habits.simulate import MAJORS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("CS", "Computer Science"), (" compsci ", "Computer Science"), ("Bio", "Biology"),
     ("ENGINEERING", "Engineering"), ("Bus.", "Business"), ("Art", "Arts")],
)
def test_normalise_major(raw, expected):
    assert normalise_major(raw) == expected


def test_normalise_major_unknown():
    assert np.isnan(normalise_major("Underwater Basket Weaving"))
    assert np.isnan(normalise_major(None))


@pytest.mark.parametrize(("raw", "expected"), [("Y", True), ("TRUE", True), ("no", False),
                                               ("N", False), (True, True)])
def test_parse_yes_no(raw, expected):
    assert parse_yes_no(raw) is expected


def test_parse_yes_no_unknown():
    assert np.isnan(parse_yes_no("maybe"))


def test_to_number_strips_units():
    out = to_number(pd.Series(["2 cups", "7.5h", "3", None, "n/a"]))
    assert out.iloc[:3].tolist() == [2.0, 7.5, 3.0]
    assert out.iloc[3:].isna().all()


def test_clean_result_is_tidy(cleaned):
    assert len(cleaned) == 400
    assert cleaned["student_id"].is_unique
    assert set(cleaned["major"].unique()) <= set(MAJORS)
    assert cleaned[["sleep_hours", "attendance_pct", "coffee_cups_day"]].notna().all().all()
    assert cleaned["sleep_hours"].max() <= 14
    assert cleaned["attendance_pct"].max() <= 100
    assert cleaned["part_time_job"].isin([True, False]).all()


def test_clean_recovers_original_values(cleaned, tidy):
    # values that were only "dressed up" (units, spelling) should come back exactly
    merged = cleaned.merge(tidy, on="student_id", suffixes=("", "_true"))
    assert (merged["major"].astype(str) == merged["major_true"]).all()
    assert (merged["coffee_cups_day"] == merged["coffee_cups_day_true"]).all()
    assert (merged["exam_score"] == merged["exam_score_true"]).all()


def test_clean_log(messy):
    _, log = clean(messy)
    steps = dict(log.steps)
    assert steps["exact duplicate rows removed"] == 8
    assert steps["sleep_hours: impossible values set to missing"] == 4
    assert "| Step |" in log.as_markdown()


def test_clean_handles_repeat_submission():
    df = pd.DataFrame({
        "student_id": ["S1", "S1"], "major": ["CS", "cs"], "year": [1, 1],
        "part_time_job": ["no", "no"], "sleep_hours": [7, 8], "study_hours_week": [10, 10],
        "social_media_hours_day": [2, 2], "coffee_cups_day": [1, 1],
        "attendance_pct": [90, 90], "exam_score": [70, 71],
    })
    out, log = clean(df)
    assert len(out) == 1
    assert dict(log.steps)["repeat submissions (same student_id) removed"] == 1
