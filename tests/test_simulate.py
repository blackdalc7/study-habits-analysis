from study_habits.simulate import MAJORS, simulate


def test_shape_and_columns(tidy):
    assert len(tidy) == 400
    assert {"student_id", "major", "exam_score", "coffee_cups_day"} <= set(tidy.columns)


def test_seeded(tidy):
    assert simulate(400, seed=2026).equals(tidy)
    assert not simulate(400, seed=1).equals(tidy)


def test_values_in_range(tidy):
    assert tidy["exam_score"].between(0, 100).all()
    assert tidy["sleep_hours"].between(3.5, 10).all()
    assert set(tidy["major"]) <= set(MAJORS)


def test_messy_has_problems(messy, tidy):
    assert len(messy) == len(tidy) + 8
    assert messy["sleep_hours"].isna().any()
    assert messy["coffee_cups_day"].astype(str).str.contains("cups").any()
    assert not set(messy["major"]) <= set(MAJORS)
