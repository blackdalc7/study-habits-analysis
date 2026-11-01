import numpy as np
import pandas as pd

from study_habits import analysis


def test_summary_columns(cleaned):
    s = analysis.summary(cleaned)
    assert list(s.columns) == ["mean", "std", "min", "50%", "max"]
    assert "exam_score" in s.index


def test_study_is_the_strongest_correlation(cleaned):
    corr = analysis.correlations(cleaned)["exam_score"].drop("exam_score")
    assert corr.idxmax() == "study_hours_week"
    assert corr["social_media_hours_day"] < 0


def test_sleep_sweet_spot(cleaned):
    assert analysis.best_sleep_bucket(cleaned) in {"7-8", "8-9"}
    table = analysis.score_by_sleep(cleaned)
    assert table.loc["<5", "mean"] < table.loc["7-8", "mean"]
    assert table["count"].sum() == len(cleaned)


def test_coffee_is_an_upside_down_u(cleaned):
    table = analysis.score_by_coffee(cleaned)
    assert table.index[-1] == "5+"
    peak = table["mean"].idxmax()
    assert peak in {"1", "2", "3"}
    assert table.loc["5+", "mean"] < table.loc[peak, "mean"]


def test_score_by_major_sorted(cleaned):
    t = analysis.score_by_major(cleaned)
    assert t["avg_score"].is_monotonic_decreasing
    assert t["students"].sum() == len(cleaned)


def test_job_table(cleaned):
    t = analysis.job_vs_no_job(cleaned)
    assert set(t.index) == {"has a job", "no job"}
    assert t.loc["has a job", "sleep_hours"] < t.loc["no job", "sleep_hours"]


def test_linear_model_recovers_a_known_line():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.random(200), "b": rng.random(200)})
    df["exam_score"] = 10 + 3 * df["a"] - 2 * df["b"]
    fit = analysis.linear_model(df, ["a", "b"])
    assert np.allclose(fit.coefficients, [3, -2])
    assert abs(fit.intercept - 10) < 1e-9
    assert fit.r_squared > 0.999
    assert fit.as_series().to_dict() == {"a": 3.0, "b": -2.0}


def test_linear_model_on_survey(cleaned):
    fit = analysis.linear_model(cleaned)
    coef = fit.as_series()
    assert coef["study_hours_week"] > 0
    assert coef["social_media_hours_day"] < 0
    assert 0.3 < fit.r_squared < 0.9
