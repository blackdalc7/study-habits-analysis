from study_habits import charts
from study_habits.report import main


def test_all_charts_render(cleaned, tmp_path):
    paths = charts.save_all(cleaned, tmp_path / "charts")
    assert len(paths) == len(charts.ALL_CHARTS)
    assert all(p.stat().st_size > 5000 for p in paths)


def test_main_writes_everything(tmp_path, capsys):
    data, docs = tmp_path / "data", tmp_path / "docs"
    assert main(["--students", "150", "--seed", "3", "--data-dir", str(data),
                 "--docs-dir", str(docs)]) == 0
    assert (data / "survey_raw.csv").exists()
    assert (data / "survey_clean.csv").exists()
    report = (docs / "REPORT.md").read_text()
    assert "## Linear model" in report
    assert "R²" in report
    assert "clean rows   : 150" in capsys.readouterr().out
