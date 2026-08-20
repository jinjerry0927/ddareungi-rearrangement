from pathlib import Path

from ddareungi_rearrangement.doctor import format_report, inspect_environment


def test_environment_report_accepts_scaffold(tmp_path: Path, monkeypatch) -> None:
    for relative_path in (
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/sample",
        "docs",
        "notebooks",
        "reports",
        "scripts",
        "tests",
    ):
        (tmp_path / relative_path).mkdir(parents=True)

    monkeypatch.setenv("SEOUL_OPEN_DATA_API_KEY", "secret-value")
    report = inspect_environment(tmp_path)

    assert report.ready
    assert report.api_key_configured
    assert report.missing_directories == ()


def test_formatted_report_never_exposes_api_key(tmp_path: Path, monkeypatch) -> None:
    secret = "never-print-this-key"
    monkeypatch.setenv("SEOUL_OPEN_DATA_API_KEY", secret)

    output = format_report(inspect_environment(tmp_path))

    assert secret not in output
    assert "CONFIGURED" in output
