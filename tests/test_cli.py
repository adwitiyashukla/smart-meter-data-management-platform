from __future__ import annotations

import json

import pytest
from conftest import apply_tiny, write_config

from meterdm import cli
from meterdm.config import load_config


def test_report_collects_every_stage_summary(analytics_project, capsys):
    config, _ = analytics_project
    path = str(config.paths.root / "config.toml")
    assert cli.main(["--config", path, "--quiet", "report"]) == 0
    target = config.paths.reports_dir / "run_report.json"
    payload = json.loads(target.read_text())
    for stage in ("gold_summary", "changepoint_summary", "theft_summary", "forecast_summary", "vee_summary"):
        assert stage in payload
    capsys.readouterr()


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        cli.run_stage("nonsense", None, None)


def test_probe_reports_failure_for_unreachable_sources(tmp_path, capsys):
    def mutate(raw):
        apply_tiny(raw)
        raw["source"]["london"]["archive_url"] = "https://127.0.0.1:1/london.zip"
        raw["source"]["london"]["mirror_url"] = "https://127.0.0.1:1/mirror.zip"
        raw["source"]["portugal"]["archive_url"] = "https://127.0.0.1:1/portugal.zip"
        raw["source"]["portugal"]["mirror_url"] = "https://127.0.0.1:1/mirror.zip"
        raw["source"]["weather"]["archive_url"] = "https://127.0.0.1:1/weather"
        raw["download"]["timeout_seconds"] = 2.0

    path = write_config(tmp_path / "project", mutate)
    assert cli.main(["--config", str(path), "probe"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {
        "london_archive",
        "london_mirror",
        "portugal_archive",
        "portugal_mirror",
        "weather",
    }
    assert all(entry["ok"] is False for entry in printed.values())


def test_synth_command_builds_the_replica(tmp_path, capsys):
    path = write_config(tmp_path / "project", apply_tiny)
    assert cli.main(["--config", str(path), "--quiet", "synth"]) == 0
    config = load_config(path)
    assert (config.paths.raw_dir / "london" / "halfhourly").is_dir()
    assert (config.paths.truth_dir / "replica_summary.json").is_file()
    capsys.readouterr()
