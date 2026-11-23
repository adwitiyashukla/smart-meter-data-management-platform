from __future__ import annotations

import json

import duckdb
import pytest

from meterdm import space


@pytest.fixture(scope="module")
def bundle(analytics_project, tmp_path_factory):
    config, _ = analytics_project
    out = tmp_path_factory.mktemp("space") / "bundle"
    return config, space.build_space(config, out), out


def test_bundle_carries_everything_the_app_needs(bundle):
    _, summary, out = bundle
    files = set(summary["files"])
    for name in ("app.py", "config.toml", "Dockerfile", "requirements.txt", "README.md", "meterdm.duckdb"):
        assert name in files, name
    for name in space.PACKAGE_MODULES:
        assert f"meterdm/{name}" in files
    assert (out / "refdata" / "uk_bank_holidays.csv").is_file()


def test_bundle_config_points_at_the_local_database(bundle):
    _, _, out = bundle
    text = (out / "config.toml").read_text()
    assert 'gold_db = "meterdm.duckdb"' in text
    assert "data/gold" not in text


def test_requirements_stay_minimal(bundle):
    _, _, out = bundle
    lines = [line for line in (out / "requirements.txt").read_text().splitlines() if line]
    assert lines == list(space.RUNTIME_REQUIREMENTS)
    assert not any("dagster" in line or "fastapi" in line for line in lines)


def test_space_readme_carries_a_real_emoji(bundle):
    config, _, out = bundle
    header = (out / "README.md").read_text(encoding="utf-8").splitlines()
    line = next(item for item in header if item.startswith("emoji:"))
    assert line == f"emoji: {space.EMOJI_NAMES[config.space.emoji]}"


def test_dockerfile_and_readme_agree_on_the_port(bundle):
    config, _, out = bundle
    docker = (out / "Dockerfile").read_text()
    readme = (out / "README.md").read_text()
    assert f"--server.port={config.space.port}" in docker
    assert f"app_port: {config.space.port}" in readme
    assert "sdk: docker" in readme


def test_database_holds_the_tables_the_app_queries(bundle):
    _, summary, out = bundle
    connection = duckdb.connect((out / "meterdm.duckdb").as_posix(), read_only=True)
    names = {row[0] for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    meters = connection.execute("SELECT count(*) FROM dim_meter").fetchone()[0]
    sampled = connection.execute("SELECT count(DISTINCT meter_id) FROM fact_interval_sample").fetchone()[0]
    connection.close()
    for table in ("dim_meter", "fact_daily_read", "fact_quality", "fact_theft_score", "fact_forecast_metric"):
        assert table in names, table
    assert meters == summary["rows"]["dim_meter"]
    assert sampled > 0


def test_summary_is_written_to_reports(bundle):
    config, summary, _ = bundle
    stored = json.loads((config.paths.reports_dir / "space_summary.json").read_text())
    assert stored["database_bytes"] == summary["database_bytes"]
    assert stored["files"] == summary["files"]


def test_lfs_attributes_track_the_database(bundle):
    _, _, out = bundle
    assert "*.duckdb filter=lfs" in (out / ".gitattributes").read_text()
