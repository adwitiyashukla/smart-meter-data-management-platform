from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import write_config

from meterdm.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_reference_config_loads():
    config = load_config(ROOT / "config.toml")
    assert config.enabled_fleets == ("london", "portugal")
    assert config.fleet("london").intervals_per_day == 48
    assert config.fleet("portugal").intervals_per_day == 96
    assert config.paths.gold_db.name.endswith(".duckdb")


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_unknown_fleet_lookup():
    config = load_config(ROOT / "config.toml")
    with pytest.raises(ConfigError, match="unknown fleet"):
        config.fleet("mars")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["run"].__setitem__("seed", -1), "must be >= 0"),
        (lambda raw: raw["run"].__setitem__("profile", "turbo"), "must be one of"),
        (lambda raw: raw["run"].pop("seed"), "missing key run.seed"),
        (lambda raw: raw["duckdb"].__setitem__("threads", 0), "must be >= 1"),
        (lambda raw: raw["duckdb"].__setitem__("memory_limit_gb", "big"), "must be a number"),
        (lambda raw: raw["duckdb"].__setitem__("preserve_insertion_order", "no"), "must be true or false"),
        (lambda raw: raw["fleet"]["london"].__setitem__("timezone", "Mars/Olympus"), "not a known timezone"),
        (lambda raw: raw["fleet"]["london"].__setitem__("interval_minutes", 7), "must divide 1440"),
        (lambda raw: raw["fleet"]["london"].__setitem__("latitude", 120.0), "must be <= 90"),
        (
            lambda raw: [
                raw["fleet"]["london"].__setitem__("enabled", False),
                raw["fleet"]["portugal"].__setitem__("enabled", False),
            ],
            "at least one fleet must be enabled",
        ),
        (lambda raw: raw["source"]["london"].__setitem__("archive_url", "http://example.org"), "must be an https url"),
        (lambda raw: raw["source"]["weather"].__setitem__("start_date", "2020-13-01"), "must be an ISO date"),
        (lambda raw: raw["source"]["weather"].__setitem__("start_date", "2030-01-01"), "before end_date"),
        (lambda raw: raw["synthetic"].__setitem__("end_date", "2010-01-01"), "before end_date"),
        (lambda raw: raw["synthetic"].__setitem__("min_span_days", 9000), "exceed the"),
        (lambda raw: raw["synthetic"].__setitem__("heating_share", 0.99), "stay below 1.0"),
        (lambda raw: raw["synthetic"].__setitem__("acorn_groups", ["Only"]), "at least 2 entries"),
        (
            lambda raw: [
                raw["changepoint"].__setitem__("grid_quantile_low", 0.5),
                raw["changepoint"].__setitem__("grid_quantile_high", 0.5),
            ],
            "below grid_quantile_high",
        ),
        (lambda raw: raw["changepoint"].__setitem__("refine_step_c", 4.0), "must not exceed grid_step_c"),
        (lambda raw: raw["cluster"].__setitem__("k_min", 20), "must not exceed cluster.k_max"),
        (lambda raw: raw["theft"].__setitem__("night_end_hour", 1), "before night_end_hour"),
        (lambda raw: raw["theft"].__setitem__("tamper_fraction", 0.9), "must be <= 0.5"),
        (lambda raw: raw["theft"].__setitem__("precision_at_k", [0]), "must be >= 1"),
        (lambda raw: raw["forecast"].__setitem__("quantiles", [0.9, 0.1]), "strictly increasing"),
        (lambda raw: raw["forecast"].__setitem__("quantiles", [0.0, 0.5]), "strictly between 0 and 1"),
        (lambda raw: raw["quality"].__setitem__("max_missing_ratio", 2.0), "must be <= 1"),
        (lambda raw: raw["vee"]["spike"].__setitem__("profile_ratio", 0.1), "must be >= 1"),
        (lambda raw: raw["vee"].pop("stuck"), "missing table"),
    ],
)
def test_invalid_values_rejected(tmp_path, mutate, message):
    path = write_config(tmp_path, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_paths_are_created(tmp_path):
    shutil.copy(ROOT / "config.toml", tmp_path / "config.toml")
    config = load_config(tmp_path / "config.toml")
    config.paths.ensure()
    assert config.paths.bronze_dir.is_dir()
    assert config.paths.gold_db.parent.is_dir()
    assert config.paths.root == tmp_path.resolve()
