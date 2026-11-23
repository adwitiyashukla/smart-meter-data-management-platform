from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from meterdm.config import load_config

ROOT = Path(__file__).resolve().parents[1]

TINY = {
    ("synthetic", "london_meters"): 12,
    ("synthetic", "portugal_meters"): 5,
    ("synthetic", "start_date"): "2012-01-01",
    ("synthetic", "end_date"): "2012-12-31",
    ("synthetic", "enrollment_spread_days"): 40,
    ("synthetic", "min_span_days"): 30,
    ("synthetic", "portugal", "creation_spread_days"): 40,
    ("synthetic", "tou", "trial_start"): "2012-02-01",
    ("synthetic", "tou", "trial_end"): "2012-11-30",
    ("source", "london", "blocks"): 3,
    ("theft", "transformer_size"): 4,
    ("download", "backoff_seconds"): 0.05,
    ("quality", "min_meters"): 2,
    ("quality", "min_days"): 30,
}


def render(item):
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, str):
        return f'"{item}"'
    if isinstance(item, list):
        return "[{}]".format(", ".join(render(entry) for entry in item))
    return repr(item)


def dump_toml(value, prefix=""):
    lines = []
    scalars = {key: item for key, item in value.items() if not isinstance(item, dict)}
    tables = {key: item for key, item in value.items() if isinstance(item, dict)}
    if scalars and prefix:
        lines.append(f"[{prefix}]")
    for key, item in scalars.items():
        lines.append(f"{key} = {render(item)}")
    if scalars:
        lines.append("")
    for key, item in tables.items():
        lines.append(dump_toml(item, key if not prefix else prefix + "." + key))
    return "\n".join(lines)


def write_config(target_dir: Path, mutate=None) -> Path:
    with (ROOT / "config.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    if mutate is not None:
        mutate(raw)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "config.toml"
    path.write_text(dump_toml(raw), encoding="ascii")
    if not (target_dir / "refdata").exists():
        shutil.copytree(ROOT / "refdata", target_dir / "refdata")
    return path


def apply_tiny(raw):
    for keys, value in TINY.items():
        table = raw
        for key in keys[:-1]:
            table = table[key]
        table[keys[-1]] = value


MEDIUM = {
    ("synthetic", "london_meters"): 240,
    ("synthetic", "portugal_meters"): 24,
    ("synthetic", "start_date"): "2011-11-01",
    ("synthetic", "end_date"): "2013-12-31",
    ("synthetic", "enrollment_spread_days"): 200,
    ("synthetic", "min_span_days"): 180,
    ("synthetic", "portugal", "creation_spread_days"): 120,
    ("synthetic", "tou", "trial_start"): "2013-01-01",
    ("synthetic", "tou", "trial_end"): "2013-11-30",
    ("source", "london", "blocks"): 6,
    ("fleet", "london", "meter_batches"): 6,
    ("fleet", "portugal", "meter_batches"): 2,
    ("theft", "transformer_size"): 20,
    ("quality", "min_meters"): 10,
    ("quality", "min_days"): 200,
    ("dtou", "min_meters_per_group"): 20,
    ("dtou", "placebo_draws"): 40,
    ("gold", "interval_sample_meters"): 40,
    ("forecast", "backtest_origins"): 3,
    ("forecast", "max_iter"): 60,
    ("forecast", "train_days"): 300,
    ("cluster", "restarts"): 4,
    ("cluster", "k_max"): 6,
    ("holdout", "batches"): 3,
    ("holdout", "gap_events"): 8,
}


def apply_medium(raw):
    for keys, value in MEDIUM.items():
        table = raw
        for key in keys[:-1]:
            table = table[key]
        table[keys[-1]] = value


@pytest.fixture(scope="session")
def warehouse_project(tmp_path_factory):
    from meterdm import bronze, synth, vee, warehouse

    config = load_config(write_config(tmp_path_factory.mktemp("warehouse"), apply_medium))
    synth.build_replica(config)
    bronze.build_bronze(config)
    vee.build_silver(config)
    warehouse.build_gold(config)
    return config


@pytest.fixture(scope="session")
def analytics_project(warehouse_project):
    from meterdm import changepoint, cluster, dtou, forecast, theft

    reports = {
        "changepoint": changepoint.build_changepoints(warehouse_project),
        "cluster": cluster.build_clusters(warehouse_project),
        "theft": theft.build_theft(warehouse_project),
        "dtou": dtou.build_dtou(warehouse_project),
        "forecast": forecast.build_forecast(warehouse_project),
    }
    return warehouse_project, reports


@pytest.fixture
def tiny_config(tmp_path):
    return load_config(write_config(tmp_path / "project", apply_tiny))


@pytest.fixture
def reference_config():
    return load_config(ROOT / "config.toml")
