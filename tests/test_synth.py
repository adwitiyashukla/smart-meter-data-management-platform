from __future__ import annotations

import hashlib
import json
from datetime import date

import duckdb
import numpy as np
import pytest
from conftest import apply_tiny, write_config

from meterdm import synth
from meterdm import timegrid as tg
from meterdm.config import load_config

INTERVAL_COLUMN = "KWH/hh (per half hour)"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("replica")
    config = load_config(write_config(root / "project", apply_tiny))
    summary = synth.build_replica(config)
    return config, summary


def _connect():
    connection = duckdb.connect()
    connection.execute("SET threads=2")
    return connection


def test_archetypes_are_normalised():
    weekday, weekend, heating = synth.household_archetypes(48)
    assert weekday.shape == (6, 48)
    assert np.all(weekday >= 0.0) and np.all(weekend >= 0.0)
    assert np.allclose(weekday.sum(axis=1), 1.0)
    assert np.allclose(weekend.sum(axis=1), 1.0)
    assert heating.sum() == pytest.approx(1.0)
    commercial = synth.commercial_archetypes(96)
    assert commercial.shape == (4, 96)
    assert np.allclose(commercial.sum(axis=1), 1.0)


def test_weather_matches_requested_shape(built):
    config, _ = built
    payload = json.loads((config.paths.raw_dir / "weather" / "london.json").read_text())
    temps = np.array(payload["hourly"]["temperature_2m"])
    shape = config.synthetic.weather["london"]
    assert payload["timezone"] == "UTC"
    assert len(payload["hourly"]["time"]) == temps.size
    assert temps.mean() == pytest.approx(shape.mean_c, abs=1.5)
    assert temps.max() > shape.mean_c + shape.seasonal_amplitude_c
    assert temps.min() < shape.mean_c - shape.seasonal_amplitude_c


def test_replica_is_reproducible(tmp_path):
    digests = []
    for run in ("first", "second"):
        config = load_config(write_config(tmp_path / run, apply_tiny))
        synth.build_replica(config)
        digest = hashlib.sha256()
        for path in sorted(config.paths.raw_dir.rglob("*")):
            if path.is_file():
                digest.update(path.name.encode("ascii"))
                digest.update(path.read_bytes())
        digests.append(digest.hexdigest())
    assert digests[0] == digests[1]


def test_london_days_have_dst_aware_slot_counts(built):
    config, _ = built
    pattern = str(config.paths.raw_dir / "london" / "halfhourly" / "*.csv")
    rows = _connect().execute(
        f"""
        SELECT local_day, max(slots) AS widest FROM (
          SELECT CAST(strptime(substr("DateTime", 1, 19), '%Y-%m-%d %H:%M:%S')
                      - INTERVAL 30 MINUTE AS DATE) AS local_day,
                 "LCLid" AS meter, count(*) AS slots
          FROM read_csv('{pattern}', header=true, all_varchar=true)
          GROUP BY 1, 2
        ) GROUP BY 1
        """
    ).fetchall()
    widest = dict(rows)
    assert widest[date(2012, 3, 25)] == 46
    assert widest[date(2012, 10, 28)] == 50
    assert widest[date(2012, 6, 1)] == 48
    for local_day, slots in rows:
        assert slots == tg.expected_intervals(local_day, "Europe/London", 30)


def test_daily_register_matches_interval_sums(built):
    config, _ = built
    pattern = str(config.paths.raw_dir / "london" / "halfhourly" / "*.csv")
    daily = str(config.paths.raw_dir / "london" / "daily_dataset.csv")
    row = _connect().execute(
        f"""
        WITH interval_side AS (
          SELECT "LCLid" AS meter,
                 CAST(strptime(substr("DateTime", 1, 19), '%Y-%m-%d %H:%M:%S') - INTERVAL 30 MINUTE AS DATE) AS day,
                 count(*) AS slots,
                 count("{INTERVAL_COLUMN}") FILTER (WHERE "{INTERVAL_COLUMN}" <> 'Null') AS present,
                 sum(TRY_CAST("{INTERVAL_COLUMN}" AS DOUBLE)) AS total
          FROM read_csv('{pattern}', header=true, all_varchar=true)
          GROUP BY 1, 2
        ), register AS (
          SELECT "LCLid" AS meter, CAST("day" AS DATE) AS day, energy_sum, energy_count
          FROM read_csv('{daily}', header=true)
        )
        SELECT count(*),
               sum(CASE WHEN slots <= energy_count THEN 1 ELSE 0 END),
               sum(CASE WHEN present = energy_count AND abs(total - energy_sum) < 1e-9 THEN 1 ELSE 0 END),
               sum(CASE WHEN present = energy_count THEN 1 ELSE 0 END)
        FROM interval_side JOIN register USING (meter, day)
        """
    ).fetchone()
    total_days, never_wider, exact_match, complete = row
    assert total_days > 1000
    assert never_wider == total_days
    assert complete > total_days * 0.5
    assert exact_match >= complete * 0.99


def test_tariff_labels_stay_inside_the_trial(built):
    config, _ = built
    path = str(config.paths.raw_dir / "london" / "tariffs.csv")
    rows = _connect().execute(
        f"""
        SELECT "Tariff",
               min(CAST(strptime(substr("TariffDateTime", 1, 19), '%Y-%m-%d %H:%M:%S') - INTERVAL 30 MINUTE AS DATE)),
               max(CAST(strptime(substr("TariffDateTime", 1, 19), '%Y-%m-%d %H:%M:%S') - INTERVAL 30 MINUTE AS DATE)),
               count(*)
        FROM read_csv('{path}', header=true, all_varchar=true)
        GROUP BY 1
        """
    ).fetchall()
    labels = {row[0]: row for row in rows}
    assert set(labels) == {"Normal", "High", "Low"}
    for label, first, last, count in rows:
        assert first >= config.synthetic.tou.trial_start, label
        assert last <= config.synthetic.tou.trial_end, label
        assert count > 0


def test_portugal_reproduces_the_documented_dst_quirk(built):
    config, summary = built
    path = config.paths.raw_dir / "portugal" / config.portugal_source.member
    lines = path.read_text().splitlines()
    header = lines[0].split(";")
    assert header[0] == '""'
    assert header[1] == '"MT_001"'
    assert len(header) == config.synthetic.portugal_meters + 1
    spring = [line for line in lines if line.startswith('"2012-03-25 01:15:00"')][0]
    autumn_before = [line for line in lines if line.startswith('"2012-10-28 00:45:00"')][0]
    autumn = [line for line in lines if line.startswith('"2012-10-28 01:15:00"')][0]
    assert set(spring.split(";")[1:]) == {"0"}
    before = float(autumn_before.split(";")[1].replace(",", "."))
    during = float(autumn.split(";")[1].replace(",", "."))
    assert during > before
    assert summary["fleets"]["portugal"]["spring_zero_slots"] == 4
    assert summary["fleets"]["portugal"]["autumn_doubled_slots"] == 4
    assert len(lines) == summary["fleets"]["portugal"]["slots"] + 1


def test_truth_files_cover_every_meter(built):
    config, summary = built
    meters = (config.paths.truth_dir / "london_meters.csv").read_text().splitlines()
    clients = (config.paths.truth_dir / "portugal_meters.csv").read_text().splitlines()
    assert len(meters) == summary["fleets"]["london"]["meters"] + 1
    assert len(clients) == summary["fleets"]["portugal"]["meters"] + 1
    assert meters[0].startswith("LCLid,stdorToU,acorn,acorn_group,transformer")
    defects = (config.paths.truth_dir / "london_defects.csv").read_text().splitlines()
    kinds = {line.split(",")[1] for line in defects[1:] if line}
    assert kinds <= {"missing", "gap", "spike", "stuck", "zero_run", "omitted"}
    assert "omitted" in kinds
    assert "missing" in kinds
