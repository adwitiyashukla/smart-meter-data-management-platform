from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest
from conftest import apply_tiny, write_config

from meterdm import bronze, db, synth, vee
from meterdm.config import load_config

METER = "MAC000002"
START = date(2012, 10, 18)
DAYS = 70


def base_profile(period: int, index: int) -> float:
    shape = 0.34 + 0.22 * math.sin(2.0 * math.pi * (period - 6) / 48.0)
    jitter = 0.013 * ((index * 37) % 19)
    return round(shape + jitter, 3)


def build_series(zone="Europe/London"):
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(zone)
    rows = []
    cursor = datetime(START.year, START.month, START.day, tzinfo=tz)
    end = cursor + timedelta(days=DAYS)
    period = 0
    day = cursor.date()
    while cursor < end:
        cursor = cursor + timedelta(minutes=30)
        local = cursor.astimezone(tz)
        stamp = local.replace(tzinfo=None)
        if (cursor - timedelta(minutes=30)).astimezone(tz).date() != day:
            day = (cursor - timedelta(minutes=30)).astimezone(tz).date()
            period = 0
        period += 1
        rows.append([stamp.strftime("%Y-%m-%d %H:%M:%S") + ".0000000", base_profile(period, len(rows))])
    return rows


def write_raw(config, rows):
    folder = config.paths.raw_dir / "london" / "halfhourly"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["LCLid,stdorToU,DateTime,KWH/hh (per half hour) "]
    for stamp, value in rows:
        text = "Null" if value is None else f"{value}"
        lines.append(f"{METER},Std,{stamp},{text}")
    (folder / "block_0.csv").write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture
def crafted(tmp_path):
    def mutate(raw):
        apply_tiny(raw)
        raw["fleet"]["portugal"]["enabled"] = False
        raw["fleet"]["london"]["meter_batches"] = 1
        raw["vee"]["min_history_days"] = 1
        raw["vee"]["day_level"]["min_like_days"] = 2

    config = load_config(write_config(tmp_path / "project", mutate))
    rows = build_series()
    marks = {}
    marks["negative"] = 100
    rows[100][1] = -0.5
    marks["out_of_range"] = 150
    rows[150][1] = 30.0
    marks["spike"] = 200
    rows[200][1] = 20.0
    marks["stuck"] = (300, 330)
    for index in range(300, 330):
        rows[index][1] = 0.777
    marks["zero_run"] = (400, 460)
    for index in range(400, 460):
        rows[index][1] = 0.0
    marks["null_gap"] = (500, 503)
    for index in range(500, 503):
        rows[index][1] = None
    marks["long_null"] = (600, 1000)
    for index in range(600, 1000):
        rows[index][1] = None
    marks["omitted"] = (1100, 1105)
    kept = [row for index, row in enumerate(rows) if not 1100 <= index < 1105]
    write_raw(config, kept)
    connection = db.connect(config)
    bronze.build_london_bronze(connection, config)
    connection.close()
    vee.build_silver(config)
    return config, marks, rows


def silver(config):
    connection = db.connect(config)
    pattern = (config.paths.silver_dir / "london" / "**" / "readings.parquet").as_posix()
    rows = connection.execute(
        f"""
        SELECT ts_local, ts_utc, kwh, kwh_raw, status, rule_id, method, generated, period, local_date
        FROM read_parquet({db.quote(pattern)}) ORDER BY ts_utc
        """
    ).fetchall()
    connection.close()
    return rows


def test_daylight_saving_repeats_map_to_distinct_instants(crafted):
    config, _, _ = crafted
    connection = db.connect(config)
    pattern = (config.paths.silver_dir / "london" / "**" / "readings.parquet").as_posix()
    total, distinct = connection.execute(
        f"SELECT count(*), count(DISTINCT ts_utc) FROM read_parquet({db.quote(pattern)})"
    ).fetchone()
    long_day = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT ts_utc), max(period)
        FROM read_parquet({db.quote(pattern)}) WHERE local_date = DATE '2012-10-28'
        """
    ).fetchone()
    connection.close()
    assert total == distinct
    assert long_day[0] == 50
    assert long_day[1] == 50
    assert long_day[2] == 50


def test_value_rules_fire_on_the_crafted_defects(crafted):
    config, marks, _ = crafted
    rows = silver(config)
    by_rule = {}
    for row in rows:
        if row[5]:
            by_rule.setdefault(row[5], []).append(row)
    assert len(by_rule[vee.RULE_NEGATIVE]) == 1
    assert len(by_rule[vee.RULE_OUT_OF_RANGE]) == 1
    assert len(by_rule[vee.RULE_SPIKE]) == 1
    assert len(by_rule[vee.RULE_STUCK]) == 30
    assert len(by_rule[vee.RULE_ZERO_RUN]) == 60
    assert by_rule[vee.RULE_NEGATIVE][0][3] == pytest.approx(-0.5)
    assert by_rule[vee.RULE_OUT_OF_RANGE][0][3] == pytest.approx(30.0)


def test_edited_values_are_replaced_and_non_negative(crafted):
    config, _, _ = crafted
    rows = silver(config)
    edited = [row for row in rows if row[4] == vee.STATUS_EDITED]
    assert edited
    for row in edited:
        assert row[2] is not None
        assert row[2] >= 0.0
        assert row[2] != row[3]
        assert row[6] in (vee.METHOD_INTERPOLATE, vee.METHOD_LIKE_DAY)


def test_short_null_gap_is_interpolated_linearly(crafted):
    config, _, original = crafted
    rows = silver(config)
    stamps = {str(row[0]): row for row in rows}
    before = original[499]
    after = original[503]
    filled = [stamps[str(datetime.fromisoformat(original[index][0][:19]))] for index in range(500, 503)]
    assert all(row[4] == vee.STATUS_ESTIMATED for row in filled)
    assert all(row[6] == vee.METHOD_INTERPOLATE for row in filled)
    step = (after[1] - before[1]) / 4.0
    for offset, row in enumerate(filled, start=1):
        assert row[2] == pytest.approx(before[1] + step * offset, abs=1e-9)


def test_omitted_rows_are_generated_and_filled(crafted):
    config, _, _ = crafted
    rows = silver(config)
    generated = [row for row in rows if row[7]]
    on_long_day = [row for row in generated if row[9] == date(2012, 10, 28)]
    assert len(generated) == 7
    assert len(on_long_day) == 2
    assert all(row[4] == vee.STATUS_ESTIMATED for row in generated)
    assert all(row[3] is None for row in generated)


def test_long_null_run_stays_missing(crafted):
    config, _, _ = crafted
    rows = silver(config)
    missing = [row for row in rows if row[4] == vee.STATUS_MISSING]
    assert len(missing) == 400
    assert all(row[2] is None for row in missing)
    assert all(row[5] == vee.RULE_MISSING for row in missing)


def test_every_change_has_an_audit_row(crafted):
    config, _, _ = crafted
    connection = db.connect(config)
    readings = (config.paths.silver_dir / "london" / "**" / "readings.parquet").as_posix()
    edits = (config.paths.silver_dir / "london" / "**" / "edits.parquet").as_posix()
    changed = connection.execute(
        f"SELECT count(*) FROM read_parquet({db.quote(readings)}) WHERE status <> '{vee.STATUS_ACTUAL}'"
    ).fetchone()[0]
    audited = connection.execute(f"SELECT count(*) FROM read_parquet({db.quote(edits)})").fetchone()[0]
    orphan = connection.execute(
        f"""
        SELECT count(*) FROM read_parquet({db.quote(readings)}) r
        WHERE r.status <> '{vee.STATUS_ACTUAL}'
          AND NOT EXISTS (
            SELECT 1 FROM read_parquet({db.quote(edits)}) e
            WHERE e.meter_id = r.meter_id AND e.ts_utc = r.ts_utc
          )
        """
    ).fetchone()[0]
    connection.close()
    assert changed > 0
    assert audited >= changed
    assert orphan == 0


def test_duplicate_reading_is_rejected(tmp_path):
    def mutate(raw):
        apply_tiny(raw)
        raw["fleet"]["portugal"]["enabled"] = False
        raw["fleet"]["london"]["meter_batches"] = 1

    config = load_config(write_config(tmp_path / "project", mutate))
    rows = build_series()
    rows.insert(50, list(rows[49]))
    write_raw(config, rows)
    connection = db.connect(config)
    bronze.build_london_bronze(connection, config)
    connection.close()
    summary = vee.build_silver(config)
    assert summary["london"]["duplicates"] == 1


def test_portugal_nonexistent_local_times_are_rejected(tmp_path):
    def mutate(raw):
        apply_tiny(raw)
        raw["fleet"]["london"]["enabled"] = False

    config = load_config(write_config(tmp_path / "project", mutate))
    synth.build_replica(config)
    bronze.build_bronze(config)
    summary = vee.build_silver(config)
    assert summary["portugal"]["nonexistent"] == 4 * config.synthetic.portugal_meters
    assert summary["portugal"]["duplicates"] == 0


def test_estimation_error_is_measured_against_the_register(tmp_path):
    config = load_config(write_config(tmp_path / "project", apply_tiny))
    synth.build_replica(config)
    bronze.build_bronze(config)
    summary = vee.build_silver(config)
    london = summary["london"]
    assert london["register_checked"] is True
    assert london["estimated_days_checked"] > 50
    assert 0.0 < london["estimation_mean_relative_error"] < 0.25
    assert london["register_flags"] < london["estimated_days_checked"]


def test_injected_defects_are_recovered(tmp_path):
    config = load_config(write_config(tmp_path / "project", apply_tiny))
    synth.build_replica(config)
    bronze.build_bronze(config)
    summary = vee.build_silver(config)
    truth = {}
    lines = (config.paths.truth_dir / "london_defects.csv").read_text().splitlines()[1:]
    for line in lines:
        if not line:
            continue
        _, kind, _, intervals = line.split(",")
        truth[kind] = truth.get(kind, 0) + int(intervals)
    rules = summary["london"]["rules"]
    blank = truth["missing"] + truth["gap"]
    assert rules[vee.RULE_MISSING] == pytest.approx(blank, rel=0.03)
    assert rules[vee.RULE_SPIKE] >= 0.6 * truth["spike"]
    assert rules[vee.RULE_SPIKE] <= 1.4 * truth["spike"]
    if truth.get("zero_run"):
        assert rules[vee.RULE_ZERO_RUN] >= 0.6 * truth["zero_run"]
    if truth.get("stuck"):
        assert rules[vee.RULE_STUCK] >= 0.6 * truth["stuck"]
