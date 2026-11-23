from __future__ import annotations

import json
from datetime import date

import pytest

from meterdm import db, warehouse


@pytest.fixture(scope="module")
def gold(warehouse_project):
    connection = db.read_only(warehouse_project)
    yield warehouse_project, connection
    connection.close()


def test_every_table_is_populated(gold):
    _, connection = gold
    for table in warehouse.TABLES:
        count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert count > 0, table


def test_quality_checks_all_pass(gold):
    config, connection = gold
    failures = connection.execute("SELECT * FROM fact_quality WHERE NOT passed").fetchall()
    assert failures == []
    report = json.loads((config.paths.reports_dir / "gold_summary.json").read_text())
    assert report["quality_failures"] == 0
    assert report["quality_checks"] >= 10


def test_expected_slots_track_daylight_saving(gold):
    _, connection = gold
    rows = dict(
        connection.execute(
            """
            SELECT local_date, expected_slots FROM dim_date
            WHERE fleet = 'london' AND local_date IN (DATE '2012-03-25', DATE '2012-10-28', DATE '2012-06-01')
            """
        ).fetchall()
    )
    assert rows[date(2012, 3, 25)] == 46
    assert rows[date(2012, 10, 28)] == 50
    assert rows[date(2012, 6, 1)] == 48


def test_daily_totals_match_the_readings(gold):
    _, connection = gold
    mismatch = connection.execute(
        """
        SELECT count(*) FROM (
            SELECT s.fleet, s.meter_id, s.local_date, sum(s.kwh) FILTER (WHERE s.status <> 'missing') AS total
            FROM fact_interval_sample s GROUP BY 1, 2, 3
        ) a JOIN fact_daily_read d USING (fleet, meter_id, local_date)
        WHERE abs(coalesce(a.total, 0) - coalesce(d.kwh, 0)) > 1e-9
        """
    ).fetchone()[0]
    assert mismatch == 0


def test_every_meter_has_one_transformer(gold):
    _, connection = gold
    spread = connection.execute(
        """
        SELECT count(*) FROM (
            SELECT meter_id, count(DISTINCT transformer_id) AS groups FROM dim_meter GROUP BY 1
        ) WHERE groups <> 1
        """
    ).fetchone()[0]
    largest = connection.execute(
        "SELECT max(members) FROM (SELECT transformer_id, count(*) AS members FROM dim_meter GROUP BY 1)"
    ).fetchone()[0]
    config, _ = gold
    assert spread == 0
    assert largest <= config.theft.transformer_size


def test_weather_covers_every_day(gold):
    _, connection = gold
    gaps = connection.execute(
        """
        SELECT count(*) FROM (SELECT DISTINCT fleet, local_date FROM fact_daily_read) d
        LEFT JOIN fact_weather_day w USING (fleet, local_date) WHERE w.local_date IS NULL
        """
    ).fetchone()[0]
    assert gaps == 0


def test_trial_window_matches_the_tariff_slots(gold):
    config, connection = gold
    report = json.loads((config.paths.reports_dir / "gold_summary.json").read_text())
    slots = report["trial"]["event_slots"]
    assert slots
    modal = connection.execute(
        "SELECT window_slots, count(*) AS n FROM fact_trial_window GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    ).fetchone()
    assert modal[0] == len(slots)
    labels = connection.execute("SELECT DISTINCT label FROM dim_tariff ORDER BY 1").fetchall()
    assert {row[0] for row in labels} == {"High", "Low", "Normal"}


def test_holiday_flags_come_from_reference_data(gold):
    _, connection = gold
    flagged = connection.execute(
        "SELECT is_holiday FROM dim_date WHERE fleet = 'london' AND local_date = DATE '2012-12-25'"
    ).fetchone()[0]
    ordinary = connection.execute(
        "SELECT is_holiday FROM dim_date WHERE fleet = 'london' AND local_date = DATE '2012-12-27'"
    ).fetchone()[0]
    assert flagged is True
    assert ordinary is False
