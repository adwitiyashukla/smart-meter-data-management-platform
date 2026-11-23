from __future__ import annotations

import numpy as np
import pytest

from meterdm import db, holdout


@pytest.fixture(scope="module")
def scored(warehouse_project):
    return warehouse_project, holdout.build_holdout(warehouse_project)


def test_planned_gaps_never_touch_or_overrun(reference_config):
    rng = np.random.default_rng(5)
    mask, runs = holdout.plan_gaps(4000, [1, 4, 24, 96], 12, rng)
    assert runs
    for start, length in runs:
        assert start >= 1
        assert start + length < 4000
        assert mask[start : start + length].all()
        assert not mask[start - 1]
        assert not mask[start + length]
    assert mask.sum() == sum(length for _, length in runs)


def test_short_gaps_are_interpolated_exactly(reference_config):
    size = 600
    stamps = np.arange(size, dtype=np.int64) * 1800
    values = 2.0 + 0.01 * np.arange(size)
    dows = np.zeros(size, dtype=np.int64)
    periods = (np.arange(size) % 48) + 1
    mask = np.zeros(size, dtype=bool)
    mask[100:103] = True
    estimates, methods = holdout.estimate_masked(
        values, stamps, dows, periods, mask, [(100, 3)], reference_config
    )
    assert list(methods[100:103]) == [holdout.METHOD_INTERPOLATE] * 3
    assert np.allclose(estimates[100:103], values[100:103], atol=1e-9)


def test_long_gaps_fall_back_to_the_like_day_profile(reference_config):
    size = 48 * 40
    stamps = np.arange(size, dtype=np.int64) * 1800
    periods = (np.arange(size) % 48) + 1
    dows = (np.arange(size) // 48) % 7
    values = 0.2 + 0.05 * periods
    mask = np.zeros(size, dtype=bool)
    length = reference_config.vee.short_gap_intervals + 20
    mask[500 : 500 + length] = True
    estimates, methods = holdout.estimate_masked(
        values, stamps, dows, periods, mask, [(500, length)], reference_config
    )
    window = slice(500, 500 + length)
    assert set(methods[window]) == {holdout.METHOD_LIKE_DAY}
    assert np.allclose(estimates[window], values[window], atol=1e-9)


def test_estimates_are_never_negative(reference_config):
    size = 400
    stamps = np.arange(size, dtype=np.int64) * 1800
    values = np.zeros(size)
    values[::2] = 0.4
    dows = np.zeros(size, dtype=np.int64)
    periods = (np.arange(size) % 48) + 1
    mask = np.zeros(size, dtype=bool)
    mask[50:52] = True
    estimates, _ = holdout.estimate_masked(values, stamps, dows, periods, mask, [(50, 2)], reference_config)
    assert np.all(estimates[50:52] >= 0.0)


def test_holdout_reports_a_usable_error(scored):
    _, report = scored
    london = report["london"]
    assert london["meters"] > 20
    assert london["readings_held_out"] > 500
    assert london["days_checked"] > 50
    assert 0.0 < london["daily_mean_relative_error"] < 0.5
    assert london["mean_absolute_error_kwh"] < london["mean_truth_kwh"]
    assert london["normalised_mean_absolute_error"] < 1.0


def test_both_estimators_are_exercised(scored):
    _, report = scored
    london = report["london"]
    assert london[f"{holdout.METHOD_INTERPOLATE}_readings"] > 0
    assert london[f"{holdout.METHOD_LIKE_DAY}_readings"] > 0


def test_metrics_are_stored_in_the_warehouse(scored):
    config, report = scored
    connection = db.read_only(config)
    stored = dict(
        connection.execute("SELECT metric, value FROM fact_holdout WHERE fleet = 'london'").fetchall()
    )
    connection.close()
    assert stored["daily_mean_relative_error"] == pytest.approx(report["london"]["daily_mean_relative_error"])
    assert stored["meters"] == report["london"]["meters"]
