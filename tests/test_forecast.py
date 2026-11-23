from __future__ import annotations

import numpy as np
import pytest

from meterdm import db, forecast


@pytest.fixture(scope="module")
def backtested(analytics_project):
    config, reports = analytics_project
    return config, reports["forecast"]


def test_pinball_matches_a_manual_calculation():
    actual = np.array([10.0, 8.0])
    predicted = np.array([9.0, 9.0])
    assert forecast.pinball(actual, predicted, 0.9) == pytest.approx((0.9 * 1.0 + 0.1 * 1.0) / 2.0)
    assert forecast.pinball(actual, predicted, 0.1) == pytest.approx((0.1 * 1.0 + 0.9 * 1.0) / 2.0)
    assert forecast.pinball(actual, actual, 0.5) == 0.0


def test_short_history_is_rejected(reference_config):
    import pyarrow as pa

    size = 100
    table = pa.table(
        {
            "ts_utc": pa.array(np.arange(size)),
            "kwh": pa.array(np.ones(size)),
            "meters": pa.array(np.ones(size)),
            "temp_c": pa.array(np.ones(size)),
            "local_hour": pa.array(np.arange(size) % 24),
            "day_of_week": pa.array(np.zeros(size)),
            "is_weekend": pa.array(np.zeros(size)),
            "is_holiday": pa.array(np.zeros(size)),
        }
    )
    with pytest.raises(forecast.ForecastError, match="hours of history"):
        forecast.build_matrix(table, reference_config)


def test_feature_matrix_lags_line_up(reference_config):
    import pyarrow as pa

    size = 1000
    load = np.arange(size, dtype=float)
    table = pa.table(
        {
            "ts_utc": pa.array(np.arange(size)),
            "kwh": pa.array(load),
            "meters": pa.array(np.ones(size)),
            "temp_c": pa.array(np.linspace(0.0, 10.0, size)),
            "local_hour": pa.array(np.arange(size) % 24),
            "day_of_week": pa.array((np.arange(size) // 24) % 7),
            "is_weekend": pa.array(np.zeros(size)),
            "is_holiday": pa.array(np.zeros(size)),
        }
    )
    _, values, _, features = forecast.build_matrix(table, reference_config)
    lag24 = features[:, forecast.FEATURE_NAMES.index("lag_24")]
    lag168 = features[:, forecast.FEATURE_NAMES.index("lag_168")]
    mean168 = features[:, forecast.FEATURE_NAMES.index("mean_168")]
    assert np.isnan(lag24[:24]).all()
    assert lag24[500] == pytest.approx(values[476])
    assert lag168[500] == pytest.approx(values[332])
    assert mean168[500] == pytest.approx(values[332:500].mean())
    assert np.isnan(mean168[:168]).all()


def test_backtest_produces_full_horizons(backtested):
    config, report = backtested
    london = report["london"]
    assert london["origins"] >= 1
    assert london["points"] == london["origins"] * config.forecast.horizon_hours


def test_model_beats_the_seasonal_naive_baseline(backtested):
    _, report = backtested
    for fleet_name, metrics in report.items():
        if metrics.get("skipped"):
            continue
        assert metrics["mape"] < metrics["naive_mape"], fleet_name
        assert metrics["skill_vs_naive"] > 0.0
        assert 0.0 < metrics["mape"] < 0.5


def test_conformal_calibration_improves_coverage(backtested):
    _, report = backtested
    for fleet_name, metrics in report.items():
        if metrics.get("skipped"):
            continue
        nominal = metrics["interval_nominal"]
        assert nominal == pytest.approx(0.8)
        calibrated = abs(metrics["interval_coverage"] - nominal)
        raw = abs(metrics["raw_interval_coverage"] - nominal)
        assert calibrated <= raw + 1e-9, fleet_name
        assert metrics["interval_width"] >= metrics["raw_interval_width"], fleet_name
        assert metrics["conformal_width"] >= 0.0
        assert metrics["interval_coverage"] >= nominal - 0.25, fleet_name
        assert 0.4 < metrics["interval_coverage"] <= 1.0


def test_backtest_rows_are_stored(backtested):
    config, report = backtested
    connection = db.read_only(config)
    rows = connection.execute(
        "SELECT count(*), count(DISTINCT origin), min(horizon), max(horizon) FROM fact_forecast_backtest"
        " WHERE fleet = 'london'"
    ).fetchone()
    metrics = dict(
        connection.execute("SELECT metric, value FROM fact_forecast_metric WHERE fleet = 'london'").fetchall()
    )
    connection.close()
    assert rows[0] == report["london"]["points"]
    assert rows[1] == report["london"]["origins"]
    assert rows[2] == 1
    assert rows[3] == config.forecast.horizon_hours
    assert metrics["mape"] == pytest.approx(report["london"]["mape"])
