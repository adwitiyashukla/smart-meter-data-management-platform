from __future__ import annotations

import numpy as np
import pytest

from meterdm import db, theft


@pytest.fixture(scope="module")
def scored(analytics_project):
    config, reports = analytics_project
    return config, reports["theft"]


@pytest.fixture
def series():
    generator = np.random.default_rng(21)
    day = 12.0 + generator.normal(0.0, 1.0, size=400)
    night = day * 0.25
    return day, night


def test_constant_bypass_scales_both_series(series, reference_config):
    day, night = series
    tampered_day, tampered_night = theft.apply_tamper(day, night, theft.TAMPER_CONSTANT, 300, reference_config)
    retain = reference_config.theft.tamper.constant_bypass_retain
    assert np.allclose(tampered_day[:300], day[:300])
    assert np.allclose(tampered_day[300:], day[300:] * retain)
    assert np.allclose(tampered_night[300:], night[300:] * retain)


def test_night_bypass_only_removes_night_energy(series, reference_config):
    day, night = series
    tampered_day, tampered_night = theft.apply_tamper(day, night, theft.TAMPER_NIGHT, 300, reference_config)
    retain = reference_config.theft.tamper.night_bypass_retain
    assert np.allclose(tampered_night[300:], night[300:] * retain)
    removed = night[300:] * (1.0 - retain)
    assert np.allclose(tampered_day[300:], day[300:] - removed)


def test_freeze_holds_a_constant_reading(series, reference_config):
    day, night = series
    tampered_day, _ = theft.apply_tamper(day, night, theft.TAMPER_FREEZE, 300, reference_config)
    assert len(set(np.round(tampered_day[300:], 9))) == 1
    assert tampered_day[300] < day[300:].mean()


def test_drift_ramps_down_to_the_final_retention(series, reference_config):
    day, night = series
    tampered_day, _ = theft.apply_tamper(day, night, theft.TAMPER_DRIFT, 300, reference_config)
    retain = reference_config.theft.tamper.drift_final_retain
    assert tampered_day[300] == pytest.approx(day[300], rel=1e-9)
    assert tampered_day[-1] == pytest.approx(day[-1] * retain, rel=1e-9)
    ratio = tampered_day[300:] / day[300:]
    assert np.all(np.diff(ratio) < 1e-9)


def test_no_tamper_leaves_the_series_alone(series, reference_config):
    day, night = series
    tampered_day, tampered_night = theft.apply_tamper(day, night, theft.TAMPER_NONE, -1, reference_config)
    assert np.allclose(tampered_day, day)
    assert np.allclose(tampered_night, night)


def test_cusum_ignores_noise_and_catches_a_drop():
    generator = np.random.default_rng(3)
    quiet = generator.normal(0.0, 1.0, size=200)
    dropped = np.concatenate([generator.normal(0.0, 1.0, size=100), generator.normal(-1.5, 1.0, size=100)])
    assert theft.cusum(quiet, 0.5, 5.0) > -2.0
    assert theft.cusum(dropped, 0.5, 5.0) < theft.cusum(quiet, 0.5, 5.0)


def test_tamper_plan_respects_the_event_window(reference_config):
    lengths = np.array([800, 700, 400, 250, 100])
    kinds, starts = theft.plan_tampering(lengths, reference_config, 7)
    settings = reference_config.theft
    for index, length in enumerate(lengths):
        if kinds[index] == theft.TAMPER_NONE:
            continue
        assert starts[index] >= length - settings.test_days
        assert length - starts[index] >= settings.tamper.min_event_days
        assert kinds[index] in theft.TAMPER_TYPES


def test_scoring_beats_chance_on_injected_tampering(scored):
    _, report = scored
    london = report["london"]
    assert london["tampered"] >= 5
    assert london["holdout_tampered"] >= 2
    assert london["roc_auc"] > 0.75
    assert london["roc_auc"] > london["baseline_roc_auc"]
    assert london["average_precision"] > london["tampered"] / london["meters_scored"]
    assert set(london["coefficients"]) == set(theft.FEATURES)


def test_variance_features_separate_a_frozen_meter(series, reference_config):
    day, night = series
    frozen, _ = theft.apply_tamper(day, night, theft.TAMPER_FREEZE, 300, reference_config)
    reference = slice(0, 300)
    test = slice(300, day.size)
    temps = np.linspace(2.0, 18.0, day.size)
    grid = np.array([10.0, 12.0, 14.0])
    clean = theft.meter_features(day, night, temps, reference, test, reference_config, grid)
    tampered = theft.meter_features(frozen, night, temps, reference, test, reference_config, grid)
    assert tampered["variation_ratio"] < 0.05
    assert clean["variation_ratio"] > 0.5
    assert tampered["flat_days_share"] > 0.9
    assert clean["flat_days_share"] < 0.1


def test_scores_are_stored_with_features(scored):
    config, report = scored
    connection = db.read_only(config)
    rows, tampered, distinct = connection.execute(
        """
        SELECT count(*), count(*) FILTER (WHERE label = 1), count(DISTINCT kind)
        FROM fact_theft_score WHERE fleet = 'london'
        """
    ).fetchone()
    connection.close()
    assert rows == report["london"]["meters_scored"]
    assert tampered == report["london"]["tampered"]
    assert distinct >= 2
