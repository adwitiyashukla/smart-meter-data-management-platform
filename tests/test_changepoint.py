from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from conftest import apply_tiny, write_config

from meterdm import changepoint as cp
from meterdm.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config(write_config.__globals__["ROOT"] / "config.toml")


@pytest.fixture
def temps():
    generator = np.random.default_rng(11)
    return np.round(np.sort(generator.uniform(-4.0, 30.0, size=700)), 2)


def grid_for(temps, config):
    return cp.candidate_grid(temps, config)


def brute_force(temps, energy, model, config, step=0.05):
    low = float(np.quantile(temps, config.changepoint.grid_quantile_low))
    high = float(np.quantile(temps, config.changepoint.grid_quantile_high))
    points = np.arange(low, high + step, step)
    best_sse = np.inf
    best_point = None
    for first in points:
        heating = np.maximum(0.0, first - temps)
        if model == cp.MODEL_HEATING:
            design = np.column_stack([np.ones_like(temps), heating])
        else:
            design = np.column_stack([np.ones_like(temps), heating, np.maximum(0.0, temps - first)])
        coefficients = np.linalg.lstsq(design, energy, rcond=None)[0]
        sse = float(((design @ coefficients - energy) ** 2).sum())
        if sse < best_sse:
            best_sse = sse
            best_point = (first,)
    return best_sse, best_point


def test_recovers_a_heating_only_model(config, temps):
    energy = 8.0 + 0.9 * np.maximum(0.0, 15.0 - temps)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    assert fit["model"] == cp.MODEL_HEATING
    assert fit["heat_cp"] == pytest.approx(15.0, abs=config.changepoint.grid_step_c)
    assert fit["heat_slope"] == pytest.approx(0.9, rel=0.02)
    assert fit["baseline_kwh_day"] == pytest.approx(8.0, rel=0.02)
    assert fit["r2"] > 0.99


def test_recovers_a_cooling_only_model(config, temps):
    energy = 5.0 + 0.7 * np.maximum(0.0, temps - 22.0)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    assert fit["model"] == cp.MODEL_COOLING
    assert fit["cool_cp"] == pytest.approx(22.0, abs=config.changepoint.grid_step_c)
    assert fit["cool_slope"] == pytest.approx(0.7, rel=0.02)


def test_recovers_a_five_parameter_model(config, temps):
    energy = 6.0 + 1.1 * np.maximum(0.0, 13.0 - temps) + 0.8 * np.maximum(0.0, temps - 23.0)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    assert fit["model"] == cp.MODEL_FIVE
    assert fit["heat_cp"] == pytest.approx(13.0, abs=config.changepoint.grid_step_c)
    assert fit["cool_cp"] == pytest.approx(23.0, abs=config.changepoint.grid_step_c)
    assert fit["heat_slope"] == pytest.approx(1.1, rel=0.02)
    assert fit["cool_slope"] == pytest.approx(0.8, rel=0.02)


def test_flat_series_selects_the_mean_model(config, temps):
    generator = np.random.default_rng(3)
    energy = 9.0 + generator.normal(0.0, 0.4, size=temps.size)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    assert fit["model"] == cp.MODEL_MEAN
    assert fit["baseline_kwh_day"] == pytest.approx(energy.mean())
    assert fit["heat_cp"] is None and fit["cool_cp"] is None


def test_coefficients_match_statsmodels(config, temps):
    generator = np.random.default_rng(5)
    energy = 7.0 + 0.85 * np.maximum(0.0, 16.0 - temps) + generator.normal(0.0, 0.5, size=temps.size)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    design = sm.add_constant(np.maximum(0.0, fit["heat_cp"] - temps))
    reference = sm.OLS(energy, design).fit()
    assert fit["baseline_kwh_day"] == pytest.approx(reference.params[0], abs=1e-8)
    assert fit["heat_slope"] == pytest.approx(reference.params[1], abs=1e-8)
    assert fit["sse"] == pytest.approx(float(reference.ssr), rel=1e-9)
    assert fit["r2"] == pytest.approx(float(reference.rsquared), abs=1e-9)


def test_enumeration_matches_a_finer_brute_force_search(config, temps):
    generator = np.random.default_rng(7)
    energy = 6.5 + 0.95 * np.maximum(0.0, 14.5 - temps) + generator.normal(0.0, 0.6, size=temps.size)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    oracle_sse, oracle_point = brute_force(temps, energy, cp.MODEL_HEATING, config)
    assert fit["model"] == cp.MODEL_HEATING
    assert abs(fit["heat_cp"] - oracle_point[0]) <= config.changepoint.grid_step_c
    assert fit["sse"] <= oracle_sse * 1.02


def test_four_parameter_matches_brute_force(config, temps):
    generator = np.random.default_rng(9)
    energy = (
        4.0
        + 0.9 * np.maximum(0.0, 18.0 - temps)
        + 0.6 * np.maximum(0.0, temps - 18.0)
        + generator.normal(0.0, 0.3, size=temps.size)
    )
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    oracle_sse, oracle_point = brute_force(temps, energy, cp.MODEL_FOUR, config)
    assert fit["model"] in (cp.MODEL_FOUR, cp.MODEL_FIVE)
    assert fit["sse"] <= oracle_sse * 1.02
    assert abs(fit["heat_cp"] - oracle_point[0]) <= 2.0 * config.changepoint.grid_step_c


def test_prediction_and_decomposition_agree(config, temps):
    energy = 6.0 + 1.1 * np.maximum(0.0, 13.0 - temps) + 0.8 * np.maximum(0.0, temps - 23.0)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    parts = cp.decompose(fit, temps)
    predicted = cp.predict(fit, temps).sum() * 365.0 / temps.size
    total = parts["baseload_kwh_year"] + parts["heating_kwh_year"] + parts["cooling_kwh_year"]
    assert total == pytest.approx(predicted, rel=1e-9)
    assert parts["heating_kwh_year"] > parts["cooling_kwh_year"] > 0.0


def test_linear_fit_attributes_weather_dependent_energy(temps):
    fit = {
        "model": cp.MODEL_LINEAR,
        "baseline_kwh_day": 24.0,
        "heat_slope": 0.55,
        "cool_slope": 0.0,
        "heat_cp": None,
        "cool_cp": None,
    }
    parts = cp.decompose(fit, temps)
    predicted = cp.predict(fit, temps).sum() * 365.0 / temps.size
    total = parts["baseload_kwh_year"] + parts["heating_kwh_year"] + parts["cooling_kwh_year"]
    assert total == pytest.approx(predicted, rel=1e-9)
    assert parts["heating_kwh_year"] > 0.0
    assert parts["cooling_kwh_year"] == pytest.approx(0.0, abs=1e-9)
    assert parts["baseload_kwh_year"] == pytest.approx(float(cp.predict(fit, temps).min()) * 365.0, rel=1e-9)


def test_flat_fit_has_no_weather_dependent_energy(config, temps):
    energy = np.full(temps.size, 7.5)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    parts = cp.decompose(fit, temps)
    assert parts["heating_kwh_year"] == pytest.approx(0.0, abs=1e-6)
    assert parts["cooling_kwh_year"] == pytest.approx(0.0, abs=1e-6)
    assert parts["baseload_kwh_year"] == pytest.approx(7.5 * 365.0, rel=1e-6)


def test_short_series_is_rejected(config):
    temps = np.linspace(0.0, 20.0, 10)
    with pytest.raises(cp.ChangepointError, match="at least"):
        cp.fit_meter(temps, np.ones(10), np.array([10.0]), config)


def test_slope_sign_constraint_blocks_negative_heating(config, temps):
    energy = 10.0 - 0.4 * np.maximum(0.0, 15.0 - temps)
    fit = cp.fit_meter(temps, energy, grid_for(temps, config), config)
    assert fit["heat_slope"] >= 0.0
    assert fit["cool_slope"] >= 0.0
    assert fit["model"] in (cp.MODEL_MEAN, cp.MODEL_LINEAR, cp.MODEL_COOLING)


def test_tiny_config_grid_is_bounded(tmp_path, temps):
    tiny = load_config(write_config(tmp_path / "project", apply_tiny))
    grid = cp.candidate_grid(temps, tiny)
    assert grid.size >= 1
    assert grid.min() >= temps.min()
    assert grid.max() <= temps.max()
