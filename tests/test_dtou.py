from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from meterdm import db, dtou


@pytest.fixture(scope="module")
def estimated(analytics_project):
    config, reports = analytics_project
    return config, reports["dtou"]


def panel(effect, units=40, periods=30, seed=1):
    generator = np.random.default_rng(seed)
    unit_effect = generator.normal(4.0, 1.2, size=units)
    period_effect = generator.normal(0.0, 0.7, size=periods)
    unit = np.repeat(np.arange(units), periods)
    period = np.tile(np.arange(periods), units)
    treated_unit = (np.arange(units) % 2 == 0)[unit]
    event = (np.arange(periods) >= periods // 2)[period]
    treatment = (treated_unit & event).astype(float)
    outcome = (
        unit_effect[unit] + period_effect[period] + effect * treatment + generator.normal(0.0, 0.3, size=unit.size)
    )
    return outcome, treatment, unit, period


def test_absorb_removes_unit_and_period_means():
    outcome, _, unit, period = panel(0.0)
    residual = dtou.absorb(outcome, unit, period)
    for code in np.unique(unit):
        assert residual[unit == code].mean() == pytest.approx(0.0, abs=1e-8)
    for code in np.unique(period):
        assert residual[period == code].mean() == pytest.approx(0.0, abs=1e-8)


def test_two_way_did_recovers_a_known_effect():
    outcome, treatment, unit, period = panel(-0.65)
    estimate = dtou.two_way_did(outcome, treatment, unit, period, cluster=unit)
    assert estimate["beta"] == pytest.approx(-0.65, abs=0.05)
    assert estimate["standard_error"] > 0.0
    assert estimate["observations"] == outcome.size
    assert estimate["clusters"] == np.unique(unit).size


def test_matches_statsmodels_with_explicit_dummies():
    outcome, treatment, unit, period = panel(-0.4, units=20, periods=16, seed=5)
    mine = dtou.two_way_did(outcome, treatment, unit, period, cluster=unit)
    design = np.column_stack(
        [
            treatment,
            np.eye(np.unique(unit).size)[unit][:, 1:],
            np.eye(np.unique(period).size)[period][:, 1:],
        ]
    )
    reference = sm.OLS(outcome, sm.add_constant(design)).fit(
        cov_type="cluster", cov_kwds={"groups": unit}, use_t=False
    )
    assert mine["beta"] == pytest.approx(float(reference.params[1]), abs=1e-8)
    assert mine["standard_error"] == pytest.approx(float(reference.bse[1]), rel=0.06)


def test_constant_treatment_is_rejected():
    outcome, _, unit, period = panel(0.0)
    with pytest.raises(dtou.DtouError, match="no variation"):
        dtou.two_way_did(outcome, np.ones_like(outcome), unit, period)


def test_trial_estimate_recovers_the_injected_elasticity(estimated):
    config, report = estimated
    assert report["available"] is True
    truth = float(
        (config.paths.truth_dir / "replica_summary.json").read_text().split('"mean_elasticity":')[1].split(",")[0]
    )
    assert report["relative_effect"] < 0.0
    assert abs(report["relative_effect"]) == pytest.approx(truth, rel=0.25)
    assert report["treated_meters"] >= config.dtou.min_meters_per_group
    assert report["control_meters"] >= config.dtou.min_meters_per_group
    assert abs(report["t_statistic"]) > 3.0


def test_placebo_effects_are_centred_on_zero(estimated):
    _, report = estimated
    assert report["placebo_draws"] > 10
    assert abs(report["placebo_mean"]) < abs(report["beta"]) / 2.0
    assert report["placebo_share_more_extreme"] < 0.1


def test_estimate_is_stored_in_the_warehouse(estimated):
    config, report = estimated
    connection = db.read_only(config)
    stored = dict(connection.execute("SELECT metric, value FROM fact_dtou_estimate").fetchall())
    connection.close()
    assert stored["beta"] == pytest.approx(report["beta"])
    assert stored["relative_effect"] == pytest.approx(report["relative_effect"])
