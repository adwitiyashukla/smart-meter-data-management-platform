from __future__ import annotations

import json

import numpy as np

from meterdm import db
from meterdm.config import Config

TREATMENT_LABEL = "ToU"
CONTROL_LABEL = "Std"


class DtouError(RuntimeError):
    pass


def absorb(values: np.ndarray, unit: np.ndarray, period: np.ndarray, iterations: int = 60, tolerance=1e-10):
    unit_count = np.bincount(unit)
    period_count = np.bincount(period)
    working = values.astype(float).copy()
    for _ in range(iterations):
        before = working.copy()
        working -= np.bincount(unit, weights=working)[unit] / unit_count[unit]
        working -= np.bincount(period, weights=working)[period] / period_count[period]
        if np.max(np.abs(working - before)) < tolerance:
            break
    return working


def two_way_did(outcome, treatment, unit, period, cluster=None, absorbed_outcome=None):
    y = absorb(outcome, unit, period) if absorbed_outcome is None else absorbed_outcome
    x = absorb(treatment, unit, period)
    denominator = float(x @ x)
    if denominator <= 1e-12:
        raise DtouError("treatment indicator has no variation after absorbing fixed effects")
    beta = float(x @ y) / denominator
    residual = y - beta * x
    observations = outcome.size
    parameters = np.unique(unit).size + np.unique(period).size
    degrees = max(observations - parameters - 1, 1)
    if cluster is None:
        variance = float(residual @ residual) / degrees / denominator
    else:
        groups = np.unique(cluster)
        meat = 0.0
        for group in groups:
            mask = cluster == group
            meat += float(x[mask] @ residual[mask]) ** 2
        correction = (groups.size / max(groups.size - 1, 1)) * ((observations - 1) / degrees)
        variance = correction * meat / denominator**2
    standard_error = float(np.sqrt(max(variance, 0.0)))
    return {
        "beta": beta,
        "standard_error": standard_error,
        "t_statistic": beta / standard_error if standard_error > 0 else 0.0,
        "observations": int(observations),
        "clusters": int(np.unique(cluster).size) if cluster is not None else 0,
    }


def load_panel(connection, config: Config):
    return connection.sql(
        """
        SELECT meter_id, local_date, tariff_group, window_kwh, day_kwh, event_day, window_slots
        FROM fact_trial_window
        WHERE window_kwh IS NOT NULL AND window_slots > 0
        ORDER BY meter_id, local_date
        """
    ).to_arrow_table()


def build_dtou(config: Config, progress=None) -> dict:
    connection = db.gold(config)
    table = load_panel(connection, config)
    if table.num_rows == 0:
        connection.close()
        report = {"available": False, "reason": "no tariff trial data in the warehouse"}
        _write(config, report)
        return report
    meters = np.array(table.column("meter_id").to_pylist())
    dates = np.array([str(value) for value in table.column("local_date").to_pylist()])
    groups = np.array(table.column("tariff_group").to_pylist())
    window = table.column("window_kwh").to_numpy(zero_copy_only=False).astype(float)
    event = np.array(table.column("event_day").to_pylist(), dtype=bool)
    slots = table.column("window_slots").to_numpy(zero_copy_only=False).astype(int)

    modal = np.bincount(slots).argmax()
    keep = slots == modal
    meters, dates, groups, window, event = meters[keep], dates[keep], groups[keep], window[keep], event[keep]
    treated_meter = groups == TREATMENT_LABEL
    floor = config.dtou.min_meters_per_group
    treated_count = int(np.unique(meters[treated_meter]).size)
    control_count = int(np.unique(meters[~treated_meter]).size)
    if treated_count < floor or control_count < floor:
        connection.close()
        report = {"available": False, "reason": "not enough meters in one of the tariff groups"}
        _write(config, report)
        return report

    unit_codes, unit_index = np.unique(meters, return_inverse=True)
    period_codes, period_index = np.unique(dates, return_inverse=True)
    treatment = (treated_meter & event).astype(float)
    absorbed = absorb(window, unit_index, period_index)
    estimate = two_way_did(
        window, treatment, unit_index, period_index, cluster=unit_index, absorbed_outcome=absorbed
    )

    control_event_mean = float(window[(~treated_meter) & event].mean())
    treated_event_mean = float(window[treated_meter & event].mean())
    treated_quiet_mean = float(window[treated_meter & ~event].mean())
    estimate["relative_effect"] = estimate["beta"] / control_event_mean if control_event_mean else 0.0
    estimate["control_event_mean_kwh"] = control_event_mean
    estimate["treated_event_mean_kwh"] = treated_event_mean
    estimate["treated_quiet_mean_kwh"] = treated_quiet_mean
    estimate["treated_meters"] = int(np.unique(meters[treated_meter]).size)
    estimate["control_meters"] = int(np.unique(meters[~treated_meter]).size)
    estimate["event_days"] = int(np.unique(dates[event]).size)
    estimate["days"] = int(period_codes.size)

    rng = np.random.default_rng([config.run.seed, int(period_codes.size)])
    event_codes = np.unique(period_index[event])
    quiet_codes = np.setdiff1d(np.arange(period_codes.size), event_codes)
    placebos = []
    if config.dtou.placebo_draws and quiet_codes.size > event_codes.size + 1:
        for _ in range(config.dtou.placebo_draws):
            fake = rng.choice(quiet_codes, size=event_codes.size, replace=False)
            fake_event = np.isin(period_index, fake)
            fake_treatment = (treated_meter & fake_event).astype(float)
            try:
                placebos.append(
                    two_way_did(
                        window, fake_treatment, unit_index, period_index, absorbed_outcome=absorbed
                    )["beta"]
                )
            except DtouError:
                continue
    if placebos:
        array = np.array(placebos)
        estimate["placebo_draws"] = int(array.size)
        estimate["placebo_mean"] = float(array.mean())
        estimate["placebo_sd"] = float(array.std(ddof=1)) if array.size > 1 else 0.0
        estimate["placebo_share_more_extreme"] = float((np.abs(array) >= abs(estimate["beta"])).mean())

    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_dtou_estimate (
            metric VARCHAR, value DOUBLE
        )
        """
    )
    connection.executemany(
        "INSERT INTO fact_dtou_estimate VALUES (?, ?)",
        [(key, float(value)) for key, value in estimate.items() if isinstance(value, int | float)],
    )
    connection.close()
    estimate["available"] = True
    if progress:
        progress(f"dtou beta={estimate['beta']:.5f} relative={estimate['relative_effect']:.4f}")
    _write(config, estimate)
    return estimate


def _write(config: Config, report):
    path = config.paths.reports_dir / "dtou_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
