from __future__ import annotations

import json

import numpy as np

from meterdm import db, vee
from meterdm.config import Config

METHOD_INTERPOLATE = vee.METHOD_INTERPOLATE
METHOD_LIKE_DAY = vee.METHOD_LIKE_DAY


class HoldoutError(RuntimeError):
    pass


def batch_files(config: Config, fleet_name: str, wanted: int):
    root = config.paths.silver_dir / fleet_name
    folders = sorted(path for path in root.glob("meter_batch=*") if path.is_dir())
    if not folders:
        raise HoldoutError(f"no silver batches for fleet {fleet_name}")
    return [folder / "readings.parquet" for folder in folders[:wanted]]


def load_actuals(connection, paths):
    pattern = [db.quote(path) for path in paths]
    union = " UNION ALL ".join(f"SELECT * FROM read_parquet({item})" for item in pattern)
    return connection.sql(
        f"""
        SELECT dense_rank() OVER (ORDER BY meter_id) - 1 AS code, meter_id, ts_utc, local_date, period, kwh
        FROM ({union})
        WHERE status = '{vee.STATUS_ACTUAL}' AND kwh IS NOT NULL AND period IS NOT NULL
        ORDER BY meter_id, ts_utc
        """
    ).to_arrow_table()


def plan_gaps(size: int, lengths, events: int, rng):
    mask = np.zeros(size, dtype=bool)
    runs = []
    for _ in range(events):
        length = int(lengths[rng.integers(0, len(lengths))])
        if size <= length + 4:
            continue
        start = int(rng.integers(1, size - length - 1))
        if mask[max(start - 1, 0) : start + length + 1].any():
            continue
        mask[start : start + length] = True
        runs.append((start, length))
    return mask, runs


def estimate_masked(values, stamps, dows, periods, mask, runs, config: Config):
    settings = config.vee
    visible = ~mask
    profile = {}
    keys = dows * 1000 + periods
    for key in np.unique(keys[visible]):
        profile[int(key)] = float(np.median(values[visible & (keys == key)]))
    estimates = np.full(values.size, np.nan)
    methods = np.empty(values.size, dtype=object)
    for start, length in runs:
        before = start - 1
        after = start + length
        window = slice(start, start + length)
        if length <= settings.short_gap_intervals and before >= 0 and after < values.size:
            span = stamps[after] - stamps[before]
            if span > 0:
                weight = (stamps[window] - stamps[before]) / span
                estimates[window] = values[before] + (values[after] - values[before]) * weight
                methods[window] = METHOD_INTERPOLATE
                continue
        for index in range(start, start + length):
            estimates[index] = profile.get(int(keys[index]), np.nan)
            methods[index] = METHOD_LIKE_DAY
    return np.maximum(estimates, 0.0), methods


def score_fleet(connection, config: Config, fleet_name: str, progress=None):
    settings = config.holdout
    paths = batch_files(config, fleet_name, settings.batches)
    table = load_actuals(connection, paths)
    if table.num_rows == 0:
        raise HoldoutError(f"no actual readings available for {fleet_name}")
    codes = table.column("code").to_numpy()
    meters = table.column("meter_id").to_pylist()
    stamps = table.column("ts_utc").to_numpy(zero_copy_only=False).astype("datetime64[s]").astype(np.int64)
    dates = table.column("local_date").to_numpy(zero_copy_only=False).astype("datetime64[D]")
    periods = table.column("period").to_numpy(zero_copy_only=False).astype(np.int64)
    values = table.column("kwh").to_numpy(zero_copy_only=False).astype(float)
    dows = (dates.astype(np.int64) + 4) % 7

    boundaries = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [codes.size]])
    lengths = np.array(settings.gap_lengths, dtype=np.int64)

    truth = []
    predicted = []
    method_of = []
    day_truth = {}
    day_predicted = {}
    covered_meters = 0
    for index, (first, last) in enumerate(zip(starts, ends, strict=True)):
        size = last - first
        if size < settings.min_readings:
            continue
        rng = np.random.default_rng([config.run.seed, settings.seed_offset, index])
        mask, runs = plan_gaps(size, lengths, settings.gap_events, rng)
        if not runs:
            continue
        slice_values = values[first:last]
        estimates, methods = estimate_masked(
            slice_values, stamps[first:last], dows[first:last], periods[first:last], mask, runs, config
        )
        usable = mask & ~np.isnan(estimates)
        if not usable.any():
            continue
        covered_meters += 1
        truth.append(slice_values[usable])
        predicted.append(estimates[usable])
        method_of.append(methods[usable])
        meter = meters[first]
        affected = dates[first:last][usable]
        for day in np.unique(affected):
            same_day = dates[first:last] == day
            key = (meter, str(day))
            day_truth[key] = float(slice_values[same_day].sum())
            replaced = slice_values[same_day].copy()
            marks = usable[same_day]
            replaced[marks] = estimates[same_day][marks]
            day_predicted[key] = float(replaced.sum())
        if progress and covered_meters % 50 == 0:
            progress(f"holdout {fleet_name} meters={covered_meters}")

    if not truth:
        raise HoldoutError(f"holdout produced no comparable readings for {fleet_name}")
    truth_values = np.concatenate(truth)
    predicted_values = np.concatenate(predicted)
    methods = np.concatenate(method_of)
    errors = np.abs(predicted_values - truth_values)
    totals_truth = np.array([day_truth[key] for key in day_truth])
    totals_predicted = np.array([day_predicted[key] for key in day_truth])
    usable_days = totals_truth > 0
    daily_error = np.abs(totals_predicted[usable_days] - totals_truth[usable_days]) / totals_truth[usable_days]

    report = {
        "meters": covered_meters,
        "batches": len(paths),
        "readings_held_out": int(truth_values.size),
        "mean_absolute_error_kwh": float(errors.mean()),
        "median_absolute_error_kwh": float(np.median(errors)),
        "mean_truth_kwh": float(truth_values.mean()),
        "normalised_mean_absolute_error": float(errors.mean() / max(truth_values.mean(), 1e-9)),
        "days_checked": int(usable_days.sum()),
        "daily_mean_relative_error": float(daily_error.mean()),
        "daily_median_relative_error": float(np.median(daily_error)),
    }
    for name in (METHOD_INTERPOLATE, METHOD_LIKE_DAY):
        chosen = methods == name
        if chosen.any():
            report[f"{name}_readings"] = int(chosen.sum())
            report[f"{name}_mean_absolute_error_kwh"] = float(errors[chosen].mean())
    return report


def build_holdout(config: Config, progress=None) -> dict:
    connection = db.gold(config)
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_holdout (fleet VARCHAR, metric VARCHAR, value DOUBLE)
        """
    )
    report = {}
    for fleet_name in config.enabled_fleets:
        try:
            outcome = score_fleet(connection, config, fleet_name, progress)
        except HoldoutError as error:
            report[fleet_name] = {"skipped": True, "reason": str(error)}
            continue
        report[fleet_name] = outcome
        connection.executemany(
            "INSERT INTO fact_holdout VALUES (?, ?, ?)",
            [(fleet_name, key, float(value)) for key, value in outcome.items() if isinstance(value, int | float)],
        )
        if progress:
            progress(
                f"holdout {fleet_name} daily error={outcome['daily_mean_relative_error']:.4f} "
                f"on {outcome['days_checked']} days"
            )
    connection.close()
    path = config.paths.reports_dir / "holdout_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
    return report
