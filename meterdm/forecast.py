from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from meterdm import db
from meterdm.config import Config

FEATURE_NAMES = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "hour_of_week",
    "temp_c",
    "temp_mean_24",
    "lag_24",
    "lag_48",
    "lag_168",
    "mean_168",
    "trend",
)
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168


class ForecastError(RuntimeError):
    pass


def load_series(connection, config: Config, fleet_name: str):
    zone = config.fleet(fleet_name).timezone
    return connection.sql(
        f"""
        WITH hourly AS (
            SELECT h.ts_utc, h.kwh, h.meters, w.temp_c,
                   CAST((h.ts_utc AT TIME ZONE 'UTC' AT TIME ZONE '{zone}') AS DATE) AS local_date,
                   hour((h.ts_utc AT TIME ZONE 'UTC' AT TIME ZONE '{zone}')) AS local_hour
            FROM fact_fleet_hourly h
            JOIN fact_weather_hour w ON w.fleet = h.fleet AND w.ts_utc = h.ts_utc
            WHERE h.fleet = ? AND h.meters > 0
        )
        SELECT hourly.ts_utc, hourly.kwh, hourly.meters, hourly.temp_c, hourly.local_hour,
               d.day_of_week, d.is_weekend, d.is_holiday
        FROM hourly LEFT JOIN dim_date d ON d.fleet = ? AND d.local_date = hourly.local_date
        ORDER BY hourly.ts_utc
        """,
        params=[fleet_name, fleet_name],
    ).to_arrow_table()


def build_matrix(table, config: Config):
    stamps = table.column("ts_utc").to_numpy(zero_copy_only=False)
    kwh = table.column("kwh").to_numpy(zero_copy_only=False).astype(float)
    meters = table.column("meters").to_numpy(zero_copy_only=False).astype(float)
    temps = table.column("temp_c").to_numpy(zero_copy_only=False).astype(float)
    hours = table.column("local_hour").to_numpy(zero_copy_only=False).astype(float)
    def numeric(name):
        return np.array([0.0 if value is None else float(value) for value in table.column(name).to_pylist()])

    dow = numeric("day_of_week")
    weekend = numeric("is_weekend")
    holiday = numeric("is_holiday")
    load = kwh / np.maximum(meters, 1.0)
    size = load.size
    if size < HOURS_PER_WEEK * 4:
        raise ForecastError(f"need at least {HOURS_PER_WEEK * 4} hours of history, got {size}")

    def lag(values, shift):
        out = np.full(size, np.nan)
        if shift < size:
            out[shift:] = values[: size - shift]
        return out

    def rolling(values, span):
        padded = np.concatenate([np.full(span, np.nan), values[:-1]])
        cumulative = np.nancumsum(np.nan_to_num(padded))
        counts = np.cumsum(~np.isnan(padded))
        out = np.full(size, np.nan)
        for index in range(size):
            high = index + span
            low = high - span
            if counts[high - 1] - (counts[low - 1] if low > 0 else 0) == span:
                out[index] = (cumulative[high - 1] - (cumulative[low - 1] if low > 0 else 0)) / span
        return out

    features = np.column_stack(
        [
            hours,
            dow,
            weekend,
            holiday,
            dow * 24.0 + hours,
            temps,
            rolling(temps, HOURS_PER_DAY),
            lag(load, HOURS_PER_DAY),
            lag(load, 2 * HOURS_PER_DAY),
            lag(load, HOURS_PER_WEEK),
            rolling(load, HOURS_PER_WEEK),
            np.arange(size, dtype=float) / (24.0 * 365.0),
        ]
    )
    return stamps, load, meters, features


def _model(settings, config, quantile=None):
    extra = {"loss": "quantile", "quantile": quantile} if quantile is not None else {}
    return HistGradientBoostingRegressor(
        max_iter=settings.max_iter,
        learning_rate=settings.learning_rate,
        max_depth=settings.max_depth,
        early_stopping=True,
        n_iter_no_change=settings.early_stopping_rounds,
        validation_fraction=settings.validation_fraction,
        random_state=config.run.seed,
        **extra,
    )


def pinball(actual, predicted, quantile):
    delta = actual - predicted
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def backtest(stamps, load, meters, features, config: Config, progress=None):
    settings = config.forecast
    horizon = settings.horizon_hours
    train_hours = settings.train_days * HOURS_PER_DAY
    step = settings.backtest_step_days * HOURS_PER_DAY
    size = load.size
    last_origin = size - horizon
    origins = [last_origin - index * step for index in range(settings.backtest_origins)]
    origins = sorted(origin for origin in origins if origin - train_hours >= HOURS_PER_WEEK)
    if not origins:
        raise ForecastError("history is too short for the configured backtest")
    records = []
    for number, origin in enumerate(origins):
        train = slice(max(origin - train_hours, 0), origin)
        test = slice(origin, origin + horizon)
        usable = ~np.isnan(features[train]).any(axis=1) & ~np.isnan(load[train])
        train_x = features[train][usable]
        train_y = load[train][usable]
        test_x = features[test]
        if train_y.size < HOURS_PER_WEEK or np.isnan(test_x).any():
            continue
        split = train_y.size
        if settings.conformal:
            split = int(train_y.size * (1.0 - settings.calibration_fraction))
        fit_x, fit_y = train_x[:split], train_y[:split]
        calibration_x, calibration_y = train_x[split:], train_y[split:]
        point = _model(settings, config).fit(train_x, train_y)
        predictions = {"point": point.predict(test_x)}
        raw = {}
        for quantile in settings.quantiles:
            model = _model(settings, config, quantile=quantile).fit(fit_x, fit_y)
            key = f"q{int(quantile * 100)}"
            predictions[key] = model.predict(test_x)
            raw[key] = predictions[key].copy()
            if settings.conformal and calibration_y.size:
                raw[f"cal_{key}"] = model.predict(calibration_x)
        width = 0.0
        if settings.conformal and calibration_y.size >= HOURS_PER_DAY:
            low_key = f"q{int(settings.quantiles[0] * 100)}"
            high_key = f"q{int(settings.quantiles[-1] * 100)}"
            scores = np.maximum(
                raw[f"cal_{low_key}"] - calibration_y, calibration_y - raw[f"cal_{high_key}"]
            )
            level = settings.quantiles[-1] - settings.quantiles[0]
            rank = min(
                int(np.ceil((calibration_y.size + 1) * level)) - 1, calibration_y.size - 1
            )
            width = float(np.sort(scores)[max(rank, 0)])
            predictions[low_key] = predictions[low_key] - width
            predictions[high_key] = predictions[high_key] + width
        naive = features[test][:, FEATURE_NAMES.index("lag_168")]
        for offset in range(horizon):
            index = origin + offset
            records.append(
                {
                    "origin": number,
                    "origin_ts": str(stamps[origin]),
                    "ts_utc": str(stamps[index]),
                    "horizon": offset + 1,
                    "actual": float(load[index]),
                    "meters": float(meters[index]),
                    "naive": float(naive[offset]),
                    "predicted": float(predictions["point"][offset]),
                    "conformal_width": width,
                    **{key: float(values[offset]) for key, values in predictions.items() if key != "point"},
                    **{
                        f"raw_{key}": float(values[offset])
                        for key, values in raw.items()
                        if not key.startswith("cal_")
                    },
                }
            )
        if progress:
            progress(f"forecast origin {number + 1}/{len(origins)}")
    return records


def summarise(records, config: Config):
    actual = np.array([row["actual"] for row in records])
    predicted = np.array([row["predicted"] for row in records])
    naive = np.array([row["naive"] for row in records])
    meters = np.array([row["meters"] for row in records])
    safe = np.maximum(np.abs(actual), 1e-9)
    metrics = {
        "points": int(actual.size),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape": float(np.mean(np.abs(actual - predicted) / safe)),
        "naive_mape": float(np.mean(np.abs(actual - naive) / safe)),
        "total_mape": float(
            np.mean(np.abs(actual * meters - predicted * meters) / np.maximum(np.abs(actual * meters), 1e-9))
        ),
    }
    metrics["skill_vs_naive"] = 1.0 - metrics["mape"] / metrics["naive_mape"] if metrics["naive_mape"] else 0.0
    for quantile in config.forecast.quantiles:
        key = f"q{int(quantile * 100)}"
        values = np.array([row[key] for row in records])
        metrics[f"pinball_{key}"] = pinball(actual, values, quantile)
    low = f"q{int(config.forecast.quantiles[0] * 100)}"
    high = f"q{int(config.forecast.quantiles[-1] * 100)}"
    if low != high:
        lower = np.array([row[low] for row in records])
        upper = np.array([row[high] for row in records])
        metrics["interval_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
        metrics["interval_nominal"] = config.forecast.quantiles[-1] - config.forecast.quantiles[0]
        metrics["interval_width"] = float(np.mean(upper - lower))
        raw_low = np.array([row.get(f"raw_{low}", row[low]) for row in records])
        raw_high = np.array([row.get(f"raw_{high}", row[high]) for row in records])
        metrics["raw_interval_coverage"] = float(np.mean((actual >= raw_low) & (actual <= raw_high)))
        metrics["raw_interval_width"] = float(np.mean(raw_high - raw_low))
        metrics["conformal_width"] = float(np.mean([row.get("conformal_width", 0.0) for row in records]))
    return metrics


def build_forecast(config: Config, progress=None) -> dict:
    connection = db.gold(config)
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_forecast_backtest (
            fleet VARCHAR, origin INTEGER, origin_ts VARCHAR, ts_utc VARCHAR, horizon INTEGER,
            actual DOUBLE, meters DOUBLE, naive DOUBLE, predicted DOUBLE, q10 DOUBLE, q50 DOUBLE, q90 DOUBLE
        )
        """
    )
    connection.execute(
        "CREATE OR REPLACE TABLE fact_forecast_metric (fleet VARCHAR, metric VARCHAR, value DOUBLE)"
    )
    report = {}
    quantile_keys = [f"q{int(value * 100)}" for value in config.forecast.quantiles]
    for fleet_name in config.enabled_fleets:
        table = load_series(connection, config, fleet_name)
        if table.num_rows == 0:
            report[fleet_name] = {"skipped": True}
            continue
        stamps, load, meters, features = build_matrix(table, config)
        records = backtest(stamps, load, meters, features, config, progress)
        if not records:
            report[fleet_name] = {"skipped": True}
            continue
        connection.executemany(
            "INSERT INTO fact_forecast_backtest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    fleet_name, row["origin"], row["origin_ts"], row["ts_utc"], row["horizon"],
                    row["actual"], row["meters"], row["naive"], row["predicted"],
                    *[row.get(key, float("nan")) for key in ("q10", "q50", "q90")],
                )
                for row in records
            ],
        )
        metrics = summarise(records, config)
        metrics["origins"] = len({row["origin"] for row in records})
        metrics["quantiles"] = quantile_keys
        connection.executemany(
            "INSERT INTO fact_forecast_metric VALUES (?, ?, ?)",
            [(fleet_name, key, float(value)) for key, value in metrics.items() if isinstance(value, int | float)],
        )
        report[fleet_name] = metrics
        if progress:
            progress(f"forecast {fleet_name} mape={metrics['mape']:.4f} naive={metrics['naive_mape']:.4f}")
    connection.close()
    path = config.paths.reports_dir / "forecast_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
    return report
