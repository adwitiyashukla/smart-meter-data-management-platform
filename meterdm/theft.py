from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from meterdm import changepoint, db
from meterdm.config import Config

TAMPER_NONE = "none"
TAMPER_CONSTANT = "constant_bypass"
TAMPER_NIGHT = "night_bypass"
TAMPER_FREEZE = "meter_freeze"
TAMPER_DRIFT = "gradual_drift"
TAMPER_TYPES = (TAMPER_CONSTANT, TAMPER_NIGHT, TAMPER_FREEZE, TAMPER_DRIFT)

FEATURES = (
    "level_ratio",
    "night_ratio",
    "residual_ratio",
    "cusum_negative",
    "transformer_loss",
    "night_share_shift",
    "variation_ratio",
    "flat_days_share",
)


class TheftError(RuntimeError):
    pass


def load_panel(connection, fleet_name: str):
    return connection.sql(
        """
        SELECT dense_rank() OVER (ORDER BY d.meter_id) - 1 AS code, d.meter_id, d.local_date,
               d.kwh, d.night_kwh, w.tavg_c, m.transformer_id
        FROM fact_daily_read d
        JOIN dim_date x USING (fleet, local_date)
        JOIN fact_weather_day w USING (fleet, local_date)
        JOIN dim_meter m USING (fleet, meter_id)
        WHERE d.fleet = ? AND d.filled_slots = x.expected_slots AND d.aggregated = 0
          AND d.kwh IS NOT NULL AND d.night_kwh IS NOT NULL
        ORDER BY d.meter_id, d.local_date
        """,
        params=[fleet_name],
    ).to_arrow_table()


def plan_tampering(lengths: np.ndarray, config: Config, seed: int):
    rng = np.random.default_rng([seed, int(lengths.sum()), int(lengths.size)])
    settings = config.theft
    count = int(lengths.size)
    tampered = rng.random(count) < settings.tamper_fraction
    kinds = np.array([TAMPER_NONE] * count, dtype=object)
    starts = np.full(count, -1, dtype=int)
    for index in np.flatnonzero(tampered):
        length = int(lengths[index])
        earliest = length - settings.test_days
        latest = length - settings.tamper.min_event_days
        if earliest < 0 or latest <= earliest:
            continue
        kinds[index] = TAMPER_TYPES[int(rng.integers(0, len(TAMPER_TYPES)))]
        starts[index] = int(rng.integers(earliest, latest))
    return kinds, starts


def apply_tamper(day_kwh, night_kwh, kind, start, config: Config):
    settings = config.theft.tamper
    day = day_kwh.copy()
    night = night_kwh.copy()
    if kind == TAMPER_NONE or start < 0 or start >= day.size:
        return day, night
    window = slice(start, day.size)
    if kind == TAMPER_CONSTANT:
        day[window] = day[window] * settings.constant_bypass_retain
        night[window] = night[window] * settings.constant_bypass_retain
    elif kind == TAMPER_NIGHT:
        removed = night[window] * (1.0 - settings.night_bypass_retain)
        night[window] = night[window] - removed
        day[window] = day[window] - removed
    elif kind == TAMPER_FREEZE:
        level = float(np.quantile(day[:start], settings.freeze_quantile)) if start > 0 else float(day.mean())
        share = float(np.mean(night[:start] / np.maximum(day[:start], 1e-9))) if start > 0 else 0.2
        day[window] = level
        night[window] = level * share
    elif kind == TAMPER_DRIFT:
        length = day.size - start
        retain = np.linspace(1.0, settings.drift_final_retain, length)
        day[window] = day[window] * retain
        night[window] = night[window] * retain
    return np.maximum(day, 0.0), np.maximum(night, 0.0)


def cusum(values: np.ndarray, k: float, limit: float) -> float:
    low = 0.0
    worst = 0.0
    for value in values:
        low = min(0.0, low + value + k)
        worst = min(worst, low)
    return float(min(worst / limit, 0.0))


def meter_features(day_kwh, night_kwh, temps, reference, test, config: Config, grid):
    settings = config.theft
    reference_day = day_kwh[reference]
    test_day = day_kwh[test]
    reference_night = night_kwh[reference]
    test_night = night_kwh[test]
    level_ratio = float(test_day.mean() / max(reference_day.mean(), 1e-9))
    night_ratio = float(np.median(test_night) / max(np.median(reference_night), 1e-9))
    reference_share = float(np.mean(reference_night / np.maximum(reference_day, 1e-9)))
    test_share = float(np.mean(test_night / np.maximum(test_day, 1e-9)))
    reference_spread = float(np.std(reference_day))
    test_spread = float(np.std(test_day))
    variation_ratio = test_spread / max(reference_spread, 1e-9)
    steps = np.abs(np.diff(test_day))
    tolerance = max(1e-4 * float(np.mean(test_day)), 1e-9)
    flat_days_share = float(np.mean(steps <= tolerance)) if steps.size else 0.0
    residual_ratio = level_ratio
    cusum_score = 0.0
    if reference_day.size >= config.changepoint.min_days:
        fit = changepoint.fit_meter(temps[reference], reference_day, grid, config)
        expected = changepoint.predict(fit, temps[test])
        expected = np.maximum(expected, 1e-9)
        residual_ratio = float((test_day / expected).mean())
        sigma = float(np.std(reference_day - changepoint.predict(fit, temps[reference])))
        if sigma > 1e-9:
            standardised = (test_day - expected) / sigma
            cusum_score = cusum(standardised, settings.cusum_k, settings.cusum_h)
    return {
        "level_ratio": level_ratio,
        "night_ratio": night_ratio,
        "residual_ratio": residual_ratio,
        "cusum_negative": cusum_score,
        "night_share_shift": test_share - reference_share,
        "variation_ratio": variation_ratio,
        "flat_days_share": flat_days_share,
    }


def build_features(connection, config: Config, fleet_name: str, progress=None):
    table = load_panel(connection, fleet_name)
    if table.num_rows == 0:
        return None
    codes = table.column("code").to_numpy()
    meters = table.column("meter_id").to_pylist()
    transformers = table.column("transformer_id").to_pylist()
    dates = table.column("local_date").to_numpy(zero_copy_only=False)
    day_kwh = table.column("kwh").to_numpy(zero_copy_only=False).astype(float)
    night_kwh = table.column("night_kwh").to_numpy(zero_copy_only=False).astype(float)
    temps_all = table.column("tavg_c").to_numpy(zero_copy_only=False).astype(float)
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [codes.size]])
    unique_meters = [meters[index] for index in starts]
    unique_transformers = [transformers[index] for index in starts]
    lengths = ends - starts
    settings = config.theft
    grid = changepoint.candidate_grid(temps_all, config)

    required = settings.min_reference_days + settings.test_days
    usable = lengths >= required
    if not np.any(usable):
        raise TheftError(f"no meter in {fleet_name} has {required} clean days")
    kinds, tamper_starts = plan_tampering(lengths, config, config.run.seed)
    kinds = np.where(usable, kinds, TAMPER_NONE)

    rows = []
    honest_totals = {}
    recorded_totals = {}
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        length = end - start
        transformer = unique_transformers[index]
        honest = day_kwh[start:end]
        if not usable[index]:
            honest_totals[transformer] = honest_totals.get(transformer, 0.0) + float(honest.sum())
            recorded_totals[transformer] = recorded_totals.get(transformer, 0.0) + float(honest.sum())
            continue
        offset = length - settings.test_days
        tampered_day, tampered_night = apply_tamper(
            honest, night_kwh[start:end], kinds[index], int(tamper_starts[index]), config
        )
        honest_totals[transformer] = honest_totals.get(transformer, 0.0) + float(honest.sum())
        recorded_totals[transformer] = recorded_totals.get(transformer, 0.0) + float(tampered_day.sum())
        reference = slice(max(offset - settings.reference_days, 0), offset)
        if offset - reference.start < settings.min_reference_days:
            reference = slice(0, offset)
        test = slice(offset, length)
        features = meter_features(
            tampered_day, tampered_night, temps_all[start:end], reference, test, config, grid
        )
        features["meter_id"] = unique_meters[index]
        features["transformer_id"] = transformer
        features["label"] = int(kinds[index] != TAMPER_NONE)
        features["kind"] = kinds[index]
        features["first_test_date"] = str(dates[start + offset])
        rows.append(features)
        if progress and index and index % 1000 == 0:
            progress(f"theft {fleet_name} meter {index}")

    losses = {}
    for transformer, honest in honest_totals.items():
        feed = honest * (1.0 + config.theft.technical_loss_ratio)
        recorded = recorded_totals.get(transformer, 0.0)
        losses[transformer] = float(1.0 - recorded / feed) if feed > 0 else 0.0
    for row in rows:
        row["transformer_loss"] = losses.get(row["transformer_id"], 0.0)
    return rows


def score(rows, config: Config, fleet_name: str):
    matrix = np.array([[row[name] for name in FEATURES] for row in rows], dtype=float)
    labels = np.array([row["label"] for row in rows], dtype=int)
    rng = np.random.default_rng([config.run.seed, len(rows)])
    holdout = rng.random(len(rows)) < config.theft.holdout_fraction
    if labels.sum() < 5 or labels[holdout].sum() < 2 or labels[~holdout].sum() < 2:
        raise TheftError(f"not enough tampered meters in {fleet_name} to score")
    scaler = StandardScaler().fit(matrix[~holdout])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(matrix[~holdout]), labels[~holdout])
    probabilities = model.predict_proba(scaler.transform(matrix))[:, 1]
    baseline = -(
        np.argsort(np.argsort(matrix[:, FEATURES.index("residual_ratio")]))
        + np.argsort(np.argsort(matrix[:, FEATURES.index("night_ratio")]))
        + np.argsort(np.argsort(matrix[:, FEATURES.index("cusum_negative")]))
    ).astype(float)
    metrics = {
        "meters_scored": int(len(rows)),
        "tampered": int(labels.sum()),
        "holdout_meters": int(holdout.sum()),
        "holdout_tampered": int(labels[holdout].sum()),
        "roc_auc": float(roc_auc_score(labels[holdout], probabilities[holdout])),
        "average_precision": float(average_precision_score(labels[holdout], probabilities[holdout])),
        "baseline_roc_auc": float(roc_auc_score(labels[holdout], baseline[holdout])),
        "coefficients": dict(zip(FEATURES, [float(value) for value in model.coef_[0]], strict=True)),
    }
    order = np.argsort(-probabilities[holdout])
    holdout_labels = labels[holdout][order]
    for k in config.theft.precision_at_k:
        cut = min(k, holdout_labels.size)
        metrics[f"precision_at_{k}"] = float(holdout_labels[:cut].mean()) if cut else 0.0
    by_kind = {}
    for kind in TAMPER_TYPES:
        mask = np.array([row["kind"] == kind for row in rows]) & holdout
        if mask.sum():
            threshold = float(np.quantile(probabilities[holdout], 0.9))
            by_kind[kind] = {
                "meters": int(mask.sum()),
                "recall_at_top_decile": float((probabilities[mask] >= threshold).mean()),
            }
    metrics["by_kind"] = by_kind
    return probabilities, holdout, metrics


def build_theft(config: Config, progress=None) -> dict:
    connection = db.gold(config)
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_theft_score (
            fleet VARCHAR, meter_id VARCHAR, transformer_id VARCHAR, label INTEGER, kind VARCHAR,
            score DOUBLE, holdout BOOLEAN, level_ratio DOUBLE, night_ratio DOUBLE, residual_ratio DOUBLE,
            cusum_negative DOUBLE, transformer_loss DOUBLE, night_share_shift DOUBLE,
            variation_ratio DOUBLE, flat_days_share DOUBLE, first_test_date VARCHAR
        )
        """
    )
    report = {}
    for fleet_name in config.enabled_fleets:
        try:
            rows = build_features(connection, config, fleet_name, progress)
        except TheftError as error:
            report[fleet_name] = {"skipped": True, "reason": str(error)}
            continue
        if not rows:
            report[fleet_name] = {"skipped": True, "reason": "no clean daily panel"}
            continue
        try:
            probabilities, holdout, metrics = score(rows, config, fleet_name)
        except TheftError as error:
            report[fleet_name] = {"skipped": True, "reason": str(error)}
            continue
        connection.executemany(
            "INSERT INTO fact_theft_score VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    fleet_name, row["meter_id"], row["transformer_id"], row["label"], row["kind"],
                    float(probabilities[index]), bool(holdout[index]), row["level_ratio"], row["night_ratio"],
                    row["residual_ratio"], row["cusum_negative"], row["transformer_loss"],
                    row["night_share_shift"], row["variation_ratio"], row["flat_days_share"],
                    row["first_test_date"],
                )
                for index, row in enumerate(rows)
            ],
        )
        report[fleet_name] = metrics
        if progress:
            progress(f"theft {fleet_name} auc={metrics['roc_auc']:.4f} ap={metrics['average_precision']:.4f}")
    connection.close()
    path = config.paths.reports_dir / "theft_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
    return report
