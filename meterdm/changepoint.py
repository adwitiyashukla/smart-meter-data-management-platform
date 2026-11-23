from __future__ import annotations

import json

import numpy as np

from meterdm import db
from meterdm.config import Config

MODEL_MEAN = "mean"
MODEL_LINEAR = "2p"
MODEL_HEATING = "3ph"
MODEL_COOLING = "3pc"
MODEL_FOUR = "4p"
MODEL_FIVE = "5p"
MODELS = (MODEL_MEAN, MODEL_LINEAR, MODEL_HEATING, MODEL_COOLING, MODEL_FOUR, MODEL_FIVE)

FIELDS = (
    "model",
    "baseline_kwh_day",
    "heat_slope",
    "cool_slope",
    "heat_cp",
    "cool_cp",
    "sse",
    "r2",
    "adj_r2",
    "cvrmse",
    "aic",
    "days",
)


class ChangepointError(RuntimeError):
    pass


def candidate_grid(temps: np.ndarray, config: Config) -> np.ndarray:
    settings = config.changepoint
    low = float(np.quantile(temps, settings.grid_quantile_low))
    high = float(np.quantile(temps, settings.grid_quantile_high))
    if high - low < settings.grid_step_c:
        return np.array([0.5 * (low + high)])
    count = int(np.floor((high - low) / settings.grid_step_c)) + 1
    return low + settings.grid_step_c * np.arange(count)


def _aic(sse: float, n: int, k: int) -> float:
    if sse <= 0.0 or n <= 0:
        return -np.inf
    return n * np.log(sse / n) + 2.0 * k


def _summary(sse, n, k, total_ss, mean_y):
    r2 = 0.0 if total_ss <= 0 else max(0.0, 1.0 - sse / total_ss)
    denominator = max(n - k, 1)
    adj = 1.0 - (1.0 - r2) * (n - 1) / denominator if n > k else 0.0
    rmse = np.sqrt(max(sse, 0.0) / n)
    cvrmse = rmse / mean_y if mean_y > 0 else np.inf
    return r2, adj, cvrmse, _aic(sse, n, k)


def _two_parameter(basis, y, n):
    sum_x = basis.sum(axis=1)
    sum_xx = (basis * basis).sum(axis=1)
    sum_xy = basis @ y
    sum_y = y.sum()
    sum_yy = float(y @ y)
    denominator = n * sum_xx - sum_x * sum_x
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denominator > 1e-12, (n * sum_xy - sum_x * sum_y) / denominator, np.nan)
    intercept = (sum_y - slope * sum_x) / n
    sse = sum_yy - intercept * sum_y - slope * sum_xy
    return intercept, slope, sse


def _three_parameter(heat, cool, y, n):
    sum_h = heat.sum(axis=1)
    sum_c = cool.sum(axis=1)
    sum_hh = (heat * heat).sum(axis=1)
    sum_cc = (cool * cool).sum(axis=1)
    sum_hc = heat @ cool.T
    sum_hy = heat @ y
    sum_cy = cool @ y
    sum_y = float(y.sum())
    sum_yy = float(y @ y)
    return {
        "sum_h": sum_h,
        "sum_c": sum_c,
        "sum_hh": sum_hh,
        "sum_cc": sum_cc,
        "sum_hc": sum_hc,
        "sum_hy": sum_hy,
        "sum_cy": sum_cy,
        "sum_y": sum_y,
        "sum_yy": sum_yy,
        "n": n,
    }


def _solve_pairs(moments, heat_index, cool_index):
    n = moments["n"]
    count = heat_index.size
    matrices = np.empty((count, 3, 3))
    vectors = np.empty((count, 3))
    sum_h = moments["sum_h"][heat_index]
    sum_c = moments["sum_c"][cool_index]
    matrices[:, 0, 0] = n
    matrices[:, 0, 1] = sum_h
    matrices[:, 0, 2] = sum_c
    matrices[:, 1, 0] = sum_h
    matrices[:, 1, 1] = moments["sum_hh"][heat_index]
    matrices[:, 1, 2] = moments["sum_hc"][heat_index, cool_index]
    matrices[:, 2, 0] = sum_c
    matrices[:, 2, 1] = matrices[:, 1, 2]
    matrices[:, 2, 2] = moments["sum_cc"][cool_index]
    vectors[:, 0] = moments["sum_y"]
    vectors[:, 1] = moments["sum_hy"][heat_index]
    vectors[:, 2] = moments["sum_cy"][cool_index]
    determinant = np.linalg.det(matrices)
    ok = np.abs(determinant) > 1e-9
    solution = np.full((count, 3), np.nan)
    if np.any(ok):
        solution[ok] = np.linalg.solve(matrices[ok], vectors[ok][..., None])[..., 0]
    sse = (
        moments["sum_yy"]
        - solution[:, 0] * vectors[:, 0]
        - solution[:, 1] * vectors[:, 1]
        - solution[:, 2] * vectors[:, 2]
    )
    return solution, sse


def fit_meter(temps: np.ndarray, energy: np.ndarray, grid: np.ndarray, config: Config) -> dict:
    settings = config.changepoint
    n = temps.size
    if n < settings.min_days:
        raise ChangepointError(f"need at least {settings.min_days} days, got {n}")
    mean_y = float(energy.mean())
    total_ss = float(((energy - mean_y) ** 2).sum())
    heat = np.maximum(0.0, grid[:, None] - temps[None, :])
    cool = np.maximum(0.0, temps[None, :] - grid[:, None])
    heat_points = (heat > 0).sum(axis=1)
    cool_points = (cool > 0).sum(axis=1)
    minimum = settings.min_points_per_segment

    best = {
        "model": MODEL_MEAN,
        "baseline_kwh_day": mean_y,
        "heat_slope": 0.0,
        "cool_slope": 0.0,
        "heat_cp": None,
        "cool_cp": None,
        "sse": total_ss,
        "aic": _aic(total_ss, n, 1),
        "params": 1,
    }

    intercept, slope, sse = _two_parameter(temps[None, :], energy, n)
    linear_aic = _aic(float(sse[0]), n, 2)
    if np.isfinite(linear_aic) and linear_aic < best["aic"]:
        best = {
            "model": MODEL_LINEAR,
            "baseline_kwh_day": float(intercept[0]),
            "heat_slope": float(max(-slope[0], 0.0)),
            "cool_slope": float(max(slope[0], 0.0)),
            "heat_cp": None,
            "cool_cp": None,
            "sse": float(sse[0]),
            "aic": linear_aic,
            "params": 2,
        }

    valid_heat = heat_points >= minimum
    valid_cool = cool_points >= minimum
    if np.any(valid_heat):
        intercept, slope, sse = _two_parameter(heat, energy, n)
        usable = valid_heat & np.isfinite(sse) & (slope >= settings.min_slope_kwh_per_degree_day)
        if np.any(usable):
            index = int(np.flatnonzero(usable)[np.argmin(sse[usable])])
            aic = _aic(float(sse[index]), n, 3)
            if aic < best["aic"]:
                best = {
                    "model": MODEL_HEATING,
                    "baseline_kwh_day": float(intercept[index]),
                    "heat_slope": float(slope[index]),
                    "cool_slope": 0.0,
                    "heat_cp": float(grid[index]),
                    "cool_cp": None,
                    "sse": float(sse[index]),
                    "aic": aic,
                    "params": 3,
                }
    if np.any(valid_cool):
        intercept, slope, sse = _two_parameter(cool, energy, n)
        usable = valid_cool & np.isfinite(sse) & (slope >= settings.min_slope_kwh_per_degree_day)
        if np.any(usable):
            index = int(np.flatnonzero(usable)[np.argmin(sse[usable])])
            aic = _aic(float(sse[index]), n, 3)
            if aic < best["aic"]:
                best = {
                    "model": MODEL_COOLING,
                    "baseline_kwh_day": float(intercept[index]),
                    "heat_slope": 0.0,
                    "cool_slope": float(slope[index]),
                    "heat_cp": None,
                    "cool_cp": float(grid[index]),
                    "sse": float(sse[index]),
                    "aic": aic,
                    "params": 3,
                }

    moments = _three_parameter(heat, cool, energy, n)
    single = np.flatnonzero(valid_heat & valid_cool)
    if single.size:
        solution, sse = _solve_pairs(moments, single, single)
        usable = (
            np.isfinite(sse)
            & (solution[:, 1] >= settings.min_slope_kwh_per_degree_day)
            & (solution[:, 2] >= settings.min_slope_kwh_per_degree_day)
        )
        if np.any(usable):
            local = int(np.flatnonzero(usable)[np.argmin(sse[usable])])
            aic = _aic(float(sse[local]), n, 4)
            if aic < best["aic"]:
                point = float(grid[single[local]])
                best = {
                    "model": MODEL_FOUR,
                    "baseline_kwh_day": float(solution[local, 0]),
                    "heat_slope": float(solution[local, 1]),
                    "cool_slope": float(solution[local, 2]),
                    "heat_cp": point,
                    "cool_cp": point,
                    "sse": float(sse[local]),
                    "aic": aic,
                    "params": 4,
                }

    heat_index, cool_index = np.nonzero(
        valid_heat[:, None] & valid_cool[None, :] & (grid[:, None] < grid[None, :])
    )
    if heat_index.size:
        solution, sse = _solve_pairs(moments, heat_index, cool_index)
        usable = (
            np.isfinite(sse)
            & (solution[:, 1] >= settings.min_slope_kwh_per_degree_day)
            & (solution[:, 2] >= settings.min_slope_kwh_per_degree_day)
        )
        if np.any(usable):
            local = int(np.flatnonzero(usable)[np.argmin(sse[usable])])
            aic = _aic(float(sse[local]), n, 5)
            if aic < best["aic"]:
                best = {
                    "model": MODEL_FIVE,
                    "baseline_kwh_day": float(solution[local, 0]),
                    "heat_slope": float(solution[local, 1]),
                    "cool_slope": float(solution[local, 2]),
                    "heat_cp": float(grid[heat_index[local]]),
                    "cool_cp": float(grid[cool_index[local]]),
                    "sse": float(sse[local]),
                    "aic": aic,
                    "params": 5,
                }

    r2, adj_r2, cvrmse, _ = _summary(best["sse"], n, best["params"], total_ss, mean_y)
    best.update({"r2": r2, "adj_r2": adj_r2, "cvrmse": cvrmse, "days": n})
    best["accepted"] = bool(cvrmse <= settings.max_cvrmse and r2 >= settings.min_r2)
    return best


def predict(fit: dict, temps: np.ndarray) -> np.ndarray:
    if fit["model"] == MODEL_LINEAR:
        return fit["baseline_kwh_day"] + (fit["cool_slope"] - fit["heat_slope"]) * temps
    values = np.full(temps.shape, fit["baseline_kwh_day"], dtype=float)
    if fit["heat_cp"] is not None:
        values = values + fit["heat_slope"] * np.maximum(0.0, fit["heat_cp"] - temps)
    if fit["cool_cp"] is not None:
        values = values + fit["cool_slope"] * np.maximum(0.0, temps - fit["cool_cp"])
    return values


def decompose(fit: dict, temps: np.ndarray) -> dict:
    days = temps.size
    if days == 0:
        return {"baseload_kwh_year": 0.0, "heating_kwh_year": 0.0, "cooling_kwh_year": 0.0}
    scale = 365.0 / days
    values = predict(fit, temps)
    floor = float(values.min())
    pivot = float(temps[int(np.argmin(values))])
    excess = values - floor
    return {
        "baseload_kwh_year": floor * days * scale,
        "heating_kwh_year": float(excess[temps < pivot].sum()) * scale,
        "cooling_kwh_year": float(excess[temps > pivot].sum()) * scale,
    }


def load_panel(connection, fleet_name: str):
    return connection.sql(
        """
        SELECT dense_rank() OVER (ORDER BY d.meter_id) - 1 AS code, d.meter_id, d.kwh, w.tavg_c
        FROM fact_daily_read d
        JOIN dim_date x USING (fleet, local_date)
        JOIN fact_weather_day w USING (fleet, local_date)
        WHERE d.fleet = ? AND d.filled_slots = x.expected_slots AND d.aggregated = 0 AND d.kwh IS NOT NULL
        ORDER BY d.meter_id, d.local_date
        """,
        params=[fleet_name],
    ).to_arrow_table()


def fit_fleet(connection, config: Config, fleet_name: str, progress=None):
    table = load_panel(connection, fleet_name)
    if table.num_rows == 0:
        return []
    codes = table.column("code").to_numpy()
    meters = table.column("meter_id").to_pylist()
    energy = table.column("kwh").to_numpy(zero_copy_only=False).astype(float)
    temps = table.column("tavg_c").to_numpy(zero_copy_only=False).astype(float)
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [codes.size]])
    grid = candidate_grid(temps, config)
    results = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end - start < config.changepoint.min_days:
            continue
        meter_temps = temps[start:end]
        meter_energy = energy[start:end]
        fit = fit_meter(meter_temps, meter_energy, grid, config)
        fit.update(decompose(fit, meter_temps))
        fit["fleet"] = fleet_name
        fit["meter_id"] = meters[start]
        results.append(fit)
        if progress and index and index % 1000 == 0:
            progress(f"changepoint {fleet_name} meter {index}")
    return results


def store(connection, rows):
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_changepoint (
            fleet VARCHAR, meter_id VARCHAR, model VARCHAR, baseline_kwh_day DOUBLE,
            heat_slope DOUBLE, cool_slope DOUBLE, heat_cp DOUBLE, cool_cp DOUBLE,
            sse DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, cvrmse DOUBLE, aic DOUBLE, days BIGINT,
            accepted BOOLEAN, baseload_kwh_year DOUBLE, heating_kwh_year DOUBLE, cooling_kwh_year DOUBLE
        )
        """
    )
    payload = [
        (
            row["fleet"], row["meter_id"], row["model"], row["baseline_kwh_day"],
            row["heat_slope"], row["cool_slope"], row["heat_cp"], row["cool_cp"],
            row["sse"], row["r2"], row["adj_r2"], row["cvrmse"], row["aic"], row["days"],
            row["accepted"], row["baseload_kwh_year"], row["heating_kwh_year"], row["cooling_kwh_year"],
        )
        for row in rows
    ]
    if payload:
        connection.executemany(
            "INSERT INTO fact_changepoint VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", payload
        )


def build_changepoints(config: Config, progress=None) -> dict:
    connection = db.gold(config)
    rows = []
    for fleet_name in config.enabled_fleets:
        fleet_rows = fit_fleet(connection, config, fleet_name, progress)
        rows.extend(fleet_rows)
        if progress:
            progress(f"changepoint {fleet_name} fitted {len(fleet_rows)} meters")
    store(connection, rows)
    report = {}
    for fleet_name in config.enabled_fleets:
        models = dict(
            connection.sql(
                "SELECT model, count(*) FROM fact_changepoint WHERE fleet = ? GROUP BY 1 ORDER BY 2 DESC",
                params=[fleet_name],
            ).fetchall()
        )
        stats = connection.sql(
            """
            SELECT count(*), avg(r2), median(cvrmse), count(*) FILTER (WHERE accepted),
                   median(heat_cp), median(cool_cp), sum(heating_kwh_year), sum(cooling_kwh_year),
                   sum(baseload_kwh_year + heating_kwh_year + cooling_kwh_year)
            FROM fact_changepoint WHERE fleet = ?
            """,
            params=[fleet_name],
        ).fetchall()[0]
        total = stats[8] or 0.0
        report[fleet_name] = {
            "meters": stats[0],
            "models": models,
            "mean_r2": stats[1],
            "median_cvrmse": stats[2],
            "accepted": stats[3],
            "median_heat_cp": stats[4],
            "median_cool_cp": stats[5],
            "heating_share_of_energy": (stats[6] / total) if total else None,
            "cooling_share_of_energy": (stats[7] / total) if total else None,
            "modelled_kwh_year": total,
        }
    connection.close()
    path = config.paths.reports_dir / "changepoint_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
    return report
