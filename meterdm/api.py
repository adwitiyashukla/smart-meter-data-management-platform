from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query

from meterdm import db
from meterdm.config import load_config

CONFIG_PATH = os.environ.get("METERDM_CONFIG", "config.toml")

app = FastAPI(title="Smart Meter Data Management Platform", version="0.1.0")


def connection():
    config = load_config(CONFIG_PATH)
    try:
        return db.read_only(config)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def rows(query: str, parameters=None):
    handle = connection()
    try:
        result = handle.execute(query, parameters or [])
        names = [column[0] for column in result.description]
        return [dict(zip(names, record, strict=True)) for record in result.fetchall()]
    finally:
        handle.close()


@app.get("/health")
def health():
    config = load_config(CONFIG_PATH)
    return {"status": "ok", "gold": config.paths.gold_db.is_file(), "fleets": list(config.enabled_fleets)}


@app.get("/quality")
def quality():
    return rows("SELECT stage, check_name, subject, value, threshold, passed FROM fact_quality ORDER BY 1, 2, 3")


@app.get("/fleets")
def fleets():
    return rows(
        """
        SELECT fleet, count(*) AS meters, sum(readings) AS readings, min(first_local_date) AS first_day,
               max(last_local_date) AS last_day, sum(total_kwh) AS total_kwh
        FROM dim_meter GROUP BY 1 ORDER BY 1
        """
    )


@app.get("/meters")
def meters(fleet: str = Query(...), limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return rows(
        """
        SELECT m.fleet, m.meter_id, m.tariff_group, m.acorn_group, m.transformer_id, m.first_local_date,
               m.last_local_date, m.readings, m.total_kwh, c.model, c.heat_cp, c.cool_cp, c.r2,
               k.cluster, t.score
        FROM dim_meter m
        LEFT JOIN fact_changepoint c USING (fleet, meter_id)
        LEFT JOIN fact_cluster k USING (fleet, meter_id)
        LEFT JOIN fact_theft_score t USING (fleet, meter_id)
        WHERE m.fleet = ? ORDER BY m.meter_id LIMIT ? OFFSET ?
        """,
        [fleet, limit, offset],
    )


@app.get("/meters/{meter_id}")
def meter(meter_id: str, fleet: str = Query(...)):
    found = rows(
        """
        SELECT m.*, c.model, c.baseline_kwh_day, c.heat_slope, c.cool_slope, c.heat_cp, c.cool_cp,
               c.r2, c.cvrmse, c.baseload_kwh_year, c.heating_kwh_year, c.cooling_kwh_year,
               k.cluster, k.distance, t.score, t.label AS tamper_label, t.kind AS tamper_kind
        FROM dim_meter m
        LEFT JOIN fact_changepoint c USING (fleet, meter_id)
        LEFT JOIN fact_cluster k USING (fleet, meter_id)
        LEFT JOIN fact_theft_score t USING (fleet, meter_id)
        WHERE m.fleet = ? AND m.meter_id = ?
        """,
        [fleet, meter_id],
    )
    if not found:
        raise HTTPException(status_code=404, detail=f"meter {meter_id} not found in fleet {fleet}")
    return found[0]


@app.get("/meters/{meter_id}/daily")
def meter_daily(meter_id: str, fleet: str = Query(...), limit: int = Query(400, ge=1, le=2000)):
    return rows(
        """
        SELECT d.local_date, d.kwh, d.night_kwh, d.actual_slots, d.filled_slots, x.expected_slots, w.tavg_c
        FROM fact_daily_read d
        JOIN dim_date x USING (fleet, local_date)
        LEFT JOIN fact_weather_day w USING (fleet, local_date)
        WHERE d.fleet = ? AND d.meter_id = ? ORDER BY d.local_date DESC LIMIT ?
        """,
        [fleet, meter_id, limit],
    )


@app.get("/meters/{meter_id}/readings")
def meter_readings(meter_id: str, fleet: str = Query(...), limit: int = Query(1000, ge=1, le=20000)):
    return rows(
        """
        SELECT ts_utc, ts_local, local_date, period, kwh, kwh_raw, status, rule_id, method
        FROM fact_interval_sample WHERE fleet = ? AND meter_id = ? ORDER BY ts_utc DESC LIMIT ?
        """,
        [fleet, meter_id, limit],
    )


@app.get("/vee/summary")
def vee_summary():
    return rows(
        """
        SELECT fleet, rule_id, method, count(*) AS readings
        FROM fact_vee_edit GROUP BY 1, 2, 3 ORDER BY 4 DESC
        """
    )


@app.get("/vee/flags")
def vee_flags(fleet: str = Query(...), limit: int = Query(100, ge=1, le=1000)):
    return rows(
        """
        SELECT meter_id, local_date, rule_id, day_kwh, reference_kwh, present, expected_slots
        FROM fact_day_flag WHERE fleet = ? ORDER BY local_date DESC LIMIT ?
        """,
        [fleet, limit],
    )


@app.get("/theft/queue")
def theft_queue(fleet: str = Query(...), limit: int = Query(50, ge=1, le=500)):
    return rows(
        """
        SELECT meter_id, transformer_id, score, label, kind, level_ratio, night_ratio, residual_ratio,
               cusum_negative, transformer_loss
        FROM fact_theft_score WHERE fleet = ? ORDER BY score DESC LIMIT ?
        """,
        [fleet, limit],
    )


@app.get("/clusters")
def clusters(fleet: str = Query(...)):
    return rows(
        """
        SELECT cluster, any_value(label) AS label, any_value(meters) AS meters,
               list(share ORDER BY is_weekend, period) AS profile
        FROM dim_cluster_profile WHERE fleet = ? GROUP BY 1 ORDER BY 1
        """,
        [fleet],
    )


@app.get("/dtou")
def dtou():
    return rows("SELECT metric, value FROM fact_dtou_estimate ORDER BY 1")


@app.get("/forecast/metrics")
def forecast_metrics():
    return rows("SELECT fleet, metric, value FROM fact_forecast_metric ORDER BY 1, 2")


@app.get("/forecast/backtest")
def forecast_backtest(fleet: str = Query(...), origin: int = Query(0, ge=0)):
    return rows(
        """
        SELECT ts_utc, horizon, actual, predicted, naive, q10, q90
        FROM fact_forecast_backtest WHERE fleet = ? AND origin = ? ORDER BY horizon
        """,
        [fleet, origin],
    )
