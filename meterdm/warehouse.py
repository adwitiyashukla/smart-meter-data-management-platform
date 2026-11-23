from __future__ import annotations

import json

from meterdm import db, vee
from meterdm.calendar_data import FLEET_FILES
from meterdm.config import Config

TABLES = (
    "dim_meter",
    "dim_date",
    "dim_tariff",
    "fact_weather_hour",
    "fact_weather_day",
    "fact_daily_read",
    "fact_period_profile",
    "fact_fleet_hourly",
    "fact_trial_window",
    "fact_interval_sample",
    "fact_vee_edit",
    "fact_day_flag",
    "fact_quality",
)


class WarehouseError(RuntimeError):
    pass


def silver_pattern(config: Config, fleet_name: str, leaf: str) -> str:
    return (config.paths.silver_dir / fleet_name / "**" / f"{leaf}.parquet").as_posix()


def _union(config: Config, leaf: str) -> str:
    parts = []
    for fleet_name in config.enabled_fleets:
        pattern = silver_pattern(config, fleet_name, leaf)
        parts.append(f"SELECT * FROM read_parquet({db.quote(pattern)})")
    if not parts:
        raise WarehouseError("no enabled fleets to load")
    return " UNION ALL ".join(parts)


def _holiday_union(config: Config) -> str:
    parts = []
    for fleet_name in config.enabled_fleets:
        path = config.paths.refdata_dir / FLEET_FILES[fleet_name]
        parts.append(
            f"SELECT '{fleet_name}' AS fleet, CAST(date AS DATE) AS local_date "
            f"FROM read_csv({db.quote(path)}, header=true)"
        )
    return " UNION ALL ".join(parts)


def _expected_slots_expression(config: Config) -> str:
    branches = []
    for fleet_name in config.enabled_fleets:
        fleet = config.fleet(fleet_name)
        zone = fleet.timezone
        branches.append(
            f"""WHEN fleet = '{fleet_name}' THEN CAST(date_diff('minute',
                CAST(CAST(local_date AS TIMESTAMP) AT TIME ZONE '{zone}' AS TIMESTAMP),
                CAST(CAST(local_date + INTERVAL 1 DAY AS TIMESTAMP) AT TIME ZONE '{zone}' AS TIMESTAMP)
            ) / {fleet.interval_minutes} AS INTEGER)"""
        )
    return "CASE " + " ".join(branches) + " END"


def build_reads(connection, config: Config, progress=None):
    connection.execute(f"CREATE OR REPLACE VIEW v_readings AS {_union(config, 'readings')}")
    night_terms = []
    for fleet_name in config.enabled_fleets:
        fleet = config.fleet(fleet_name)
        per_hour = 60 // fleet.interval_minutes
        low = config.theft.night_start_hour * per_hour
        high = config.theft.night_end_hour * per_hour
        night_terms.append(f"(fleet = '{fleet_name}' AND period > {low} AND period <= {high})")
    night = " OR ".join(night_terms)
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE fact_daily_read AS
        SELECT fleet, meter_id, local_date,
               sum(kwh) FILTER (WHERE status <> '{vee.STATUS_MISSING}') AS kwh,
               sum(kwh) FILTER (WHERE status <> '{vee.STATUS_MISSING}' AND ({night})) AS night_kwh,
               count(*) FILTER (WHERE status = '{vee.STATUS_ACTUAL}') AS actual_slots,
               count(*) FILTER (WHERE status <> '{vee.STATUS_MISSING}') AS filled_slots,
               count(*) AS slots,
               max(CAST(dst_aggregated AS INTEGER)) AS aggregated
        FROM v_readings GROUP BY 1, 2, 3
        """
    )
    if progress:
        progress("gold fact_daily_read built")
    profile_terms = " OR ".join(
        f"(fleet = '{name}' AND period <= {config.fleet(name).intervals_per_day})"
        for name in config.enabled_fleets
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE fact_period_profile AS
        SELECT fleet, meter_id, dayofweek(local_date) IN (0, 6) AS is_weekend, period,
               avg(kwh) AS mean_kwh, count(*) AS observations
        FROM v_readings
        WHERE status <> '{vee.STATUS_MISSING}' AND ({profile_terms})
        GROUP BY 1, 2, 3, 4
        """
    )
    if progress:
        progress("gold fact_period_profile built")
    meter_terms = " ".join(
        f"WHEN fleet = '{name}' THEN count(*) * {config.fleet(name).interval_minutes} / 60.0"
        for name in config.enabled_fleets
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE fact_fleet_hourly AS
        SELECT fleet, date_trunc('hour', ts_utc) AS ts_utc, sum(kwh) AS kwh,
               CASE {meter_terms} END AS meters
        FROM v_readings WHERE status <> '{vee.STATUS_MISSING}' GROUP BY 1, 2
        """
    )


def build_dimensions(connection, config: Config):
    households = config.paths.bronze_dir / "london_households.parquet"
    if households.is_file():
        household_sql = f"""
            SELECT trim("LCLid") AS meter_id, trim("Acorn") AS acorn, trim("Acorn_grouped") AS acorn_group
            FROM read_parquet({db.quote(households)})
        """
    else:
        household_sql = """
            SELECT CAST(NULL AS VARCHAR) AS meter_id, CAST(NULL AS VARCHAR) AS acorn,
                   CAST(NULL AS VARCHAR) AS acorn_group WHERE false
        """
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE dim_meter AS
        WITH spans AS (
            SELECT fleet, meter_id,
                   any_value(tariff_group) AS tariff_group,
                   min(local_date) AS first_local_date,
                   max(local_date) AS last_local_date,
                   count(*) AS readings,
                   count(*) FILTER (WHERE status = '{vee.STATUS_ACTUAL}') AS actual_readings,
                   sum(kwh) AS total_kwh
            FROM v_readings GROUP BY 1, 2
        ), joined AS (
            SELECT s.*, h.acorn, coalesce(h.acorn_group, 'unclassified') AS acorn_group
            FROM spans s LEFT JOIN ({household_sql}) h USING (meter_id)
        )
        SELECT *,
               fleet || '-' || acorn_group || '-' || CAST(
                   (row_number() OVER (PARTITION BY fleet, acorn_group ORDER BY meter_id) - 1)
                   // {config.theft.transformer_size} AS INTEGER
               ) AS transformer_id,
               date_diff('day', first_local_date, last_local_date) + 1 AS span_days
        FROM joined
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE dim_date AS
        WITH days AS (SELECT DISTINCT fleet, local_date FROM fact_daily_read),
             holidays AS ({_holiday_union(config)})
        SELECT d.fleet, d.local_date,
               year(d.local_date) AS year, month(d.local_date) AS month,
               dayofweek(d.local_date) AS day_of_week,
               dayofweek(d.local_date) IN (0, 6) AS is_weekend,
               (h.local_date IS NOT NULL) AS is_holiday,
               {_expected_slots_expression(config)} AS expected_slots
        FROM days d LEFT JOIN holidays h USING (fleet, local_date)
        """
    )
    tariffs = config.paths.bronze_dir / "london_tariffs.parquet"
    if tariffs.is_file():
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE dim_tariff AS
            SELECT 'london' AS fleet,
                   CAST(strptime(substr(trim("TariffDateTime"), 1, 19), '%Y-%m-%d %H:%M:%S') AS TIMESTAMP) AS ts_local,
                   trim("Tariff") AS label
            FROM read_parquet({db.quote(tariffs)})
            WHERE strptime(substr(trim("TariffDateTime"), 1, 19), '%Y-%m-%d %H:%M:%S') IS NOT NULL
            """
        )
    else:
        connection.execute(
            """
            CREATE OR REPLACE TABLE dim_tariff AS
            SELECT CAST(NULL AS VARCHAR) AS fleet, CAST(NULL AS TIMESTAMP) AS ts_local,
                   CAST(NULL AS VARCHAR) AS label WHERE false
            """
        )


def build_weather(connection, config: Config):
    parts = []
    for fleet_name in config.enabled_fleets:
        path = config.paths.bronze_dir / "weather" / f"{fleet_name}.parquet"
        if not path.is_file():
            raise WarehouseError(f"weather bronze missing for {fleet_name}: {path}")
        parts.append(f"SELECT '{fleet_name}' AS fleet, ts_utc, temp_c FROM read_parquet({db.quote(path)})")
    connection.execute("CREATE OR REPLACE TABLE fact_weather_hour AS " + " UNION ALL ".join(parts))
    day_parts = []
    for fleet_name in config.enabled_fleets:
        zone = config.fleet(fleet_name).timezone
        day_parts.append(
            f"""
            SELECT '{fleet_name}' AS fleet,
                   CAST((ts_utc AT TIME ZONE 'UTC' AT TIME ZONE '{zone}') AS DATE) AS local_date,
                   avg(temp_c) AS tavg_c, min(temp_c) AS tmin_c, max(temp_c) AS tmax_c, count(*) AS hours
            FROM fact_weather_hour WHERE fleet = '{fleet_name}' GROUP BY 1, 2
            """
        )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE fact_weather_day AS
        SELECT *, greatest(0.0, {config.gold.hdd_base_c} - tavg_c) AS hdd,
                  greatest(0.0, tavg_c - {config.gold.cdd_base_c}) AS cdd
        FROM ({" UNION ALL ".join(day_parts)})
        """
    )


def build_trial(connection, config: Config):
    frequency = connection.execute(
        """
        SELECT CAST(ts_local AS TIME) AS slot, count(*) AS days
        FROM dim_tariff WHERE label = 'High' GROUP BY 1
        """
    ).fetchall()
    busiest = max((row[1] for row in frequency), default=0)
    slots = sorted(str(row[0]) for row in frequency if row[1] == busiest)
    if not slots:
        connection.execute(
            """
            CREATE OR REPLACE TABLE fact_trial_window AS
            SELECT CAST(NULL AS VARCHAR) AS fleet, CAST(NULL AS VARCHAR) AS meter_id,
                   CAST(NULL AS DATE) AS local_date, CAST(NULL AS VARCHAR) AS tariff_group,
                   CAST(NULL AS DOUBLE) AS window_kwh, CAST(NULL AS DOUBLE) AS day_kwh,
                   CAST(NULL AS BOOLEAN) AS event_day, CAST(NULL AS BIGINT) AS window_slots
            WHERE false
            """
        )
        return {"event_slots": [], "event_days": 0}
    window = ", ".join(f"TIME '{value}'" for value in slots)
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE dim_event_day AS
        SELECT CAST(ts_local - INTERVAL 30 MINUTE AS DATE) AS local_date
        FROM dim_tariff
        WHERE label = 'High' AND CAST(ts_local AS TIME) IN ({window})
        GROUP BY 1
        HAVING count(*) = {len(slots)}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE fact_trial_window AS
        SELECT r.fleet, r.meter_id, r.local_date, any_value(r.tariff_group) AS tariff_group,
               sum(r.kwh) FILTER (WHERE CAST(r.ts_local AS TIME) IN ({window})) AS window_kwh,
               sum(r.kwh) AS day_kwh,
               (e.local_date IS NOT NULL) AS event_day,
               count(*) FILTER (WHERE CAST(r.ts_local AS TIME) IN ({window})) AS window_slots
        FROM v_readings r
        LEFT JOIN dim_event_day e ON r.local_date = e.local_date
        WHERE r.fleet = 'london' AND r.status <> '{vee.STATUS_MISSING}'
          AND r.local_date BETWEEN (SELECT min(local_date) FROM dim_event_day)
                               AND (SELECT max(local_date) FROM dim_event_day)
        GROUP BY 1, 2, 3, e.local_date
        """
    )
    event_days = connection.execute("SELECT count(*) FROM dim_event_day").fetchone()[0]
    return {"event_slots": slots, "event_days": event_days}


def build_audit(connection, config: Config):
    connection.execute(f"CREATE OR REPLACE TABLE fact_vee_edit AS {_union(config, 'edits')}")
    connection.execute(f"CREATE OR REPLACE TABLE fact_day_flag AS {_union(config, 'day_flags')}")


def build_interval_sample(connection, config: Config):
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE dim_sample_meter AS
        SELECT fleet, meter_id FROM (
            SELECT fleet, meter_id,
                   row_number() OVER (PARTITION BY fleet ORDER BY hash(meter_id)) AS pick
            FROM dim_meter
        ) WHERE pick <= {config.gold.interval_sample_meters}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_interval_sample AS
        SELECT r.fleet, r.meter_id, r.ts_utc, r.ts_local, r.local_date, r.period, r.kwh, r.kwh_raw,
               r.status, r.rule_id, r.method
        FROM v_readings r JOIN dim_sample_meter s USING (fleet, meter_id)
        """
    )


def build_quality(connection, config: Config) -> list:
    checks = []

    def record(stage, name, subject, value, threshold, passed):
        checks.append((stage, name, subject, float(value), float(threshold), bool(passed)))

    rules = config.quality
    rows = connection.execute(
        """
        SELECT fleet, count(*) AS readings, count(DISTINCT meter_id) AS meters,
               count(DISTINCT local_date) AS days
        FROM fact_daily_read GROUP BY 1
        """
    ).fetchall()
    for fleet_name, _readings, meters, days in rows:
        record("coverage", "meters_present", fleet_name, meters, rules.min_meters, meters >= rules.min_meters)
        record("coverage", "days_present", fleet_name, days, rules.min_days, days >= rules.min_days)
    missing = connection.execute(
        """
        SELECT fleet, sum(slots - filled_slots) * 1.0 / nullif(sum(slots), 0) FROM fact_daily_read GROUP BY 1
        """
    ).fetchall()
    for fleet_name, ratio in missing:
        value = float(ratio or 0.0)
        record("vee", "unrecovered_ratio", fleet_name, value, rules.max_missing_ratio, value <= rules.max_missing_ratio)
    negatives = connection.execute(
        """
        SELECT fleet, count(*) FILTER (WHERE kwh < 0) * 1.0 / nullif(count(*), 0) FROM fact_interval_sample GROUP BY 1
        """
    ).fetchall()
    for fleet_name, ratio in negatives:
        value = float(ratio or 0.0)
        record("gold", "negative_ratio", fleet_name, value, rules.max_negative_ratio, value <= rules.max_negative_ratio)
    duplicate = connection.execute(
        """
        SELECT fleet, count(*) FROM (
            SELECT fleet, meter_id, ts_utc, count(*) AS n FROM fact_interval_sample GROUP BY 1, 2, 3 HAVING n > 1
        ) GROUP BY 1
        """
    ).fetchall()
    seen = {row[0]: row[1] for row in duplicate}
    for fleet_name in config.enabled_fleets:
        value = float(seen.get(fleet_name, 0))
        record("gold", "duplicate_instants", fleet_name, value, 0.0, value == 0.0)
    orphan = connection.execute(
        """
        SELECT count(*) FROM fact_daily_read d LEFT JOIN dim_date x USING (fleet, local_date)
        WHERE x.local_date IS NULL
        """
    ).fetchone()[0]
    record("gold", "days_without_dimension", "all", orphan, 0.0, orphan == 0)
    weather_gap = connection.execute(
        """
        SELECT count(*) FROM (SELECT DISTINCT fleet, local_date FROM fact_daily_read) d
        LEFT JOIN fact_weather_day w USING (fleet, local_date)
        WHERE w.local_date IS NULL
        """
    ).fetchone()[0]
    record("gold", "days_without_weather", "all", weather_gap, 0.0, weather_gap == 0)
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_quality (
            run_id VARCHAR, stage VARCHAR, check_name VARCHAR, subject VARCHAR,
            value DOUBLE, threshold DOUBLE, passed BOOLEAN
        )
        """
    )
    connection.executemany(
        "INSERT INTO fact_quality VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(config.run.run_id, *check) for check in checks],
    )
    return checks


def build_gold(config: Config, progress=None) -> dict:
    config.paths.ensure()
    if config.paths.gold_db.exists():
        config.paths.gold_db.unlink()
    connection = db.gold(config)
    build_reads(connection, config, progress)
    if progress:
        progress("gold reads built")
    build_dimensions(connection, config)
    build_weather(connection, config)
    if progress:
        progress("gold dimensions and weather built")
    trial = build_trial(connection, config)
    build_audit(connection, config)
    build_interval_sample(connection, config)
    if progress:
        progress("gold audit, trial and sample built")
    checks = build_quality(connection, config)
    counts = {}
    for table in TABLES:
        counts[table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    connection.close()
    summary = {
        "tables": counts,
        "trial": trial,
        "quality_checks": len(checks),
        "quality_failures": sum(1 for check in checks if not check[5]),
        "failed": [
            {"stage": check[0], "check": check[1], "subject": check[2], "value": check[3], "threshold": check[4]}
            for check in checks
            if not check[5]
        ],
    }
    path = config.paths.reports_dir / "gold_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    return summary
