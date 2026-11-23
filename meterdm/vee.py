from __future__ import annotations

import json

from meterdm import db
from meterdm.config import Config

RULE_MISSING = "R01_missing"
RULE_NEGATIVE = "R02_negative"
RULE_SPIKE = "R03_spike"
RULE_STUCK = "R04_stuck"
RULE_ZERO_RUN = "R05_zero_run"
RULE_HIGH_DAY = "R06_high_day"
RULE_LOW_DAY = "R07_low_day"
RULE_REGISTER = "R08_register"
RULE_DUPLICATE = "R09_duplicate"
RULE_OUT_OF_RANGE = "R10_out_of_range"
RULE_NONEXISTENT = "R11_nonexistent_local"
RULE_OUT_OF_SERVICE = "R12_out_of_service"

VALUE_RULES = (RULE_OUT_OF_RANGE, RULE_NEGATIVE, RULE_SPIKE, RULE_STUCK, RULE_ZERO_RUN)
DAY_RULES = (RULE_HIGH_DAY, RULE_LOW_DAY, RULE_REGISTER)
STRUCTURAL_RULES = (RULE_DUPLICATE, RULE_NONEXISTENT, RULE_OUT_OF_SERVICE)

STATUS_ACTUAL = "actual"
STATUS_ESTIMATED = "estimated"
STATUS_EDITED = "edited"
STATUS_MISSING = "missing"

METHOD_INTERPOLATE = "interpolate"
METHOD_LIKE_DAY = "like_day"
METHOD_NONE = "none"


class VeeError(RuntimeError):
    pass


def _local(expression: str, zone: str) -> str:
    return f"(({expression}) AT TIME ZONE 'UTC' AT TIME ZONE '{zone}')"


def _day_start_utc(zone: str) -> str:
    return f"CAST(CAST(local_date AS TIMESTAMP) AT TIME ZONE '{zone}' AS TIMESTAMP)"


def batch_paths(config: Config, fleet_name: str):
    root = config.paths.bronze_dir / fleet_name
    return sorted(path for path in root.glob("meter_batch=*") if path.is_dir())


def prepare_batch(connection, config: Config, fleet_name: str, batch_dir):
    fleet = config.fleet(fleet_name)
    zone = fleet.timezone
    minutes = fleet.interval_minutes
    settings = config.vee
    pattern = (batch_dir / "*.parquet").as_posix()

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_resolved AS
        WITH ranked AS (
            SELECT meter_id, ts_local, kwh AS kwh_raw, tariff_group,
                   row_number() OVER (PARTITION BY meter_id, ts_local ORDER BY kwh NULLS LAST) AS occurrence,
                   count(*) OVER (PARTITION BY meter_id, ts_local) AS occurrences
            FROM read_parquet({db.quote(pattern)})
        ), stamped AS (
            SELECT *, ts_local AT TIME ZONE '{zone}' AS inst_late FROM ranked
        )
        SELECT meter_id, ts_local, kwh_raw, tariff_group, occurrence, occurrences,
               (((inst_late - INTERVAL 1 HOUR) AT TIME ZONE '{zone}') = ts_local) AS ambiguous,
               ((inst_late AT TIME ZONE '{zone}') <> ts_local) AS nonexistent,
               inst_late
        FROM stamped
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_indexed AS
        WITH accepted AS (
            SELECT meter_id, ts_local, kwh_raw, tariff_group,
                   CAST(
                       CASE WHEN ambiguous AND occurrence = 1 THEN inst_late - INTERVAL 1 HOUR ELSE inst_late END
                       AS TIMESTAMP
                   ) AS ts_utc,
                   (ambiguous AND occurrences = 1) AS dst_aggregated
            FROM t_resolved
            WHERE NOT nonexistent AND (occurrence = 1 OR (occurrence = 2 AND ambiguous))
        )
        SELECT *,
               CAST({_local("ts_utc", zone)} - INTERVAL {minutes} MINUTE AS DATE) AS local_date
        FROM accepted
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_scoped AS
        SELECT meter_id, ts_utc, ts_local, local_date,
               CAST(date_diff('minute', {_day_start_utc(zone)}, ts_utc) // {minutes} AS INTEGER) AS period,
               kwh_raw, tariff_group, dst_aggregated
        FROM t_indexed
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_service AS
        SELECT meter_id,
               min(ts_utc) FILTER (WHERE kwh_raw > 0) AS first_live,
               max(ts_utc) FILTER (WHERE kwh_raw > 0) AS last_live
        FROM t_scoped GROUP BY 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_present AS
        SELECT s.* FROM t_scoped s JOIN t_service v USING (meter_id)
        WHERE v.first_live IS NOT NULL AND s.ts_utc >= v.first_live AND s.ts_utc <= v.last_live
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_rejected AS
        SELECT meter_id, ts_local, kwh_raw,
               CASE WHEN nonexistent THEN '{RULE_NONEXISTENT}' ELSE '{RULE_DUPLICATE}' END AS rule_id
        FROM t_resolved
        WHERE nonexistent OR NOT (occurrence = 1 OR (occurrence = 2 AND ambiguous))
        """
    )
    connection.execute(
        f"""
        INSERT INTO t_rejected
        SELECT s.meter_id, s.ts_local, s.kwh_raw, '{RULE_OUT_OF_SERVICE}'
        FROM t_scoped s LEFT JOIN t_service v USING (meter_id)
        WHERE v.first_live IS NULL OR s.ts_utc < v.first_live OR s.ts_utc > v.last_live
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_fill AS
        WITH ordered AS (
            SELECT meter_id, ts_utc, tariff_group,
                   lead(ts_utc) OVER (PARTITION BY meter_id ORDER BY ts_utc) AS next_ts
            FROM t_present
        ), gaps AS (
            SELECT meter_id, ts_utc, next_ts, tariff_group,
                   CAST(date_diff('minute', ts_utc, next_ts) // {minutes} AS BIGINT) - 1 AS missing_slots
            FROM ordered WHERE next_ts IS NOT NULL
        )
        SELECT meter_id, tariff_group,
               unnest(generate_series(
                   ts_utc + INTERVAL {minutes} MINUTE,
                   next_ts - INTERVAL {minutes} MINUTE,
                   INTERVAL {minutes} MINUTE
               )) AS ts_utc
        FROM gaps
        WHERE missing_slots BETWEEN 1 AND {settings.max_estimate_intervals}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_stats_first AS
        SELECT meter_id, median(kwh_raw) AS med, avg(kwh_raw) AS mean_kwh, count(kwh_raw) AS present
        FROM t_present WHERE kwh_raw IS NOT NULL GROUP BY 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_stats AS
        SELECT p.meter_id, any_value(s.med) AS med, any_value(s.mean_kwh) AS mean_kwh,
               any_value(s.present) AS present, median(abs(p.kwh_raw - s.med)) AS mad
        FROM t_present p JOIN t_stats_first s USING (meter_id)
        WHERE p.kwh_raw IS NOT NULL
        GROUP BY 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_profile_raw AS
        SELECT meter_id, dayofweek(local_date) AS dow, period, median(kwh_raw) AS profile_kwh
        FROM t_present WHERE kwh_raw IS NOT NULL GROUP BY 1, 2, 3
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_runs AS
        WITH marked AS (
            SELECT meter_id, ts_utc, kwh_raw,
                   CASE WHEN kwh_raw IS NOT DISTINCT FROM lag(kwh_raw) OVER w THEN 0 ELSE 1 END AS starts
            FROM t_present
            WINDOW w AS (PARTITION BY meter_id ORDER BY ts_utc)
        ), islands AS (
            SELECT *, sum(starts) OVER (
                PARTITION BY meter_id ORDER BY ts_utc ROWS UNBOUNDED PRECEDING
            ) AS island
            FROM marked
        )
        SELECT meter_id, ts_utc, island,
               count(*) OVER (PARTITION BY meter_id, island) AS run_length
        FROM islands
        """
    )


def flag_values(connection, config: Config, fleet_name: str):
    fleet = config.fleet(fleet_name)
    settings = config.vee
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_flags AS
        SELECT p.meter_id, p.ts_utc,
            CASE
                WHEN p.kwh_raw IS NULL THEN NULL
                WHEN p.kwh_raw > {fleet.physical_max_kwh_per_interval} THEN '{RULE_OUT_OF_RANGE}'
                WHEN p.kwh_raw < 0 THEN '{RULE_NEGATIVE}'
                WHEN {str(settings.spike.enabled).lower()}
                     AND pr.profile_kwh > 0
                     AND p.kwh_raw > {settings.spike.profile_ratio} * pr.profile_kwh
                     AND p.kwh_raw > {settings.spike.floor_kwh} THEN '{RULE_SPIKE}'
                WHEN {str(settings.zero_run.enabled).lower()}
                     AND p.kwh_raw = 0
                     AND r.run_length >= {settings.zero_run.run_length}
                     AND s.mean_kwh > {settings.zero_run.min_history_mean_kwh} THEN '{RULE_ZERO_RUN}'
                WHEN {str(settings.stuck.enabled).lower()}
                     AND p.kwh_raw > {settings.stuck.min_kwh}
                     AND r.run_length >= {settings.stuck.run_length} THEN '{RULE_STUCK}'
                ELSE NULL
            END AS rule_id
        FROM t_present p
        LEFT JOIN t_stats s USING (meter_id)
        LEFT JOIN t_runs r USING (meter_id, ts_utc)
        LEFT JOIN t_profile_raw pr
          ON p.meter_id = pr.meter_id AND dayofweek(p.local_date) = pr.dow AND p.period = pr.period
        """
    )


def flag_days(connection, config: Config, fleet_name: str, register_available: bool):
    fleet = config.fleet(fleet_name)
    settings = config.vee.day_level
    zone = fleet.timezone
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_daily AS
        SELECT meter_id, local_date,
               sum(kwh) FILTER (WHERE status <> '{STATUS_MISSING}') AS day_kwh,
               count(*) FILTER (WHERE status = '{STATUS_ACTUAL}') AS actual_slots,
               count(*) FILTER (WHERE status <> '{STATUS_MISSING}') AS filled_slots,
               count(*) AS slots,
               CAST(date_diff('minute',
                    {_day_start_utc(zone)},
                    CAST(CAST(local_date + INTERVAL 1 DAY AS TIMESTAMP) AT TIME ZONE '{zone}' AS TIMESTAMP)
               ) // {fleet.interval_minutes} AS INTEGER) AS expected_slots,
               max(CAST(dst_aggregated AS INTEGER)) AS aggregated
        FROM t_silver GROUP BY 1, 2
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_like_day AS
        SELECT meter_id, dayofweek(local_date) AS dow, median(day_kwh) AS med_day, count(*) AS like_days
        FROM t_daily WHERE actual_slots = expected_slots GROUP BY 1, 2
        HAVING count(*) >= {settings.min_like_days}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_day_flags AS
        SELECT d.meter_id, d.local_date,
               CASE
                   WHEN {str(settings.enabled).lower()} AND l.med_day > 0
                        AND d.filled_slots = d.expected_slots
                        AND d.day_kwh > {settings.high_ratio} * l.med_day THEN '{RULE_HIGH_DAY}'
                   WHEN {str(settings.enabled).lower()} AND l.med_day > 0
                        AND d.filled_slots = d.expected_slots
                        AND d.day_kwh < {settings.low_ratio} * l.med_day THEN '{RULE_LOW_DAY}'
                   ELSE NULL
               END AS rule_id,
               d.day_kwh, l.med_day, d.filled_slots AS present, d.expected_slots, d.aggregated
        FROM t_daily d
        LEFT JOIN t_like_day l ON d.meter_id = l.meter_id AND dayofweek(d.local_date) = l.dow
        """
    )
    if not register_available:
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE t_register_flags AS
            SELECT CAST(NULL AS VARCHAR) AS meter_id, CAST(NULL AS DATE) AS local_date,
                   CAST(NULL AS DOUBLE) AS day_kwh, CAST(NULL AS DOUBLE) AS register_kwh,
                   CAST(NULL AS BIGINT) AS present, CAST(NULL AS BIGINT) AS register_count
            WHERE false
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE t_estimation_error AS
            SELECT CAST(NULL AS VARCHAR) AS meter_id, CAST(NULL AS DATE) AS local_date,
                   CAST(NULL AS DOUBLE) AS day_kwh, CAST(NULL AS DOUBLE) AS register_kwh,
                   CAST(NULL AS BIGINT) AS estimated_slots, CAST(NULL AS DOUBLE) AS relative_error
            WHERE false
            """
        )
        return
    register = config.paths.bronze_dir / "london_daily.parquet"
    rules = config.vee.register
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_register_flags AS
        WITH book AS (
            SELECT trim("LCLid") AS meter_id, CAST(trim("day") AS DATE) AS local_date,
                   TRY_CAST(trim("energy_sum") AS DOUBLE) AS register_kwh,
                   TRY_CAST(trim("energy_count") AS BIGINT) AS register_count
            FROM read_parquet({db.quote(register)})
        )
        SELECT d.meter_id, d.local_date, d.day_kwh, b.register_kwh, d.filled_slots AS present, b.register_count
        FROM t_daily d JOIN book b USING (meter_id, local_date)
        WHERE d.aggregated = 0
          AND (
              d.filled_slots <> b.register_count
              OR (
                  d.actual_slots = d.expected_slots
                  AND abs(d.day_kwh - b.register_kwh) >
                      greatest({rules.tolerance_kwh}, {rules.tolerance_ratio} * abs(b.register_kwh))
              )
          )
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_estimation_error AS
        WITH book AS (
            SELECT trim("LCLid") AS meter_id, CAST(trim("day") AS DATE) AS local_date,
                   TRY_CAST(trim("energy_sum") AS DOUBLE) AS register_kwh,
                   TRY_CAST(trim("energy_count") AS BIGINT) AS register_count
            FROM read_parquet({db.quote(register)})
        )
        SELECT d.meter_id, d.local_date, d.day_kwh, b.register_kwh,
               d.expected_slots - d.actual_slots AS estimated_slots,
               abs(d.day_kwh - b.register_kwh) / nullif(abs(b.register_kwh), 0) AS relative_error
        FROM t_daily d JOIN book b USING (meter_id, local_date)
        WHERE d.aggregated = 0
          AND d.actual_slots < d.expected_slots
          AND d.filled_slots = d.expected_slots
          AND b.register_count = d.expected_slots
          AND b.register_kwh > 0
        """
    )


def estimate(connection, config: Config, fleet_name: str):
    fleet = config.fleet(fleet_name)
    settings = config.vee
    zone = fleet.timezone
    minutes = fleet.interval_minutes
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_series AS
        SELECT p.meter_id, p.ts_utc, p.ts_local, p.local_date, p.period, p.kwh_raw, p.tariff_group,
               p.dst_aggregated, false AS generated, f.rule_id AS value_rule,
               CASE WHEN f.rule_id IS NULL THEN p.kwh_raw ELSE NULL END AS kwh_clean
        FROM t_present p LEFT JOIN t_flags f USING (meter_id, ts_utc)
        UNION ALL
        SELECT g.meter_id, g.ts_utc,
               CAST({_local("g.ts_utc", zone)} AS TIMESTAMP) AS ts_local,
               CAST({_local("g.ts_utc", zone)} - INTERVAL {minutes} MINUTE AS DATE) AS local_date,
               CAST(NULL AS INTEGER) AS period,
               CAST(NULL AS DOUBLE) AS kwh_raw, g.tariff_group, false AS dst_aggregated, true AS generated,
               CAST(NULL AS VARCHAR) AS value_rule, CAST(NULL AS DOUBLE) AS kwh_clean
        FROM t_fill g
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_profile AS
        SELECT meter_id, dayofweek(local_date) AS dow, period, median(kwh_clean) AS profile_kwh
        FROM t_series WHERE kwh_clean IS NOT NULL AND period IS NOT NULL GROUP BY 1, 2, 3
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_filled AS
        WITH marked AS (
            SELECT *,
                   CASE WHEN (kwh_clean IS NULL) IS NOT DISTINCT FROM (lag(kwh_clean) OVER w IS NULL)
                        THEN 0 ELSE 1 END AS starts
            FROM t_series
            WINDOW w AS (PARTITION BY meter_id ORDER BY ts_utc)
        ), islands AS (
            SELECT *, sum(starts) OVER (
                PARTITION BY meter_id ORDER BY ts_utc ROWS UNBOUNDED PRECEDING
            ) AS island
            FROM marked
        ), sized AS (
            SELECT *, count(*) OVER (PARTITION BY meter_id, island) AS island_length
            FROM islands
        )
        SELECT meter_id, ts_utc, ts_local, local_date,
               CASE WHEN period IS NOT NULL THEN period
                    ELSE CAST(date_diff('minute', {_day_start_utc(zone)}, ts_utc) // {minutes} AS INTEGER)
               END AS period,
               kwh_raw, tariff_group, dst_aggregated, generated, value_rule, kwh_clean, island_length,
               last_value(kwh_clean IGNORE NULLS) OVER back AS prev_value,
               last_value(CASE WHEN kwh_clean IS NOT NULL THEN ts_utc END IGNORE NULLS) OVER back AS prev_ts,
               first_value(kwh_clean IGNORE NULLS) OVER ahead AS next_value,
               first_value(CASE WHEN kwh_clean IS NOT NULL THEN ts_utc END IGNORE NULLS) OVER ahead AS next_ts
        FROM sized
        WINDOW back AS (PARTITION BY meter_id ORDER BY ts_utc ROWS UNBOUNDED PRECEDING),
               ahead AS (PARTITION BY meter_id ORDER BY ts_utc ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_silver AS
        WITH joined AS (
            SELECT f.*, p.profile_kwh,
                   CASE
                       WHEN f.prev_ts IS NULL OR f.next_ts IS NULL OR f.next_ts = f.prev_ts THEN NULL
                       ELSE f.prev_value + (f.next_value - f.prev_value)
                            * (epoch(f.ts_utc) - epoch(f.prev_ts)) / (epoch(f.next_ts) - epoch(f.prev_ts))
                   END AS interpolated
            FROM t_filled f
            LEFT JOIN t_profile p
              ON f.meter_id = p.meter_id AND dayofweek(f.local_date) = p.dow AND f.period = p.period
        )
        SELECT meter_id, ts_utc, ts_local, local_date, period, tariff_group, kwh_raw, dst_aggregated, generated,
               CASE
                   WHEN kwh_clean IS NOT NULL THEN kwh_clean
                   WHEN island_length <= {settings.short_gap_intervals} AND interpolated IS NOT NULL
                        THEN greatest(interpolated, 0.0)
                   WHEN island_length <= {settings.max_estimate_intervals} AND profile_kwh IS NOT NULL
                        THEN greatest(profile_kwh, 0.0)
                   ELSE NULL
               END AS kwh,
               CASE
                   WHEN kwh_clean IS NOT NULL THEN '{STATUS_ACTUAL}'
                   WHEN island_length <= {settings.short_gap_intervals} AND interpolated IS NOT NULL
                        THEN CASE WHEN value_rule IS NULL THEN '{STATUS_ESTIMATED}' ELSE '{STATUS_EDITED}' END
                   WHEN island_length <= {settings.max_estimate_intervals} AND profile_kwh IS NOT NULL
                        THEN CASE WHEN value_rule IS NULL THEN '{STATUS_ESTIMATED}' ELSE '{STATUS_EDITED}' END
                   ELSE '{STATUS_MISSING}'
               END AS status,
               CASE
                   WHEN kwh_clean IS NOT NULL THEN CAST(NULL AS VARCHAR)
                   WHEN value_rule IS NOT NULL THEN value_rule
                   ELSE '{RULE_MISSING}'
               END AS rule_id,
               CASE
                   WHEN kwh_clean IS NOT NULL THEN '{METHOD_NONE}'
                   WHEN island_length <= {settings.short_gap_intervals} AND interpolated IS NOT NULL
                        THEN '{METHOD_INTERPOLATE}'
                   WHEN island_length <= {settings.max_estimate_intervals} AND profile_kwh IS NOT NULL
                        THEN '{METHOD_LIKE_DAY}'
                   ELSE '{METHOD_NONE}'
               END AS method
        FROM joined
        """
    )


def write_batch(connection, config: Config, fleet_name: str, batch_label: str):
    silver = config.paths.silver_dir / fleet_name / f"meter_batch={batch_label}"
    silver.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (SELECT '{fleet_name}' AS fleet, * FROM t_silver)
        TO {db.quote(silver / 'readings.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT '{fleet_name}' AS fleet, meter_id, ts_utc, rule_id, method, kwh_raw AS original_kwh, kwh AS new_kwh
            FROM t_silver WHERE status <> '{STATUS_ACTUAL}'
            UNION ALL
            SELECT '{fleet_name}', meter_id, CAST(ts_local AS TIMESTAMP), rule_id, '{METHOD_NONE}', kwh_raw, NULL
            FROM t_rejected
        ) TO {db.quote(silver / 'edits.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT '{fleet_name}' AS fleet, meter_id, local_date, rule_id, day_kwh, med_day AS reference_kwh,
                   present, expected_slots
            FROM t_day_flags WHERE rule_id IS NOT NULL
            UNION ALL
            SELECT '{fleet_name}', meter_id, local_date, '{RULE_REGISTER}', day_kwh, register_kwh,
                   present, register_count
            FROM t_register_flags
        ) TO {db.quote(silver / 'day_flags.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def batch_counts(connection) -> dict:
    row = connection.execute(
        f"""
        SELECT count(*), count(*) FILTER (WHERE status = '{STATUS_ACTUAL}'),
               count(*) FILTER (WHERE status = '{STATUS_ESTIMATED}'),
               count(*) FILTER (WHERE status = '{STATUS_EDITED}'),
               count(*) FILTER (WHERE status = '{STATUS_MISSING}'),
               count(*) FILTER (WHERE generated),
               count(DISTINCT meter_id)
        FROM t_silver
        """
    ).fetchone()
    rejected = connection.execute(
        f"""
        SELECT count(*) FILTER (WHERE rule_id = '{RULE_DUPLICATE}'),
               count(*) FILTER (WHERE rule_id = '{RULE_NONEXISTENT}'),
               count(*) FILTER (WHERE rule_id = '{RULE_OUT_OF_SERVICE}')
        FROM t_rejected
        """
    ).fetchone()
    days = connection.execute(
        "SELECT count(*) FILTER (WHERE rule_id IS NOT NULL) FROM t_day_flags"
    ).fetchone()[0]
    register = connection.execute("SELECT count(*) FROM t_register_flags").fetchone()[0]
    accuracy = connection.execute(
        """
        SELECT count(*), sum(relative_error), sum(estimated_slots)
        FROM t_estimation_error WHERE relative_error IS NOT NULL
        """
    ).fetchone()
    rules = dict(
        connection.execute(
            f"SELECT rule_id, count(*) FROM t_silver WHERE status <> '{STATUS_ACTUAL}' GROUP BY 1"
        ).fetchall()
    )
    return {
        "rows": row[0],
        "actual": row[1],
        "estimated": row[2],
        "edited": row[3],
        "missing": row[4],
        "generated": row[5],
        "meters": row[6],
        "duplicates": rejected[0],
        "nonexistent": rejected[1],
        "out_of_service": rejected[2],
        "day_flags": days,
        "register_flags": register,
        "estimated_days_checked": accuracy[0] or 0,
        "estimated_days_error_sum": float(accuracy[1] or 0.0),
        "estimated_slots_checked": int(accuracy[2] or 0),
        "rules": rules,
    }


def build_silver(config: Config, progress=None) -> dict:
    if not config.vee.enabled:
        raise VeeError("vee is disabled in config, silver cannot be built")
    config.paths.ensure()
    connection = db.connect(config)
    summary = {}
    for fleet_name in config.enabled_fleets:
        batches = batch_paths(config, fleet_name)
        if not batches:
            raise VeeError(f"no bronze batches found for fleet {fleet_name}")
        register_available = (
            fleet_name == "london"
            and config.fleet(fleet_name).has_daily_register
            and (config.paths.bronze_dir / "london_daily.parquet").is_file()
        )
        totals = {}
        for index, batch_dir in enumerate(batches):
            label = batch_dir.name.split("=", 1)[1]
            prepare_batch(connection, config, fleet_name, batch_dir)
            flag_values(connection, config, fleet_name)
            estimate(connection, config, fleet_name)
            flag_days(connection, config, fleet_name, register_available)
            write_batch(connection, config, fleet_name, label)
            counts = batch_counts(connection)
            for key, value in counts.items():
                if key == "rules":
                    bucket = totals.setdefault("rules", {})
                    for rule, amount in value.items():
                        bucket[rule] = bucket.get(rule, 0) + amount
                else:
                    totals[key] = totals.get(key, 0) + value
            if progress and (index % 16 == 0 or index == len(batches) - 1):
                progress(f"vee {fleet_name} batch {index + 1}/{len(batches)} rows={totals['rows']}")
        totals["register_checked"] = register_available
        if totals.get("estimated_days_checked"):
            totals["estimation_mean_relative_error"] = round(
                totals["estimated_days_error_sum"] / totals["estimated_days_checked"], 6
            )
        summary[fleet_name] = totals
    connection.close()
    path = config.paths.silver_dir / "vee_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    return summary
