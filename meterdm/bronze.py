from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from meterdm import db
from meterdm.config import Config

LONDON_METER_KEYS = ("lclid",)
LONDON_STAMP_KEYS = ("datetime", "tstp")
LONDON_GROUP_KEYS = ("stdortou",)


class BronzeError(RuntimeError):
    pass


def _describe(connection, relation: str):
    rows = connection.execute(f"DESCRIBE SELECT * FROM {relation} LIMIT 0").fetchall()
    return [row[0] for row in rows]


def _pick(names, keys, prefixes=()):
    lookup = {name.strip().lower(): name for name in names}
    for key in keys:
        if key in lookup:
            return lookup[key]
    for lowered, original in lookup.items():
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return original
    return None


def london_reader(config: Config) -> str:
    pattern = (config.paths.raw_dir / "london" / "halfhourly" / "*.csv").as_posix()
    return f"read_csv({db.quote(pattern)}, header=true, all_varchar=true)"


def london_projection(connection, config: Config):
    relation = london_reader(config)
    names = _describe(connection, relation)
    meter = _pick(names, LONDON_METER_KEYS)
    stamp = _pick(names, LONDON_STAMP_KEYS)
    value = _pick(names, (), prefixes=("kwh/hh", "energy(kwh", "energy_khw", "kwh"))
    group = _pick(names, LONDON_GROUP_KEYS)
    missing = [label for label, found in (("meter", meter), ("timestamp", stamp), ("value", value)) if found is None]
    if missing:
        raise BronzeError(f"london source is missing {', '.join(missing)} columns, found {names}")
    minutes = config.fleet("london").interval_minutes
    group_expression = f'trim("{group}")' if group else "CAST(NULL AS VARCHAR)"
    return (
        relation,
        f"""
        SELECT
            trim("{meter}") AS meter_id,
            try_strptime(substr(trim("{stamp}"), 1, 19), '%Y-%m-%d %H:%M:%S') AS ts_local,
            TRY_CAST(replace(trim("{value}"), ',', '.') AS DOUBLE) AS kwh,
            {group_expression} AS tariff_group
        FROM {relation}
        """,
        minutes,
    )


def build_london_bronze(connection, config: Config, progress=None) -> dict:
    relation, projection, minutes = london_projection(connection, config)
    batches = config.fleet("london").meter_batches
    target = config.paths.bronze_dir / "london"
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT meter_id, ts_local, kwh, tariff_group,
                   strftime(ts_local - INTERVAL {minutes} MINUTE, '%Y-%m') AS year_month,
                   CAST(hash(meter_id) % {batches} AS INTEGER) AS meter_batch
            FROM ({projection})
            WHERE ts_local IS NOT NULL
        ) TO {db.quote(target)}
        (FORMAT PARQUET, PARTITION_BY (meter_batch), OVERWRITE_OR_IGNORE, COMPRESSION ZSTD)
        """
    )
    stats = connection.execute(
        f"""
        SELECT count(*), count(kwh), count(DISTINCT meter_id), min(ts_local), max(ts_local)
        FROM read_parquet({db.quote((target / '**' / '*.parquet').as_posix())})
        """
    ).fetchone()
    unparsed = connection.execute(
        f"SELECT count(*) FROM ({projection}) WHERE ts_local IS NULL"
    ).fetchone()[0]
    if progress:
        progress(f"bronze london rows={stats[0]} meters={stats[2]} unparsed={unparsed}")
    batch_count = connection.execute(
        f"""
        SELECT count(DISTINCT meter_batch)
        FROM read_parquet({db.quote((target / '**' / '*.parquet').as_posix())}, hive_partitioning=true)
        """
    ).fetchone()[0]
    return {
        "rows": stats[0],
        "values": stats[1],
        "meters": stats[2],
        "batches": batch_count,
        "first_local": str(stats[3]),
        "last_local": str(stats[4]),
        "unparsed_timestamps": unparsed,
    }


def build_portugal_bronze(connection, config: Config, progress=None) -> dict:
    source = config.paths.raw_dir / "portugal" / config.portugal_source.member
    if not source.is_file():
        raise BronzeError(f"portugal source file missing: {source}")
    delimiter = config.portugal_source.delimiter
    reader = f"read_csv({db.quote(source)}, header=true, delim='{delimiter}', all_varchar=true)"
    names = _describe(connection, reader)
    first = names[0]
    minutes = config.fleet("portugal").interval_minutes
    batches = config.fleet("portugal").meter_batches
    target = config.paths.bronze_dir / "portugal"
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            WITH source AS (
                SELECT "{first}" AS ts_text, * EXCLUDE ("{first}") FROM {reader}
            ), tall AS (
                SELECT
                    try_strptime(trim(ts_text), '%Y-%m-%d %H:%M:%S') AS ts_local,
                    meter_id,
                    TRY_CAST(replace(kw_text, '{config.portugal_source.decimal}', '.') AS DOUBLE) AS kw
                FROM source UNPIVOT (kw_text FOR meter_id IN (COLUMNS(* EXCLUDE (ts_text))))
            )
            SELECT meter_id, ts_local, kw * {minutes} / 60.0 AS kwh,
                   CAST(NULL AS VARCHAR) AS tariff_group,
                   strftime(ts_local - INTERVAL {minutes} MINUTE, '%Y-%m') AS year_month,
                   CAST(hash(meter_id) % {batches} AS INTEGER) AS meter_batch
            FROM tall
            WHERE ts_local IS NOT NULL
        ) TO {db.quote(target)}
        (FORMAT PARQUET, PARTITION_BY (meter_batch), OVERWRITE_OR_IGNORE, COMPRESSION ZSTD)
        """
    )
    stats = connection.execute(
        f"""
        SELECT count(*), count(kwh), count(DISTINCT meter_id), min(ts_local), max(ts_local)
        FROM read_parquet({db.quote((target / '**' / '*.parquet').as_posix())})
        """
    ).fetchone()
    if progress:
        progress(f"bronze portugal rows={stats[0]} meters={stats[2]}")
    batch_count = connection.execute(
        f"""
        SELECT count(DISTINCT meter_batch)
        FROM read_parquet({db.quote((target / '**' / '*.parquet').as_posix())}, hive_partitioning=true)
        """
    ).fetchone()[0]
    return {
        "rows": stats[0],
        "values": stats[1],
        "meters": stats[2],
        "batches": batch_count,
        "first_local": str(stats[3]),
        "last_local": str(stats[4]),
        "unparsed_timestamps": 0,
    }


def build_weather_bronze(config: Config, progress=None) -> dict:
    out = {}
    for name in config.enabled_fleets:
        source = config.paths.raw_dir / "weather" / f"{name}.json"
        if not source.is_file():
            raise BronzeError(f"weather file missing: {source}")
        payload = json.loads(source.read_text())
        hourly = payload.get("hourly", {})
        stamps = hourly.get("time")
        values = hourly.get(config.weather_source.hourly_variable)
        if not stamps or not values or len(stamps) != len(values):
            raise BronzeError(f"weather file {source} does not carry a usable hourly series")
        table = pa.table(
            {
                "ts_utc": pa.array(stamps).cast(pa.timestamp("s")),
                "temp_c": pa.array([None if value is None else float(value) for value in values], pa.float64()),
            }
        )
        target = config.paths.bronze_dir / "weather" / f"{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd")
        out[name] = {"hours": table.num_rows, "path": str(target)}
        if progress:
            progress(f"bronze weather {name} hours={table.num_rows}")
    return out


def build_reference_bronze(connection, config: Config, progress=None) -> dict:
    base = config.paths.raw_dir / "london"
    jobs = {
        "households": base / "informations_households.csv",
        "daily": base / "daily_dataset.csv",
        "tariffs": base / "tariffs.csv",
    }
    out = {}
    for name, source in jobs.items():
        if not source.is_file():
            out[name] = {"rows": 0, "available": False}
            continue
        target = config.paths.bronze_dir / f"london_{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"""
            COPY (SELECT * FROM read_csv({db.quote(source)}, header=true, all_varchar=true))
            TO {db.quote(target)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows = connection.execute(f"SELECT count(*) FROM read_parquet({db.quote(target)})").fetchone()[0]
        out[name] = {"rows": rows, "available": True}
        if progress:
            progress(f"bronze london_{name} rows={rows}")
    return out


def build_bronze(config: Config, progress=None) -> dict:
    config.paths.ensure()
    connection = db.connect(config)
    summary = {}
    if config.fleet("london").enabled:
        summary["london"] = build_london_bronze(connection, config, progress)
        summary["reference"] = build_reference_bronze(connection, config, progress)
    if config.fleet("portugal").enabled:
        summary["portugal"] = build_portugal_bronze(connection, config, progress)
    summary["weather"] = build_weather_bronze(config, progress)
    connection.close()
    path = Path(config.paths.bronze_dir) / "bronze_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    return summary
