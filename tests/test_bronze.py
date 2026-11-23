from __future__ import annotations

import json

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from conftest import apply_tiny, write_config

from meterdm import bronze, db, synth
from meterdm.config import load_config


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    config = load_config(write_config(root / "project", apply_tiny))
    replica = synth.build_replica(config)
    summary = bronze.build_bronze(config)
    return config, replica, summary


def test_row_counts_match_the_replica(built):
    config, replica, summary = built
    assert summary["london"]["rows"] == replica["fleets"]["london"]["rows"]
    assert summary["portugal"]["rows"] == replica["fleets"]["portugal"]["rows"]
    assert summary["london"]["meters"] == replica["fleets"]["london"]["meters"]
    assert summary["portugal"]["meters"] == replica["fleets"]["portugal"]["meters"]
    assert summary["london"]["unparsed_timestamps"] == 0


def test_parquet_is_readable_without_duckdb(built):
    config, _, summary = built
    dataset = ds.dataset(config.paths.bronze_dir / "london", format="parquet", partitioning="hive")
    assert dataset.count_rows() == summary["london"]["rows"]
    names = set(dataset.schema.names)
    assert {"meter_id", "ts_local", "kwh", "tariff_group", "year_month", "meter_batch"} <= names


def test_meter_batches_are_disjoint(built):
    config, _, summary = built
    connection = db.connect(config)
    pattern = (config.paths.bronze_dir / "london" / "**" / "*.parquet").as_posix()
    spread = connection.execute(
        f"""
        SELECT count(*) FROM (
            SELECT meter_id, count(DISTINCT meter_batch) AS batches
            FROM read_parquet({db.quote(pattern)}, hive_partitioning=true)
            GROUP BY 1
        ) WHERE batches <> 1
        """
    ).fetchone()[0]
    connection.close()
    assert spread == 0
    assert summary["london"]["batches"] >= 1


def test_partitions_follow_the_interval_ending_convention(built):
    config, _, _ = built
    connection = db.connect(config)
    pattern = (config.paths.bronze_dir / "london" / "**" / "*.parquet").as_posix()
    wrong = connection.execute(
        f"""
        SELECT count(*) FROM read_parquet({db.quote(pattern)}, hive_partitioning=true)
        WHERE year_month <> strftime(ts_local - INTERVAL 30 MINUTE, '%Y-%m')
        """
    ).fetchone()[0]
    boundary = connection.execute(
        f"""
        SELECT year_month FROM read_parquet({db.quote(pattern)}, hive_partitioning=true)
        WHERE ts_local = TIMESTAMP '2012-02-01 00:00:00' LIMIT 1
        """
    ).fetchone()
    connection.close()
    assert wrong == 0
    assert boundary is not None and boundary[0] == "2012-01"


def test_missing_reads_survive_as_nulls(built):
    config, _, summary = built
    connection = db.connect(config)
    pattern = (config.paths.bronze_dir / "london" / "**" / "*.parquet").as_posix()
    nulls, zeros = connection.execute(
        f"""
        SELECT count(*) FILTER (WHERE kwh IS NULL), count(*) FILTER (WHERE kwh = 0.0)
        FROM read_parquet({db.quote(pattern)})
        """
    ).fetchone()
    connection.close()
    assert nulls == summary["london"]["rows"] - summary["london"]["values"]
    assert nulls > 0
    assert zeros >= 0


def test_portugal_power_is_converted_to_energy(built):
    config, _, _ = built
    source = config.paths.raw_dir / "portugal" / config.portugal_source.member
    lines = source.read_text().splitlines()
    header = lines[0].split(";")
    stamp = lines[1].split(";")[0].strip('"')
    first_client = header[1].strip('"')
    raw_kw = float(lines[1].split(";")[1].replace(",", "."))
    connection = db.connect(config)
    pattern = (config.paths.bronze_dir / "portugal" / "**" / "*.parquet").as_posix()
    value = connection.execute(
        f"""
        SELECT kwh FROM read_parquet({db.quote(pattern)})
        WHERE meter_id = ? AND ts_local = strptime(?, '%Y-%m-%d %H:%M:%S')
        """,
        [first_client, stamp],
    ).fetchone()
    connection.close()
    assert value is not None
    assert value[0] == pytest.approx(raw_kw * 15.0 / 60.0, rel=1e-9)


def test_weather_bronze_matches_the_json(built):
    config, _, summary = built
    payload = json.loads((config.paths.raw_dir / "weather" / "london.json").read_text())
    table = pq.read_table(config.paths.bronze_dir / "weather" / "london.parquet")
    assert table.num_rows == len(payload["hourly"]["time"])
    assert summary["weather"]["london"]["hours"] == table.num_rows
    assert table.column("temp_c")[0].as_py() == pytest.approx(payload["hourly"]["temperature_2m"][0])


def test_alternate_column_names_are_detected(tmp_path):
    config = load_config(write_config(tmp_path / "project", apply_tiny))
    folder = config.paths.raw_dir / "london" / "halfhourly"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "block_0.csv").write_text(
        "LCLid,tstp,energy(kWh/hh)\n"
        "MAC000002,2012-01-01 00:30:00.0000000,0.100\n"
        "MAC000002,2012-01-01 01:00:00.0000000,Null\n"
        "MAC000002,not a timestamp,0.300\n",
        encoding="ascii",
    )
    connection = db.connect(config)
    result = bronze.build_london_bronze(connection, config)
    connection.close()
    assert result["rows"] == 2
    assert result["values"] == 1
    assert result["unparsed_timestamps"] == 1


def test_missing_value_column_is_reported(tmp_path):
    config = load_config(write_config(tmp_path / "project", apply_tiny))
    folder = config.paths.raw_dir / "london" / "halfhourly"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "block_0.csv").write_text("LCLid,tstp\nMAC000002,2012-01-01 00:30:00.0000000\n", encoding="ascii")
    connection = db.connect(config)
    with pytest.raises(bronze.BronzeError, match="missing value columns"):
        bronze.build_london_bronze(connection, config)
    connection.close()


def test_reference_tables_are_optional(tmp_path):
    config = load_config(write_config(tmp_path / "project", apply_tiny))
    (config.paths.raw_dir / "london").mkdir(parents=True, exist_ok=True)
    connection = db.connect(config)
    result = bronze.build_reference_bronze(connection, config)
    connection.close()
    assert result["daily"]["available"] is False
    assert result["tariffs"]["available"] is False
