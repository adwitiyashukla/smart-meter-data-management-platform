from __future__ import annotations

import duckdb

from meterdm.config import Config


def connect(config: Config, database: str | None = None):
    connection = duckdb.connect(database if database is not None else ":memory:")
    connection.execute("SET TimeZone='UTC'")
    connection.execute(f"SET threads={config.duckdb.threads}")
    connection.execute(f"SET memory_limit='{config.duckdb.memory_limit_gb}GB'")
    config.paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory='{config.paths.tmp_dir.as_posix()}'")
    flag = "true" if config.duckdb.preserve_insertion_order else "false"
    connection.execute(f"SET preserve_insertion_order={flag}")
    return connection


def read_only(config: Config):
    if not config.paths.gold_db.is_file():
        raise FileNotFoundError(f"gold database not found: {config.paths.gold_db}")
    connection = duckdb.connect(config.paths.gold_db.as_posix(), read_only=True)
    connection.execute("SET TimeZone='UTC'")
    connection.execute(f"SET threads={config.duckdb.threads}")
    return connection


def gold(config: Config):
    config.paths.gold_db.parent.mkdir(parents=True, exist_ok=True)
    return connect(config, config.paths.gold_db.as_posix())


def quote(path) -> str:
    text = str(path).replace("'", "''")
    return f"'{text}'"
