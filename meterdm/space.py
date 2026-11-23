from __future__ import annotations

import json
import shutil
from pathlib import Path

from meterdm import db
from meterdm.config import Config

RUNTIME_REQUIREMENTS = ("streamlit>=1.50", "plotly>=5.20", "duckdb>=1.1", "pandas>=2.1")
PACKAGE_MODULES = ("__init__.py", "config.py", "db.py")
SMALL_TABLES = (
    "dim_meter",
    "dim_date",
    "dim_tariff",
    "fact_weather_day",
    "fact_daily_read",
    "fact_period_profile",
    "fact_fleet_hourly",
    "fact_trial_window",
    "fact_day_flag",
    "fact_quality",
    "fact_changepoint",
    "fact_cluster",
    "dim_cluster_profile",
    "fact_theft_score",
    "fact_dtou_estimate",
    "fact_forecast_metric",
)

DOCKERFILE = """FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    STREAMLIT_SERVER_HEADLESS=true \\
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \\
    METERDM_CONFIG=/app/config.toml

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE {port}
CMD ["streamlit", "run", "app.py", "--server.port={port}", "--server.address=0.0.0.0"]
"""

EMOJI_NAMES = {
    "zap": "\U000026a1",
    "bulb": "\U0001f4a1",
    "chart": "\U0001f4ca",
    "plug": "\U0001f50c",
    "house": "\U0001f3e0",
}

SPACE_README = """---
title: {title}
emoji: {emoji}
colorFrom: blue
colorTo: gray
sdk: docker
app_port: {port}
pinned: false
---

# {title}

Meter data management and revenue protection over smart meter interval reads.
The warehouse shipped with this Space was built by the pipeline in
https://github.com/adwitiyashukla/smart-meter-data-management-platform
"""


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_sources(config: Config, out_dir: Path):
    root = source_root()
    package = out_dir / "meterdm"
    package.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_MODULES:
        shutil.copy(root / "meterdm" / name, package / name)
    shutil.copy(root / "app.py", out_dir / "app.py")
    if (root / ".streamlit" / "config.toml").is_file():
        (out_dir / ".streamlit").mkdir(exist_ok=True)
        shutil.copy(root / ".streamlit" / "config.toml", out_dir / ".streamlit" / "config.toml")
    if Path(config.paths.refdata_dir).is_dir():
        shutil.copytree(config.paths.refdata_dir, out_dir / "refdata", dirs_exist_ok=True)


def _write_config(config: Config, out_dir: Path):
    text = (config.paths.root / "config.toml").read_text(encoding="ascii")
    text = text.replace('gold_db = "data/gold/meterdm.duckdb"', 'gold_db = "meterdm.duckdb"')
    (out_dir / "config.toml").write_text(text, encoding="ascii")


def _slim_database(config: Config, out_dir: Path) -> dict:
    target = out_dir / "meterdm.duckdb"
    if target.exists():
        target.unlink()
    connection = db.connect(config, target.as_posix())
    connection.execute(f"ATTACH '{config.paths.gold_db.as_posix()}' AS gold (READ_ONLY)")
    counts = {}
    for table in SMALL_TABLES:
        exists = connection.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE database_name = 'gold' AND table_name = ?", [table]
        ).fetchone()[0]
        if not exists:
            continue
        connection.execute(f"CREATE TABLE {table} AS SELECT * FROM gold.{table}")
        counts[table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    connection.execute(
        f"""
        CREATE TABLE dim_sample_meter AS
        SELECT fleet, meter_id FROM (
            SELECT fleet, meter_id, row_number() OVER (PARTITION BY fleet ORDER BY hash(meter_id)) AS pick
            FROM gold.dim_sample_meter
        ) WHERE pick <= {config.space.interval_meters}
        """
    )
    connection.execute(
        """
        CREATE TABLE fact_interval_sample AS
        SELECT s.* FROM gold.fact_interval_sample s JOIN dim_sample_meter USING (fleet, meter_id)
        """
    )
    connection.execute(
        f"""
        CREATE TABLE fact_vee_edit AS
        SELECT * FROM gold.fact_vee_edit USING SAMPLE {config.space.edit_rows} ROWS (reservoir, {config.run.seed})
        """
    )
    connection.execute(
        f"""
        CREATE TABLE fact_forecast_backtest AS
        SELECT * FROM gold.fact_forecast_backtest
        WHERE origin IN (SELECT DISTINCT origin FROM gold.fact_forecast_backtest ORDER BY origin DESC
                         LIMIT {config.space.backtest_origins})
        """
    )
    for table in ("dim_sample_meter", "fact_interval_sample", "fact_vee_edit", "fact_forecast_backtest"):
        counts[table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    connection.execute("DETACH gold")
    connection.execute("CHECKPOINT")
    connection.close()
    return counts


def build_space(config: Config, out_dir, progress=None) -> dict:
    if not config.paths.gold_db.is_file():
        raise FileNotFoundError(f"gold database not found: {config.paths.gold_db}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _copy_sources(config, out_dir)
    _write_config(config, out_dir)
    counts = _slim_database(config, out_dir)
    (out_dir / "requirements.txt").write_text("\n".join(RUNTIME_REQUIREMENTS) + "\n", encoding="ascii")
    (out_dir / "Dockerfile").write_text(DOCKERFILE.format(port=config.space.port), encoding="ascii")
    (out_dir / "README.md").write_text(
        SPACE_README.format(
            title=config.space.title,
            emoji=EMOJI_NAMES.get(config.space.emoji, config.space.emoji),
            port=config.space.port,
        ),
        encoding="utf-8",
    )
    (out_dir / ".gitattributes").write_text("*.duckdb filter=lfs diff=lfs merge=lfs -text\n", encoding="ascii")
    size = (out_dir / "meterdm.duckdb").stat().st_size
    summary = {
        "path": str(out_dir),
        "database_bytes": size,
        "database_mb": round(size / 1_048_576, 2),
        "files": sorted(item.relative_to(out_dir).as_posix() for item in out_dir.rglob("*") if item.is_file()),
        "rows": counts,
    }
    (config.paths.reports_dir).mkdir(parents=True, exist_ok=True)
    (config.paths.reports_dir / "space_summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    if progress:
        progress(f"space bundle written to {out_dir} database={summary['database_mb']} MB")
    return summary
