from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from meterdm import (
    bronze,
    changepoint,
    cluster,
    dtou,
    forecast,
    holdout,
    ingest,
    space,
    synth,
    theft,
    vee,
    warehouse,
)
from meterdm.config import load_config

STAGES = ("bronze", "vee", "gold", "holdout", "changepoint", "cluster", "theft", "dtou", "forecast")


def reporter(quiet: bool):
    start = time.time()

    def note(message):
        if not quiet:
            print(f"[{time.time() - start:8.1f}s] {message}", flush=True)

    return note


def run_stage(name: str, config, note):
    if name == "bronze":
        return bronze.build_bronze(config, note)
    if name == "vee":
        return vee.build_silver(config, note)
    if name == "gold":
        return warehouse.build_gold(config, note)
    if name == "holdout":
        return holdout.build_holdout(config, note)
    if name == "changepoint":
        return changepoint.build_changepoints(config, note)
    if name == "cluster":
        return cluster.build_clusters(config, note)
    if name == "theft":
        return theft.build_theft(config, note)
    if name == "dtou":
        return dtou.build_dtou(config, note)
    if name == "forecast":
        return forecast.build_forecast(config, note)
    raise ValueError(f"unknown stage {name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="meterdm")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--profile", choices=["full", "sample"])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    subparsers.add_parser("fetch")
    subparsers.add_parser("synth")
    extract = subparsers.add_parser("extract")
    extract.add_argument("--london")
    extract.add_argument("--portugal")
    for name in STAGES:
        subparsers.add_parser(name)
    analytics = subparsers.add_parser("analytics")
    analytics.add_argument("--skip", nargs="*", default=[])
    pipeline = subparsers.add_parser("run")
    pipeline.add_argument("--from-stage", default="bronze", choices=STAGES)
    pipeline.add_argument("--synthetic", action="store_true")
    bundle = subparsers.add_parser("space")
    bundle.add_argument("--out", default="space")
    subparsers.add_parser("report")
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config, arguments.profile)
    note = reporter(arguments.quiet)
    started = time.time()

    if arguments.command == "probe":
        result = ingest.probe_sources(config)
        print(json.dumps(result, indent=2))
        return 0 if all(entry["ok"] for entry in result.values()) else 1
    if arguments.command == "fetch":
        result = ingest.fetch_all(config, note)
    elif arguments.command == "extract":
        result = {}
        if arguments.london:
            result["london"] = ingest.extract_london(Path(arguments.london), config, note)
        if arguments.portugal:
            result["portugal"] = ingest.extract_portugal(Path(arguments.portugal), config, note)
        if not result:
            parser.error("extract needs --london or --portugal")
    elif arguments.command == "synth":
        result = synth.build_replica(config, note)
    elif arguments.command == "analytics":
        result = {}
        for name in ("holdout", "changepoint", "cluster", "theft", "dtou", "forecast"):
            if name in arguments.skip:
                continue
            result[name] = run_stage(name, config, note)
    elif arguments.command == "run":
        result = {}
        if arguments.synthetic:
            result["synth"] = synth.build_replica(config, note)
        start_index = STAGES.index(arguments.from_stage)
        for name in STAGES[start_index:]:
            result[name] = run_stage(name, config, note)
    elif arguments.command == "space":
        result = space.build_space(config, arguments.out, note)
    elif arguments.command == "report":
        result = collect_report(config)
        failures = result.get("gold_summary", {}).get("quality_failures", 0)
        if failures:
            print(json.dumps(result.get("gold_summary", {}).get("failed", []), indent=2))
            return 1
    else:
        result = run_stage(arguments.command, config, note)

    note(f"{arguments.command} finished in {time.time() - started:.1f}s")
    if not arguments.quiet:
        print(json.dumps(result, indent=2, default=str))
    return 0


def collect_report(config) -> dict:
    report = {}
    for path in sorted(Path(config.paths.reports_dir).glob("*.json")):
        report[path.stem] = json.loads(path.read_text())
    for path in sorted(Path(config.paths.silver_dir).glob("*.json")):
        report[path.stem] = json.loads(path.read_text())
    for path in sorted(Path(config.paths.bronze_dir).glob("*.json")):
        report[path.stem] = json.loads(path.read_text())
    target = Path(config.paths.reports_dir) / "run_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="ascii")
    return report


if __name__ == "__main__":
    sys.exit(main())
