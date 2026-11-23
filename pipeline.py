from __future__ import annotations

import os

from dagster import AssetExecutionContext, Definitions, asset

from meterdm import bronze, changepoint, cluster, dtou, forecast, synth, theft, vee, warehouse
from meterdm.config import load_config

CONFIG_PATH = os.environ.get("METERDM_CONFIG", "config.toml")


def settings():
    return load_config(CONFIG_PATH)


def note(context: AssetExecutionContext):
    return context.log.info


@asset
def replica(context: AssetExecutionContext) -> dict:
    return synth.build_replica(settings(), note(context))


@asset(deps=[replica])
def bronze_layer(context: AssetExecutionContext) -> dict:
    return bronze.build_bronze(settings(), note(context))


@asset(deps=[bronze_layer])
def silver_layer(context: AssetExecutionContext) -> dict:
    return vee.build_silver(settings(), note(context))


@asset(deps=[silver_layer])
def gold_layer(context: AssetExecutionContext) -> dict:
    return warehouse.build_gold(settings(), note(context))


@asset(deps=[gold_layer])
def changepoint_models(context: AssetExecutionContext) -> dict:
    return changepoint.build_changepoints(settings(), note(context))


@asset(deps=[gold_layer])
def load_clusters(context: AssetExecutionContext) -> dict:
    return cluster.build_clusters(settings(), note(context))


@asset(deps=[changepoint_models])
def revenue_protection(context: AssetExecutionContext) -> dict:
    return theft.build_theft(settings(), note(context))


@asset(deps=[gold_layer])
def tariff_trial(context: AssetExecutionContext) -> dict:
    return dtou.build_dtou(settings(), note(context))


@asset(deps=[gold_layer])
def load_forecast(context: AssetExecutionContext) -> dict:
    return forecast.build_forecast(settings(), note(context))


defs = Definitions(
    assets=[
        replica,
        bronze_layer,
        silver_layer,
        gold_layer,
        changepoint_models,
        load_clusters,
        revenue_protection,
        tariff_trial,
        load_forecast,
    ]
)
