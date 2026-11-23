from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def running(analytics_project):
    config, _ = analytics_project
    os.environ["METERDM_CONFIG"] = str(config.paths.root / "config.toml")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.run()
    return app


def test_app_runs_without_exceptions(running):
    assert not running.exception
    assert running.error == []


def test_sidebar_offers_both_fleets(running):
    options = running.sidebar.selectbox[0].options
    assert "london" in options
    assert "portugal" in options


def test_overview_metrics_are_rendered(running):
    labels = [item.label for item in running.metric]
    assert "Meters" in labels
    assert "Readings" in labels
    assert "Energy" in labels


def test_every_tab_renders_a_chart(running):
    assert len(running.tabs) == 7
    titles = [tab.label for tab in running.tabs]
    assert titles == [
        "Overview",
        "Validation",
        "Meter",
        "Load shapes",
        "Revenue protection",
        "Tariff trial",
        "Forecast",
    ]


def test_dataframes_are_populated(running):
    assert running.dataframe
    assert any(len(frame.value) > 0 for frame in running.dataframe)


def test_switching_fleet_keeps_the_app_healthy(running):
    running.sidebar.selectbox[0].set_value("portugal").run()
    assert not running.exception
