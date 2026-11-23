from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meterdm import api


@pytest.fixture(scope="module")
def client(analytics_project):
    config, _ = analytics_project
    api.CONFIG_PATH = str(config.paths.root / "config.toml")
    return TestClient(api.app)


def test_health_reports_the_warehouse(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["gold"] is True
    assert "london" in payload["fleets"]


def test_quality_endpoint_is_all_green(client):
    payload = client.get("/quality").json()
    assert payload
    assert all(row["passed"] for row in payload)


def test_fleets_endpoint_lists_both_fleets(client):
    payload = client.get("/fleets").json()
    names = {row["fleet"] for row in payload}
    assert names == {"london", "portugal"}
    for row in payload:
        assert row["meters"] > 0
        assert row["readings"] > 0


def test_meter_listing_and_detail_agree(client):
    listing = client.get("/meters", params={"fleet": "london", "limit": 5}).json()
    assert len(listing) == 5
    first = listing[0]
    detail = client.get(f"/meters/{first['meter_id']}", params={"fleet": "london"}).json()
    assert detail["meter_id"] == first["meter_id"]
    assert detail["readings"] == first["readings"]


def test_unknown_meter_returns_404(client):
    response = client.get("/meters/MAC999999", params={"fleet": "london"})
    assert response.status_code == 404


def test_readings_are_ordered_and_limited(client):
    listing = client.get("/meters", params={"fleet": "london", "limit": 40}).json()
    sampled = client.get("/meters", params={"fleet": "london", "limit": 200}).json()
    assert len(listing) == 40
    candidate = None
    for row in sampled:
        payload = client.get(
            f"/meters/{row['meter_id']}/readings", params={"fleet": "london", "limit": 10}
        ).json()
        if payload:
            candidate = payload
            break
    assert candidate is not None
    stamps = [row["ts_utc"] for row in candidate]
    assert stamps == sorted(stamps, reverse=True)
    assert {"status", "kwh", "rule_id"} <= set(candidate[0])


def test_theft_queue_is_sorted_by_score(client):
    payload = client.get("/theft/queue", params={"fleet": "london", "limit": 20}).json()
    scores = [row["score"] for row in payload]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in scores)


def test_cluster_profiles_are_returned(client):
    payload = client.get("/clusters", params={"fleet": "london"}).json()
    assert payload
    for row in payload:
        assert row["meters"] > 0
        assert len(row["profile"]) == 96


def test_dtou_and_forecast_endpoints(client):
    trial = {row["metric"]: row["value"] for row in client.get("/dtou").json()}
    metrics = client.get("/forecast/metrics").json()
    backtest = client.get("/forecast/backtest", params={"fleet": "london", "origin": 0}).json()
    assert trial["relative_effect"] < 0.0
    assert any(row["metric"] == "mape" for row in metrics)
    assert backtest
    assert [row["horizon"] for row in backtest] == sorted(row["horizon"] for row in backtest)


def test_vee_endpoints_expose_the_audit(client):
    summary = client.get("/vee/summary").json()
    flags = client.get("/vee/flags", params={"fleet": "london", "limit": 10}).json()
    assert summary
    assert any(row["rule_id"] for row in summary)
    assert isinstance(flags, list)
