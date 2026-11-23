from __future__ import annotations

import pathlib

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from meterdm import cluster, db
from meterdm.config import load_config


@pytest.fixture(scope="module")
def clustered(analytics_project):
    config, reports = analytics_project
    return config, reports["cluster"]


@pytest.fixture
def blobs():
    generator = np.random.default_rng(4)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    return np.vstack([centre + generator.normal(0.0, 0.5, size=(80, 2)) for centre in centres])


def test_kmeans_plus_plus_is_deterministic(blobs):
    first = cluster.kmeans_plus_plus(blobs, 3, np.random.default_rng(2))
    second = cluster.kmeans_plus_plus(blobs, 3, np.random.default_rng(2))
    assert np.allclose(first, second)
    assert first.shape == (3, 2)


def test_lloyd_reaches_a_fixed_point(blobs):
    centres = cluster.kmeans_plus_plus(blobs, 3, np.random.default_rng(1))
    centres, labels, inertia = cluster.lloyd(blobs, centres, 200, 1e-12)
    for index in range(3):
        members = blobs[labels == index]
        assert np.allclose(centres[index], members.mean(axis=0), atol=1e-9)
    assert inertia > 0.0
    assert set(np.unique(labels)) == {0, 1, 2}


def test_inertia_matches_sklearn_from_the_same_start(blobs):
    start = cluster.kmeans_plus_plus(blobs, 3, np.random.default_rng(6))
    mine, _, inertia = cluster.lloyd(blobs, start.copy(), 300, 1e-12)
    reference = KMeans(n_clusters=3, init=start, n_init=1, max_iter=300, tol=1e-12).fit(blobs)
    assert inertia == pytest.approx(float(reference.inertia_), rel=1e-6)
    ordered_mine = mine[np.lexsort(mine.T)]
    ordered_reference = reference.cluster_centers_[np.lexsort(reference.cluster_centers_.T)]
    assert np.allclose(ordered_mine, ordered_reference, atol=1e-6)


def test_silhouette_matches_sklearn(blobs):
    centres = cluster.kmeans_plus_plus(blobs, 3, np.random.default_rng(8))
    _, labels, _ = cluster.lloyd(blobs, centres, 200, 1e-12)
    mine = cluster.silhouette(blobs, labels, blobs.shape[0], 8)
    reference = float(silhouette_score(blobs, labels))
    assert mine == pytest.approx(reference, abs=1e-9)


def test_profile_labels_follow_the_shape():
    intervals = 48
    hours = (np.arange(intervals) + 0.5) * 24.0 / intervals
    flat = np.ones(intervals) / intervals
    night = np.where(hours < 6.0, 4.0, 0.4)
    evening = np.where((hours >= 17.0) & (hours < 21.0), 5.0, 0.4)
    midday = np.where((hours >= 11.0) & (hours < 15.0), 5.0, 0.4)
    doubled = np.concatenate
    assert cluster.label_profile(doubled([flat, flat]), intervals) == cluster.LABEL_FLAT
    assert cluster.label_profile(doubled([night / night.sum(), flat]), intervals) == cluster.LABEL_NIGHT
    assert cluster.label_profile(doubled([evening / evening.sum(), flat]), intervals) == cluster.LABEL_EVENING
    assert cluster.label_profile(doubled([midday / midday.sum(), flat]), intervals) == cluster.LABEL_DAYTIME


def test_shared_shapes_get_distinct_labels():
    intervals = 48
    hours = (np.arange(intervals) + 0.5) * 24.0 / intervals
    flat = np.ones(intervals) / intervals
    early = np.where((hours >= 17.0) & (hours < 19.0), 5.0, 0.4)
    late = np.where((hours >= 19.0) & (hours < 21.0), 5.0, 0.4)
    centres = np.vstack(
        [
            np.concatenate([early / early.sum(), flat]),
            np.concatenate([late / late.sum(), flat]),
            np.concatenate([flat, flat]),
        ]
    )
    names = cluster.label_clusters(centres, intervals)
    assert len(set(names)) == len(names)
    assert names[2] == cluster.LABEL_FLAT
    assert names[0].startswith(cluster.LABEL_EVENING)
    assert names[1].startswith(cluster.LABEL_EVENING)
    assert cluster.peak_clock(centres[0], intervals) == "17:15"
    assert cluster.peak_clock(centres[1], intervals) == "19:15"


def test_tied_flat_shapes_are_split_by_window():
    intervals = 48
    hours = (np.arange(intervals) + 0.5) * 24.0 / intervals
    night_tilt = np.where(hours < 6.0, 1.15, 0.95)
    midday_tilt = np.where((hours >= 9.0) & (hours < 16.0), 1.15, 0.95)
    centres = np.vstack(
        [
            np.concatenate([night_tilt / night_tilt.sum(), night_tilt / night_tilt.sum()]),
            np.concatenate([midday_tilt / midday_tilt.sum(), midday_tilt / midday_tilt.sum()]),
        ]
    )
    names = cluster.label_clusters(centres, intervals)
    assert names == ["flat, more night", "flat, more midday"]


def test_unique_shapes_keep_plain_labels():
    intervals = 48
    hours = (np.arange(intervals) + 0.5) * 24.0 / intervals
    flat = np.ones(intervals) / intervals
    evening = np.where((hours >= 17.0) & (hours < 21.0), 5.0, 0.4)
    midday = np.where((hours >= 11.0) & (hours < 15.0), 5.0, 0.4)
    centres = np.vstack(
        [
            np.concatenate([evening / evening.sum(), flat]),
            np.concatenate([midday / midday.sum(), flat]),
        ]
    )
    assert cluster.label_clusters(centres, intervals) == [cluster.LABEL_EVENING, cluster.LABEL_DAYTIME]


def test_clusters_are_written_for_every_fleet(clustered):
    config, report = clustered
    connection = db.read_only(config)
    rows = connection.execute(
        "SELECT fleet, count(*) AS meters, count(DISTINCT cluster) AS clusters FROM fact_cluster GROUP BY 1"
    ).fetchall()
    profiles = connection.execute(
        "SELECT fleet, cluster, count(*) AS points FROM dim_cluster_profile GROUP BY 1, 2"
    ).fetchall()
    connection.close()
    seen = {row[0]: row for row in rows}
    for fleet_name in config.enabled_fleets:
        if report[fleet_name].get("skipped"):
            continue
        assert seen[fleet_name][1] == report[fleet_name]["meters"]
        assert seen[fleet_name][2] == report[fleet_name]["k"]
    for _, _, points in profiles:
        assert points in (96, 192)


def test_chosen_k_maximises_silhouette(clustered):
    _, report = clustered
    for entry in report.values():
        if entry.get("skipped"):
            continue
        scan = entry["scan"]
        best = max(scan, key=lambda item: item["silhouette"])
        assert best["k"] == entry["k"]
        assert entry["silhouette"] == pytest.approx(best["silhouette"])
        inertias = [item["inertia"] for item in scan]
        assert inertias == sorted(inertias, reverse=True)


def test_profiles_sum_to_one_per_day_type(clustered):
    config, _ = clustered
    connection = db.read_only(config)
    totals = connection.execute(
        "SELECT fleet, cluster, is_weekend, sum(share) FROM dim_cluster_profile GROUP BY 1, 2, 3"
    ).fetchall()
    connection.close()
    for _, _, _, total in totals:
        assert total == pytest.approx(1.0, abs=1e-6)


def test_reference_config_still_loads():
    config = load_config(pathlib.Path(cluster.__file__).resolve().parents[1] / "config.toml")
    assert config.cluster.k_min >= 2
