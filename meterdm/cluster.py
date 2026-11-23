from __future__ import annotations

import json

import numpy as np

from meterdm import db
from meterdm.config import Config

LABEL_NIGHT = "night heavy"
LABEL_FLAT = "flat"
LABEL_EVENING = "evening peak"
LABEL_MORNING_EVENING = "morning and evening"
LABEL_DAYTIME = "daytime"
LABEL_LATE_EVENING = "late evening"
WINDOWS = (
    ("night", 0.0, 6.0),
    ("morning", 6.0, 9.0),
    ("midday", 9.0, 16.0),
    ("evening", 16.0, 21.0),
    ("late evening", 21.0, 24.0),
)


class ClusterError(RuntimeError):
    pass


def kmeans_plus_plus(points: np.ndarray, k: int, rng) -> np.ndarray:
    count = points.shape[0]
    centres = np.empty((k, points.shape[1]))
    first = int(rng.integers(0, count))
    centres[0] = points[first]
    closest = ((points - centres[0]) ** 2).sum(axis=1)
    for index in range(1, k):
        total = closest.sum()
        if total <= 0:
            centres[index] = points[int(rng.integers(0, count))]
        else:
            pick = int(np.searchsorted(np.cumsum(closest / total), rng.random()))
            centres[index] = points[min(pick, count - 1)]
        closest = np.minimum(closest, ((points - centres[index]) ** 2).sum(axis=1))
    return centres


def assign(points: np.ndarray, centres: np.ndarray):
    squares = (points**2).sum(axis=1)[:, None] - 2.0 * points @ centres.T + (centres**2).sum(axis=1)[None, :]
    labels = np.argmin(squares, axis=1)
    return labels, np.maximum(squares[np.arange(points.shape[0]), labels], 0.0)


def lloyd(points: np.ndarray, centres: np.ndarray, max_iter: int, tolerance: float):
    for _ in range(max_iter):
        labels, distances = assign(points, centres)
        updated = np.empty_like(centres)
        for index in range(centres.shape[0]):
            members = points[labels == index]
            updated[index] = members.mean(axis=0) if members.size else centres[index]
        shift = float(((updated - centres) ** 2).sum())
        centres = updated
        if shift <= tolerance:
            break
    labels, distances = assign(points, centres)
    return centres, labels, float(distances.sum())


def fit_kmeans(points: np.ndarray, k: int, config: Config, seed: int):
    settings = config.cluster
    best = None
    for restart in range(settings.restarts):
        rng = np.random.default_rng([seed, k, restart])
        centres = kmeans_plus_plus(points, k, rng)
        centres, labels, inertia = lloyd(points, centres, settings.max_iter, settings.tolerance)
        if best is None or inertia < best[2]:
            best = (centres, labels, inertia)
    return best


def silhouette(points: np.ndarray, labels: np.ndarray, sample: int, seed: int) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return -1.0
    rng = np.random.default_rng([seed, int(points.shape[0])])
    if points.shape[0] > sample:
        chosen = rng.choice(points.shape[0], size=sample, replace=False)
    else:
        chosen = np.arange(points.shape[0])
    subset = points[chosen]
    subset_labels = labels[chosen]
    distances = np.sqrt(
        np.maximum(
            (subset**2).sum(axis=1)[:, None] - 2.0 * subset @ subset.T + (subset**2).sum(axis=1)[None, :], 0.0
        )
    )
    scores = np.zeros(subset.shape[0])
    for index in range(subset.shape[0]):
        own = subset_labels == subset_labels[index]
        own[index] = False
        if not np.any(own):
            scores[index] = 0.0
            continue
        inside = distances[index, own].mean()
        outside = np.inf
        for other in unique:
            if other == subset_labels[index]:
                continue
            mask = subset_labels == other
            if np.any(mask):
                outside = min(outside, distances[index, mask].mean())
        scores[index] = 0.0 if outside == np.inf else (outside - inside) / max(inside, outside)
    return float(scores.mean())


def label_profile(profile: np.ndarray, intervals_per_day: int) -> str:
    weekday = profile[:intervals_per_day]
    total = weekday.sum()
    if total <= 0:
        return LABEL_FLAT
    hours = (np.arange(intervals_per_day) + 0.5) * 24.0 / intervals_per_day
    night = weekday[hours < 6.0].sum() / total
    peak_hour = float(hours[int(np.argmax(weekday))])
    ratio = float(weekday.max() / (total / intervals_per_day))
    if ratio < 1.30:
        return LABEL_FLAT
    if night > 0.28:
        return LABEL_NIGHT
    if 9.0 <= peak_hour < 16.0:
        return LABEL_DAYTIME
    if peak_hour >= 21.0 or peak_hour < 4.0:
        return LABEL_LATE_EVENING
    morning = weekday[(hours >= 5.0) & (hours < 11.0)].max()
    evening = weekday[(hours >= 16.0) & (hours < 23.0)].max()
    if evening > 1.25 * morning:
        return LABEL_EVENING
    return LABEL_MORNING_EVENING


def peak_clock(profile: np.ndarray, intervals_per_day: int) -> str:
    weekday = profile[:intervals_per_day]
    minutes = int(round((float(np.argmax(weekday)) + 0.5) * 1440.0 / intervals_per_day)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def window_shares(profile: np.ndarray, intervals_per_day: int) -> np.ndarray:
    weekday = profile[:intervals_per_day]
    total = float(weekday.sum())
    if total <= 0:
        return np.zeros(len(WINDOWS))
    hours = (np.arange(intervals_per_day) + 0.5) * 24.0 / intervals_per_day
    return np.array(
        [float(weekday[(hours >= low) & (hours < high)].sum()) / total for _, low, high in WINDOWS]
    )


def window_words(centres: np.ndarray, members: list, intervals_per_day: int) -> list:
    shares = np.vstack([window_shares(centres[index], intervals_per_day) for index in members])
    average = shares.mean(axis=0)
    taken = set()
    words = []
    for row in shares:
        order = np.argsort(-(row - average))
        word = WINDOWS[int(order[0])][0]
        for position in order:
            candidate = WINDOWS[int(position)][0]
            if candidate not in taken:
                word = candidate
                break
        taken.add(word)
        words.append(word)
    return words


def label_clusters(centres: np.ndarray, intervals_per_day: int) -> list:
    base = [label_profile(centres[index], intervals_per_day) for index in range(len(centres))]
    names = list(base)
    for name in sorted(set(base)):
        members = [index for index in range(len(base)) if base[index] == name]
        if len(members) < 2:
            continue
        clocks = [peak_clock(centres[index], intervals_per_day) for index in members]
        if name != LABEL_FLAT and len(set(clocks)) == len(members):
            for index, clock in zip(members, clocks, strict=True):
                names[index] = f"{name} at {clock}"
            continue
        words = window_words(centres, members, intervals_per_day)
        for index, word in zip(members, words, strict=True):
            names[index] = f"{name}, more {word}"
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    seen = {}
    for index, name in enumerate(names):
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            names[index] = f"{name} {seen[name]}"
    return names


def load_profiles(connection, config: Config, fleet_name: str):
    intervals = config.fleet(fleet_name).intervals_per_day
    table = connection.sql(
        """
        WITH eligible AS (
            SELECT fleet, meter_id FROM fact_daily_read
            WHERE fleet = ? AND kwh IS NOT NULL
            GROUP BY 1, 2 HAVING count(*) >= ?
        )
        SELECT p.meter_id, p.is_weekend, p.period, p.mean_kwh
        FROM fact_period_profile p JOIN eligible e USING (fleet, meter_id)
        WHERE p.fleet = ?
        ORDER BY p.meter_id, p.is_weekend, p.period
        """,
        params=[fleet_name, config.cluster.min_days_per_meter, fleet_name],
    ).to_arrow_table()
    if table.num_rows == 0:
        return [], np.zeros((0, 2 * intervals))
    meters = table.column("meter_id").to_pylist()
    weekend = np.asarray(table.column("is_weekend").to_pylist(), dtype=bool)
    period = table.column("period").to_numpy(zero_copy_only=False).astype(int)
    values = table.column("mean_kwh").to_numpy(zero_copy_only=False).astype(float)
    unique_meters = sorted(set(meters))
    index = {meter: position for position, meter in enumerate(unique_meters)}
    matrix = np.zeros((len(unique_meters), 2 * intervals))
    rows = np.array([index[meter] for meter in meters])
    columns = np.where(weekend, intervals, 0) + np.clip(period - 1, 0, intervals - 1)
    np.add.at(matrix, (rows, columns), values)
    weekday_total = matrix[:, :intervals].sum(axis=1, keepdims=True)
    weekend_total = matrix[:, intervals:].sum(axis=1, keepdims=True)
    keep = (weekday_total[:, 0] > 0) & (weekend_total[:, 0] > 0)
    matrix[:, :intervals] = np.divide(
        matrix[:, :intervals], np.where(weekday_total > 0, weekday_total, 1.0)
    )
    matrix[:, intervals:] = np.divide(
        matrix[:, intervals:], np.where(weekend_total > 0, weekend_total, 1.0)
    )
    kept = [meter for meter, flag in zip(unique_meters, keep, strict=True) if flag]
    return kept, matrix[keep]


def choose_k(points: np.ndarray, config: Config, seed: int, progress=None):
    settings = config.cluster
    outcomes = []
    for k in range(settings.k_min, settings.k_max + 1):
        if k >= points.shape[0]:
            break
        centres, labels, inertia = fit_kmeans(points, k, config, seed)
        score = silhouette(points, labels, 1500, seed)
        outcomes.append({"k": k, "inertia": inertia, "silhouette": score, "centres": centres, "labels": labels})
        if progress:
            progress(f"cluster k={k} inertia={inertia:.4f} silhouette={score:.4f}")
    if not outcomes:
        raise ClusterError("not enough meters to cluster")
    return max(outcomes, key=lambda item: item["silhouette"]), outcomes


def build_clusters(config: Config, progress=None) -> dict:
    connection = db.gold(config)
    connection.execute(
        """
        CREATE OR REPLACE TABLE fact_cluster (
            fleet VARCHAR, meter_id VARCHAR, cluster INTEGER, distance DOUBLE, k INTEGER
        )
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE dim_cluster_profile (
            fleet VARCHAR, cluster INTEGER, is_weekend BOOLEAN, period INTEGER, share DOUBLE,
            meters BIGINT, label VARCHAR
        )
        """
    )
    report = {}
    for fleet_name in config.enabled_fleets:
        meters, points = load_profiles(connection, config, fleet_name)
        if len(meters) < config.cluster.k_min + 1:
            report[fleet_name] = {"meters": len(meters), "skipped": True}
            continue
        best, outcomes = choose_k(points, config, config.run.seed, progress)
        labels = best["labels"]
        centres = best["centres"]
        intervals = config.fleet(fleet_name).intervals_per_day
        names = label_clusters(centres, intervals)
        distances = np.sqrt(assign(points, centres)[1])
        connection.executemany(
            "INSERT INTO fact_cluster VALUES (?, ?, ?, ?, ?)",
            [
                (fleet_name, meters[index], int(labels[index]), float(distances[index]), int(best["k"]))
                for index in range(len(meters))
            ],
        )
        profile_rows = []
        for cluster in range(best["k"]):
            size = int((labels == cluster).sum())
            name = names[cluster]
            for position in range(2 * intervals):
                profile_rows.append(
                    (
                        fleet_name,
                        cluster,
                        position >= intervals,
                        (position % intervals) + 1,
                        float(centres[cluster][position]),
                        size,
                        name,
                    )
                )
        connection.executemany("INSERT INTO dim_cluster_profile VALUES (?, ?, ?, ?, ?, ?, ?)", profile_rows)
        report[fleet_name] = {
            "meters": len(meters),
            "k": best["k"],
            "inertia": best["inertia"],
            "silhouette": best["silhouette"],
            "sizes": [int((labels == cluster).sum()) for cluster in range(best["k"])],
            "labels": list(names),
            "scan": [
                {"k": item["k"], "inertia": item["inertia"], "silhouette": item["silhouette"]}
                for item in outcomes
            ],
        }
        if progress:
            progress(f"cluster {fleet_name} chose k={best['k']} for {len(meters)} meters")
    connection.close()
    path = config.paths.reports_dir / "cluster_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="ascii")
    return report
