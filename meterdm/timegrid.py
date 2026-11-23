from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

SECONDS_PER_MINUTE = 60


def utc_instants(start: date, end: date, interval_minutes: int, zone_name: str) -> np.ndarray:
    zone = ZoneInfo(zone_name)
    first = datetime(start.year, start.month, start.day, tzinfo=zone)
    last = datetime(end.year, end.month, end.day, tzinfo=zone) + timedelta(days=1)
    start_utc = np.datetime64(first.astimezone(UTC).replace(tzinfo=None), "s")
    end_utc = np.datetime64(last.astimezone(UTC).replace(tzinfo=None), "s")
    step = np.timedelta64(interval_minutes * SECONDS_PER_MINUTE, "s")
    return np.arange(start_utc + step, end_utc + step, step, dtype="datetime64[s]")


def utc_offset_seconds(instants: np.ndarray, zone_name: str) -> np.ndarray:
    zone = ZoneInfo(zone_name)
    if instants.size == 0:
        return np.zeros(0, dtype=np.int32)
    boundaries = _transition_indices(instants, zone)
    offsets = np.empty(instants.size, dtype=np.int32)
    for lo, hi in boundaries:
        moment = instants[lo].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        offsets[lo:hi] = int(moment.astimezone(zone).utcoffset().total_seconds())
    return offsets


def _transition_indices(instants, zone):
    def offset_at(index):
        moment = instants[index].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        return int(moment.astimezone(zone).utcoffset().total_seconds())

    spans = []
    lo = 0
    size = instants.size
    while lo < size:
        current = offset_at(lo)
        low, high = lo, size
        while low < high:
            mid = (low + high) // 2
            if offset_at(mid) == current:
                low = mid + 1
            else:
                high = mid
        if low <= lo:
            low = lo + 1
        spans.append((lo, low))
        lo = low
    return spans


def to_local_wall(instants: np.ndarray, zone_name: str) -> np.ndarray:
    offsets = utc_offset_seconds(instants, zone_name)
    return instants + offsets.astype("timedelta64[s]")


def naive_grid(start: date, end: date, interval_minutes: int) -> np.ndarray:
    first = np.datetime64(start.isoformat(), "s")
    last = np.datetime64((end + timedelta(days=1)).isoformat(), "s")
    step = np.timedelta64(interval_minutes * SECONDS_PER_MINUTE, "s")
    return np.arange(first + step, last + step, step, dtype="datetime64[s]")


def lcl_strings(wall: np.ndarray) -> np.ndarray:
    base = np.datetime_as_string(wall, unit="s")
    return np.char.add(np.char.replace(base, "T", " "), ".0000000")


def plain_strings(wall: np.ndarray) -> np.ndarray:
    base = np.datetime_as_string(wall, unit="s")
    return np.char.replace(base, "T", " ")


def local_dates(wall: np.ndarray, interval_minutes: int) -> np.ndarray:
    step = np.timedelta64(interval_minutes * SECONDS_PER_MINUTE, "s")
    return (wall - step).astype("datetime64[D]")


def day_index(wall: np.ndarray, interval_minutes: int, origin: date) -> np.ndarray:
    days = local_dates(wall, interval_minutes)
    return (days - np.datetime64(origin.isoformat(), "D")).astype(np.int64)


def day_length_hours(day: date, zone_name: str) -> float:
    zone = ZoneInfo(zone_name)
    midnight = datetime(day.year, day.month, day.day, tzinfo=zone)
    following = datetime.combine(day + timedelta(days=1), midnight.timetz().replace(tzinfo=zone))
    delta = following.astimezone(UTC) - midnight.astimezone(UTC)
    return delta.total_seconds() / 3600.0


def expected_intervals(day: date, zone_name: str, interval_minutes: int) -> int:
    return int(round(day_length_hours(day, zone_name) * 60.0 / interval_minutes))


def short_days(start: date, end: date, zone_name: str):
    return _odd_days(start, end, zone_name, lambda hours: hours < 24.0)


def long_days(start: date, end: date, zone_name: str):
    return _odd_days(start, end, zone_name, lambda hours: hours > 24.0)


def _odd_days(start, end, zone_name, predicate):
    cursor = start
    found = []
    while cursor <= end:
        hours = day_length_hours(cursor, zone_name)
        if predicate(hours):
            found.append((cursor, hours))
        cursor = cursor + timedelta(days=1)
    return found
