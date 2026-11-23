from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from meterdm import timegrid as tg
from meterdm.calendar_data import holiday_mask, load_holidays
from meterdm.config import Config

ACORN_BY_GROUP = {
    "Affluent": ("ACORN-A", "ACORN-B", "ACORN-C", "ACORN-D", "ACORN-E"),
    "Comfortable": ("ACORN-F", "ACORN-G", "ACORN-H", "ACORN-I", "ACORN-J", "ACORN-K"),
    "Adversity": ("ACORN-L", "ACORN-M", "ACORN-N", "ACORN-O", "ACORN-P", "ACORN-Q"),
}
LONDON_COLUMNS = ("LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) ")
DAILY_COLUMNS = ("LCLid", "day", "energy_sum", "energy_count", "energy_mean", "energy_max", "energy_min")
TARIFF_COLUMNS = ("TariffDateTime", "Tariff")
NORMAL_PRICE = 0
HIGH_PRICE = 1
LOW_PRICE = 2
PRICE_LABELS = ("Normal", "High", "Low")
HEAT_BASE_C = 15.5
COOL_BASE_C = 21.0
HEATING_PENETRATION = 0.55
COOLING_PENETRATION = 0.12
DAY_NOISE_CV = 0.16


def _hour_axis(intervals_per_day: int) -> np.ndarray:
    return (np.arange(intervals_per_day) + 0.5) * 24.0 / intervals_per_day


def _bump(hours, center, width, height):
    delta = np.abs(hours - center)
    delta = np.minimum(delta, 24.0 - delta)
    return height * np.exp(-0.5 * (delta / width) ** 2)


def household_archetypes(intervals_per_day: int):
    hours = _hour_axis(intervals_per_day)
    weekday = np.array(
        [
            0.35 + _bump(hours, 7.8, 1.1, 0.85) + _bump(hours, 19.0, 1.7, 1.95),
            0.30 + _bump(hours, 7.2, 0.9, 1.65) + _bump(hours, 18.5, 1.4, 1.30),
            0.95 + _bump(hours, 13.0, 4.5, 0.30),
            0.30 + _bump(hours, 2.5, 2.3, 2.40) + _bump(hours, 19.5, 1.5, 0.70),
            0.45 + _bump(hours, 11.5, 3.2, 0.95) + _bump(hours, 19.0, 1.9, 1.10),
            0.25 + _bump(hours, 22.8, 2.0, 1.75) + _bump(hours, 9.5, 1.6, 0.55),
        ]
    )
    weekend = np.array(
        [
            0.40 + _bump(hours, 10.0, 1.8, 0.90) + _bump(hours, 19.8, 2.0, 1.75),
            0.35 + _bump(hours, 9.5, 1.5, 1.25) + _bump(hours, 19.5, 1.7, 1.35),
            0.95 + _bump(hours, 13.5, 4.5, 0.35),
            0.32 + _bump(hours, 2.5, 2.3, 2.30) + _bump(hours, 20.5, 1.7, 0.80),
            0.50 + _bump(hours, 12.5, 3.6, 1.05) + _bump(hours, 20.0, 2.1, 1.00),
            0.28 + _bump(hours, 23.6, 2.2, 1.70) + _bump(hours, 11.0, 2.0, 0.60),
        ]
    )
    heating = 0.55 + _bump(hours, 7.0, 1.6, 1.40) + _bump(hours, 18.5, 2.4, 1.60)
    weekday = weekday / weekday.sum(axis=1, keepdims=True)
    weekend = weekend / weekend.sum(axis=1, keepdims=True)
    return weekday, weekend, heating / heating.sum()


def commercial_archetypes(intervals_per_day: int):
    hours = _hour_axis(intervals_per_day)
    shapes = np.array(
        [
            np.ones_like(hours),
            0.30 + _bump(hours, 13.0, 3.4, 1.70),
            0.35 + _bump(hours, 16.0, 4.6, 1.35),
            0.80 + _bump(hours, 3.0, 3.0, 0.85),
        ]
    )
    return shapes / shapes.sum(axis=1, keepdims=True)


ARCHETYPES_WEEKDAY, ARCHETYPES_WEEKEND, HEATING_SHAPE = household_archetypes(48)


def _ar1(size, phi, sd, rng):
    out = np.empty(size)
    out[0] = rng.normal(0.0, sd)
    innovation = rng.normal(0.0, sd * np.sqrt(max(1.0 - phi * phi, 1e-9)), size=size)
    for index in range(1, size):
        out[index] = phi * out[index - 1] + innovation[index]
    return out


def hourly_weather(config: Config, fleet_name: str, rng):
    shape = config.synthetic.weather[fleet_name]
    first = config.synthetic.start_date - timedelta(days=2)
    last = config.synthetic.end_date + timedelta(days=2)
    hours = np.arange(
        np.datetime64(first.isoformat() + "T00:00", "s"),
        np.datetime64((last + timedelta(days=1)).isoformat() + "T00:00", "s"),
        np.timedelta64(3600, "s"),
        dtype="datetime64[s]",
    )
    day = hours.astype("datetime64[D]")
    year_start = hours.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (day - year_start).astype(np.int64) + 1
    hour_of_day = ((hours - day) / np.timedelta64(1, "h")).astype(np.float64)
    seasonal = shape.mean_c - shape.seasonal_amplitude_c * np.cos(
        2.0 * np.pi * (day_of_year - shape.peak_day_of_year) / 365.25
    )
    diurnal = -0.5 * shape.diurnal_range_c * np.cos(2.0 * np.pi * (hour_of_day - 3.0) / 24.0)
    noise = _ar1(hours.size, shape.noise_phi, shape.noise_sd_c, rng)
    return hours, np.round(seasonal + diurnal + noise, 1)


def write_weather_json(path: Path, latitude: float, longitude: float, hours, temps):
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "generationtime_ms": 0.0,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 0.0,
        "hourly_units": {"time": "iso8601", "temperature_2m": "C"},
        "hourly": {
            "time": np.datetime_as_string(hours, unit="m").tolist(),
            "temperature_2m": [round(float(value), 1) for value in temps],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="ascii")


def daily_mean_temperature(hours, temps, zone_name):
    offsets = tg.utc_offset_seconds(hours, zone_name)
    local_days = (hours + offsets.astype("timedelta64[s]")).astype("datetime64[D]")
    unique, inverse = np.unique(local_days, return_inverse=True)
    totals = np.bincount(inverse, weights=temps)
    counts = np.bincount(inverse)
    return unique, totals / counts


def _day_axis(config: Config):
    start = np.datetime64(config.synthetic.start_date.isoformat(), "D")
    end = np.datetime64(config.synthetic.end_date.isoformat(), "D")
    return np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")


def _align_temperature(days, unique_days, values):
    positions = np.searchsorted(unique_days, days)
    if np.any(positions >= unique_days.size) or np.any(unique_days[positions] != days):
        raise ValueError("weather series does not cover the requested day axis")
    return values[positions]


def _tariff_codes(config: Config, wall, intervals_per_day, rng):
    tou = config.synthetic.tou
    days = tg.local_dates(wall, 24 * 60 // intervals_per_day)
    hour = ((wall - days.astype("datetime64[s]")) / np.timedelta64(1, "h")).astype(np.float64)
    hour = np.where(hour == 0.0, 24.0, hour)
    start_hour = hour - 24.0 / intervals_per_day
    trial = (days >= np.datetime64(tou.trial_start.isoformat(), "D")) & (
        days <= np.datetime64(tou.trial_end.isoformat(), "D")
    )
    weekday = (days.astype("datetime64[D]").astype(np.int64) + 4) % 7 < 5
    unique_days = np.unique(days[trial])
    picked = rng.random(unique_days.size) < (tou.event_days_per_week / 7.0)
    event_days = unique_days[picked]
    is_event = np.isin(days, event_days) & trial & weekday
    codes = np.zeros(wall.size, dtype=np.int8)
    high_window = (start_hour >= tou.high_start_hour) & (start_hour < tou.high_end_hour)
    low_window = (start_hour >= tou.low_start_hour) | (start_hour < tou.low_end_hour)
    codes[is_event & high_window] = HIGH_PRICE
    codes[is_event & low_window] = LOW_PRICE
    return codes, trial


def _london_meter_table(config: Config, rng, n_days, tavg):
    synth = config.synthetic
    count = synth.london_meters
    ids = np.array([f"MAC{index:06d}" for index in range(2, count + 2)], dtype="U9")
    per_block = int(np.ceil(count / config.london_source.blocks))
    block = np.arange(count) // per_block
    tou = rng.random(count) < synth.tou_share
    groups = np.array(synth.acorn_groups)[rng.integers(0, len(synth.acorn_groups), size=count)]
    acorn = np.array(
        [ACORN_BY_GROUP[group][rng.integers(0, len(ACORN_BY_GROUP[group]))] for group in groups], dtype="U8"
    )
    start_day = rng.integers(0, synth.enrollment_spread_days + 1, size=count)
    base = rng.lognormal(synth.base_kwh_per_day_lognorm_mu, synth.base_kwh_per_day_lognorm_sigma, size=count)
    heat_cp = np.clip(rng.normal(HEAT_BASE_C, 1.8, size=count), 11.0, 19.0)
    cool_cp = np.clip(rng.normal(COOL_BASE_C, 1.5, size=count), 17.5, 26.0)
    cool_cp = np.maximum(cool_cp, heat_cp + 1.5)
    typical_hdd = float(np.maximum(0.0, HEAT_BASE_C - tavg).mean() * 365.0)
    typical_cdd = float(np.maximum(0.0, tavg - COOL_BASE_C).mean() * 365.0)
    has_heat = rng.random(count) < HEATING_PENETRATION
    has_cool = rng.random(count) < COOLING_PENETRATION
    heat_slope = has_heat * synth.heating_share * base * 365.0 / max(typical_hdd, 1.0) * rng.gamma(4.0, 0.25, count)
    cool_slope = has_cool * synth.cooling_share * base * 365.0 / max(typical_cdd, 1.0) * rng.gamma(4.0, 0.25, count)
    weights = rng.gamma(0.6, 1.0, size=(count, 6))
    weights = weights / weights.sum(axis=1, keepdims=True)
    weekend = np.clip(rng.normal(1.06, 0.08, size=count), 0.8, 1.4)
    drift = rng.normal(0.0, 0.03, size=count)
    elasticity = np.where(
        tou, np.clip(rng.normal(synth.tou.elasticity_mean, synth.tou.elasticity_sd, size=count), 0.0, 0.6), 0.0
    )
    transformer = np.arange(count) // config.theft.transformer_size
    return {
        "id": ids,
        "block": block,
        "tou": tou,
        "acorn": acorn,
        "group": groups,
        "start_day": np.minimum(start_day, max(n_days - config.synthetic.min_span_days, 0)),
        "base": base,
        "heat_cp": heat_cp,
        "cool_cp": cool_cp,
        "heat_slope": heat_slope,
        "cool_slope": cool_slope,
        "weights": weights,
        "weekend": weekend,
        "drift": drift,
        "elasticity": elasticity,
        "transformer": transformer,
    }


def _apply_defects(values, config: Config, rng):
    synth = config.synthetic
    defects = synth.defects
    size = values.size
    counts = {"missing": 0, "gap": 0, "spike": 0, "stuck": 0, "zero_run": 0}
    events = {"missing": 0, "gap": 0, "spike": 0, "stuck": 0, "zero_run": 0}

    spikes = rng.poisson(defects.spike_rate_per_meter)
    if spikes:
        where = rng.integers(0, size, size=spikes)
        multiplier = rng.uniform(defects.spike_multiplier_low, defects.spike_multiplier_high, size=spikes)
        values[where] = values[where] * multiplier
        counts["spike"] += int(spikes)
        events["spike"] += int(spikes)

    if rng.random() < defects.stuck_probability:
        length = int(rng.integers(defects.stuck_min_intervals, defects.stuck_max_intervals))
        start = int(rng.integers(0, max(size - length, 1)))
        values[start : start + length] = values[start]
        counts["stuck"] += length
        events["stuck"] += 1

    if rng.random() < defects.zero_run_probability:
        length = int(rng.integers(defects.zero_run_min_intervals, defects.zero_run_max_intervals))
        start = int(rng.integers(0, max(size - length, 1)))
        values[start : start + length] = 0.0
        counts["zero_run"] += length
        events["zero_run"] += 1

    burst_starts = np.flatnonzero(rng.random(size) < synth.gap_burst_rate)
    for start in burst_starts:
        length = int(rng.integers(2, synth.gap_burst_max_intervals))
        values[start : start + length] = np.nan
        counts["gap"] += min(length, size - start)
        events["gap"] += 1

    single = (rng.random(size) < synth.missing_read_rate) & ~np.isnan(values)
    values[single] = np.nan
    counts["missing"] += int(single.sum())
    events["missing"] += int(single.sum())
    return counts, events


def _kwh_text(values):
    column = pa.array(values)
    column = pc.if_else(pc.is_nan(column), pa.scalar(None, pa.float64()), column)
    return pc.fill_null(pc.cast(column, pa.string()), "Null")


def _write_csv(path: Path, header, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write((",".join(header) + "\n").encode("ascii"))
        pacsv.write_csv(table, handle, pacsv.WriteOptions(include_header=False, quoting_style="none"))


def build_london(config: Config, rng, progress):
    fleet = config.fleet("london")
    intervals = fleet.intervals_per_day
    instants = tg.utc_instants(
        config.synthetic.start_date, config.synthetic.end_date, fleet.interval_minutes, fleet.timezone
    )
    wall = tg.to_local_wall(instants, fleet.timezone)
    days = _day_axis(config)
    n_days = days.size
    day_index = tg.day_index(wall, fleet.interval_minutes, config.synthetic.start_date)
    slots_per_day = np.bincount(day_index, minlength=n_days)
    day_start = np.concatenate([[0], np.cumsum(slots_per_day)[:-1]])
    slot_in_day = np.arange(wall.size) - day_start[day_index]
    profile_pos = (slot_in_day * intervals) // slots_per_day[day_index]

    hours, temps = hourly_weather(config, "london", rng)
    write_weather_json(
        config.paths.raw_dir / "weather" / "london.json", fleet.latitude, fleet.longitude, hours, temps
    )
    unique_days, daily_temp = daily_mean_temperature(hours, temps, fleet.timezone)
    tavg = _align_temperature(days, unique_days, daily_temp)

    holidays = load_holidays(config.paths.refdata_dir, "london")
    is_holiday = holiday_mask(days, holidays)
    weekday_number = (days.astype(np.int64) + 4) % 7
    is_weekend = weekday_number >= 5
    holiday_factor = np.where(is_holiday, 1.05, 1.0)

    codes, _ = _tariff_codes(config, wall, intervals, rng)
    meters = _london_meter_table(config, rng, n_days, tavg)
    timestamps = pa.array(tg.lcl_strings(wall))
    tariff_labels = np.array(PRICE_LABELS)

    truth_rows = []
    defect_rows = []
    daily_parts = []
    blocks = int(meters["block"].max()) + 1
    total_rows = 0
    for block in range(blocks):
        members = np.flatnonzero(meters["block"] == block)
        id_parts = []
        tou_parts = []
        ts_parts = []
        value_parts = []
        for index in members:
            start = int(day_start[meters["start_day"][index]])
            length = wall.size - start
            local_day = day_index[start:] - int(meters["start_day"][index])
            positions = profile_pos[start:]
            meter_days = np.arange(int(meters["start_day"][index]), n_days)
            temp_days = tavg[meter_days]
            hdd = np.maximum(0.0, meters["heat_cp"][index] - temp_days)
            cdd = np.maximum(0.0, temp_days - meters["cool_cp"][index])
            level = (
                meters["base"][index]
                * np.where(is_weekend[meter_days], meters["weekend"][index], 1.0)
                * holiday_factor[meter_days]
                * (1.0 + meters["drift"][index] * np.arange(meter_days.size) / 365.0)
            )
            daily = level + meters["heat_slope"][index] * hdd + meters["cool_slope"][index] * cdd
            daily = daily * rng.lognormal(0.0, DAY_NOISE_CV, size=meter_days.size)
            daily = np.maximum(daily, 0.15 * meters["base"][index])

            shape_weekday = meters["weights"][index] @ ARCHETYPES_WEEKDAY
            shape_weekend = meters["weights"][index] @ ARCHETYPES_WEEKEND
            raw = np.where(is_weekend[meter_days][local_day], shape_weekend[positions], shape_weekday[positions])
            blend = 0.15 * np.clip((meters["heat_cp"][index] - temp_days) / 10.0, 0.0, 1.0)
            raw = raw * (1.0 - blend[local_day]) + HEATING_SHAPE[positions] * blend[local_day]
            raw = raw * rng.lognormal(0.0, config.synthetic.noise_cv, size=length)
            day_total = np.bincount(local_day, weights=raw, minlength=meter_days.size)
            values = daily[local_day] * raw / day_total[local_day]

            if meters["tou"][index]:
                meter_codes = codes[start:]
                high = meter_codes == HIGH_PRICE
                low = meter_codes == LOW_PRICE
                cut = values[high] * meters["elasticity"][index]
                values[high] = values[high] - cut
                moved = np.bincount(local_day[high], weights=cut, minlength=meter_days.size)
                low_count = np.bincount(local_day[low], minlength=meter_days.size)
                share = np.where(
                    low_count > 0,
                    config.synthetic.tou.shift_fraction * moved / np.maximum(low_count, 1),
                    0.0,
                )
                values[low] = values[low] + share[local_day[low]]

            values = np.round(values, 3)
            totals = np.bincount(local_day, weights=values, minlength=meter_days.size)
            counts = np.bincount(local_day, minlength=meter_days.size)
            maxima = np.zeros(meter_days.size)
            np.maximum.at(maxima, local_day, values)
            minima = np.full(meter_days.size, np.inf)
            np.minimum.at(minima, local_day, values)
            daily_parts.append(
                pa.table(
                    {
                        "LCLid": pa.array(np.full(meter_days.size, meters["id"][index])),
                        "day": pa.array(days[meter_days].astype(str)),
                        "energy_sum": pa.array(np.round(totals, 3)),
                        "energy_count": pa.array(counts.astype(np.int32)),
                        "energy_mean": pa.array(np.round(totals / np.maximum(counts, 1), 6)),
                        "energy_max": pa.array(np.round(maxima, 3)),
                        "energy_min": pa.array(np.round(minima, 3)),
                    }
                )
            )

            counts_map, events_map = _apply_defects(values, config, rng)
            gone = np.isnan(values) & (rng.random(length) < config.synthetic.omitted_read_share)
            keep = ~gone
            counts_map["omitted"] = int(gone.sum())
            events_map["omitted"] = int(gone.sum())
            for kind, interval_count in counts_map.items():
                if interval_count:
                    defect_rows.append((meters["id"][index], kind, events_map[kind], interval_count))

            kept = int(keep.sum())
            id_parts.append(pa.array(np.full(kept, meters["id"][index])))
            tou_parts.append(pa.array(np.full(kept, "ToU" if meters["tou"][index] else "Std")))
            ts_parts.append(timestamps.slice(start, length).filter(pa.array(keep)))
            value_parts.append(values[keep])
            total_rows += kept
            truth_rows.append(
                (
                    meters["id"][index],
                    "ToU" if meters["tou"][index] else "Std",
                    meters["acorn"][index],
                    meters["group"][index],
                    int(meters["transformer"][index]),
                    str(days[meters["start_day"][index]]),
                    round(float(meters["base"][index]), 4),
                    round(float(meters["heat_slope"][index]), 5),
                    round(float(meters["cool_slope"][index]), 5),
                    round(float(meters["heat_cp"][index]), 3),
                    round(float(meters["cool_cp"][index]), 3),
                    round(float(meters["elasticity"][index]), 4),
                )
            )

        table = pa.table(
            {
                LONDON_COLUMNS[0]: pa.concat_arrays(id_parts),
                LONDON_COLUMNS[1]: pa.concat_arrays(tou_parts),
                LONDON_COLUMNS[2]: pa.concat_arrays(ts_parts),
                LONDON_COLUMNS[3]: _kwh_text(np.concatenate(value_parts)),
            }
        )
        _write_csv(config.paths.raw_dir / "london" / "halfhourly" / f"block_{block}.csv", LONDON_COLUMNS, table)
        if progress and (block % 16 == 0 or block == blocks - 1):
            progress(f"london block {block + 1}/{blocks} rows={total_rows}")

    households = pa.table(
        {
            "LCLid": pa.array(meters["id"]),
            "stdorToU": pa.array(np.where(meters["tou"], "ToU", "Std")),
            "Acorn": pa.array(meters["acorn"]),
            "Acorn_grouped": pa.array(meters["group"]),
            "file": pa.array(np.array([f"block_{value}" for value in meters["block"]])),
        }
    )
    _write_csv(
        config.paths.raw_dir / "london" / "informations_households.csv",
        ("LCLid", "stdorToU", "Acorn", "Acorn_grouped", "file"),
        households,
    )
    daily = pa.concat_tables(daily_parts)
    _write_csv(config.paths.raw_dir / "london" / "daily_dataset.csv", DAILY_COLUMNS, daily)

    trial = codes != NORMAL_PRICE
    tariff_mask = np.zeros(wall.size, dtype=bool)
    tariff_days = tg.local_dates(wall, fleet.interval_minutes)
    window = (tariff_days >= np.datetime64(config.synthetic.tou.trial_start.isoformat(), "D")) & (
        tariff_days <= np.datetime64(config.synthetic.tou.trial_end.isoformat(), "D")
    )
    tariff_mask[window] = True
    tariffs = pa.table(
        {
            "TariffDateTime": timestamps.filter(pa.array(tariff_mask)),
            "Tariff": pa.array(tariff_labels[codes[tariff_mask]]),
        }
    )
    _write_csv(config.paths.raw_dir / "london" / "tariffs.csv", TARIFF_COLUMNS, tariffs)

    truth_dir = config.paths.truth_dir
    truth_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "LCLid,stdorToU,acorn,acorn_group,transformer,first_day,base_kwh_day,"
        "heat_slope,cool_slope,heat_cp,cool_cp,elasticity\n"
    )
    (truth_dir / "london_meters.csv").write_text(
        header + "\n".join(",".join(str(field) for field in row) for row in truth_rows) + "\n", encoding="ascii"
    )
    (truth_dir / "london_defects.csv").write_text(
        "LCLid,kind,events,intervals\n"
        + "\n".join(",".join(str(field) for field in row) for row in defect_rows)
        + "\n",
        encoding="ascii",
    )
    return {
        "fleet": "london",
        "meters": int(config.synthetic.london_meters),
        "rows": int(total_rows),
        "blocks": int(blocks),
        "days": int(n_days),
        "tou_meters": int(meters["tou"].sum()),
        "high_price_intervals": int((codes == HIGH_PRICE).sum()),
        "mean_elasticity": round(float(meters["elasticity"][meters["tou"]].mean()), 5),
        "trial_intervals": int(trial.sum()),
    }


def _day_member(local_day, entries):
    if not entries:
        return np.zeros(local_day.size, dtype=bool)
    marks = np.array([np.datetime64(day.isoformat(), "D") for day, _ in entries], dtype="datetime64[D]")
    return np.isin(local_day, marks)


def build_portugal(config: Config, rng, progress):
    fleet = config.fleet("portugal")
    settings = config.synthetic.portugal
    intervals = fleet.intervals_per_day
    wall = tg.naive_grid(config.synthetic.start_date, config.synthetic.end_date, fleet.interval_minutes)
    days = _day_axis(config)
    n_days = days.size
    n_slots = wall.size
    count = config.synthetic.portugal_meters

    hours, temps = hourly_weather(config, "portugal", rng)
    write_weather_json(
        config.paths.raw_dir / "weather" / "portugal.json", fleet.latitude, fleet.longitude, hours, temps
    )
    unique_days, daily_temp = daily_mean_temperature(hours, temps, fleet.timezone)
    tavg = _align_temperature(days, unique_days, daily_temp)

    holidays = load_holidays(config.paths.refdata_dir, "portugal")
    is_holiday = holiday_mask(days, holidays)
    weekday_number = (days.astype(np.int64) + 4) % 7
    is_weekend = weekday_number >= 5

    day_of_slot = tg.day_index(wall, fleet.interval_minutes, config.synthetic.start_date)
    slot_of_day = np.arange(n_slots) % intervals
    archetypes = commercial_archetypes(intervals)

    base = rng.lognormal(settings.base_kwh_per_day_lognorm_mu, settings.base_kwh_per_day_lognorm_sigma, size=count)
    heat_cp = np.clip(rng.normal(14.0, 2.0, size=count), 9.0, 18.0)
    cool_cp = np.clip(rng.normal(21.0, 1.8, size=count), 18.5, 27.0)
    typical_hdd = float(np.maximum(0.0, 14.0 - tavg).mean() * 365.0)
    typical_cdd = float(np.maximum(0.0, tavg - 21.0).mean() * 365.0)
    heat_slope = settings.heating_share * base * 365.0 / max(typical_hdd, 1.0) * rng.gamma(4.0, 0.25, count)
    cool_slope = settings.cooling_share * base * 365.0 / max(typical_cdd, 1.0) * rng.gamma(4.0, 0.25, count)
    weights = rng.gamma(0.7, 1.0, size=(count, archetypes.shape[0]))
    weights = weights / weights.sum(axis=1, keepdims=True)
    weekend_factor = np.clip(rng.normal(0.72, 0.18, size=count), 0.2, 1.2)
    created = np.where(
        rng.random(count) < settings.late_client_fraction,
        rng.integers(1, settings.creation_spread_days + 1, size=count),
        0,
    )

    matrix = np.zeros((n_slots, count), dtype=np.float32)
    for index in range(count):
        level = (
            base[index]
            * np.where(is_weekend, weekend_factor[index], 1.0)
            * np.where(is_holiday, weekend_factor[index], 1.0)
        )
        daily = (
            level
            + heat_slope[index] * np.maximum(0.0, heat_cp[index] - tavg)
            + cool_slope[index] * np.maximum(0.0, tavg - cool_cp[index])
        )
        daily = daily * rng.lognormal(0.0, DAY_NOISE_CV, size=n_days)
        daily[: created[index]] = 0.0
        shape = weights[index] @ archetypes
        raw = shape[slot_of_day] * rng.lognormal(0.0, settings.noise_cv, size=n_slots)
        day_total = np.bincount(day_of_slot, weights=raw, minlength=n_days)
        matrix[:, index] = (daily[day_of_slot] * raw / day_total[day_of_slot] * (60.0 / fleet.interval_minutes)).astype(
            np.float32
        )
        if progress and index and index % 100 == 0:
            progress(f"portugal client {index}/{count}")

    minute_of_day = ((wall - wall.astype("datetime64[D]")) / np.timedelta64(1, "m")).astype(np.int64)
    dst_slots = np.isin(minute_of_day, np.array([75, 90, 105, 120]))
    local_day = wall.astype("datetime64[D]")
    short_days = tg.short_days(config.synthetic.start_date, config.synthetic.end_date, fleet.timezone)
    long_days = tg.long_days(config.synthetic.start_date, config.synthetic.end_date, fleet.timezone)
    spring = dst_slots & _day_member(local_day, short_days)
    autumn = dst_slots & _day_member(local_day, long_days)
    matrix[spring, :] = 0.0
    matrix[autumn, :] = matrix[autumn, :] * 2.0

    names = [f"MT_{index + 1:03d}" for index in range(count)]
    stamps = pa.array(np.char.add(np.char.add('"', tg.plain_strings(wall)), '"'))
    header = ('"";' + ";".join(f'"{name}"' for name in names) + "\n").encode("ascii")
    target = config.paths.raw_dir / "portugal" / config.portugal_source.member
    target.parent.mkdir(parents=True, exist_ok=True)
    chunk = 8192
    with target.open("wb") as handle:
        handle.write(header)
        for lo in range(0, n_slots, chunk):
            hi = min(lo + chunk, n_slots)
            columns = [stamps.slice(lo, hi - lo)]
            for index in range(count):
                column = pa.array(np.round(matrix[lo:hi, index].astype(np.float64), 3))
                columns.append(pc.replace_substring(pc.cast(column, pa.string()), ".", ","))
            lines = pc.binary_join_element_wise(*columns, config.portugal_source.delimiter)
            handle.write(("\n".join(lines.to_pylist()) + "\n").encode("ascii"))
        if progress:
            progress(f"portugal rows written {n_slots}")

    rows = [
        (
            names[index],
            str(days[created[index]]),
            round(float(base[index]), 3),
            round(float(heat_slope[index]), 5),
            round(float(cool_slope[index]), 5),
            round(float(heat_cp[index]), 3),
            round(float(cool_cp[index]), 3),
        )
        for index in range(count)
    ]
    config.paths.truth_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.truth_dir / "portugal_meters.csv").write_text(
        "client,first_day,base_kwh_day,heat_slope,cool_slope,heat_cp,cool_cp\n"
        + "\n".join(",".join(str(field) for field in row) for row in rows)
        + "\n",
        encoding="ascii",
    )
    return {
        "fleet": "portugal",
        "meters": int(count),
        "rows": int(n_slots) * int(count),
        "slots": int(n_slots),
        "days": int(n_days),
        "spring_zero_slots": int(spring.sum()),
        "autumn_doubled_slots": int(autumn.sum()),
    }


def build_replica(config: Config, progress=None):
    config.paths.ensure()
    summary = {"seed": config.run.seed, "fleets": {}}
    for name in config.enabled_fleets:
        stream = int.from_bytes(hashlib.blake2b(name.encode("ascii"), digest_size=4).digest(), "big")
        rng = np.random.default_rng([config.run.seed, stream])
        if name == "london":
            summary["fleets"][name] = build_london(config, rng, progress)
        elif name == "portugal":
            summary["fleets"][name] = build_portugal(config, rng, progress)
        else:
            raise ValueError(f"no synthetic generator for fleet {name!r}")
    path = config.paths.truth_dir / "replica_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    return summary
