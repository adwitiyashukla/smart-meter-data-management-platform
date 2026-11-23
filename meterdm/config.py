from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ConfigError(ValueError):
    pass


def _table(parent, key, path):
    if key not in parent:
        raise ConfigError(f"missing table [{_join(path, key)}]")
    value = parent[key]
    if not isinstance(value, dict):
        raise ConfigError(f"[{_join(path, key)}] must be a table")
    return value


def _join(path, key):
    return key if not path else path + "." + key


def _value(table, key, path):
    if key not in table:
        raise ConfigError(f"missing key {_join(path, key)}")
    return table[key]


def _int(table, key, path, low=None, high=None):
    value = _value(table, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{_join(path, key)} must be an integer, got {value!r}")
    _bounds(value, key, path, low, high)
    return value


def _float(table, key, path, low=None, high=None):
    value = _value(table, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{_join(path, key)} must be a number, got {value!r}")
    value = float(value)
    _bounds(value, key, path, low, high)
    return value


def _bounds(value, key, path, low, high):
    if low is not None and value < low:
        raise ConfigError(f"{_join(path, key)} must be >= {low}, got {value}")
    if high is not None and value > high:
        raise ConfigError(f"{_join(path, key)} must be <= {high}, got {value}")


def _bool(table, key, path):
    value = _value(table, key, path)
    if not isinstance(value, bool):
        raise ConfigError(f"{_join(path, key)} must be true or false, got {value!r}")
    return value


def _str(table, key, path, choices=None, allow_empty=False):
    value = _value(table, key, path)
    if not isinstance(value, str):
        raise ConfigError(f"{_join(path, key)} must be a string, got {value!r}")
    if not allow_empty and not value.strip():
        raise ConfigError(f"{_join(path, key)} must not be empty")
    if choices is not None and value not in choices:
        raise ConfigError(f"{_join(path, key)} must be one of {sorted(choices)}, got {value!r}")
    return value


def _str_list(table, key, path, min_len=1):
    value = _value(table, key, path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{_join(path, key)} must be a list of strings")
    if len(value) < min_len:
        raise ConfigError(f"{_join(path, key)} must have at least {min_len} entries")
    return list(value)


def _int_list(table, key, path, low=None, min_len=1):
    value = _value(table, key, path)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ConfigError(f"{_join(path, key)} must be a list of integers")
    if len(value) < min_len:
        raise ConfigError(f"{_join(path, key)} must have at least {min_len} entries")
    for item in value:
        _bounds(item, key, path, low, None)
    return list(value)


def _float_list(table, key, path, low=None, high=None, min_len=1):
    value = _value(table, key, path)
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        raise ConfigError(f"{_join(path, key)} must be a list of numbers")
    if len(value) < min_len:
        raise ConfigError(f"{_join(path, key)} must have at least {min_len} entries")
    out = [float(item) for item in value]
    for item in out:
        _bounds(item, key, path, low, high)
    return out


def _url(table, key, path):
    value = _str(table, key, path)
    if not value.startswith("https://"):
        raise ConfigError(f"{_join(path, key)} must be an https url, got {value!r}")
    return value


def _date(table, key, path):
    value = _str(table, key, path)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{_join(path, key)} must be an ISO date, got {value!r}") from exc


def _zone(table, key, path):
    value = _str(table, key, path)
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"{_join(path, key)} is not a known timezone: {value!r}") from exc
    return value


@dataclass(frozen=True)
class RunConfig:
    seed: int
    profile: str
    run_id: str


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    refdata_dir: Path
    raw_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_db: Path
    tmp_dir: Path
    reports_dir: Path
    truth_dir: Path

    def ensure(self):
        for target in (self.raw_dir, self.bronze_dir, self.silver_dir, self.tmp_dir, self.reports_dir, self.truth_dir):
            target.mkdir(parents=True, exist_ok=True)
        self.gold_db.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DuckConfig:
    memory_limit_gb: float
    threads: int
    preserve_insertion_order: bool


@dataclass(frozen=True)
class DownloadConfig:
    timeout_seconds: float
    retries: int
    backoff_seconds: float
    chunk_bytes: int
    min_bytes: int
    reject_html: bool


@dataclass(frozen=True)
class FleetConfig:
    name: str
    enabled: bool
    meter_batches: int
    timezone: str
    interval_minutes: int
    latitude: float
    longitude: float
    physical_max_kwh_per_interval: float
    has_daily_register: bool
    has_tariff_trial: bool

    @property
    def intervals_per_day(self) -> int:
        return (24 * 60) // self.interval_minutes

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class LondonSourceConfig:
    kind: str
    blocks: int
    archive_url: str
    mirror_url: str
    tariff_url: str
    expected_columns: tuple


@dataclass(frozen=True)
class PortugalSourceConfig:
    kind: str
    archive_url: str
    mirror_url: str
    member: str
    delimiter: str
    decimal: str


@dataclass(frozen=True)
class WeatherSourceConfig:
    kind: str
    archive_url: str
    hourly_variable: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class WeatherShapeConfig:
    mean_c: float
    seasonal_amplitude_c: float
    peak_day_of_year: int
    diurnal_range_c: float
    noise_sd_c: float
    noise_phi: float


@dataclass(frozen=True)
class PortugalSynthConfig:
    base_kwh_per_day_lognorm_mu: float
    base_kwh_per_day_lognorm_sigma: float
    heating_share: float
    cooling_share: float
    creation_spread_days: int
    late_client_fraction: float
    noise_cv: float


@dataclass(frozen=True)
class TouTrialConfig:
    trial_start: date
    trial_end: date
    high_start_hour: int
    high_end_hour: int
    low_start_hour: int
    low_end_hour: int
    event_days_per_week: int
    elasticity_mean: float
    elasticity_sd: float
    shift_fraction: float


@dataclass(frozen=True)
class DefectConfig:
    spike_rate_per_meter: float
    spike_multiplier_low: float
    spike_multiplier_high: float
    stuck_probability: float
    stuck_min_intervals: int
    stuck_max_intervals: int
    zero_run_probability: float
    zero_run_min_intervals: int
    zero_run_max_intervals: int


@dataclass(frozen=True)
class SyntheticConfig:
    london_meters: int
    portugal_meters: int
    start_date: date
    end_date: date
    enrollment_spread_days: int
    min_span_days: int
    tou_share: float
    acorn_groups: tuple
    base_kwh_per_day_lognorm_mu: float
    base_kwh_per_day_lognorm_sigma: float
    heating_share: float
    cooling_share: float
    missing_read_rate: float
    omitted_read_share: float
    gap_burst_rate: float
    gap_burst_max_intervals: int
    noise_cv: float
    weather: dict
    portugal: PortugalSynthConfig
    tou: TouTrialConfig
    defects: DefectConfig


@dataclass(frozen=True)
class SpikeRuleConfig:
    enabled: bool
    profile_ratio: float
    floor_kwh: float


@dataclass(frozen=True)
class StuckRuleConfig:
    enabled: bool
    run_length: int
    min_kwh: float


@dataclass(frozen=True)
class ZeroRunRuleConfig:
    enabled: bool
    run_length: int
    min_history_mean_kwh: float


@dataclass(frozen=True)
class DayLevelRuleConfig:
    enabled: bool
    high_ratio: float
    low_ratio: float
    min_like_days: int


@dataclass(frozen=True)
class RegisterRuleConfig:
    enabled: bool
    tolerance_kwh: float
    tolerance_ratio: float


@dataclass(frozen=True)
class VeeConfig:
    enabled: bool
    short_gap_intervals: int
    like_day_window_weeks: int
    max_estimate_intervals: int
    min_history_days: int
    spike: SpikeRuleConfig
    stuck: StuckRuleConfig
    zero_run: ZeroRunRuleConfig
    day_level: DayLevelRuleConfig
    register: RegisterRuleConfig


@dataclass(frozen=True)
class ChangepointConfig:
    grid_step_c: float
    grid_quantile_low: float
    grid_quantile_high: float
    min_days: int
    min_points_per_segment: int
    min_slope_kwh_per_degree_day: float
    max_cvrmse: float
    min_r2: float
    refine_step_c: float


@dataclass(frozen=True)
class ClusterConfig:
    k_min: int
    k_max: int
    restarts: int
    max_iter: int
    tolerance: float
    min_days_per_meter: int


@dataclass(frozen=True)
class TamperConfig:
    constant_bypass_retain: float
    night_bypass_retain: float
    freeze_quantile: float
    drift_final_retain: float
    min_event_days: int


@dataclass(frozen=True)
class TheftConfig:
    tamper_fraction: float
    holdout_fraction: float
    reference_days: int
    min_reference_days: int
    test_days: int
    night_start_hour: int
    night_end_hour: int
    cusum_k: float
    cusum_h: float
    transformer_size: int
    technical_loss_ratio: float
    loss_tolerance: float
    precision_at_k: tuple
    tamper: TamperConfig


@dataclass(frozen=True)
class DtouConfig:
    event_price_label: str
    pre_days: int
    post_days: int
    placebo_draws: int
    cluster_standard_errors: bool
    min_meters_per_group: int


@dataclass(frozen=True)
class ForecastConfig:
    horizon_hours: int
    backtest_origins: int
    backtest_step_days: int
    train_days: int
    quantiles: tuple
    max_iter: int
    learning_rate: float
    max_depth: int
    early_stopping_rounds: int
    validation_fraction: float
    calibration_fraction: float
    conformal: bool


@dataclass(frozen=True)
class GoldConfig:
    interval_sample_meters: int
    hdd_base_c: float
    cdd_base_c: float
    profile_min_days: int


@dataclass(frozen=True)
class SpaceConfig:
    interval_meters: int
    edit_rows: int
    backtest_origins: int
    port: int
    title: str
    emoji: str


@dataclass(frozen=True)
class HoldoutConfig:
    batches: int
    gap_events: int
    gap_lengths: tuple
    min_readings: int
    seed_offset: int


@dataclass(frozen=True)
class QualityConfig:
    max_missing_ratio: float
    max_duplicate_ratio: float
    max_negative_ratio: float
    min_meters: int
    min_days: int


@dataclass(frozen=True)
class Config:
    run: RunConfig
    paths: PathsConfig
    duckdb: DuckConfig
    download: DownloadConfig
    fleets: dict
    london_source: LondonSourceConfig
    portugal_source: PortugalSourceConfig
    weather_source: WeatherSourceConfig
    synthetic: SyntheticConfig
    gold: GoldConfig
    holdout: HoldoutConfig
    space: SpaceConfig
    vee: VeeConfig
    changepoint: ChangepointConfig
    cluster: ClusterConfig
    theft: TheftConfig
    dtou: DtouConfig
    forecast: ForecastConfig
    quality: QualityConfig

    def fleet(self, name: str) -> FleetConfig:
        if name not in self.fleets:
            raise ConfigError(f"unknown fleet {name!r}")
        return self.fleets[name]

    @property
    def enabled_fleets(self) -> tuple:
        return tuple(name for name, cfg in self.fleets.items() if cfg.enabled)


def _paths(table, root):
    def resolve(key):
        return (root / _str(table, key, "paths")).resolve()

    return PathsConfig(
        root=root.resolve(),
        refdata_dir=resolve("refdata_dir"),
        raw_dir=resolve("raw_dir"),
        bronze_dir=resolve("bronze_dir"),
        silver_dir=resolve("silver_dir"),
        gold_db=resolve("gold_db"),
        tmp_dir=resolve("tmp_dir"),
        reports_dir=resolve("reports_dir"),
        truth_dir=resolve("truth_dir"),
    )


def _fleet(table, name):
    path = "fleet." + name
    interval = _int(table, "interval_minutes", path, low=1, high=60)
    if (24 * 60) % interval != 0:
        raise ConfigError(f"{path}.interval_minutes must divide 1440, got {interval}")
    return FleetConfig(
        name=name,
        enabled=_bool(table, "enabled", path),
        meter_batches=_int(table, "meter_batches", path, low=1, high=4096),
        timezone=_zone(table, "timezone", path),
        interval_minutes=interval,
        latitude=_float(table, "latitude", path, low=-90.0, high=90.0),
        longitude=_float(table, "longitude", path, low=-180.0, high=180.0),
        physical_max_kwh_per_interval=_float(table, "physical_max_kwh_per_interval", path, low=0.001),
        has_daily_register=_bool(table, "has_daily_register", path),
        has_tariff_trial=_bool(table, "has_tariff_trial", path),
    )


def _vee(table):
    return VeeConfig(
        enabled=_bool(table, "enabled", "vee"),
        short_gap_intervals=_int(table, "short_gap_intervals", "vee", low=1, high=192),
        like_day_window_weeks=_int(table, "like_day_window_weeks", "vee", low=1, high=52),
        max_estimate_intervals=_int(table, "max_estimate_intervals", "vee", low=1),
        min_history_days=_int(table, "min_history_days", "vee", low=1),
        spike=SpikeRuleConfig(
            enabled=_bool(_table(table, "spike", "vee"), "enabled", "vee.spike"),
            profile_ratio=_float(_table(table, "spike", "vee"), "profile_ratio", "vee.spike", low=1.0),
            floor_kwh=_float(_table(table, "spike", "vee"), "floor_kwh", "vee.spike", low=0.0),
        ),
        stuck=StuckRuleConfig(
            enabled=_bool(_table(table, "stuck", "vee"), "enabled", "vee.stuck"),
            run_length=_int(_table(table, "stuck", "vee"), "run_length", "vee.stuck", low=2),
            min_kwh=_float(_table(table, "stuck", "vee"), "min_kwh", "vee.stuck", low=0.0),
        ),
        zero_run=ZeroRunRuleConfig(
            enabled=_bool(_table(table, "zero_run", "vee"), "enabled", "vee.zero_run"),
            run_length=_int(_table(table, "zero_run", "vee"), "run_length", "vee.zero_run", low=2),
            min_history_mean_kwh=_float(
                _table(table, "zero_run", "vee"), "min_history_mean_kwh", "vee.zero_run", low=0.0
            ),
        ),
        day_level=DayLevelRuleConfig(
            enabled=_bool(_table(table, "day_level", "vee"), "enabled", "vee.day_level"),
            high_ratio=_float(_table(table, "day_level", "vee"), "high_ratio", "vee.day_level", low=1.0),
            low_ratio=_float(_table(table, "day_level", "vee"), "low_ratio", "vee.day_level", low=0.0, high=1.0),
            min_like_days=_int(_table(table, "day_level", "vee"), "min_like_days", "vee.day_level", low=1),
        ),
        register=RegisterRuleConfig(
            enabled=_bool(_table(table, "register", "vee"), "enabled", "vee.register"),
            tolerance_kwh=_float(_table(table, "register", "vee"), "tolerance_kwh", "vee.register", low=0.0),
            tolerance_ratio=_float(
                _table(table, "register", "vee"), "tolerance_ratio", "vee.register", low=0.0, high=1.0
            ),
        ),
    )


SAMPLE_MAP = (
    ("raw_dir", ("paths", "raw_dir")),
    ("bronze_dir", ("paths", "bronze_dir")),
    ("silver_dir", ("paths", "silver_dir")),
    ("gold_db", ("paths", "gold_db")),
    ("tmp_dir", ("paths", "tmp_dir")),
    ("reports_dir", ("paths", "reports_dir")),
    ("truth_dir", ("paths", "truth_dir")),
    ("london_meters", ("synthetic", "london_meters")),
    ("portugal_meters", ("synthetic", "portugal_meters")),
    ("start_date", ("synthetic", "start_date")),
    ("end_date", ("synthetic", "end_date")),
    ("enrollment_spread_days", ("synthetic", "enrollment_spread_days")),
    ("min_span_days", ("synthetic", "min_span_days")),
    ("creation_spread_days", ("synthetic", "portugal", "creation_spread_days")),
    ("trial_start", ("synthetic", "tou", "trial_start")),
    ("trial_end", ("synthetic", "tou", "trial_end")),
    ("london_blocks", ("source", "london", "blocks")),
    ("london_meter_batches", ("fleet", "london", "meter_batches")),
    ("portugal_meter_batches", ("fleet", "portugal", "meter_batches")),
    ("interval_sample_meters", ("gold", "interval_sample_meters")),
    ("transformer_size", ("theft", "transformer_size")),
    ("quality_min_meters", ("quality", "min_meters")),
    ("quality_min_days", ("quality", "min_days")),
    ("dtou_min_meters_per_group", ("dtou", "min_meters_per_group")),
    ("forecast_backtest_origins", ("forecast", "backtest_origins")),
    ("forecast_max_iter", ("forecast", "max_iter")),
    ("forecast_train_days", ("forecast", "train_days")),
    ("cluster_restarts", ("cluster", "restarts")),
)


def apply_sample(raw):
    sample = _table(raw, "sample", "")
    for key, target in SAMPLE_MAP:
        if key not in sample:
            raise ConfigError(f"missing key sample.{key}")
        table = raw
        for step in target[:-1]:
            table = _table(table, step, "")
        table[target[-1]] = sample[key]


def load_config(path="config.toml", profile=None) -> Config:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if profile is not None:
        _table(raw, "run", "")["profile"] = profile
    if _table(raw, "run", "").get("profile") == "sample":
        apply_sample(raw)

    root = config_path.parent
    run_table = _table(raw, "run", "")
    duck_table = _table(raw, "duckdb", "")
    download_table = _table(raw, "download", "")
    fleet_table = _table(raw, "fleet", "")
    source_table = _table(raw, "source", "")
    synthetic_table = _table(raw, "synthetic", "")
    changepoint_table = _table(raw, "changepoint", "")
    cluster_table = _table(raw, "cluster", "")
    theft_table = _table(raw, "theft", "")
    dtou_table = _table(raw, "dtou", "")
    forecast_table = _table(raw, "forecast", "")
    gold_table = _table(raw, "gold", "")
    space_table = _table(raw, "space", "")
    holdout_table = _table(raw, "holdout", "")
    quality_table = _table(raw, "quality", "")

    fleets = {}
    for name in sorted(fleet_table):
        fleets[name] = _fleet(_table(fleet_table, name, "fleet"), name)
    if not fleets:
        raise ConfigError("at least one [fleet.<name>] table is required")
    if not any(cfg.enabled for cfg in fleets.values()):
        raise ConfigError("at least one fleet must be enabled")

    london_table = _table(source_table, "london", "source")
    portugal_table = _table(source_table, "portugal", "source")
    weather_table = _table(source_table, "weather", "source")

    weather_start = _date(weather_table, "start_date", "source.weather")
    weather_end = _date(weather_table, "end_date", "source.weather")
    if weather_start >= weather_end:
        raise ConfigError("source.weather.start_date must be before end_date")

    synth_start = _date(synthetic_table, "start_date", "synthetic")
    synth_end = _date(synthetic_table, "end_date", "synthetic")
    if synth_start >= synth_end:
        raise ConfigError("synthetic.start_date must be before end_date")
    span_days = (synth_end - synth_start).days
    min_span = _int(synthetic_table, "min_span_days", "synthetic", low=1)
    spread = _int(synthetic_table, "enrollment_spread_days", "synthetic", low=0)
    if min_span + spread > span_days:
        raise ConfigError(
            f"synthetic.min_span_days plus enrollment_spread_days ({min_span + spread}) "
            f"exceed the {span_days} day window"
        )
    heating_share = _float(synthetic_table, "heating_share", "synthetic", low=0.0, high=1.0)
    cooling_share = _float(synthetic_table, "cooling_share", "synthetic", low=0.0, high=1.0)
    if heating_share + cooling_share >= 1.0:
        raise ConfigError("synthetic.heating_share plus cooling_share must stay below 1.0")

    weather_shapes = {}
    synth_weather_table = _table(synthetic_table, "weather", "synthetic")
    for name in sorted(synth_weather_table):
        shape_path = "synthetic.weather." + name
        shape_table = _table(synth_weather_table, name, "synthetic.weather")
        weather_shapes[name] = WeatherShapeConfig(
            mean_c=_float(shape_table, "mean_c", shape_path, low=-40.0, high=50.0),
            seasonal_amplitude_c=_float(shape_table, "seasonal_amplitude_c", shape_path, low=0.0, high=40.0),
            peak_day_of_year=_int(shape_table, "peak_day_of_year", shape_path, low=1, high=366),
            diurnal_range_c=_float(shape_table, "diurnal_range_c", shape_path, low=0.0, high=40.0),
            noise_sd_c=_float(shape_table, "noise_sd_c", shape_path, low=0.0, high=20.0),
            noise_phi=_float(shape_table, "noise_phi", shape_path, low=0.0, high=0.999),
        )
    for name in fleets:
        if name not in weather_shapes:
            raise ConfigError(f"missing table [synthetic.weather.{name}] for fleet {name}")

    portugal_synth_table = _table(synthetic_table, "portugal", "synthetic")
    pt_heating = _float(portugal_synth_table, "heating_share", "synthetic.portugal", low=0.0, high=1.0)
    pt_cooling = _float(portugal_synth_table, "cooling_share", "synthetic.portugal", low=0.0, high=1.0)
    if pt_heating + pt_cooling >= 1.0:
        raise ConfigError("synthetic.portugal heating_share plus cooling_share must stay below 1.0")

    tou_table = _table(synthetic_table, "tou", "synthetic")
    trial_start = _date(tou_table, "trial_start", "synthetic.tou")
    trial_end = _date(tou_table, "trial_end", "synthetic.tou")
    if trial_start >= trial_end:
        raise ConfigError("synthetic.tou.trial_start must be before trial_end")
    if trial_start < synth_start or trial_end > synth_end:
        raise ConfigError("synthetic.tou trial window must sit inside the synthetic window")
    high_start = _int(tou_table, "high_start_hour", "synthetic.tou", low=0, high=23)
    high_end = _int(tou_table, "high_end_hour", "synthetic.tou", low=1, high=24)
    if high_start >= high_end:
        raise ConfigError("synthetic.tou.high_start_hour must be before high_end_hour")

    defect_table = _table(synthetic_table, "defects", "synthetic")
    stuck_min = _int(defect_table, "stuck_min_intervals", "synthetic.defects", low=2)
    stuck_max = _int(defect_table, "stuck_max_intervals", "synthetic.defects", low=2)
    if stuck_min >= stuck_max:
        raise ConfigError("synthetic.defects.stuck_min_intervals must be below stuck_max_intervals")
    zero_min = _int(defect_table, "zero_run_min_intervals", "synthetic.defects", low=2)
    zero_max = _int(defect_table, "zero_run_max_intervals", "synthetic.defects", low=2)
    if zero_min >= zero_max:
        raise ConfigError("synthetic.defects.zero_run_min_intervals must be below zero_run_max_intervals")
    spike_low = _float(defect_table, "spike_multiplier_low", "synthetic.defects", low=1.0)
    spike_high = _float(defect_table, "spike_multiplier_high", "synthetic.defects", low=1.0)
    if spike_low >= spike_high:
        raise ConfigError("synthetic.defects.spike_multiplier_low must be below spike_multiplier_high")

    grid_low = _float(changepoint_table, "grid_quantile_low", "changepoint", low=0.0, high=0.5)
    grid_high = _float(changepoint_table, "grid_quantile_high", "changepoint", low=0.5, high=1.0)
    if grid_low >= grid_high:
        raise ConfigError("changepoint.grid_quantile_low must be below grid_quantile_high")
    grid_step = _float(changepoint_table, "grid_step_c", "changepoint", low=0.01, high=5.0)
    refine_step = _float(changepoint_table, "refine_step_c", "changepoint", low=0.001, high=5.0)
    if refine_step > grid_step:
        raise ConfigError("changepoint.refine_step_c must not exceed grid_step_c")

    k_min = _int(cluster_table, "k_min", "cluster", low=2)
    k_max = _int(cluster_table, "k_max", "cluster", low=2)
    if k_min > k_max:
        raise ConfigError("cluster.k_min must not exceed cluster.k_max")

    if _int(theft_table, "min_reference_days", "theft", low=30) > _int(theft_table, "reference_days", "theft", low=30):
        raise ConfigError("theft.min_reference_days must not exceed theft.reference_days")
    night_start = _int(theft_table, "night_start_hour", "theft", low=0, high=23)
    night_end = _int(theft_table, "night_end_hour", "theft", low=1, high=24)
    if night_start >= night_end:
        raise ConfigError("theft.night_start_hour must be before night_end_hour")
    tamper_table = _table(theft_table, "tamper", "theft")

    quantiles = _float_list(forecast_table, "quantiles", "forecast", low=0.0, high=1.0, min_len=1)
    if any(value <= 0.0 or value >= 1.0 for value in quantiles):
        raise ConfigError("forecast.quantiles must lie strictly between 0 and 1")
    if quantiles != sorted(quantiles) or len(set(quantiles)) != len(quantiles):
        raise ConfigError("forecast.quantiles must be strictly increasing")

    return Config(
        run=RunConfig(
            seed=_int(run_table, "seed", "run", low=0),
            profile=_str(run_table, "profile", "run", choices={"full", "sample"}),
            run_id=_str(run_table, "run_id", "run"),
        ),
        paths=_paths(_table(raw, "paths", ""), root),
        duckdb=DuckConfig(
            memory_limit_gb=_float(duck_table, "memory_limit_gb", "duckdb", low=0.25, high=512.0),
            threads=_int(duck_table, "threads", "duckdb", low=1, high=256),
            preserve_insertion_order=_bool(duck_table, "preserve_insertion_order", "duckdb"),
        ),
        download=DownloadConfig(
            timeout_seconds=_float(download_table, "timeout_seconds", "download", low=1.0),
            retries=_int(download_table, "retries", "download", low=1, high=20),
            backoff_seconds=_float(download_table, "backoff_seconds", "download", low=0.0),
            chunk_bytes=_int(download_table, "chunk_bytes", "download", low=4096),
            min_bytes=_int(download_table, "min_bytes", "download", low=1),
            reject_html=_bool(download_table, "reject_html", "download"),
        ),
        fleets=fleets,
        london_source=LondonSourceConfig(
            kind=_str(london_table, "kind", "source.london", choices={"lcl"}),
            blocks=_int(london_table, "blocks", "source.london", low=1, high=1000),
            archive_url=_url(london_table, "archive_url", "source.london"),
            mirror_url=_url(london_table, "mirror_url", "source.london"),
            tariff_url=_url(london_table, "tariff_url", "source.london"),
            expected_columns=tuple(_str_list(london_table, "expected_columns", "source.london", min_len=3)),
        ),
        portugal_source=PortugalSourceConfig(
            kind=_str(portugal_table, "kind", "source.portugal", choices={"uci"}),
            archive_url=_url(portugal_table, "archive_url", "source.portugal"),
            mirror_url=_url(portugal_table, "mirror_url", "source.portugal"),
            member=_str(portugal_table, "member", "source.portugal"),
            delimiter=_str(portugal_table, "delimiter", "source.portugal"),
            decimal=_str(portugal_table, "decimal", "source.portugal"),
        ),
        weather_source=WeatherSourceConfig(
            kind=_str(weather_table, "kind", "source.weather", choices={"open-meteo"}),
            archive_url=_url(weather_table, "archive_url", "source.weather"),
            hourly_variable=_str(weather_table, "hourly_variable", "source.weather"),
            start_date=weather_start,
            end_date=weather_end,
        ),
        synthetic=SyntheticConfig(
            london_meters=_int(synthetic_table, "london_meters", "synthetic", low=1),
            portugal_meters=_int(synthetic_table, "portugal_meters", "synthetic", low=1),
            start_date=synth_start,
            end_date=synth_end,
            enrollment_spread_days=spread,
            min_span_days=min_span,
            tou_share=_float(synthetic_table, "tou_share", "synthetic", low=0.0, high=1.0),
            acorn_groups=tuple(_str_list(synthetic_table, "acorn_groups", "synthetic", min_len=2)),
            base_kwh_per_day_lognorm_mu=_float(synthetic_table, "base_kwh_per_day_lognorm_mu", "synthetic"),
            base_kwh_per_day_lognorm_sigma=_float(
                synthetic_table, "base_kwh_per_day_lognorm_sigma", "synthetic", low=0.01, high=3.0
            ),
            heating_share=heating_share,
            cooling_share=cooling_share,
            missing_read_rate=_float(synthetic_table, "missing_read_rate", "synthetic", low=0.0, high=0.5),
            omitted_read_share=_float(synthetic_table, "omitted_read_share", "synthetic", low=0.0, high=1.0),
            gap_burst_rate=_float(synthetic_table, "gap_burst_rate", "synthetic", low=0.0, high=0.5),
            gap_burst_max_intervals=_int(synthetic_table, "gap_burst_max_intervals", "synthetic", low=1),
            noise_cv=_float(synthetic_table, "noise_cv", "synthetic", low=0.0, high=2.0),
            weather=weather_shapes,
            portugal=PortugalSynthConfig(
                base_kwh_per_day_lognorm_mu=_float(
                    portugal_synth_table, "base_kwh_per_day_lognorm_mu", "synthetic.portugal"
                ),
                base_kwh_per_day_lognorm_sigma=_float(
                    portugal_synth_table, "base_kwh_per_day_lognorm_sigma", "synthetic.portugal", low=0.01, high=3.0
                ),
                heating_share=pt_heating,
                cooling_share=pt_cooling,
                creation_spread_days=_int(portugal_synth_table, "creation_spread_days", "synthetic.portugal", low=0),
                late_client_fraction=_float(
                    portugal_synth_table, "late_client_fraction", "synthetic.portugal", low=0.0, high=1.0
                ),
                noise_cv=_float(portugal_synth_table, "noise_cv", "synthetic.portugal", low=0.0, high=2.0),
            ),
            tou=TouTrialConfig(
                trial_start=trial_start,
                trial_end=trial_end,
                high_start_hour=high_start,
                high_end_hour=high_end,
                low_start_hour=_int(tou_table, "low_start_hour", "synthetic.tou", low=0, high=23),
                low_end_hour=_int(tou_table, "low_end_hour", "synthetic.tou", low=0, high=23),
                event_days_per_week=_int(tou_table, "event_days_per_week", "synthetic.tou", low=1, high=7),
                elasticity_mean=_float(tou_table, "elasticity_mean", "synthetic.tou", low=0.0, high=0.9),
                elasticity_sd=_float(tou_table, "elasticity_sd", "synthetic.tou", low=0.0, high=0.5),
                shift_fraction=_float(tou_table, "shift_fraction", "synthetic.tou", low=0.0, high=1.0),
            ),
            defects=DefectConfig(
                spike_rate_per_meter=_float(defect_table, "spike_rate_per_meter", "synthetic.defects", low=0.0),
                spike_multiplier_low=spike_low,
                spike_multiplier_high=spike_high,
                stuck_probability=_float(defect_table, "stuck_probability", "synthetic.defects", low=0.0, high=1.0),
                stuck_min_intervals=stuck_min,
                stuck_max_intervals=stuck_max,
                zero_run_probability=_float(
                    defect_table, "zero_run_probability", "synthetic.defects", low=0.0, high=1.0
                ),
                zero_run_min_intervals=zero_min,
                zero_run_max_intervals=zero_max,
            ),
        ),
        gold=GoldConfig(
            interval_sample_meters=_int(gold_table, "interval_sample_meters", "gold", low=1),
            hdd_base_c=_float(gold_table, "hdd_base_c", "gold", low=-20.0, high=40.0),
            cdd_base_c=_float(gold_table, "cdd_base_c", "gold", low=-20.0, high=40.0),
            profile_min_days=_int(gold_table, "profile_min_days", "gold", low=1),
        ),
        holdout=HoldoutConfig(
            batches=_int(holdout_table, "batches", "holdout", low=1, high=512),
            gap_events=_int(holdout_table, "gap_events", "holdout", low=1, high=500),
            gap_lengths=tuple(_int_list(holdout_table, "gap_lengths", "holdout", low=1)),
            min_readings=_int(holdout_table, "min_readings", "holdout", low=100),
            seed_offset=_int(holdout_table, "seed_offset", "holdout", low=0),
        ),
        space=SpaceConfig(
            interval_meters=_int(space_table, "interval_meters", "space", low=1),
            edit_rows=_int(space_table, "edit_rows", "space", low=1),
            backtest_origins=_int(space_table, "backtest_origins", "space", low=1),
            port=_int(space_table, "port", "space", low=1, high=65535),
            title=_str(space_table, "title", "space"),
            emoji=_str(space_table, "emoji", "space"),
        ),
        vee=_vee(_table(raw, "vee", "")),
        changepoint=ChangepointConfig(
            grid_step_c=grid_step,
            grid_quantile_low=grid_low,
            grid_quantile_high=grid_high,
            min_days=_int(changepoint_table, "min_days", "changepoint", low=10),
            min_points_per_segment=_int(changepoint_table, "min_points_per_segment", "changepoint", low=2),
            min_slope_kwh_per_degree_day=_float(
                changepoint_table, "min_slope_kwh_per_degree_day", "changepoint", low=0.0
            ),
            max_cvrmse=_float(changepoint_table, "max_cvrmse", "changepoint", low=0.0),
            min_r2=_float(changepoint_table, "min_r2", "changepoint", low=0.0, high=1.0),
            refine_step_c=refine_step,
        ),
        cluster=ClusterConfig(
            k_min=k_min,
            k_max=k_max,
            restarts=_int(cluster_table, "restarts", "cluster", low=1, high=200),
            max_iter=_int(cluster_table, "max_iter", "cluster", low=1, high=10000),
            tolerance=_float(cluster_table, "tolerance", "cluster", low=0.0),
            min_days_per_meter=_int(cluster_table, "min_days_per_meter", "cluster", low=1),
        ),
        theft=TheftConfig(
            tamper_fraction=_float(theft_table, "tamper_fraction", "theft", low=0.0, high=0.5),
            holdout_fraction=_float(theft_table, "holdout_fraction", "theft", low=0.05, high=0.95),
            reference_days=_int(theft_table, "reference_days", "theft", low=30),
            min_reference_days=_int(theft_table, "min_reference_days", "theft", low=30),
            test_days=_int(theft_table, "test_days", "theft", low=14),
            night_start_hour=night_start,
            night_end_hour=night_end,
            cusum_k=_float(theft_table, "cusum_k", "theft", low=0.0),
            cusum_h=_float(theft_table, "cusum_h", "theft", low=0.1),
            transformer_size=_int(theft_table, "transformer_size", "theft", low=2),
            technical_loss_ratio=_float(theft_table, "technical_loss_ratio", "theft", low=0.0, high=0.5),
            loss_tolerance=_float(theft_table, "loss_tolerance", "theft", low=0.0, high=0.5),
            precision_at_k=tuple(_int_list(theft_table, "precision_at_k", "theft", low=1)),
            tamper=TamperConfig(
                constant_bypass_retain=_float(
                    tamper_table, "constant_bypass_retain", "theft.tamper", low=0.0, high=0.99
                ),
                night_bypass_retain=_float(tamper_table, "night_bypass_retain", "theft.tamper", low=0.0, high=0.99),
                freeze_quantile=_float(tamper_table, "freeze_quantile", "theft.tamper", low=0.0, high=1.0),
                drift_final_retain=_float(tamper_table, "drift_final_retain", "theft.tamper", low=0.0, high=0.99),
                min_event_days=_int(tamper_table, "min_event_days", "theft.tamper", low=7),
            ),
        ),
        dtou=DtouConfig(
            event_price_label=_str(dtou_table, "event_price_label", "dtou"),
            pre_days=_int(dtou_table, "pre_days", "dtou", low=1),
            post_days=_int(dtou_table, "post_days", "dtou", low=0),
            placebo_draws=_int(dtou_table, "placebo_draws", "dtou", low=0, high=100000),
            cluster_standard_errors=_bool(dtou_table, "cluster_standard_errors", "dtou"),
            min_meters_per_group=_int(dtou_table, "min_meters_per_group", "dtou", low=2),
        ),
        forecast=ForecastConfig(
            horizon_hours=_int(forecast_table, "horizon_hours", "forecast", low=1, high=168),
            backtest_origins=_int(forecast_table, "backtest_origins", "forecast", low=1, high=520),
            backtest_step_days=_int(forecast_table, "backtest_step_days", "forecast", low=1, high=90),
            train_days=_int(forecast_table, "train_days", "forecast", low=28),
            quantiles=tuple(quantiles),
            max_iter=_int(forecast_table, "max_iter", "forecast", low=10, high=10000),
            learning_rate=_float(forecast_table, "learning_rate", "forecast", low=0.001, high=1.0),
            max_depth=_int(forecast_table, "max_depth", "forecast", low=1, high=64),
            early_stopping_rounds=_int(forecast_table, "early_stopping_rounds", "forecast", low=1, high=1000),
            validation_fraction=_float(forecast_table, "validation_fraction", "forecast", low=0.01, high=0.5),
            calibration_fraction=_float(forecast_table, "calibration_fraction", "forecast", low=0.05, high=0.5),
            conformal=_bool(forecast_table, "conformal", "forecast"),
        ),
        quality=QualityConfig(
            max_missing_ratio=_float(quality_table, "max_missing_ratio", "quality", low=0.0, high=1.0),
            max_duplicate_ratio=_float(quality_table, "max_duplicate_ratio", "quality", low=0.0, high=1.0),
            max_negative_ratio=_float(quality_table, "max_negative_ratio", "quality", low=0.0, high=1.0),
            min_meters=_int(quality_table, "min_meters", "quality", low=1),
            min_days=_int(quality_table, "min_days", "quality", low=1),
        ),
    )
