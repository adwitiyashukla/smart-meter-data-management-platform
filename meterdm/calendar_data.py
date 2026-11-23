from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np

UK_FILE = "uk_bank_holidays.csv"
PT_FILE = "pt_holidays.csv"

FLEET_FILES = {"london": UK_FILE, "portugal": PT_FILE}


def load_holidays(refdata_dir: Path, fleet: str) -> frozenset:
    if fleet not in FLEET_FILES:
        raise KeyError(f"no holiday file mapped for fleet {fleet!r}")
    path = Path(refdata_dir) / FLEET_FILES[fleet]
    if not path.is_file():
        raise FileNotFoundError(f"holiday reference file missing: {path}")
    days = set()
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["date", "name"]:
            raise ValueError(f"unexpected holiday columns in {path}: {reader.fieldnames}")
        for row in reader:
            days.add(date.fromisoformat(row["date"]))
    if not days:
        raise ValueError(f"holiday reference file is empty: {path}")
    return frozenset(days)


def holiday_mask(days: np.ndarray, holidays: frozenset) -> np.ndarray:
    if not holidays:
        return np.zeros(days.shape, dtype=bool)
    marks = np.array(sorted(np.datetime64(day.isoformat(), "D") for day in holidays), dtype="datetime64[D]")
    return np.isin(days.astype("datetime64[D]"), marks)
