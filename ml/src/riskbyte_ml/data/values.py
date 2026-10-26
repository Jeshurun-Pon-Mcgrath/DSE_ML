"""Analysis-only views; these functions never mutate source data."""

import math

import numpy as np
import pandas as pd

from riskbyte_ml.data.config import DatasetConfig


def empty_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().eq("").fillna(False)


def missing_mask(series: pd.Series, config: DatasetConfig) -> pd.Series:
    return series.isna() | empty_mask(series) | series.isin(config.na_values)


def numeric_view(series: pd.Series, config: DatasetConfig) -> tuple[pd.Series, int, int, int]:
    missing = missing_mask(series, config)
    parsed = pd.to_numeric(series.where(~missing), errors="coerce").astype("float64")
    invalid = int((parsed.isna() & ~missing).sum())
    infinite = int(np.isinf(parsed.to_numpy()).sum())
    finite = parsed.where(np.isfinite(parsed.to_numpy()))
    return finite, int(missing.sum()), invalid, infinite


def parse_timestamps(series: pd.Series, config: DatasetConfig) -> pd.Series:
    values = series.where(~missing_mask(series, config))
    if config.timestamp_unit is not None:
        numeric = pd.to_numeric(values, errors="coerce").astype("float64")
        numeric = numeric.where(np.isfinite(numeric.to_numpy()))
        if config.timestamp_kind == "elapsed":
            return pd.to_timedelta(numeric, unit=config.timestamp_unit, errors="coerce")
        return pd.to_datetime(
            numeric,
            unit=config.timestamp_unit,
            origin=config.timestamp_origin,
            errors="coerce",
            utc=True,
        ).dt.tz_convert(config.timezone)
    # Naive textual timestamps are interpreted in the explicitly configured timezone.
    # Offset-bearing timestamps retain their meaning and are converted to that timezone.
    parsed = pd.to_datetime(values, format=config.timestamp_format, errors="coerce", utc=True)
    if config.timezone == "UTC":
        return parsed
    aware = (
        values.astype("string")
        .str.contains(r"[T ].*(?:Z|[+-]\d{2}(?::?\d{2})?)$", regex=True, case=False)
        .fillna(False)
    )
    local_values = pd.to_datetime(
        values.where(~aware), format=config.timestamp_format, errors="coerce"
    )
    localized = local_values.dt.tz_localize(config.timezone, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert(config.timezone).where(aware, localized)


def safe_float(value) -> float | None:
    if value is None:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        number = float(value)
    return number if math.isfinite(number) else None


def percentage(count: int, total: int) -> float:
    return count * 100.0 / total if total else 0.0
