"""Finite-only descriptive statistics and report-only outlier analysis."""

import numpy as np
import pandas as pd

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.models import NumericStatistics, OutlierSummary
from riskbyte_ml.data.values import numeric_view, safe_float


def descriptive_statistics(series: pd.Series, config: DatasetConfig) -> NumericStatistics:
    finite, missing, invalid, infinite = numeric_view(series, config)
    values = finite.dropna().to_numpy(dtype=float)
    result = NumericStatistics(
        count=len(values), missing_count=missing, invalid_count=invalid, infinite_count=infinite
    )
    if not len(values):
        return result
    # Scaling keeps means, quantile interpolation and standardized moments stable.
    scale = float(np.max(np.abs(values))) or 1.0
    scaled = values / scale
    mean = scaled.mean()
    quartiles = np.quantile(scaled, [0.25, 0.5, 0.75])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        variance = scaled.var(ddof=1) if len(values) > 1 else None
        std = np.sqrt(variance) if variance is not None else None
        skewness = None
        if len(values) > 2 and std is not None and std > 0:
            n = len(values)
            skewness = n / ((n - 1) * (n - 2)) * np.sum(((scaled - mean) / std) ** 3)
        updates = dict(
            mean=safe_float(mean * scale),
            median=safe_float(quartiles[1] * scale),
            minimum=safe_float(values.min()),
            maximum=safe_float(values.max()),
            variance=safe_float(variance * scale * scale) if variance is not None else None,
            standard_deviation=safe_float(std * scale) if std is not None else None,
            q1=safe_float(quartiles[0] * scale),
            q3=safe_float(quartiles[2] * scale),
            iqr=safe_float((quartiles[2] - quartiles[0]) * scale),
            skewness=safe_float(skewness),
            percentiles={
                f"{p:g}": safe_float(np.quantile(scaled, p / 100) * scale)
                for p in config.percentiles
            },
        )
    return NumericStatistics.model_validate({**result.model_dump(), **updates})


def outlier_overview(series: pd.Series, config: DatasetConfig) -> OutlierSummary:
    finite, _, _, _ = numeric_view(series, config)
    values = finite.dropna().to_numpy(dtype=float)
    result = OutlierSummary(
        finite_count=len(values),
        iqr_multiplier=config.iqr_multiplier,
        lower_bound=None,
        upper_bound=None,
        iqr_outlier_count=0,
        zscore_threshold=config.zscore_threshold,
        zscore_outlier_count=0,
        zscore_reason="insufficient_finite_observations",
    )
    if not len(values):
        return result
    scale = float(np.max(np.abs(values))) or 1.0
    scaled = values / scale
    q1, q3 = np.quantile(scaled, [0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - config.iqr_multiplier * iqr, q3 + config.iqr_multiplier * iqr
    standard_deviation = scaled.std(ddof=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        zcount = (
            int(
                (
                    np.abs((scaled - scaled.mean()) / standard_deviation) > config.zscore_threshold
                ).sum()
            )
            if standard_deviation > 0
            else 0
        )
        updates = dict(
            lower_bound=safe_float(lower * scale),
            upper_bound=safe_float(upper * scale),
            iqr_outlier_count=int(((scaled < lower) | (scaled > upper)).sum()),
            zscore_outlier_count=zcount,
            zscore_reason=None if standard_deviation > 0 else "zero_variance",
        )
    return OutlierSummary.model_validate({**result.model_dump(), **updates})
