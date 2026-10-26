"""Strictly prior event-time history; no outcome or full-dataset aggregates."""

import bisect
import math

import numpy as np
import pandas as pd

from riskbyte_ml.datasets.synthetic_riskbyte.config import CATEGORICAL, NUMERIC, digest

FEATURE_PIPELINE_VERSION = "synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1"
WINDOWS = {"5m": 300, "30m": 1800, "1h": 3600, "24h": 86400}
NOVELTY = {
    "merchant": "merchant_id",
    "merchant_category": "merchant_category",
    "device": "device_id",
    "billing_region": "billing_region",
    "product": "product_code",
}


def finite_numeric(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan, copy=True
    )
    values[~np.isfinite(values)] = np.nan
    return values


def known(value):
    return not pd.isna(value) and bool(str(value).strip())


def behavioural_features(frame: pd.DataFrame, *, horizon_seconds=90 * 86400) -> pd.DataFrame:
    """Features align with input rows. Equal-time peers read the same prior state.

    Validation/test history may include earlier observed transactions, never labels.
    This models sequential online availability, including for the random comparator.
    """
    if not frame.index.is_unique or horizon_seconds <= 0:
        raise ValueError("Unique row indexes and positive configured horizon required")
    times = finite_numeric(frame["TransactionDT"])
    amounts = finite_numeric(frame["TransactionAmt"])
    entities = frame["behavioural_entity_id"].astype("string").to_numpy()
    if not np.isfinite(times).all() or (times < 0).any() or not all(known(e) for e in entities):
        raise ValueError(
            "Behaviour requires finite nonnegative elapsed times and complete entity keys"
        )
    contexts = {
        name: frame[col].to_numpy() if col in frame else np.full(len(frame), None)
        for name, col in NOVELTY.items()
    }
    results = [None] * len(frame)
    groups = {}
    for i, e in enumerate(entities):
        groups.setdefault(str(e), []).append(i)
    for rows in groups.values():
        rows.sort(key=lambda i: times[i])
        prior_times, sorted_amounts = [], []
        seen = {name: set() for name in NOVELTY}
        mean, m2, valid = 0.0, 0.0, 0
        cursor = 0
        while cursor < len(rows):
            stop = cursor + 1
            t = times[rows[cursor]]
            while stop < len(rows) and times[rows[stop]] == t:
                stop += 1
            count = len(prior_times)
            std = math.sqrt(max(0.0, m2 / valid)) if valid else None
            median = (
                (sorted_amounts[(valid - 1) // 2] + sorted_amounts[valid // 2]) / 2
                if valid
                else None
            )
            for i in rows[cursor:stop]:
                a = amounts[i]
                usable = valid > 0 and np.isfinite(a)
                r = {
                    "prior_transaction_count": count,
                    "behavioural_history_count": count,
                    "is_new_behavioural_entity": count == 0,
                    "behaviour_feature_available": count > 0,
                    "amount_history_available": usable,
                    "prior_valid_amount_count": valid,
                    "prior_mean_amount": mean if valid else None,
                    "prior_median_amount": median,
                    "prior_amount_std": std,
                    "prior_min_amount": sorted_amounts[0] if valid else None,
                    "prior_max_amount": sorted_amounts[-1] if valid else None,
                    "amount_minus_prior_mean": a - mean if usable else None,
                    "amount_ratio_to_prior_mean": a / mean if usable and mean != 0 else None,
                    "amount_zscore_prior": (a - mean) / std if usable and std and std > 0 else None,
                    "time_since_previous_transaction": t - prior_times[-1] if count else None,
                    "elapsed_day_index": math.floor(t / 86400),
                    "relative_time_position": t / horizon_seconds,
                }
                for window, seconds in WINDOWS.items():
                    # Window is [t-window, t); boundary inclusive, current time excluded.
                    r[f"transactions_prior_{window}"] = count - bisect.bisect_left(
                        prior_times, t - seconds
                    )
                for name in NOVELTY:
                    value = contexts[name][i]
                    available = count > 0 and known(value) and bool(seen[name])
                    r[f"{name}_novelty_available"] = available
                    r[f"{name}_seen_before"] = str(value) in seen[name] if available else None
                results[i] = {
                    k: (v if not isinstance(v, float | np.floating) or np.isfinite(v) else None)
                    for k, v in r.items()
                }
            # Commit only after every same-time event has been assessed.
            for i in rows[cursor:stop]:
                prior_times.append(t)
                a = amounts[i]
                if np.isfinite(a):
                    valid += 1
                    delta = a - mean
                    mean += delta / valid
                    m2 += delta * (a - mean)
                    bisect.insort(sorted_amounts, a)
                for name in NOVELTY:
                    value = contexts[name][i]
                    if known(value):
                        seen[name].add(str(value))
            cursor = stop
    return pd.DataFrame(results, index=frame.index)


def build_features(frame: pd.DataFrame, *, horizon_seconds=90 * 86400):
    """Allowlist of pre-outcome fields; unknown/outcome fields never propagate."""
    raw = pd.DataFrame(index=frame.index)
    for name in NUMERIC:
        if name in frame:
            raw[name] = finite_numeric(frame[name])
    for name in CATEGORICAL:
        if name in frame:
            raw[name] = frame[name].astype("string").replace(r"^\s*$", pd.NA, regex=True)
    for name in frame:
        if name.startswith("V") and name[1:].isdigit() and 1 <= int(name[1:]) <= 20:
            raw[name] = finite_numeric(frame[name])
    result = pd.concat([raw, behavioural_features(frame, horizon_seconds=horizon_seconds)], axis=1)
    if "TransactionID" in frame:
        result.attrs["row_order_fingerprint"] = digest(frame.TransactionID.astype(str).tolist())
    result.attrs["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION
    result.attrs["relative_time_horizon_seconds"] = horizon_seconds
    return result
