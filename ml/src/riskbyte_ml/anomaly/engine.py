"""Numeric Isolation Forest contracts and training-only empirical score mappings."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from sklearn.ensemble import IsolationForest

from riskbyte_ml.data.models import ReportModel
from riskbyte_ml.datasets.synthetic_riskbyte.config import EXCLUDED
from riskbyte_ml.features.behaviour import finite_numeric
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID

ANOMALY_VERSION = "anomaly-detection-v1"
INTERPRETABLE = (
    "TransactionAmt",
    "amount_ratio_to_prior_mean",
    "amount_zscore_prior",
    "amount_minus_prior_mean",
    "transactions_prior_5m",
    "transactions_prior_30m",
    "transactions_prior_1h",
    "transactions_prior_24h",
    "time_since_previous_transaction",
    "failed_attempts_before",
    "merchant_seen_before",
    "device_seen_before",
    "billing_region_seen_before",
    "product_seen_before",
    "account_age_days",
    "identity_age_days",
    "is_international",
    "behavioural_history_count",
    "is_new_behavioural_entity",
    "behaviour_feature_available",
    "device_novelty_available",
)


class AnomalyConfig(ReportModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    n_estimators: int = Field(default=150, ge=10, le=500)
    max_samples: int = Field(default=1024, ge=32, le=8192)
    n_jobs: int = Field(default=2, ge=1, le=8)
    max_legitimate_rate: float = Field(default=0.02, gt=0, lt=0.1)
    max_hard_negative_rate: float = Field(default=0.10, gt=0, lt=0.3)
    contamination_candidates: list[float | str] = Field(
        default_factory=lambda: ["auto", 0.005, 0.01, 0.02, 0.03, 0.05]
    )

    @model_validator(mode="after")
    def contamination_valid(self):
        if not self.contamination_candidates or any(
            value != "auto" and (not isinstance(value, float) or not 0 < value <= 0.5)
            for value in self.contamination_candidates
        ):
            raise ValueError("Contamination must be auto or a fraction in (0, .5]")
        return self


@dataclass
class EmpiricalNormalizer:
    reference: np.ndarray

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError("Finite training normalization reference required")
        return cls(np.sort(finite.copy()))

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        # Strict-left CDF: zero/minimum and large zero-mass components map to zero.
        result = np.searchsorted(self.reference, values, side="left") / len(self.reference)
        return np.where(np.isfinite(values), np.clip(result, 0, 1), np.nan)


def feature_contract(features, family):
    if family == "interpretable":
        names = [name for name in INTERPRETABLE if name in features]
    elif family == "broad_numeric":
        names = [
            name
            for name in features
            if pd.api.types.is_numeric_dtype(features[name])
            and name not in {"elapsed_day_index", "relative_time_position"}
        ]
    else:
        raise ValueError("Unknown anomaly feature family")
    if not names or set(names) & EXCLUDED:
        raise ValueError("Invalid anomaly feature contract")
    return names


@dataclass
class NumericPreparation:
    names: list[str]
    medians: np.ndarray

    @classmethod
    def fit(cls, features, names):
        if set(names) & EXCLUDED or not set(names) <= set(features):
            raise ValueError("Forbidden/missing anomaly feature")
        values = np.column_stack([finite_numeric(features[name]) for name in names])
        medians = np.array(
            [
                np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
                for column in values.T
            ]
        )
        return cls(list(names), medians)

    def transform(self, features):
        values = np.column_stack([finite_numeric(features[name]) for name in self.names])
        missing = ~np.isfinite(values)
        prepared = np.clip(np.where(missing, self.medians, values), -1e12, 1e12)
        return np.column_stack([prepared, missing.astype(float)]).astype(np.float32)


@dataclass
class IsolationEngine:
    model: IsolationForest
    preparation: NumericPreparation
    normalization: EmpiricalNormalizer
    population: str
    feature_family: str
    version: str = ANOMALY_VERSION
    dataset_fingerprint: str = FROZEN_FINGERPRINT
    feature_version: str = FROZEN_FEATURE_VERSION
    split_id: str = FROZEN_SPLIT_ID

    @classmethod
    def fit(cls, training_features, names, config, *, population, feature_family):
        """Accept X only. Cohort selection belongs outside fitting, using TRAIN labels only."""
        preparation = NumericPreparation.fit(training_features, names)
        matrix = preparation.transform(training_features)
        model = IsolationForest(
            n_estimators=config.n_estimators,
            max_samples=min(config.max_samples, len(matrix)),
            contamination="auto",
            random_state=config.seed,
            n_jobs=config.n_jobs,
        ).fit(matrix)
        normalization = EmpiricalNormalizer.fit(-model.score_samples(matrix))
        return cls(model, preparation, normalization, population, feature_family)

    def score(self, features):
        original = self.model.score_samples(self.preparation.transform(features))
        return original, self.normalization.transform(-original)
