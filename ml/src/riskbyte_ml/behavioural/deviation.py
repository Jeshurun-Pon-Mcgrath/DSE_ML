"""Transparent deviation from strictly prior context; not a fraud probability."""

from dataclasses import dataclass

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from riskbyte_ml.anomaly.engine import EmpiricalNormalizer
from riskbyte_ml.data.models import ReportModel
from riskbyte_ml.features.behaviour import finite_numeric
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID

BEHAVIOUR_VERSION = "behavioural-deviation-v1"
SIGNALS = {
    "amount_deviation": "amount_ratio_to_prior_mean",
    "amount_zscore": "amount_zscore_prior",
    "transaction_velocity": "transactions_prior_30m",
    "long_window_velocity": "transactions_prior_24h",
    "short_time_gap": "time_since_previous_transaction",
    "failed_attempts": "failed_attempts_before",
    "merchant_novelty": "merchant_seen_before",
    "device_novelty": "device_seen_before",
    "region_novelty": "billing_region_seen_before",
    "product_novelty": "product_seen_before",
}
NOVELTY = {name for name in SIGNALS if name.endswith("novelty")}


class BehaviourConfig(ReportModel):
    weights: dict[str, FiniteFloat] = Field(default_factory=lambda: dict.fromkeys(SIGNALS, 1.0))
    history_saturation: FiniteFloat = Field(default=20.0, gt=0)
    minimum_history: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_weights(self):
        if set(self.weights) != set(SIGNALS) or any(v < 0 for v in self.weights.values()):
            raise ValueError("All component weights must be declared and nonnegative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("Positive total weight required")
        return self


class Component(ReportModel):
    signal: str
    source: str
    score: FiniteFloat | None = Field(ge=0, le=1)
    raw_value: FiniteFloat | None
    contribution: FiniteFloat | None = Field(ge=0, le=1)
    direction: str = "deviation_increasing; not evidence of fraud by itself"
    window: str | None = None
    available: bool


class BehaviourResult(ReportModel):
    behaviour_score: FiniteFloat | None = Field(ge=0, le=1)
    behaviour_confidence: FiniteFloat = Field(ge=0, le=1)
    history_count: int
    history_available: bool
    available_weight_fraction: FiniteFloat
    components: list[Component]


def severity(name, raw):
    if name == "amount_deviation":
        return np.where(raw >= 0, np.abs(np.log(np.maximum(raw, 1e-6))), np.nan)
    if name == "amount_zscore":
        return np.abs(raw)
    if name == "short_time_gap":
        return np.where(raw >= 0, 1 / (1 + np.maximum(raw, 0)), np.nan)
    return raw


@dataclass
class DeviationEngine:
    config: BehaviourConfig
    normalizers: dict[str, EmpiricalNormalizer]
    version: str = BEHAVIOUR_VERSION
    dataset_fingerprint: str = FROZEN_FINGERPRINT
    feature_version: str = FROZEN_FEATURE_VERSION
    split_id: str = FROZEN_SPLIT_ID

    @classmethod
    def fit(cls, training_features, config=None):
        """Training feature distribution only, no labels or evaluation annotations."""
        config = config or BehaviourConfig()
        normalizers = {}
        for name, source in SIGNALS.items():
            if name not in NOVELTY and source in training_features:
                values = severity(name, finite_numeric(training_features[source]))
                if np.isfinite(values).any():
                    normalizers[name] = EmpiricalNormalizer.fit(values)
        return cls(config, normalizers)

    def score_arrays(self, features):
        raw, components = {}, {}
        for name, source in SIGNALS.items():
            values = (
                finite_numeric(features[source])
                if source in features
                else np.full(len(features), np.nan)
            )
            raw[name] = values
            if name in NOVELTY:
                components[name] = np.where(np.isin(values, [0, 1]), 1 - values, np.nan)
            elif name in self.normalizers:
                components[name] = self.normalizers[name].transform(severity(name, values))
            else:
                components[name] = np.full(len(features), np.nan)
        count = finite_numeric(features["behavioural_history_count"])
        history = np.isfinite(count) & (count >= self.config.minimum_history)
        total = sum(self.config.weights.values())
        available = sum(
            self.config.weights[name] * np.isfinite(values) for name, values in components.items()
        )
        numerator = sum(
            self.config.weights[name] * np.nan_to_num(values, nan=0)
            for name, values in components.items()
        )
        scores = np.divide(numerator, available, out=np.zeros(len(features)), where=available > 0)
        scores = np.where(history & (available > 0), np.clip(scores, 0, 1), np.nan)
        confidence = np.where(
            history, count / (count + self.config.history_saturation) * available / total, 0.0
        )
        return scores, np.clip(confidence, 0, 1), raw, components, available

    def explain(self, features, position):
        score, confidence, raw, components, weights = self.score_arrays(features.iloc[[position]])

        def finite(value):
            return float(value) if np.isfinite(value) else None

        count = finite_numeric(features["behavioural_history_count"])[position]
        return BehaviourResult(
            behaviour_score=finite(score[0]),
            behaviour_confidence=float(confidence[0]),
            history_count=int(count),
            history_available=bool(count >= self.config.minimum_history),
            available_weight_fraction=float(weights[0] / sum(self.config.weights.values())),
            components=[
                Component(
                    signal=name,
                    source=SIGNALS[name],
                    score=finite(values[0]),
                    raw_value=finite(raw[name][0]),
                    available=bool(np.isfinite(values[0])),
                    contribution=finite(values[0] * self.config.weights[name] / weights[0])
                    if weights[0] and np.isfinite(score[0])
                    else None,
                    window="30m"
                    if name == "transaction_velocity"
                    else "24h"
                    if name == "long_window_velocity"
                    else None,
                )
                for name, values in components.items()
            ],
        )
