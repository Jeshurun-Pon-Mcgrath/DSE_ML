"""Provisional internal types, not a finalized transaction API or data schema."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

type FeatureVector = Mapping[str, float]
type Record = Mapping[str, object]


@dataclass(frozen=True)
class ComponentSignal:
    name: Literal["classifier", "anomaly", "behavioural"]
    score: float
    version: str


@dataclass(frozen=True)
class Explanation:
    target: str
    feature_contributions: Mapping[str, float]
    baseline: float


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    signals: tuple[ComponentSignal, ...]
    explanation: Explanation | None
    model_bundle_version: str
    abrf_config_version: str
