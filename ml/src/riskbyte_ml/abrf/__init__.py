"""Legacy foundation configuration/protocol. Phase 6 lives in contracts/engine/pipeline."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Protocol

from riskbyte_ml.contracts import ComponentSignal


@dataclass(frozen=True)
class ABRFConfig:
    """Neutral starting weights only; must be calibrated and validated experimentally."""

    classifier_weight: float = 1 / 3
    anomaly_weight: float = 1 / 3
    behavioural_weight: float = 1 / 3
    version: str = "provisional-v0"

    def __post_init__(self) -> None:
        weights = (self.classifier_weight, self.anomaly_weight, self.behavioural_weight)
        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("ABRF weights must be finite and non-negative")
        if not isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("ABRF weights must sum to one")
        if not self.version.strip():
            raise ValueError("ABRF configuration requires a version")


class RiskFusion(Protocol):
    def combine(self, signals: Sequence[ComponentSignal], config: ABRFConfig) -> float:
        """Combine experimentally validated component signals; implementation deferred."""
        ...
