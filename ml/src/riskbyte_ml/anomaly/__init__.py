"""Anomaly signal contract; raw anomaly scores are not fraud probabilities."""

from typing import Protocol

from riskbyte_ml.contracts import ComponentSignal, FeatureVector


class AnomalyDetector(Protocol):
    def score(self, features: FeatureVector) -> ComponentSignal:
        """Return a consistently oriented, experimentally normalized anomaly signal."""
        ...
