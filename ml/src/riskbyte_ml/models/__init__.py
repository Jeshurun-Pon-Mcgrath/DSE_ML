"""Classifier inference and offline training contracts."""

from collections.abc import Sequence
from typing import Protocol

from riskbyte_ml.contracts import ComponentSignal, FeatureVector


class Classifier(Protocol):
    def predict(self, features: FeatureVector) -> ComponentSignal:
        """Return a calibrated classifier signal with its artifact version."""
        ...


class Trainer(Protocol):
    def fit(self, features: Sequence[FeatureVector], labels: Sequence[int]) -> Classifier:
        """Fit offline on an approved training partition only."""
        ...
