"""Behavioural engine contract; history must precede the assessed event."""

from collections.abc import Sequence
from typing import Protocol

from riskbyte_ml.contracts import ComponentSignal, FeatureVector, Record


class BehaviouralEngine(Protocol):
    def score(self, features: FeatureVector, history: Sequence[Record]) -> ComponentSignal:
        """Return a versioned behavioural signal using only authorized past context."""
        ...
