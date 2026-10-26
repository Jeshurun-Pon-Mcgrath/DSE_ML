"""Offline evaluation contract; metrics and thresholds require approved data."""

from collections.abc import Mapping, Sequence
from typing import Protocol


class Evaluator(Protocol):
    def evaluate(self, labels: Sequence[int], scores: Sequence[float]) -> Mapping[str, float]:
        """Report metrics with an explicitly versioned split and score interpretation."""
        ...
