"""Preprocessing contracts; no fitted transformations exist in Phase 0."""

from typing import Protocol

from riskbyte_ml.contracts import Record


class Preprocessor(Protocol):
    def transform(self, record: Record) -> Record:
        """Apply the immutable preprocessing fitted only on training data."""
        ...
