"""Feature engineering boundary; feature definitions await an approved dataset."""

from typing import Protocol

from riskbyte_ml.contracts import FeatureVector, Record


class FeatureEngineer(Protocol):
    def transform(self, record: Record) -> FeatureVector:
        """Create ordered, versioned features without future information."""
        ...
