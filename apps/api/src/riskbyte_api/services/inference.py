"""Reserved integration boundary for the future ML pipeline.

The service will coordinate classifier, anomaly detector, behavioural engine,
ABRF, explanations and persistence. No scoring is available in Phase 0.
"""

from typing import Protocol

from riskbyte_ml.contracts import FeatureVector, RiskAssessment


class InferenceService(Protocol):
    def assess(self, features: FeatureVector) -> RiskAssessment:
        """Return a versioned assessment using a validated, loaded model bundle."""
        ...
