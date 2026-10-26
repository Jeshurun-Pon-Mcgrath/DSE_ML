"""Four-layer explainability, with optional genuine SHAP and explicit unavailable states."""

from typing import Protocol

from riskbyte_ml.contracts import Explanation, FeatureVector


class Explainer(Protocol):
    def explain(self, features: FeatureVector) -> Explanation:
        """Explain a specified model output against a versioned background dataset."""
        ...
