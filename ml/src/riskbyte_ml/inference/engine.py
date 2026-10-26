"""Load trusted frozen components once and score aligned pre-outcome features."""

import joblib

from riskbyte_ml.abrf.contracts import A_THRESHOLD, P_THRESHOLD, FusionInput
from riskbyte_ml.abrf.engine import fuse, fusion_arrays
from riskbyte_ml.anomaly.pipeline import BASELINE_HASH, BASELINE_ID
from riskbyte_ml.explainability.engine import (
    confidence_explanation,
    formula_trace,
    ranked_components,
)
from riskbyte_ml.explainability.inputs import verify_selection
from riskbyte_ml.explainability.metadata import encoded_metadata, top_features
from riskbyte_ml.explainability.pipeline import verify_outputs
from riskbyte_ml.explainability.schemas import (
    SHAP_VERSION,
    AnomalyExplanation,
    BehaviourExplanation,
    ExplanationBundle,
    SupervisedExplanation,
    UnifiedExplanation,
)
from riskbyte_ml.explainability.supervised import TreeShap
from riskbyte_ml.features.behaviour import build_features
from riskbyte_ml.models.supervised.artifacts import load_classifier
from riskbyte_ml.models.supervised.estimators import positive_probability
from riskbyte_ml.preprocessing.pipeline import load_preprocessor
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import read_json


class FrozenSystem:
    def __init__(self, config, root):
        self.bundle, _, self.provenance = verify_selection(config, root)
        snapshot = root / "explainability"
        explanations = ExplanationBundle.model_validate_json(
            (snapshot / "analysis_bundle.json").read_bytes()
        )
        verify_outputs(snapshot, explanations)
        if explanations.version != SHAP_VERSION or explanations.shap_status != "AVAILABLE":
            raise ValueError("Frozen SHAP-complete snapshot required")
        manifest = read_json(root / "features" / "feature_manifest.json")
        family = manifest["families"]["tree_boosting"]
        self.preprocessor = load_preprocessor(
            root / "preprocessing" / family["preprocessing_artifact"],
            FROZEN_FINGERPRINT,
            FROZEN_SPLIT_ID,
        )
        registry = read_json(root / "models" / "model_registry.json")
        self.classifier = load_classifier(
            root / "models" / registry["provenance"]["run_id"][7:] / (BASELINE_ID + ".joblib"),
            BASELINE_HASH,
            FROZEN_FINGERPRINT,
            FROZEN_FEATURE_VERSION,
        )
        self.names = self.classifier.feature_names
        self.metadata = encoded_metadata(manifest, self.names)
        if self.names != self.preprocessor.feature_names:
            raise ValueError("Frozen feature alignment differs")
        self.anomaly = joblib.load(root / "phase5" / "isolation_forest.joblib")
        self.behaviour = joblib.load(root / "phase5" / "behavioural_engine.joblib")
        self.shap = TreeShap(self.classifier.estimator, self.names)
        if self.shap.library_version != "0.53.0" or self.shap.status != "AVAILABLE":
            raise ValueError("Verified SHAP 0.53.0 required")
        self.fusion_config = self.bundle.selected

    def score(self, features):
        matrix = self.preprocessor.transform(features)
        raw = positive_probability(self.classifier.estimator, matrix)
        p = self.classifier.probability(matrix)
        original, a = self.anomaly.score(features)
        b, c, _, _, _ = self.behaviour.score_arrays(features)
        r, weights, contributions = fusion_arrays(
            p, a, b, c, self.fusion_config.weights, self.fusion_config.confidence_curve
        )
        return {
            "P": p,
            "A": a,
            "B": b,
            "confidence": c,
            "risk": r,
            "raw": raw,
            "original_anomaly": original,
            "weights": weights,
            "contributions": contributions,
            "matrix": matrix,
        }

    def explain(self, transaction, feature, cached_scores=None, position=0):
        features = feature.reset_index(drop=True)
        scores = (
            self.score(features)
            if cached_scores is None
            else {name: value[position : position + 1] for name, value in cached_scores.items()}
        )
        b = self.behaviour.explain(features, 0)
        record = FusionInput(
            transaction_id=str(transaction.TransactionID),
            supervised_probability=float(scores["P"][0]),
            supervised_prediction=bool(scores["P"][0] >= P_THRESHOLD),
            anomaly_score=float(scores["A"][0]),
            anomaly_prediction=bool(scores["A"][0] >= A_THRESHOLD),
            behaviour_score=b.behaviour_score,
            behaviour_confidence=b.behaviour_confidence,
            behaviour_available=b.behaviour_score is not None,
        )
        trace = fuse(record, self.fusion_config)
        phi, base, residual = self.shap.values(scores["matrix"])
        positive, negative = top_features(scores["matrix"].toarray()[0], phi[0], self.metadata)
        behaviour = BehaviourExplanation(
            **b.model_copy(update={"components": ranked_components(b.components)}).model_dump(),
            version=self.behaviour.version,
            confidence_explanation=confidence_explanation(
                b, features.iloc[0], transaction, self.behaviour.config
            ),
        )
        return UnifiedExplanation(
            version=SHAP_VERSION,
            transaction_id=record.transaction_id,
            risk_summary={
                "final_risk_score": trace.final_risk_score,
                "risk_level": trace.risk_level,
                "investigation_flag": trace.investigation_flag,
            },
            supervised=SupervisedExplanation(
                model_id=self.classifier.model_id,
                model_version=self.classifier.model_version,
                raw_probability=float(scores["raw"][0]),
                calibrated_probability=record.supervised_probability,
                calibration_version="supervised-classification-v1/sigmoid",
                threshold=P_THRESHOLD,
                prediction=record.supervised_prediction,
                shap_status="AVAILABLE",
                base_value=base,
                additivity_residual=float(residual[0]),
                top_positive_features=positive,
                top_negative_features=negative,
            ),
            behaviour=behaviour,
            anomaly=AnomalyExplanation(
                version=self.anomaly.version,
                score=record.anomaly_score,
                original_model_score=float(scores["original_anomaly"][0]),
                threshold=A_THRESHOLD,
                flag=record.anomaly_prediction,
                effective_abrf_weight=0.0,
                observable_signals=[],
            ),
            fusion=trace,
            formula_trace=formula_trace(record, self.fusion_config),
        )


def future_features(history, incoming):
    """Frozen strictly-prior semantics, including equal-time batching."""
    import pandas as pd

    combined = pd.concat([history, incoming], ignore_index=True)
    return build_features(combined).iloc[len(history) :].reset_index(drop=True)
