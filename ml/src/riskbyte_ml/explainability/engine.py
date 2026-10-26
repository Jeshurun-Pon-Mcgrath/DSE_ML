"""Four separate evidence layers, all tied to exact frozen component outputs."""

import numpy as np
import pandas as pd

from riskbyte_ml.abrf.contracts import A_THRESHOLD, P_THRESHOLD, FusionInput
from riskbyte_ml.abrf.engine import confidence_transform, fuse
from riskbyte_ml.explainability.inputs import equal_scores
from riskbyte_ml.explainability.metadata import top_features
from riskbyte_ml.explainability.schemas import (
    SHAP_VERSION,
    VERSION,
    AnomalyExplanation,
    BehaviourExplanation,
    ConfidenceExplanation,
    FormulaTrace,
    SupervisedExplanation,
    UnifiedExplanation,
)


def ranked_components(components):
    return sorted(
        components,
        key=lambda x: (-(x.contribution if x.contribution is not None else -1), x.signal),
    )


def confidence_explanation(result, row, context, config):
    merchant = bool(row.merchant_novelty_available)
    device = bool(row.device_novelty_available)
    identity = not pd.isna(context.device_id) and bool(str(context.device_id).strip())
    n = result.history_count
    return ConfidenceExplanation(
        prior_transaction_count=n,
        history_saturation=config.history_saturation,
        available_weight_fraction=result.available_weight_fraction,
        merchant_history_available=merchant,
        device_history_available=device,
        current_identity_available=identity,
        supporting_measurements=[
            f"{n} strictly prior transactions; saturation parameter {config.history_saturation:g}.",
            f"Available component weight fraction: {result.available_weight_fraction:.6f}.",
            f"Prior/current merchant comparison available: {merchant}.",
            f"Prior/current device comparison available: {device}.",
            f"Current identity context available: {identity}.",
            "Confidence is evidence availability, not confidence that fraud occurred.",
        ],
    )


def formula_trace(record, config):
    g = float(confidence_transform([record.behaviour_confidence], config.confidence_curve)[0])
    if not record.behaviour_available:
        g = 0.0
    w = config.weights
    return FormulaTrace(
        transformed_confidence=g,
        denominator=w.wp + w.wa + w.wb * g,
        unnormalized_contributions={
            "supervised": w.wp * record.supervised_probability,
            "anomaly": w.wa * record.anomaly_score,
            "behaviour": w.wb * g * record.behaviour_score if record.behaviour_available else 0.0,
        },
    )


class ExplanationEngine:
    """Cache explainer and computed validation scores; no refitting or label-dependent logic."""

    def __init__(self, inputs, top_k=5, snapshot_version=VERSION):
        if not 1 <= top_k <= 20:
            raise ValueError("Top-K must be between 1 and 20")
        self.inputs, self.top_k = inputs, top_k
        if snapshot_version not in {VERSION, SHAP_VERSION}:
            raise ValueError("Unsupported explanation snapshot version")
        self.snapshot_version = snapshot_version

    def explain(self, position):
        inputs = self.inputs
        if not 0 <= position < len(inputs.features):
            raise ValueError("Only the verified validation projection may be explained")
        row = inputs.scores.iloc[position]
        behaviour = inputs.behaviour.explain(inputs.features.iloc[[position]], 0)
        equal_scores(
            [
                behaviour.behaviour_score if behaviour.behaviour_score is not None else np.nan,
                behaviour.behaviour_confidence,
            ],
            [row.B, row.behaviour_confidence],
            "local behaviour",
        )
        record = FusionInput(
            transaction_id=str(row.TransactionID),
            supervised_probability=float(row.P),
            supervised_prediction=bool(row.P >= P_THRESHOLD),
            anomaly_score=float(row.A),
            anomaly_prediction=bool(row.A >= A_THRESHOLD),
            behaviour_score=behaviour.behaviour_score,
            behaviour_available=behaviour.behaviour_score is not None,
            behaviour_confidence=behaviour.behaviour_confidence,
        )
        trace = fuse(record, inputs.bundle.selected)
        saved = inputs.fusion_scores.iloc[position]
        equal_scores([trace.final_risk_score], [saved.final_risk_score], "fusion")
        if (
            trace.risk_level != saved.risk_level
            or trace.investigation_flag != saved.investigation_flag
            or ";".join(trace.reason_codes) != saved.reason_codes
        ):
            raise ValueError("Local explanation disagrees with certified ABRF trace")
        for key in ["supervised", "anomaly", "behaviour"]:
            equal_scores(
                [trace.effective_weights[key], trace.weighted_contributions[key]],
                [saved["effective_weight_" + key], saved["contribution_" + key]],
                key,
            )
        decomposition = inputs.shap.values(inputs.matrix[position : position + 1])
        increasing, decreasing, base, residual = [], [], None, None
        if decomposition is not None:
            values, base, residuals = decomposition
            increasing, decreasing = top_features(
                inputs.matrix[position].toarray()[0], values[0], inputs.feature_metadata, self.top_k
            )
            residual = float(residuals[0])
        confidence = confidence_explanation(
            behaviour,
            inputs.features.iloc[position],
            inputs.context.iloc[position],
            inputs.behaviour.config,
        )
        b = BehaviourExplanation(
            **behaviour.model_copy(
                update={"components": ranked_components(behaviour.components)}
            ).model_dump(),
            version=inputs.behaviour.version,
            confidence_explanation=confidence,
        )
        signals = [
            {
                "signal": component.signal,
                "source": component.source,
                "observed_value": component.raw_value,
                "behavioural_deviation": component.score,
                "scope": "Observable context, not Isolation Forest feature attribution",
            }
            for component in b.components
            if component.available and component.source in inputs.anomaly.preparation.names
        ][:5]
        return UnifiedExplanation(
            version=self.snapshot_version,
            transaction_id=record.transaction_id,
            risk_summary={
                "final_risk_score": trace.final_risk_score,
                "risk_level": trace.risk_level,
                "investigation_flag": trace.investigation_flag,
            },
            supervised=SupervisedExplanation(
                model_version=inputs.classifier.model_version,
                model_id=inputs.classifier.model_id,
                raw_probability=float(inputs.raw_probability[position]),
                calibrated_probability=record.supervised_probability,
                calibration_version=inputs.provenance["calibration_version"],
                threshold=P_THRESHOLD,
                prediction=record.supervised_prediction,
                shap_status=inputs.shap.status,
                unavailable_reason=inputs.shap.reason,
                base_value=base,
                additivity_residual=residual,
                top_positive_features=increasing,
                top_negative_features=decreasing,
            ),
            behaviour=b,
            anomaly=AnomalyExplanation(
                version=inputs.anomaly.version,
                score=record.anomaly_score,
                original_model_score=float(inputs.original_anomaly[position]),
                threshold=A_THRESHOLD,
                flag=record.anomaly_prediction,
                effective_abrf_weight=trace.effective_weights["anomaly"],
                observable_signals=signals,
            ),
            fusion=trace,
            formula_trace=formula_trace(record, inputs.bundle.selected),
        )
