"""Outcome-free unified explanation contract; evaluation labels live outside it."""

import math
from typing import Generic, Literal, TypeVar

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from riskbyte_ml.abrf.contracts import FusionTrace
from riskbyte_ml.behavioural.deviation import BehaviourResult
from riskbyte_ml.data.models import ReportModel

VERSION = "explainability-v1"
SHAP_VERSION = "explainability-v1.1-shap"
NOTICE = (
    "This environment uses synthetically generated financial transaction data for "
    "academic fraud-detection research and demonstration."
)
NON_CAUSAL = (
    "SHAP describes feature contributions to model output, not causation, criminal intent, "
    "fraud guilt or why a human performed a transaction."
)


class FeatureExplanation(ReportModel):
    feature: str
    display_name: str
    feature_value: FiniteFloat
    shap_value: FiniteFloat
    direction: Literal["increases_classifier_risk", "decreases_classifier_risk"]
    interpretability: Literal["INTERPRETABLE", "PARTIALLY_INTERPRETABLE", "ANONYMIZED"]
    manifest_interpretability: str
    source_columns: list[str]
    feature_family: str
    raw_or_derived: str
    statement: str

    @model_validator(mode="after")
    def direction_matches_sign(self):
        if self.shap_value == 0 or (self.shap_value > 0) != (
            self.direction == "increases_classifier_risk"
        ):
            raise ValueError("SHAP direction must match its nonzero contribution")
        return self


class SupervisedExplanation(ReportModel):
    model_version: str
    model_id: str
    explained_model: Literal["underlying_decision_tree"] = "underlying_decision_tree"
    raw_probability: FiniteFloat = Field(ge=0, le=1)
    calibrated_probability: FiniteFloat = Field(ge=0, le=1)
    raw_tree_probability: FiniteFloat | None = Field(default=None, ge=0, le=1)
    calibrated_supervised_probability: FiniteFloat | None = Field(default=None, ge=0, le=1)
    calibrated_probability_shap_explained: Literal[False] = False
    calibration_version: str
    threshold: FiniteFloat
    prediction: bool
    shap_status: Literal["AVAILABLE", "UNAVAILABLE_WITH_REASON"]
    unavailable_reason: str | None = None
    base_value: FiniteFloat | None = None
    additivity_residual: FiniteFloat | None = None
    top_positive_features: list[FeatureExplanation] = Field(default_factory=list)
    top_negative_features: list[FeatureExplanation] = Field(default_factory=list)

    @model_validator(mode="after")
    def truthful_status(self):
        if self.raw_tree_probability is None:
            self.raw_tree_probability = self.raw_probability
        if self.calibrated_supervised_probability is None:
            self.calibrated_supervised_probability = self.calibrated_probability
        if self.raw_tree_probability != self.raw_probability or (
            self.calibrated_supervised_probability != self.calibrated_probability
        ):
            raise ValueError("Probability aliases must preserve raw/calibrated distinction")
        if self.shap_status == "UNAVAILABLE_WITH_REASON":
            if (
                not self.unavailable_reason
                or self.base_value is not None
                or (self.top_positive_features or self.top_negative_features)
            ):
                raise ValueError("Unavailable SHAP must not contain invented attributions")
        elif self.base_value is None or self.additivity_residual is None:
            raise ValueError("Available SHAP requires a verified decomposition")
        return self


class ConfidenceExplanation(ReportModel):
    prior_transaction_count: int = Field(ge=0)
    history_saturation: FiniteFloat = Field(gt=0)
    available_weight_fraction: FiniteFloat = Field(ge=0, le=1)
    merchant_history_available: bool
    device_history_available: bool
    current_identity_available: bool
    methodology: str = "n / (n + history_saturation) * available_component_weight_fraction"
    supporting_measurements: list[str]
    history_duration_used: Literal[False] = False


class BehaviourExplanation(BehaviourResult):
    version: str
    confidence_explanation: ConfidenceExplanation
    score_semantics: str = "Behavioural deviation score, not fraud probability"


class AnomalyExplanation(ReportModel):
    version: str
    score: FiniteFloat = Field(ge=0, le=1)
    original_model_score: FiniteFloat
    threshold: FiniteFloat
    flag: bool
    diagnostic_only: Literal[True] = True
    effective_abrf_weight: FiniteFloat = Field(ge=0, le=0)
    influenced_final_risk: Literal[False] = False
    observable_signals: list[dict[str, JsonValue]]
    scope: str = "Diagnostic anomaly evidence; observable signals are not model attributions"


class FormulaTrace(ReportModel):
    formula: str = "R = (wp*P + wa*A + wb*g(Cb)*B)/(wp + wa + wb*g(Cb))"
    confidence_rule: Literal["linear when confidence >= 0.5; otherwise zero"] = (
        "linear when confidence >= 0.5; otherwise zero"
    )
    transformed_confidence: FiniteFloat = Field(ge=0, le=1)
    denominator: FiniteFloat = Field(gt=0)
    unnormalized_contributions: dict[str, FiniteFloat]
    unavailable_behaviour_rule: str = (
        "Exclude unavailable B and renormalize P/A; missing is not safe"
    )


class RiskSummary(ReportModel):
    final_risk_score: FiniteFloat = Field(ge=0, le=1)
    risk_level: Literal["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"]
    investigation_flag: bool


class UnifiedExplanation(ReportModel):
    version: Literal["explainability-v1", "explainability-v1.1-shap"] = VERSION
    transaction_id: str
    risk_summary: RiskSummary
    supervised: SupervisedExplanation
    behaviour: BehaviourExplanation
    anomaly: AnomalyExplanation
    fusion: FusionTrace
    formula_trace: FormulaTrace
    limitations: list[str] = Field(default_factory=lambda: [NOTICE, NON_CAUSAL])

    @model_validator(mode="after")
    def consistent(self):
        f = self.fusion
        if (
            f.transaction_id != self.transaction_id
            or f.supervised_probability != self.supervised.calibrated_probability
            or f.anomaly_score != self.anomaly.score
            or f.behaviour_score != self.behaviour.behaviour_score
            or f.behaviour_confidence != self.behaviour.behaviour_confidence
            or f.effective_weights["anomaly"] != 0
            or f.weighted_contributions["anomaly"] != 0
            or self.risk_summary.model_dump()
            != {
                "final_risk_score": f.final_risk_score,
                "risk_level": f.risk_level,
                "investigation_flag": f.investigation_flag,
            }
        ):
            raise ValueError("Explanation layers disagree with frozen fusion output")
        formula = self.formula_trace
        expected_g = (
            f.behaviour_confidence
            if f.behaviour_available and (f.behaviour_confidence >= 0.5)
            else 0.0
        )
        expected_denominator = (
            f.base_weights.wp + f.base_weights.wa + f.base_weights.wb * expected_g
        )
        if (
            f.base_weights.model_dump() != {"wp": 0.9, "wa": 0.0, "wb": 0.1}
            or formula.transformed_confidence != expected_g
            or formula.denominator != expected_denominator
            or set(formula.unnormalized_contributions) != {"supervised", "anomaly", "behaviour"}
            or not math.isclose(
                sum(formula.unnormalized_contributions.values()) / formula.denominator,
                f.final_risk_score,
                abs_tol=1e-12,
                rel_tol=0,
            )
        ):
            raise ValueError("Formula trace does not reproduce frozen fusion")
        if not all(
            math.isclose(
                formula.unnormalized_contributions[name] / formula.denominator,
                f.weighted_contributions[name],
                abs_tol=1e-12,
                rel_tol=0,
            )
            for name in ["supervised", "anomaly", "behaviour"]
        ):
            raise ValueError("Formula contributions disagree with frozen trace")
        if f.investigation_flag != (f.final_risk_score >= f.decision_threshold):
            raise ValueError("Investigation decision disagrees with threshold")
        return self


class ExplanationBundle(ReportModel):
    version: Literal["explainability-v1", "explainability-v1.1-shap"] = VERSION
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    generated_at: str
    provenance: dict[str, JsonValue]
    shap_status: Literal["AVAILABLE", "UNAVAILABLE_WITH_REASON"]
    artifacts: dict[str, str]
    performance: dict[str, JsonValue]
    isolation_audit: dict[str, JsonValue]
    future_holdout_status: Literal["PREPARATION_ONLY_NOT_GENERATED"] = (
        "PREPARATION_ONLY_NOT_GENERATED"
    )

    @model_validator(mode="after")
    def isolation(self):
        if self.version == SHAP_VERSION and self.shap_status != "AVAILABLE":
            raise ValueError("SHAP-complete snapshot must contain genuine SHAP")
        if self.isolation_audit.get("original_temporal_test_accessed") is not False:
            raise ValueError("Original test explanation is prohibited")
        if self.isolation_audit.get("future_holdout_generated") is not False:
            raise ValueError("Fresh holdout generation was not authorized")
        return self


T = TypeVar("T")


class ExplanationResponse(ReportModel, Generic[T]):
    status: Literal[
        "dataset_not_configured",
        "explainability_not_generated",
        "explainability_ready",
        "explainability_partial",
        "explainability_unavailable",
    ]
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    data: T | None = None
    reason: str | None = None
