"""Strict, outcome-free live fusion contracts and explicit research configuration."""

import math
from typing import Generic, Literal, TypeVar

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from riskbyte_ml.data.models import ReportModel

ABRF_VERSION = "abrf-v1"
P_THRESHOLD = 0.3665406264377367
A_THRESHOLD = 0.9626184600557802
B_THRESHOLD = 0.482367645255099
type ConfidenceCurve = Literal["linear", "gated_0.5", "gated_0.75"]
type PolicyName = Literal["BALANCED", "HIGH_RECALL", "STRICT"]


class Weights(ReportModel):
    wp: FiniteFloat = Field(ge=0, le=1)
    wa: FiniteFloat = Field(ge=0, le=1)
    wb: FiniteFloat = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_sum(self):
        if not math.isclose(self.wp + self.wa + self.wb, 1, abs_tol=1e-9):
            raise ValueError("Weights must sum to one")
        if self.wp + self.wa <= 0:
            raise ValueError("P/A fallback must remain available when behaviour is unavailable")
        return self


class Evidence(ReportModel):
    signal: Literal[
        "amount_deviation",
        "amount_zscore",
        "transaction_velocity",
        "long_window_velocity",
        "device_novelty",
        "merchant_novelty",
        "region_novelty",
        "product_novelty",
        "failed_attempts",
        "short_time_gap",
    ]
    score: FiniteFloat | None = Field(default=None, ge=0, le=1)
    available: bool

    @model_validator(mode="after")
    def availability(self):
        if self.available != (self.score is not None):
            raise ValueError("Evidence availability must match its score")
        return self


class FusionInput(ReportModel):
    transaction_id: str = Field(min_length=1)
    supervised_probability: FiniteFloat = Field(ge=0, le=1)
    supervised_prediction: bool
    anomaly_score: FiniteFloat = Field(ge=0, le=1)
    anomaly_prediction: bool
    behaviour_score: FiniteFloat | None = Field(default=None, ge=0, le=1)
    behaviour_confidence: FiniteFloat = Field(ge=0, le=1)
    behaviour_available: bool
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistency(self):
        if self.supervised_prediction != (self.supervised_probability >= P_THRESHOLD):
            raise ValueError("Supervised prediction must use its frozen threshold")
        if self.anomaly_prediction != (self.anomaly_score >= A_THRESHOLD):
            raise ValueError("Anomaly prediction must use its frozen threshold")
        if self.behaviour_available != (self.behaviour_score is not None):
            raise ValueError("Behaviour availability must match its score")
        if not self.behaviour_available and self.behaviour_confidence != 0:
            raise ValueError("Unavailable behaviour requires zero confidence, not zero risk")
        return self


class OperatingPolicy(ReportModel):
    objective: Literal["maximize_f1", "maximize_recall", "minimize_false_positives"]
    max_fpr: FiniteFloat = Field(ge=0, le=1)
    max_hard_negative_fpr: FiniteFloat = Field(ge=0, le=1)
    minimum_recall: FiniteFloat = Field(default=0.0, ge=0, le=1)


class SearchConfig(ReportModel):
    grid_divisions: int = Field(default=10, ge=5, le=20)
    confidence_curves: list[ConfidenceCurve] = Field(
        default_factory=lambda: ["linear", "gated_0.5", "gated_0.75"]
    )
    default_policy: PolicyName = "BALANCED"
    policies: dict[str, OperatingPolicy] = Field(
        default_factory=lambda: {
            "BALANCED": OperatingPolicy(
                objective="maximize_f1", max_fpr=0.01, max_hard_negative_fpr=0.05
            ),
            "HIGH_RECALL": OperatingPolicy(
                objective="maximize_recall", max_fpr=0.02, max_hard_negative_fpr=0.10
            ),
            "STRICT": OperatingPolicy(
                objective="minimize_false_positives",
                max_fpr=0.005,
                max_hard_negative_fpr=0.03,
                minimum_recall=0.60,
            ),
        }
    )
    preserve_scenarios: list[str] = Field(
        default_factory=lambda: ["ACCOUNT_TAKEOVER", "MIXED_MODE"]
    )
    risk_band_quantiles: list[FiniteFloat] = Field(default_factory=lambda: [0.5, 0.9, 0.99, 0.999])
    reason_evidence_cutoff: FiniteFloat = Field(default=0.95, ge=0, le=1)
    limited_confidence_cutoff: FiniteFloat = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_policy(self):
        if set(self.policies) != {"BALANCED", "HIGH_RECALL", "STRICT"}:
            raise ValueError("Three explicit operating policies required")
        if (
            len(set(self.confidence_curves)) != len(self.confidence_curves)
            or not self.confidence_curves
        ):
            raise ValueError("Distinct confidence curves required")
        q = self.risk_band_quantiles
        if len(q) != 4 or not all(0 < x < 1 for x in q) or sorted(set(q)) != q:
            raise ValueError("Four increasing training-distribution risk quantiles required")
        if not set(self.preserve_scenarios) <= {"ACCOUNT_TAKEOVER", "MIXED_MODE"}:
            raise ValueError("Only the prespecified strong-scenario protections are supported")
        return self


class FusionConfig(ReportModel):
    version: Literal["abrf-v1"] = ABRF_VERSION
    weights: Weights
    confidence_curve: ConfidenceCurve
    policy: PolicyName
    decision_threshold: FiniteFloat = Field(ge=0, le=1.000001)
    risk_boundaries: list[FiniteFloat]
    reason_evidence_cutoff: FiniteFloat = Field(default=0.95, ge=0, le=1)
    limited_confidence_cutoff: FiniteFloat = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def bands(self):
        b = self.risk_boundaries
        if len(b) != 4 or sorted(set(b)) != b or not all(0 < x < 1 for x in b):
            raise ValueError("Four strictly increasing internal risk boundaries required")
        return self


class FusionTrace(ReportModel):
    transaction_id: str
    version: str
    supervised_probability: FiniteFloat
    anomaly_score: FiniteFloat
    behaviour_score: FiniteFloat | None
    behaviour_confidence: FiniteFloat
    behaviour_available: bool
    base_weights: Weights
    effective_weights: dict[str, FiniteFloat]
    weighted_contributions: dict[str, FiniteFloat]
    final_risk_score: FiniteFloat = Field(ge=0, le=1)
    risk_level: Literal["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"]
    decision_threshold: FiniteFloat
    investigation_flag: bool
    reason_codes: list[str]
    score_semantics: str = "RiskByte fused risk score; not fraud probability or evidence of guilt"


class ABRFBundle(ReportModel):
    version: Literal["abrf-v1"] = ABRF_VERSION
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    generated_at: str
    provenance: dict[str, JsonValue]
    search_configuration: SearchConfig
    selected: FusionConfig
    selected_candidate_id: str
    selection_rationale: str
    validation: dict[str, JsonValue]
    ablation: list[dict[str, JsonValue]]
    policies: dict[str, JsonValue]
    constraint_study: dict[str, JsonValue]
    confidence_study: dict[str, JsonValue]
    scenario_analysis: dict[str, JsonValue]
    hard_negative_analysis: dict[str, JsonValue]
    support_analysis: dict[str, JsonValue]
    improvement: dict[str, JsonValue]
    search_summary: dict[str, JsonValue]
    leakage_audit: dict[str, JsonValue]
    artifact_hashes: dict[str, str]
    future_holdout_status: Literal["PREPARATION_ONLY_NOT_GENERATED"] = (
        "PREPARATION_ONLY_NOT_GENERATED"
    )

    @model_validator(mode="after")
    def isolated(self):
        if self.leakage_audit.get("original_temporal_test_accessed") is not False:
            raise ValueError("Original temporal test use is prohibited")
        if self.leakage_audit.get("future_holdout_generated") is not False:
            raise ValueError("Fresh holdout generation was not authorized")
        return self


T = TypeVar("T")


class ABRFResponse(ReportModel, Generic[T]):
    status: Literal[
        "dataset_not_configured", "abrf_not_generated", "abrf_unavailable", "abrf_ready"
    ]
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    data: T | None = None
    reason: str | None = None
