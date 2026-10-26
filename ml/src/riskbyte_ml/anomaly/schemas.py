"""Path-free strict report envelopes; no raw transactions in HTTP responses."""

from typing import Generic, Literal, TypeVar

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from riskbyte_ml.data.models import ReportModel
from riskbyte_ml.models.supervised.schemas import Confusion, Subgroup
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID

T = TypeVar("T")


class DetectionMetrics(ReportModel):
    rows: int
    score_available: int
    unavailable_fraud: int
    unavailable_policy: str
    threshold: FiniteFloat
    precision: FiniteFloat = Field(ge=0, le=1)
    recall: FiniteFloat = Field(ge=0, le=1)
    f1: FiniteFloat = Field(ge=0, le=1)
    pr_auc: FiniteFloat | None
    roc_auc: FiniteFloat | None
    auc_population: str
    fpr: FiniteFloat | None
    confusion: Confusion
    hard_negatives: Subgroup
    hard_negative_types: dict[str, Subgroup]
    scenario_recall: dict[str, Subgroup]
    score_semantics: str


class Phase5Bundle(ReportModel):
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    dataset_fingerprint: str
    feature_version: str
    split_id: str
    anomaly_version: str
    behaviour_version: str
    generated_at: str
    configuration: dict[str, JsonValue]
    baseline: dict[str, JsonValue]
    selected: dict[str, JsonValue]
    study: list[dict[str, JsonValue]]
    anomaly_metrics: dict[str, JsonValue]
    behaviour_metrics: dict[str, JsonValue]
    distributions: dict[str, JsonValue]
    complementarity: dict[str, JsonValue]
    investigations: dict[str, JsonValue]
    behavioural_components: dict[str, JsonValue]
    leakage_audit: dict[str, JsonValue]
    artifact_hashes: dict[str, str]

    @model_validator(mode="after")
    def frozen_provenance(self):
        if (
            self.dataset_fingerprint != FROZEN_FINGERPRINT
            or self.feature_version != FROZEN_FEATURE_VERSION
            or self.split_id != FROZEN_SPLIT_ID
            or self.anomaly_version != "anomaly-detection-v1"
            or self.behaviour_version != "behavioural-deviation-v1"
            or self.leakage_audit.get("original_temporal_test_accessed") is not False
        ):
            raise ValueError("Incompatible frozen Phase 5 provenance or development scope")
        DetectionMetrics.model_validate(self.anomaly_metrics)
        return self


class Phase5Response(ReportModel, Generic[T]):
    status: Literal[
        "dataset_not_configured", "phase5_not_generated", "phase5_unavailable", "phase5_ready"
    ]
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    reason: str | None = None
    data: T | None = Field(default=None)
