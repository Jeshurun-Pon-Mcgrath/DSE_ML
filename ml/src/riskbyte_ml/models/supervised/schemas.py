"""Finite statistical/evaluation contracts, also used as strict OpenAPI models."""

from datetime import datetime
from typing import Literal

from pydantic import Field, FiniteFloat, model_serializer, model_validator

from riskbyte_ml.data.models import ReportModel

type Policy = Literal["BALANCED", "HIGH_RECALL", "LOW_FALSE_POSITIVE"]
type Parameters = dict[str, str | bool | int | FiniteFloat | None]


class ExperimentConfig(ReportModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    search_candidates: int = Field(default=4, ge=2, le=8)
    n_jobs: int = Field(default=2, ge=1, le=8)
    policy: Policy = "BALANCED"
    calibration_fraction: FiniteFloat = Field(default=0.5, gt=0, lt=1)
    knn_sample_size: int = Field(default=5000, ge=50, le=20000)
    knn_benchmark_queries: int = Field(default=256, ge=20, le=1000)
    knn_max_pairs: int = Field(default=100_000_000, ge=1)
    knn_max_projected_seconds: FiniteFloat = Field(default=60.0, gt=0)
    working_memory_mb: int = Field(default=64, ge=16, le=256)


class Confusion(ReportModel):
    tp: int
    fp: int
    tn: int
    fn: int


class Metrics(ReportModel):
    rows: int
    fraud_count: int
    prevalence: FiniteFloat
    threshold: FiniteFloat
    precision: FiniteFloat
    recall: FiniteFloat
    f1: FiniteFloat
    pr_auc: FiniteFloat | None
    pr_auc_definition: Literal["average precision; non-interpolated"] = (
        "average precision; non-interpolated"
    )
    roc_auc: FiniteFloat | None
    brier_score: FiniteFloat | None
    accuracy: FiniteFloat
    specificity: FiniteFloat | None
    fpr: FiniteFloat | None
    fnr: FiniteFloat | None
    confusion: Confusion
    probability_note: str


class Subgroup(ReportModel):
    count: int
    flagged: int
    rate: FiniteFloat | None


class Evaluation(ReportModel):
    metrics: Metrics
    hard_negatives: Subgroup
    hard_negative_types: dict[str, Subgroup]
    scenario_recall: dict[str, Subgroup]
    prediction_seconds: FiniteFloat | None
    seconds_per_transaction: FiniteFloat | None


class CurvePoint(ReportModel):
    x: FiniteFloat
    y: FiniteFloat
    threshold: FiniteFloat | None = None


class ReliabilityBin(ReportModel):
    lower: FiniteFloat
    upper: FiniteFloat
    count: int
    mean_probability: FiniteFloat | None
    observed_prevalence: FiniteFloat | None


class Curves(ReportModel):
    pr: list[CurvePoint]
    roc: list[CurvePoint]
    reliability: list[ReliabilityBin]
    source_points: dict[str, int]
    note: str = "Curve points bounded for display; AUC uses full predictions, not display sampling"


class ThresholdPoint(ReportModel):
    threshold: FiniteFloat
    precision: FiniteFloat
    recall: FiniteFloat
    f1: FiniteFloat
    fpr: FiniteFloat
    fnr: FiniteFloat
    predicted_positive_count: int
    hard_negative_fpr: FiniteFloat | None


class ThresholdChoice(ReportModel):
    policy: Policy
    chosen: ThresholdPoint
    constraints_met: bool
    criterion: str


class ThresholdAnalysis(ReportModel):
    source: Literal["later validation selection block"] = "later validation selection block"
    candidate_count: int
    points: list[ThresholdPoint]
    policies: dict[str, ThresholdChoice]


class CalibrationChoice(ReportModel):
    method: Literal["none", "sigmoid", "isotonic", "rule_score"]
    fitting_source: str
    selection_source: str
    candidates: dict[str, Metrics]
    excluded_methods: dict[str, str]
    reasoning: str


class SearchTrial(ReportModel):
    parameters: Parameters
    fit_seconds: FiniteFloat
    calibration_block_metrics: Metrics
    warnings: list[str]


class Timing(ReportModel):
    search_training_seconds: FiniteFloat
    chosen_fit_seconds: FiniteFloat
    calibration_seconds: FiniteFloat
    train_prediction_seconds: FiniteFloat
    validation_prediction_seconds: FiniteFloat


class FeatureImportance(ReportModel):
    feature: str
    importance: FiniteFloat
    interpretability_class: str


class KNNBenchmark(ReportModel):
    sample_size: int
    fraud_count: int
    query_count: int
    fit_seconds: FiniteFloat
    prediction_seconds: FiniteFloat
    matrix_memory_bytes: int
    full_pair_count: int
    projected_full_prediction_seconds: FiniteFloat
    constrained: bool
    reason: str
    metrics: Metrics


class ModelRecord(ReportModel):
    model_id: str
    algorithm: str
    version: str
    status: Literal[
        "TRAINED", "COMPUTATIONALLY_CONSTRAINED", "DEPENDENCY_UNAVAILABLE", "RULE_BASELINE"
    ]
    limitation: str | None = None
    trained_at: datetime | None = None
    dataset_fingerprint: str
    feature_version: str
    split_version: str
    split_id: str
    preprocessing_id: str | None = None
    parameters: Parameters = Field(default_factory=dict)
    class_imbalance_strategy: str
    is_selected_supervised_model: bool = False
    selected_threshold: FiniteFloat | None = None
    threshold_policy: Policy = "BALANCED"
    calibration: CalibrationChoice | None = None
    train_metrics: Metrics | None = None
    validation: Evaluation | None = None
    selection: Evaluation | None = None
    default_threshold_metrics: Metrics | None = None
    test_metrics: Evaluation | None = None
    timing: Timing | None = None
    search_trials: list[SearchTrial] = Field(default_factory=list)
    feature_importance: list[FeatureImportance] = Field(default_factory=list)
    threshold_analysis: ThresholdAnalysis | None = None
    knn_benchmark: KNNBenchmark | None = None
    validation_errors: "ErrorAnalysis | None" = None
    artifact_name: str | None = None
    artifact_hash: str | None = None


class ModelProvenance(ReportModel):
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    dataset_fingerprint: str
    feature_version: str
    split_id: str
    feature_manifest_hash: str
    split_file_hash: str
    model_version: str
    registry_version: str
    run_id: str
    generated_at: datetime
    dependency_versions: dict[str, str]


class SelectionFreeze(ReportModel):
    model_id: str
    threshold: FiniteFloat
    calibration_method: str
    artifact_hash: str
    run_id: str
    frozen_at: datetime
    ranking: list[str]
    rationale: str


class ErrorGroup(ReportModel):
    count: int
    hard_negative_count: int
    missing_identity_count: int
    median_amount: FiniteFloat | None
    median_amount_ratio: FiniteFloat | None
    median_velocity_30m: FiniteFloat | None
    new_device_count: int
    new_merchant_count: int
    high_amount_count: int
    velocity_burst_count: int
    scenarios: dict[str, int]
    hard_negative_types: dict[str, int]


class ErrorAnalysis(ReportModel):
    false_positives: ErrorGroup
    false_negatives: ErrorGroup
    threshold_definitions: str


class RandomComparison(ReportModel):
    algorithm: str
    split_id: str
    validation: Evaluation
    temporal_selection_pr_auc: FiniteFloat
    preprocessing_id: str
    training_seconds: FiniteFloat
    note: str


class ModelRegistry(ReportModel):
    provenance: ModelProvenance
    models: list[ModelRecord]
    selection_freeze: SelectionFreeze
    partition_prevalence: dict[str, FiniteFloat]
    partition_counts: dict[str, int]
    validation_blocks: dict[str, int]
    selected_test_curves: Curves
    selected_test_errors: ErrorAnalysis
    random_comparisons: list[RandomComparison]
    leakage_audit: dict[str, str]
    limitations: list[str]

    @model_validator(mode="after")
    def unique_selection(self):
        selected = [m for m in self.models if m.is_selected_supervised_model]
        if len(selected) != 1 or selected[0].model_id != self.selection_freeze.model_id:
            raise ValueError("Exactly one frozen supervised model must be selected")
        if selected[0].algorithm == "Rule Baseline" or selected[0].test_metrics is None:
            raise ValueError("Selected classifier requires final test evaluation")
        if (
            selected[0].selected_threshold != self.selection_freeze.threshold
            or selected[0].artifact_hash != self.selection_freeze.artifact_hash
            or selected[0].calibration.method != self.selection_freeze.calibration_method
        ):
            raise ValueError("Selected classifier metadata must match frozen operating state")
        if len({m.model_id for m in self.models}) != len(self.models):
            raise ValueError("Unique model IDs required")
        if any(
            m.dataset_fingerprint != self.provenance.dataset_fingerprint
            or m.feature_version != self.provenance.feature_version
            or m.split_id != self.provenance.split_id
            for m in self.models
        ):
            raise ValueError("All model records must match registry provenance")
        return self


class ModelResponse[T](ReportModel):
    status: Literal[
        "models_ready", "models_not_trained", "dataset_not_configured", "models_unavailable"
    ]
    data: T | None = None
    provenance: ModelProvenance | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        ready = self.status == "models_ready"
        if ready != (self.data is not None and self.provenance is not None):
            raise ValueError("Ready model reports require data and provenance")
        if not ready and (self.data is not None or self.provenance is not None):
            raise ValueError("Unavailable reports cannot contain data")
        return self

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if self.status != "models_ready":
            result.pop("data", None)
            result.pop("provenance", None)
        if self.reason is None:
            result.pop("reason", None)
        return result


class ModelOverview(ReportModel):
    model_id: str
    algorithm: str
    status: str
    version: str
    limitation: str | None
    dataset_fingerprint: str
    feature_version: str
    selected_threshold: FiniteFloat | None
    is_selected_supervised_model: bool


class ModelComparisonRow(ModelOverview):
    selection_metrics: Metrics | None
    validation_metrics: Metrics | None
    hard_negative_performance: Subgroup | None
    scenario_recall: dict[str, Subgroup]
    timing: Timing | None
    class_imbalance_strategy: str


class ModelComparison(ReportModel):
    rows: list[ModelComparisonRow]
    pr_auc_ranking: list[str]
    hard_negative_fpr_ranking: list[str]
    selection_rationale: str
    note: str


class ModelMetricView(ReportModel):
    train: Metrics | None
    validation: Evaluation | None
    selection: Evaluation | None
    final_test: Evaluation | None
    calibration: CalibrationChoice | None
    test_availability_reason: str | None


class ScenarioRecallView(ReportModel):
    validation: dict[str, Subgroup]
    selection: dict[str, Subgroup]
    final_test: dict[str, Subgroup] | None
    test_availability_reason: str | None
