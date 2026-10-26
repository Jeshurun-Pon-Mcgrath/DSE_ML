"""Strict, path-free integration metadata, audits and proposed feature contracts."""

from typing import Literal

from pydantic import FiniteFloat

from riskbyte_ml.data.models import NumericStatistics, QualityWarning, ReportModel
from riskbyte_ml.datasets.join_models import JoinAudit as JoinAudit
from riskbyte_ml.datasets.join_models import MemoryAudit as MemoryAudit

FeatureClass = Literal[
    "INTERPRETABLE",
    "PARTIALLY_INTERPRETABLE",
    "ANONYMIZED",
    "IDENTIFIER",
    "TEMPORAL",
    "TARGET",
    "POTENTIAL_BEHAVIOURAL_KEY",
    "EXCLUDED_CANDIDATE",
]
Support = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]


class FeatureMetadata(ReportModel):
    column: str
    classifications: list[FeatureClass]
    explanation: str
    observed_dtype: str
    missing_percentage: FiniteFloat
    unique_count: int
    review_required: bool = True


class EntityStudy(ReportModel):
    name: str
    fields: list[str]
    status: Literal["AVAILABLE", "UNAVAILABLE_WITH_REASON"]
    reason: str | None
    eligible_rows: int
    missing_rows: int
    missing_percentage: FiniteFloat
    missing_by_field: dict[str, int]
    unique_entities: int | None
    median_transactions_per_entity: FiniteFloat | None
    mean_transactions_per_entity: FiniteFloat | None
    maximum_transactions: int | None
    singleton_entity_percentage: FiniteFloat | None
    repeated_history_row_percentage: FiniteFloat | None
    rows_with_strictly_prior_elapsed_time_percentage: FiniteFloat | None
    proxy_variation_entity_percentage: dict[str, FiniteFloat]
    statistically_useful: bool
    collision_risk: str
    interpretation_limitation: str


class BehaviouralFeasibility(ReportModel):
    feature: str
    support: Support
    required_fields: list[str]
    requires_constructed_entity: bool
    explanation: str


class ContractFeature(ReportModel):
    name: str
    source_fields: list[str]
    support: Support
    rule: str


class FeatureContract(ReportModel):
    version: str
    proposed_only: Literal[True] = True
    raw_predictive_features: list[str]
    temporal_derived_features: list[ContractFeature]
    behavioural_derived_features: list[ContractFeature]
    missingness_indicators: list[str]
    categorical_features: list[str]
    anonymized_engineered_features: list[str]
    identifier_excluded_fields: list[str]
    target_field: str
    history_policy: str
    preprocessing_policy: str


class OrderingBucket(ReportModel):
    bucket: int
    row_count: int
    fraud_count: int
    fraud_prevalence: FiniteFloat


class OrderingReview(ReportModel):
    column: str
    status: Literal["AVAILABLE", "UNAVAILABLE_WITH_REASON"]
    reason: str | None
    numeric_observations: int
    file_order_monotonic: bool | None
    target_correlation: FiniteFloat | None
    buckets: list[OrderingBucket]
    explanation: str


class CleaningReview(ReportModel):
    extreme_missingness_columns: list[str]
    moderate_missingness_columns: list[str]
    possible_missingness_indicators: list[str]
    extreme_threshold_percentage: FiniteFloat = 90
    moderate_threshold_percentage: FiniteFloat = 20
    recommendation: str


class IntegrationReport(ReportModel):
    dataset_name: str
    status: Literal["joined"]
    fingerprint: str
    source_fingerprints: dict[str, str]
    adapter_version: str
    configuration_fingerprint: str
    join_audit: JoinAudit
    memory: MemoryAudit
    feature_catalog: list[FeatureMetadata]
    entity_key_study: list[EntityStudy]
    recommended_behavioural_groups: list[str]
    behavioural_feasibility: list[BehaviouralFeasibility]
    leakage_candidates: list[QualityWarning]
    ordering_review: list[OrderingReview]
    id_elapsed_rank_correlation: FiniteFloat | None
    cleaning_review: CleaningReview
    elapsed_seconds: NumericStatistics
    temporal_semantics: str
    feature_contract: FeatureContract
