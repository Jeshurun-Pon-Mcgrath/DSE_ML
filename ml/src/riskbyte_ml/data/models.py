"""Strict JSON-safe report models shared by the ML package and API schemas."""

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

T = TypeVar("T")


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Candidate(ReportModel):
    column: str
    reason: str
    inferred: Literal[True] = True


class RoleSelection(ReportModel):
    column: str | None
    available: bool
    reason: str | None = None


class ColumnMetadata(ReportModel):
    name: str
    storage_dtype: str
    inferred_type: Literal["numeric", "categorical", "boolean", "datetime", "unknown"]
    unique_count: int
    non_missing_count: int
    dominant_percentage: FiniteFloat | None
    constant: bool
    near_constant: bool
    inference_reason: str


class SchemaReport(ReportModel):
    row_count: int
    column_count: int
    columns: list[ColumnMetadata]
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_like_columns: list[str]
    constant_columns: list[str]
    near_constant_columns: list[str]
    high_cardinality_categorical_columns: list[str]
    potential_identifiers: list[Candidate]
    potential_targets: list[Candidate]
    potential_timestamps: list[Candidate]
    target: RoleSelection
    timestamp: RoleSelection
    amount: RoleSelection
    identifiers: list[RoleSelection]


class QualityWarning(ReportModel):
    code: str
    severity: Literal["INFO", "WARNING", "ERROR"]
    column: str | None = None
    rows_affected: int | None = None
    explanation: str
    inferred: bool = False


class MissingSummary(ReportModel):
    count: int
    percentage: FiniteFloat
    empty_string_count: int


class DuplicateSummary(ReportModel):
    extra_rows: int
    rows_in_duplicate_groups: int
    identifier_extra_values: dict[str, int]


class TargetDistribution(ReportModel):
    status: Literal["not_configured", "valid", "invalid"]
    column: str | None = None
    positive_class: str | None = None
    negative_class: str | None = None
    fraud_count: int | None = None
    legitimate_count: int | None = None
    fraud_percentage: FiniteFloat | None = None
    legitimate_percentage: FiniteFloat | None = None
    positive_class_prevalence: FiniteFloat | None = None
    imbalance_ratio: FiniteFloat | None = None
    minority_class: str | None = None
    majority_class: str | None = None
    missing_count: int = 0
    unsupported_count: int = 0
    valid_count: int = 0
    percentage_denominator: Literal["valid_labels"] = "valid_labels"


class QualityReport(ReportModel):
    validation_status: Literal["PASSED", "FAILED"]
    missing_values: dict[str, MissingSummary]
    duplicates: DuplicateSummary
    invalid_numeric_values: dict[str, int]
    infinite_numeric_values: dict[str, int]
    malformed_timestamps: dict[str, int]
    target: TargetDistribution
    warnings: list[QualityWarning]


class NumericStatistics(ReportModel):
    count: int
    missing_count: int
    invalid_count: int
    infinite_count: int
    mean: FiniteFloat | None = None
    median: FiniteFloat | None = None
    minimum: FiniteFloat | None = None
    maximum: FiniteFloat | None = None
    variance: FiniteFloat | None = None
    standard_deviation: FiniteFloat | None = None
    q1: FiniteFloat | None = None
    q3: FiniteFloat | None = None
    iqr: FiniteFloat | None = None
    percentiles: dict[str, FiniteFloat | None] = Field(default_factory=dict)
    skewness: FiniteFloat | None = None
    variance_ddof: Literal[1] = 1


class OutlierSummary(ReportModel):
    finite_count: int
    iqr_multiplier: FiniteFloat
    lower_bound: FiniteFloat | None
    upper_bound: FiniteFloat | None
    iqr_outlier_count: int
    zscore_threshold: FiniteFloat
    zscore_outlier_count: int
    zscore_reason: str | None = None
    zscore_ddof: Literal[0] = 0


class Frequency(ReportModel):
    value: str
    count: int
    percentage: FiniteFloat


class CategorySummary(ReportModel):
    non_missing_count: int
    unique_count: int
    frequencies: list[Frequency]
    omitted_count: int


class TimeBucket(ReportModel):
    bucket: int
    label: str
    total_count: int
    fraud_count: int
    fraud_percentage: FiniteFloat | None


class CorrelationMatrix(ReportModel):
    method: Literal["pearson"] = "pearson"
    columns: list[str]
    excluded_columns: list[str]
    values: dict[str, dict[str, FiniteFloat | None]]
    pair_counts: dict[str, dict[str, int]]


class Analysis(ReportModel, Generic[T]):
    status: Literal["AVAILABLE", "UNAVAILABLE_WITH_REASON"]
    data: T | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def check_state(self):
        if self.status == "AVAILABLE" and (self.data is None or self.reason is not None):
            raise ValueError("Available analysis requires data and no unavailable reason")
        if self.status == "UNAVAILABLE_WITH_REASON" and (self.data is not None or not self.reason):
            raise ValueError("Unavailable analysis requires a reason and no data")
        return self


class EDAReport(ReportModel):
    class_distribution: Analysis[TargetDistribution]
    amount_distribution: Analysis[NumericStatistics]
    fraud_amount_comparison: Analysis[dict[str, NumericStatistics]]
    fraud_by_hour: Analysis[list[TimeBucket]]
    fraud_by_day_of_week: Analysis[list[TimeBucket]]
    correlation: Analysis[CorrelationMatrix]
    outliers: Analysis[dict[str, OutlierSummary]]
    missing_data: Analysis[dict[str, MissingSummary]]
    numeric_distributions: Analysis[dict[str, NumericStatistics]]
    categorical_frequencies: Analysis[dict[str, CategorySummary]]


class CleaningOperation(ReportModel):
    operation: Literal[
        "remove_duplicates", "normalize_empty_strings", "coerce_numeric", "parse_datetime"
    ]
    column: str | None = None
    reason: str = Field(min_length=1)
    errors: Literal["raise", "coerce"] = "raise"

    @model_validator(mode="after")
    def check_column(self):
        if (self.operation == "remove_duplicates") != (self.column is None):
            raise ValueError("Only whole-row duplicate removal uses no column")
        return self


class CleaningPlan(ReportModel):
    source_fingerprint: str
    operations: list[CleaningOperation]
    automatically_applied: Literal[False] = False


class DataState(ReportModel):
    row_count: int
    column_count: int
    dtype: str | None = None
    missing_count: int | None = None
    unique_count: int | None = None


class AuditRecord(ReportModel):
    operation: str
    affected_column: str | None
    rows_affected: int
    before: DataState
    after: DataState
    reason: str
    errors_policy: str


class CleaningAudit(ReportModel):
    source_fingerprint: str
    processed_fingerprint: str
    generated_at: datetime
    operations: list[AuditRecord]


class DatasetProfile(ReportModel):
    dataset_name: str
    fingerprint: str
    configuration_fingerprint: str
    generated_at: datetime
    pipeline_version: str
    library_versions: dict[str, str]
    row_count: int
    column_count: int
    memory_usage_bytes: int
    schema_metadata: SchemaReport
    descriptive_statistics: dict[str, NumericStatistics]
    quality: QualityReport
    unique_counts: dict[str, int]
    data_type_summary: dict[str, list[str]]
    warnings: list[QualityWarning]
    cleaning_plan: CleaningPlan
    eda: EDAReport


class LocalDatasetProfile(DatasetProfile):
    source_path: str
