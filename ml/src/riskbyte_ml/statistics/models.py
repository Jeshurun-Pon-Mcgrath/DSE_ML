"""Strict, finite, bounded statistical artifacts shared with OpenAPI."""

from datetime import datetime
from typing import Literal

from pydantic import Field, FiniteFloat, model_serializer, model_validator

from riskbyte_ml.data.models import NumericStatistics, ReportModel


class SummaryStatistics(NumericStatistics):
    kurtosis: FiniteFloat | None = None
    kurtosis_definition: Literal["unbiased Fisher excess; normal=0"] = (
        "unbiased Fisher excess; normal=0"
    )
    distribution_note: str


class Provenance(ReportModel):
    finding_label: Literal["Synthetic-data finding"] = "Synthetic-data finding"
    dataset_fingerprint: str
    feature_pipeline_version: str
    split_id: str
    artifact_version: str
    generated_at: datetime
    primary_inference_partition: Literal["train"] = "train"
    limitations: list[str]


class DescriptiveReport(ReportModel):
    groups: dict[str, dict[str, SummaryStatistics]]
    group_counts: dict[str, int]
    fraud_vs_legitimate_effects: dict[str, "EffectSize"]
    scope: Literal["full dataset; exploratory"] = "full dataset; exploratory"


class EffectSize(ReportModel):
    method: Literal["Cliffs_delta", "Cramers_V"]
    value: FiniteFloat | None
    magnitude: str
    direction: str
    odds_ratio: FiniteFloat | None = None
    odds_ratio_reason: str | None = None


class TestOutcome(ReportModel):
    status: Literal["AVAILABLE", "UNAVAILABLE_WITH_REASON"]
    reason: str | None = None
    method: str
    statistic: FiniteFloat | None = None
    p_value: FiniteFloat | None = Field(default=None, ge=0, le=1)
    p_value_underflow: bool = False
    effect_size: EffectSize | None = None
    assumption_checks: list[str]
    group_counts: dict[str, int]
    excluded_counts: dict[str, int]
    group_summaries: dict[str, SummaryStatistics] = Field(default_factory=dict)
    contingency_table: list[list[int]] | None = None
    table_layout: str | None = None

    @model_validator(mode="after")
    def validate_availability(self):
        if self.status == "AVAILABLE" and (self.p_value is None or self.reason is not None):
            raise ValueError("Available tests require a p-value and no unavailable reason")
        if self.status == "UNAVAILABLE_WITH_REASON" and (
            not self.reason or self.p_value is not None
        ):
            raise ValueError("Unavailable tests require a reason and no p-value")
        return self


class Hypothesis(ReportModel):
    id: str
    question: str
    feature: str
    h0: str
    h1: str
    alpha: FiniteFloat
    test_selection_reasoning: str
    outcome: TestOutcome
    adjusted_p_value: FiniteFloat | None = Field(default=None, ge=0, le=1)
    decision: Literal["REJECT_H0_NOMINALLY", "DO_NOT_REJECT_H0", "UNAVAILABLE"]
    practical_interpretation: str


class HypothesisReport(ReportModel):
    correction: Literal["Benjamini-Hochberg FDR"] = "Benjamini-Hochberg FDR"
    family_size: int
    scope: Literal["training partition; exploratory nominal inference"] = (
        "training partition; exploratory nominal inference"
    )
    hypotheses: list[Hypothesis]


class CorrelationPair(ReportModel):
    first: str
    second: str
    pair_count: int
    pearson: FiniteFloat | None
    spearman: FiniteFloat | None
    reason: str | None = None


class CorrelationReport(ReportModel):
    columns: list[str]
    pairs: list[CorrelationPair]
    scope: Literal["training partition; exploratory association, not causation"] = (
        "training partition; exploratory association, not causation"
    )
    selection_reason: str


class Prevalence(ReportModel):
    rows: int
    fraud_count: int
    prevalence: FiniteFloat | None
    confidence_level: FiniteFloat
    lower: FiniteFloat | None
    upper: FiniteFloat | None
    interval_method: Literal["Wilson score; nominal independent Bernoulli approximation"] = (
        "Wilson score; nominal independent Bernoulli approximation"
    )


class PrevalenceReport(ReportModel):
    partitions: dict[str, Prevalence]
    interpretation: str


class NoveltySummary(ReportModel):
    available_count: int
    unavailable_count: int
    new_count: int
    new_fraction: FiniteFloat | None


class GroupAnalysis(ReportModel):
    count: int
    proportion: FiniteFloat
    numeric: dict[str, SummaryStatistics]
    novelty: dict[str, NoveltySummary]


class ScenarioReport(ReportModel):
    scenarios: dict[str, GroupAnalysis]
    scope: Literal["full dataset; evaluation-only synthetic ground truth"] = (
        "full dataset; evaluation-only synthetic ground truth"
    )


class HardNegativeReport(ReportModel):
    groups: dict[str, GroupAnalysis]
    hard_negative_types: dict[str, GroupAnalysis]
    fraud_vs_hard_negative_effects: dict[str, EffectSize]
    scope: Literal["full dataset; exploratory, metadata excluded from model inputs"] = (
        "full dataset; exploratory, metadata excluded from model inputs"
    )


class TemporalBucket(ReportModel):
    elapsed_day_index: int
    prevalence: Prevalence
    scenario_counts: dict[str, int]


class TemporalReport(ReportModel):
    buckets: list[TemporalBucket]
    partition_group_summaries: dict[str, dict[str, SummaryStatistics]]
    partition_fraud_association: TestOutcome
    interpretation: str


class StatisticalBundle(ReportModel):
    provenance: Provenance
    descriptive: DescriptiveReport
    hypotheses: HypothesisReport
    correlations: CorrelationReport
    prevalence: PrevalenceReport
    scenarios: ScenarioReport
    hard_negatives: HardNegativeReport
    temporal: TemporalReport


class StatisticsSummary(ReportModel):
    provenance: Provenance
    descriptive: DescriptiveReport
    hypothesis_count: int
    nominal_rejections: int


class StatisticsResponse[T](ReportModel):
    status: Literal[
        "statistics_ready",
        "dataset_not_configured",
        "statistics_not_generated",
        "statistics_unavailable",
    ]
    data: T | None = None
    provenance: Provenance | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def consistent_state(self):
        ready = self.status == "statistics_ready"
        if ready != (self.data is not None and self.provenance is not None):
            raise ValueError("Ready statistics require data and provenance")
        if not ready and (self.data is not None or self.provenance is not None):
            raise ValueError("Unavailable statistics must not contain data")
        if (self.status == "statistics_unavailable") != (self.reason is not None):
            raise ValueError("Unsupported datasets require an explicit reason")
        return self

    @model_serializer(mode="wrap")
    def serialize_response(self, handler):
        result = handler(self)
        if self.status != "statistics_ready":
            result.pop("data", None)
            result.pop("provenance", None)
        if self.reason is None:
            result.pop("reason", None)
        return result
