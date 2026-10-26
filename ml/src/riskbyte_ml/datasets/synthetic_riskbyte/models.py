"""Path-free public synthetic dataset provenance and measured availability."""

from typing import Literal

from pydantic import FiniteFloat

from riskbyte_ml.data.models import ReportModel
from riskbyte_ml.datasets.join_models import JoinAudit, MemoryAudit


class SyntheticMetadata(ReportModel):
    dataset_name: str
    dataset_type: Literal["synthetic"] = "synthetic"
    description: Literal["IEEE-CIS-inspired synthetic financial transaction dataset"] = (
        "IEEE-CIS-inspired synthetic financial transaction dataset"
    )
    notice: str
    generator_version: str
    seed: int
    dataset_fingerprint: str
    transaction_rows: int
    identity_rows: int
    joined_rows: int
    fraud_count: int
    fraud_prevalence: FiniteFloat
    identity_coverage: FiniteFloat
    entity_count: int
    merchant_count: int
    device_count: int
    feature_pipeline_version: str
    recommended_split_strategy: Literal["temporal"] = "temporal"
    split_strategy: Literal["temporal", "random_stratified"] | None = None
    behavioural_feature_availability: dict[
        str, Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]
    ]
    feature_classes: dict[str, list[str]]
    join_audit: JoinAudit
    memory: MemoryAudit
