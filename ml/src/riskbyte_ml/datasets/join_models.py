"""Dataset-independent two-table audit schemas."""

from typing import Literal

from pydantic import FiniteFloat

from riskbyte_ml.data.models import ReportModel


class JoinAudit(ReportModel):
    status: Literal["JOINED", "REJECTED"]
    transaction_rows_before: int
    identity_rows_before: int
    joined_rows: int | None
    unique_transaction_ids: int
    unique_identity_ids: int
    duplicate_transaction_keys: int
    duplicate_identity_keys: int
    duplicate_transaction_extra_rows: int
    duplicate_identity_extra_rows: int
    missing_transaction_keys: int
    missing_identity_keys: int
    matched_identity_rows: int | None
    unmatched_transaction_rows: int | None
    unused_identity_rows: int | None
    identity_coverage_percentage: FiniteFloat | None
    columns_added: list[str]
    target_integrity_verified: bool
    join_kind: Literal["left"] = "left"
    cardinality: Literal["one_to_one"] = "one_to_one"


class MemoryAudit(ReportModel):
    transaction_before_bytes: int
    transaction_after_bytes: int
    identity_before_bytes: int
    identity_after_bytes: int
    joined_before_bytes: int
    joined_after_bytes: int
    optimized_columns: list[str]
    peak_process_memory_bytes: None = None
    measurement: str = (
        "Pandas deep memory estimates; before table optimization sums measured string batches; "
        "after measures retained frames; not process peak RAM"
    )
