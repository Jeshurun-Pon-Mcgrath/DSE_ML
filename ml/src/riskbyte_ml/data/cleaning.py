"""Explicit, copy-based cleaning with source-bound plans and before/after audits."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import LoadedDataset
from riskbyte_ml.data.models import (
    AuditRecord,
    CleaningAudit,
    CleaningOperation,
    CleaningPlan,
    DataState,
    QualityReport,
)
from riskbyte_ml.data.values import empty_mask, missing_mask, numeric_view, parse_timestamps


def propose_cleaning(dataset: LoadedDataset, quality: QualityReport) -> CleaningPlan:
    operations = []
    if quality.duplicates.extra_rows:
        operations.append(
            CleaningOperation(
                operation="remove_duplicates",
                reason="Review duplicate rows; identical transactions may be distinct events",
            )
        )
    for name, summary in quality.missing_values.items():
        if summary.empty_string_count:
            operations.append(
                CleaningOperation(
                    operation="normalize_empty_strings",
                    column=name,
                    reason="Review normalization of observed empty strings to missing values",
                )
            )
    for name in dict.fromkeys((*dataset.config.numeric_columns, dataset.config.amount_column)):
        if name is not None and name in dataset.frame.columns:
            operations.append(
                CleaningOperation(
                    operation="coerce_numeric",
                    column=name,
                    reason="Explicit numeric role; invalid or infinite values raise by default",
                )
            )
    if dataset.config.timestamp_column in dataset.frame.columns:
        operations.append(
            CleaningOperation(
                operation="parse_datetime",
                column=dataset.config.timestamp_column,
                reason="Explicit timestamp role; use configured format/unit/timezone, raising "
                "on malformed values",
            )
        )
    return CleaningPlan(source_fingerprint=dataset.fingerprint, operations=operations)


def state(frame: pd.DataFrame, column: str | None, config) -> DataState:
    if column is None:
        return DataState(row_count=len(frame), column_count=len(frame.columns))
    return DataState(
        row_count=len(frame),
        column_count=len(frame.columns),
        dtype=str(frame[column].dtype),
        missing_count=int(missing_mask(frame[column], config).sum()),
        unique_count=int(frame.loc[~missing_mask(frame[column], config), column].nunique()),
    )


def apply_cleaning(
    dataset: LoadedDataset, plan: CleaningPlan
) -> tuple[pd.DataFrame, CleaningAudit]:
    if plan.source_fingerprint != dataset.fingerprint:
        raise DatasetError(
            "dataset_cleaning_plan_stale", "Cleaning plan does not match source fingerprint", 409
        )
    frame = dataset.frame.copy(deep=True)
    audit = []
    for operation in plan.operations:
        name = operation.column
        if name is not None and name not in frame.columns:
            raise DatasetError("dataset_cleaning_column_missing", "Cleaning column is absent")
        before = state(frame, name, dataset.config)
        if operation.operation == "remove_duplicates":
            affected = int(frame.duplicated().sum())
            frame = frame.drop_duplicates(keep="first").copy()
        elif operation.operation == "normalize_empty_strings":
            mask = empty_mask(frame[name])
            affected = int(mask.sum())
            frame[name] = frame[name].mask(mask, pd.NA)
        elif operation.operation == "coerce_numeric":
            converted, missing, invalid, infinite = numeric_view(frame[name], dataset.config)
            if (invalid or infinite) and operation.errors == "raise":
                raise DatasetError(
                    "dataset_cleaning_numeric_invalid",
                    "Numeric cleaning encountered invalid or infinite values",
                )
            affected = int((~frame[name].isna()).sum())
            frame[name] = converted
        else:
            if name != dataset.config.timestamp_column:
                raise DatasetError(
                    "dataset_cleaning_timestamp_unconfigured",
                    "Timestamp cleaning requires its explicit column configuration",
                )
            try:
                converted = parse_timestamps(frame[name], dataset.config)
            except (ValueError, TypeError, OverflowError):
                raise DatasetError(
                    "dataset_cleaning_timestamp_invalid", "Timestamp configuration cannot be parsed"
                ) from None
            invalid = int((converted.isna() & ~missing_mask(frame[name], dataset.config)).sum())
            if invalid and operation.errors == "raise":
                raise DatasetError(
                    "dataset_cleaning_timestamp_invalid",
                    "Timestamp cleaning encountered malformed values",
                )
            affected = int((~frame[name].isna()).sum())
            frame[name] = converted
        audit.append(
            AuditRecord(
                operation=operation.operation,
                affected_column=name,
                rows_affected=affected,
                before=before,
                after=state(frame, name, dataset.config),
                reason=operation.reason,
                errors_policy=operation.errors,
            )
        )
    output_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return frame, CleaningAudit(
        source_fingerprint=dataset.fingerprint,
        processed_fingerprint="sha256:" + hashlib.sha256(output_bytes).hexdigest(),
        generated_at=datetime.now(UTC),
        operations=audit,
    )


def save_processed(
    dataset: LoadedDataset, frame: pd.DataFrame, audit: CleaningAudit, output: Path
) -> Path:
    source = dataset.config.source_path.resolve()
    output = output.resolve()
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    if source in {output, audit_path}:
        raise DatasetError(
            "dataset_source_overwrite_forbidden", "Processed output must differ from source"
        )
    if output.exists() or audit_path.exists():
        raise DatasetError("dataset_output_exists", "Output already exists; choose a new path", 409)
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if "sha256:" + hashlib.sha256(content).hexdigest() != audit.processed_fingerprint:
        raise DatasetError(
            "dataset_cleaning_audit_mismatch", "Processed data does not match its audit"
        )
    write_artifact(output, content, overwrite=False)
    try:
        write_artifact(audit_path, audit.model_dump_json(indent=2).encode("utf-8"), overwrite=False)
    except DatasetError:
        output.unlink(missing_ok=True)
        raise
    return audit_path
