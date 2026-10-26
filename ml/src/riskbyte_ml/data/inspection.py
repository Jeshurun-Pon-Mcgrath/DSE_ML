"""Conservative schema heuristics; inferred candidates never select operational roles."""

import re

import pandas as pd

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.ingestion import LoadedDataset
from riskbyte_ml.data.models import Candidate, ColumnMetadata, RoleSelection, SchemaReport
from riskbyte_ml.data.values import missing_mask, numeric_view, percentage

ID_HINT = re.compile(r"(^|_)id($|_)|identifier|account|customer|nameorig|namedest", re.I)
TIME_HINT = re.compile(r"timestamp|datetime|(^|_)(time|date|step)($|_)|_at$|transactiondt", re.I)
TARGET_HINT = re.compile(r"fraud|(^|_)(target|label|class|outcome)($|_)", re.I)
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$"


def role(column: str | None, columns) -> RoleSelection:
    if column is None:
        return RoleSelection(column=None, available=False, reason="field_not_configured")
    exists = column in columns
    return RoleSelection(
        column=column, available=exists, reason=None if exists else "configured_field_absent"
    )


def inspect_schema(dataset: LoadedDataset) -> SchemaReport:
    frame, config = dataset.frame, dataset.config
    metadata, identifiers, targets, timestamps = [], [], [], []
    high_cardinality = []
    for name in frame.columns:
        series = frame[name]
        present = series[~missing_mask(series, config)].astype("string")
        count, unique = len(present), int(present.nunique())
        numeric, _, invalid, infinite = numeric_view(series, config)
        numerical_count = int(numeric.notna().sum()) + infinite
        sample = present.iloc[:1000]
        datetime_fraction = 0.0
        if len(sample) and sample.str.match(DATE_PATTERN).mean() >= 0.95:
            datetime_fraction = float(
                pd.to_datetime(sample, format="ISO8601", errors="coerce", utc=True).notna().mean()
            )
        kind, reason = (
            "categorical",
            "Non-missing values are not consistently numeric, boolean or ISO dates",
        )
        if not count:
            kind, reason = "unknown", "No non-missing observations"
        elif name in config.categorical_columns:
            kind, reason = (
                "categorical",
                "Explicit categorical role; numeric codes are not measures",
            )
        elif name in config.numeric_columns or name == config.amount_column:
            kind, reason = (
                "numeric",
                "Explicit numeric analysis role; source strings remain unchanged",
            )
        elif present.str.lower().isin(["true", "false"]).all():
            kind, reason = "boolean", "All observed values are textual true/false (inferred)"
        elif datetime_fraction >= 0.95:
            kind, reason = (
                "datetime",
                "At least 95% of the deterministic ISO-date sample parses (inferred)",
            )
        elif numerical_count / count >= 0.95 and numerical_count:
            kind, reason = (
                "numeric",
                "At least 95% of non-missing observations parse numerically (inferred)",
            )
        dominant = int(present.value_counts().iloc[0]) if count else 0
        constant = count > 0 and unique == 1
        near = (
            unique > 1
            and count >= config.heuristic_min_rows
            and dominant / count >= config.near_constant_threshold
        )
        metadata.append(
            ColumnMetadata(
                name=name,
                storage_dtype=str(series.dtype),
                inferred_type=kind,
                unique_count=unique,
                non_missing_count=count,
                dominant_percentage=percentage(dominant, count) if count else None,
                constant=constant,
                near_constant=near,
                inference_reason=reason,
            )
        )
        if ID_HINT.search(name):
            identifiers.append(
                Candidate(
                    column=name, reason="Identifier-like column name; meaning requires confirmation"
                )
            )
        elif count >= config.heuristic_min_rows and unique / count >= config.unique_ratio_threshold:
            identifiers.append(
                Candidate(
                    column=name,
                    reason="Nearly unique values; possible identifier or continuous measurement",
                )
            )
        if TARGET_HINT.search(name):
            targets.append(
                Candidate(
                    column=name,
                    reason="Outcome-like name; confirm labels and configure both target classes",
                )
            )
        elif count >= config.heuristic_min_rows and unique == 2:
            targets.append(
                Candidate(
                    column=name,
                    reason="Two observed values; many binary predictors are not targets",
                )
            )
        if TIME_HINT.search(name) or kind == "datetime":
            timestamps.append(
                Candidate(
                    column=name,
                    reason="Time-like name or ISO dates; confirm units, timezone and semantics",
                )
            )
        if kind == "categorical" and unique >= config.high_cardinality_threshold:
            high_cardinality.append(name)
    groups = {
        kind: [column.name for column in metadata if column.inferred_type == kind]
        for kind in ("numeric", "categorical", "boolean", "datetime")
    }
    return SchemaReport(
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=metadata,
        numeric_columns=groups["numeric"],
        categorical_columns=groups["categorical"],
        boolean_columns=groups["boolean"],
        datetime_like_columns=groups["datetime"],
        constant_columns=[column.name for column in metadata if column.constant],
        near_constant_columns=[column.name for column in metadata if column.near_constant],
        high_cardinality_categorical_columns=high_cardinality,
        potential_identifiers=identifiers,
        potential_targets=targets,
        potential_timestamps=timestamps,
        target=role(config.target_column, frame.columns),
        timestamp=role(config.timestamp_column, frame.columns),
        amount=role(config.amount_column, frame.columns),
        identifiers=[role(name, frame.columns) for name in config.identifier_columns],
    )


def analytical_numeric_columns(schema: SchemaReport, config: DatasetConfig) -> list[str]:
    """Configured identifiers/target are not continuous measurements."""
    excluded = {*config.identifier_columns, config.target_column}
    return [name for name in schema.numeric_columns if name not in excluded]
