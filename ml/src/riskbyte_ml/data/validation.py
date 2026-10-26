"""Reusable measured checks and explicitly inferred leakage warnings."""

import re

import numpy as np
import pandas as pd

from riskbyte_ml.data.ingestion import LoadedDataset
from riskbyte_ml.data.models import (
    DuplicateSummary,
    MissingSummary,
    QualityReport,
    QualityWarning,
    SchemaReport,
    TargetDistribution,
)
from riskbyte_ml.data.values import (
    empty_mask,
    missing_mask,
    numeric_view,
    parse_timestamps,
    percentage,
)


def target_distribution(dataset: LoadedDataset) -> TargetDistribution:
    frame, config = dataset.frame, dataset.config
    if config.target_column is None:
        return TargetDistribution(status="not_configured")
    series = frame[config.target_column]
    missing = missing_mask(series, config)
    positive = series.eq(config.positive_class).fillna(False) & ~missing
    negative = series.eq(config.negative_class).fillna(False) & ~missing
    fraud, legitimate = int(positive.sum()), int(negative.sum())
    invalid = int((~missing & ~positive & ~negative).sum())
    valid = fraud + legitimate
    counts = [fraud, legitimate]
    tied = fraud == legitimate
    return TargetDistribution(
        status="invalid" if invalid or int(missing.sum()) or not valid else "valid",
        column=config.target_column,
        positive_class=config.positive_class,
        negative_class=config.negative_class,
        fraud_count=fraud,
        legitimate_count=legitimate,
        fraud_percentage=percentage(fraud, valid) if valid else None,
        legitimate_percentage=percentage(legitimate, valid) if valid else None,
        positive_class_prevalence=fraud / valid if valid else None,
        imbalance_ratio=max(counts) / min(counts) if min(counts) else None,
        minority_class=None
        if tied
        else config.positive_class
        if fraud < legitimate
        else config.negative_class,
        majority_class=None
        if tied
        else config.positive_class
        if fraud > legitimate
        else config.negative_class,
        missing_count=int(missing.sum()),
        unsupported_count=invalid,
        valid_count=valid,
    )


def leakage_warnings(
    dataset: LoadedDataset, schema: SchemaReport, target: TargetDistribution
) -> list[QualityWarning]:
    frame, config = dataset.frame, dataset.config
    if config.target_column is None:
        return []
    warnings = []
    y = frame[config.target_column]
    for name in frame.columns:
        if name == config.target_column:
            continue
        series = frame[name]
        reason, code, count = None, None, None
        if (
            series.astype("string").eq(y.astype("string")).fillna(False)
            | (series.isna() & y.isna())
        ).all():
            reason, code = (
                "Directly duplicates target; review source timing and provenance",
                "leakage_target_duplicate",
            )
            count = len(frame)
        elif re.search(
            r"fraud.?status|final.?decision|label.?observed|chargeback|"
            r"confirmed.?at|resolved.?at|review.?result|outcome",
            name,
            re.I,
        ):
            reason, code = (
                "Outcome/post-outcome-like name; may not exist at inference time",
                "leakage_outcome_name",
            )
        elif target.status == "valid":
            mask = ~missing_mask(series, config) & ~missing_mask(y, config)
            n = int(mask.sum())
            if n >= config.heuristic_min_rows:
                binary = y[mask].eq(config.positive_class).astype(float)
                if name in schema.numeric_columns:
                    numeric, _, _, _ = numeric_view(series[mask], config)
                    paired = numeric.notna()
                    if (
                        int(paired.sum()) >= config.heuristic_min_rows
                        and numeric[paired].nunique() > 1
                        and binary[paired].nunique() > 1
                    ):
                        # Scale before correlation to avoid overflow for extreme finite inputs.
                        scaled = numeric[paired] / max(float(numeric[paired].abs().max()), 1.0)
                        correlation = scaled.corr(binary[paired])
                        if (
                            np.isfinite(correlation)
                            and abs(correlation) >= config.leakage_correlation_threshold
                        ):
                            reason, code = (
                                "Near-perfect numeric relationship; investigate possible leakage",
                                "leakage_target_relationship",
                            )
                            count = int(paired.sum())
                else:
                    pairs = pd.DataFrame({"x": series[mask], "y": binary})
                    grouped = pairs.groupby("x", observed=True)["y"].agg(["size", "sum"])
                    if 1 < len(grouped) <= min(20, n // 5) and grouped["size"].min() >= 2:
                        purity = (
                            np.maximum(grouped["sum"], grouped["size"] - grouped["sum"]).sum() / n
                        )
                        if purity >= config.leakage_correlation_threshold:
                            reason, code = (
                                "Categories nearly determine target; investigate source provenance",
                                "leakage_target_relationship",
                            )
                            count = n
        if reason:
            warnings.append(
                QualityWarning(
                    code=code,
                    severity="WARNING",
                    column=name,
                    rows_affected=count,
                    explanation=reason,
                    inferred=True,
                )
            )
    return warnings


def validate_dataset(dataset: LoadedDataset, schema: SchemaReport) -> QualityReport:
    frame, config = dataset.frame, dataset.config
    warnings, missing, invalid_numbers, infinities, malformed = [], {}, {}, {}, {}

    def warn(code, explanation, column=None, count=None, severity="WARNING", inferred=False):
        warnings.append(
            QualityWarning(
                code=code,
                severity=severity,
                column=column,
                rows_affected=count,
                explanation=explanation,
                inferred=inferred,
            )
        )

    for name in frame.columns:
        count = int(missing_mask(frame[name], config).sum())
        empty = int(empty_mask(frame[name]).sum())
        missing[name] = MissingSummary(
            count=count, percentage=percentage(count, len(frame)), empty_string_count=empty
        )
        if count:
            warn(
                "missing_values",
                "Missing values require an explicit policy; no rows were removed",
                name,
                count,
            )
        if empty:
            warn(
                "empty_strings",
                "Blank or whitespace-only fields are preserved and counted as missing",
                name,
                empty,
            )
        if name in schema.numeric_columns:
            _, _, bad, inf = numeric_view(frame[name], config)
            invalid_numbers[name], infinities[name] = bad, inf
            if bad:
                warn(
                    "invalid_numeric_values",
                    "Values fail numeric parsing; no coercion was applied",
                    name,
                    bad,
                    "ERROR"
                    if name in config.numeric_columns or name == config.amount_column
                    else "WARNING",
                )
            if inf:
                warn(
                    "infinite_numeric_values",
                    "Infinite values are excluded from numeric summaries, not deleted",
                    name,
                    inf,
                )
    duplicate_extra = int(frame.duplicated().sum())
    duplicate_groups = int(frame.duplicated(keep=False).sum())
    if duplicate_extra:
        warn(
            "duplicate_rows",
            "Repeated complete rows found; removal requires an explicit cleaning operation",
            count=duplicate_extra,
        )
    id_duplicates = {}
    for selection in [schema.timestamp, schema.amount, *schema.identifiers]:
        if selection.column is not None and not selection.available:
            warn(
                "configured_optional_field_absent",
                "Configured optional field is absent; dependent analyses are unavailable",
                selection.column,
            )
    for name in config.numeric_columns:
        if name not in frame.columns:
            warn(
                "configured_numeric_field_absent",
                "Explicit numeric field is absent",
                name,
                severity="ERROR",
            )
    for name in config.identifier_columns:
        if name in frame.columns:
            observed = frame.loc[~missing_mask(frame[name], config), name]
            extra = int(observed.duplicated().sum())
            id_duplicates[name] = extra
            if extra:
                unique = name in config.unique_identifier_columns
                warn(
                    "duplicate_identifier_values",
                    "Repeated identifier values; repetition can be valid for entity identifiers",
                    name,
                    extra,
                    "ERROR" if unique else "INFO",
                )
    if schema.timestamp.available:
        try:
            parsed = parse_timestamps(frame[config.timestamp_column], config)
        except (ValueError, TypeError, OverflowError):
            parsed = pd.Series(pd.NaT, index=frame.index)
        bad = int((parsed.isna() & ~missing_mask(frame[config.timestamp_column], config)).sum())
        malformed[config.timestamp_column] = bad
        if bad:
            warn(
                "malformed_timestamps",
                "Values do not parse under configured format/unit/timezone; source is unchanged",
                config.timestamp_column,
                bad,
                "ERROR",
            )
    for column in schema.columns:
        if column.constant:
            warn(
                "constant_column",
                "Only one non-missing value; review usefulness without automatic deletion",
                column.name,
            )
        if column.near_constant:
            warn(
                "near_constant_column",
                "Dominant value exceeds configured threshold",
                column.name,
                inferred=True,
            )
        if column.name in schema.high_cardinality_categorical_columns:
            warn(
                "extreme_cardinality",
                "Categorical distinct count exceeds configured threshold",
                column.name,
            )
        if (
            column.non_missing_count >= config.heuristic_min_rows
            and column.unique_count / column.non_missing_count >= config.unique_ratio_threshold
        ):
            warn(
                "suspiciously_unique",
                "Nearly unique values may be identifiers or valid continuous measurements",
                column.name,
                inferred=True,
            )
    target = target_distribution(dataset)
    if target.status == "invalid":
        warn(
            "invalid_target_values",
            "Target contains missing or unsupported labels; class-dependent EDA is unavailable",
            config.target_column,
            target.missing_count + target.unsupported_count,
            "ERROR",
        )
    if target.status == "valid":
        if target.imbalance_ratio is None:
            warn(
                "single_class_target",
                "Only one configured class is observed; imbalance ratio is undefined",
                config.target_column,
            )
        elif target.imbalance_ratio >= config.imbalance_warning_ratio:
            warn(
                "class_imbalance",
                "Majority/minority count ratio exceeds configured threshold",
                config.target_column,
            )
    warnings.extend(leakage_warnings(dataset, schema, target))
    return QualityReport(
        validation_status="FAILED" if any(w.severity == "ERROR" for w in warnings) else "PASSED",
        missing_values=missing,
        duplicates=DuplicateSummary(
            extra_rows=duplicate_extra,
            rows_in_duplicate_groups=duplicate_groups,
            identifier_extra_values=id_duplicates,
        ),
        invalid_numeric_values=invalid_numbers,
        infinite_numeric_values=infinities,
        malformed_timestamps=malformed,
        target=target,
        warnings=warnings,
    )
