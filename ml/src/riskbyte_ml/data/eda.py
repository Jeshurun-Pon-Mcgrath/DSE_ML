"""Truthful structured EDA; no plot, field or target semantics are fabricated."""

import pandas as pd

from riskbyte_ml.data.ingestion import LoadedDataset
from riskbyte_ml.data.inspection import analytical_numeric_columns
from riskbyte_ml.data.models import (
    CategorySummary,
    CorrelationMatrix,
    EDAReport,
    Frequency,
    QualityReport,
    SchemaReport,
    TimeBucket,
)
from riskbyte_ml.data.statistics import descriptive_statistics, outlier_overview
from riskbyte_ml.data.values import (
    missing_mask,
    numeric_view,
    parse_timestamps,
    percentage,
    safe_float,
)


def available(data):
    return {"status": "AVAILABLE", "data": data}


def unavailable(reason):
    return {"status": "UNAVAILABLE_WITH_REASON", "reason": reason}


def categorical_frequencies(dataset: LoadedDataset, schema: SchemaReport):
    frame, config = dataset.frame, dataset.config
    # Identifier-like values need not be disclosed to consumers of aggregate EDA.
    excluded = {*config.identifier_columns, *(c.column for c in schema.potential_identifiers)}
    result = {}
    for name in schema.categorical_columns + schema.boolean_columns:
        if name in excluded:
            continue
        present = frame.loc[~missing_mask(frame[name], config), name]
        counts = present.value_counts()
        # Stable tie order makes frequency output reproducible across runs.
        items = sorted(
            ((str(value), int(count)) for value, count in counts.items() if count > 0),
            key=lambda x: (-x[1], x[0]),
        )
        top = items[: config.frequency_limit]
        result[name] = CategorySummary(
            non_missing_count=len(present),
            unique_count=len(items),
            frequencies=[
                Frequency(value=v, count=n, percentage=percentage(n, len(present))) for v, n in top
            ],
            omitted_count=len(present) - sum(n for _, n in top),
        )
    return (
        available(result)
        if result
        else unavailable("no_suitable_non_identifier_categorical_fields")
    )


def correlation_matrix(dataset: LoadedDataset, schema: SchemaReport):
    names = [
        name
        for name in analytical_numeric_columns(schema, dataset.config)
        if name not in schema.constant_columns
    ]
    selected = names[: dataset.config.correlation_column_limit]
    if len(selected) < 2:
        return unavailable("at_least_two_nonconstant_numeric_fields_required")
    numbers = pd.DataFrame(
        {name: numeric_view(dataset.frame[name], dataset.config)[0] for name in selected}
    )
    # Scale each column before pairwise Pearson calculations to avoid squaring large inputs.
    scaled = numbers / numbers.abs().max().replace(0, 1)
    matrix = scaled.corr(min_periods=3)
    masks = numbers.notna().astype("int64")
    pairs = masks.T.dot(masks)
    return available(
        CorrelationMatrix(
            columns=selected,
            excluded_columns=[name for name in dataset.frame.columns if name not in selected],
            values={a: {b: safe_float(matrix.loc[a, b]) for b in selected} for a in selected},
            pair_counts={a: {b: int(pairs.loc[a, b]) for b in selected} for a in selected},
        )
    )


def temporal_fraud_frequency(
    dataset: LoadedDataset, schema: SchemaReport, quality: QualityReport, period: str
):
    config, frame = dataset.config, dataset.frame
    if quality.target.status != "valid":
        return unavailable("a_valid_explicit_binary_target_is_required")
    if not schema.timestamp.available:
        return unavailable(schema.timestamp.reason)
    if config.timestamp_kind == "elapsed":
        return unavailable("elapsed_time_has_no_confirmed_calendar_anchor")
    if quality.malformed_timestamps.get(config.timestamp_column, 0):
        return unavailable("configured_timestamps_contain_invalid_values")
    try:
        dates = parse_timestamps(frame[config.timestamp_column], config)
    except (ValueError, TypeError, OverflowError):
        return unavailable("timestamp_configuration_cannot_be_parsed")
    valid = dates.notna()
    if not int(valid.sum()):
        return unavailable("no_valid_timestamps")
    bucket = dates.dt.hour if period == "hour" else dates.dt.dayofweek
    names = (
        [f"{i:02d}:00" for i in range(24)]
        if period == "hour"
        else ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    result = []
    for index, label in enumerate(names):
        mask = valid & bucket.eq(index)
        total = int(mask.sum())
        fraud = int((mask & frame[config.target_column].eq(config.positive_class)).sum())
        result.append(
            TimeBucket(
                bucket=index,
                label=label,
                total_count=total,
                fraud_count=fraud,
                fraud_percentage=percentage(fraud, total) if total else None,
            )
        )
    return available(result)


def generate_eda(dataset: LoadedDataset, schema: SchemaReport, quality: QualityReport, statistics):
    frame, config = dataset.frame, dataset.config
    target_valid = quality.target.status == "valid"
    class_distribution = (
        available(quality.target)
        if target_valid
        else unavailable("a_valid_explicit_binary_target_is_required")
    )
    amount_distribution = unavailable(schema.amount.reason or "no_finite_amount_values")
    comparison = unavailable("a_valid_explicit_target_and_finite_amount_field_are_required")
    if schema.amount.available:
        stats = descriptive_statistics(frame[config.amount_column], config)
        amount_distribution = (
            available(stats) if stats.count else unavailable("no_finite_amount_values")
        )
        if target_valid and stats.count:
            groups = {
                "fraud": descriptive_statistics(
                    frame.loc[
                        frame[config.target_column].eq(config.positive_class), config.amount_column
                    ],
                    config,
                ),
                "legitimate": descriptive_statistics(
                    frame.loc[
                        frame[config.target_column].eq(config.negative_class), config.amount_column
                    ],
                    config,
                ),
            }
            comparison = (
                available(groups)
                if all(value.count for value in groups.values())
                else unavailable("both_target_classes_need_finite_amount_observations")
            )
    outliers = {name: outlier_overview(frame[name], config) for name in statistics}
    finite_stats = {name: value for name, value in statistics.items() if value.count}
    return EDAReport(
        class_distribution=class_distribution,
        amount_distribution=amount_distribution,
        fraud_amount_comparison=comparison,
        fraud_by_hour=temporal_fraud_frequency(dataset, schema, quality, "hour"),
        fraud_by_day_of_week=temporal_fraud_frequency(dataset, schema, quality, "weekday"),
        correlation=correlation_matrix(dataset, schema),
        outliers=available(outliers)
        if any(item.finite_count for item in outliers.values())
        else unavailable("no_finite_numeric_observations"),
        missing_data=available(quality.missing_values),
        numeric_distributions=available(finite_stats)
        if finite_stats
        else unavailable("no_finite_numeric_observations"),
        categorical_frequencies=categorical_frequencies(dataset, schema),
    )
