"""Descriptive ordering/leakage and missingness review; no predictive model fitting."""

import pandas as pd

from riskbyte_ml.data.models import QualityWarning
from riskbyte_ml.data.values import numeric_view, safe_float
from riskbyte_ml.datasets.ieee_cis.config import ELAPSED, JOIN_KEY, TARGET
from riskbyte_ml.datasets.ieee_cis.models import CleaningReview, OrderingBucket, OrderingReview


def ordering_reviews(dataset, profile):
    frame, config = dataset.frame, dataset.config
    reviews = []
    views = {}
    for name in (JOIN_KEY, ELAPSED):
        values, _, _, _ = numeric_view(frame[name], config)
        views[name] = values
        valid = values.notna()
        correlation, buckets = None, []
        reason = None
        if profile.quality.target.status != "valid":
            reason = "valid_binary_target_required"
        elif int(valid.sum()) < 3 or values[valid].nunique() < 2:
            reason = "insufficient_nonconstant_numeric_observations"
        else:
            binary = frame.loc[valid, TARGET].eq(config.positive_class).astype(float)
            scaled = values[valid] / max(float(values[valid].abs().max()), 1)
            if binary.nunique() > 1:
                correlation = safe_float(scaled.corr(binary))
            # Quantile boundaries keep equal values together; never split identical elapsed times.
            grouped = (
                pd.DataFrame(
                    {
                        "bucket": pd.qcut(
                            values[valid], q=min(10, values[valid].nunique()), duplicates="drop"
                        ),
                        "fraud": binary,
                    }
                )
                .groupby("bucket", observed=True, sort=True)["fraud"]
                .agg(["size", "sum"])
            )
            buckets = [
                OrderingBucket(
                    bucket=i,
                    row_count=int(row["size"]),
                    fraud_count=int(row["sum"]),
                    fraud_prevalence=float(row["sum"] / row["size"]),
                )
                for i, (_, row) in enumerate(grouped.iterrows())
                if row["size"]
            ]
        reviews.append(
            OrderingReview(
                column=name,
                status="UNAVAILABLE_WITH_REASON" if reason else "AVAILABLE",
                reason=reason,
                numeric_observations=int(valid.sum()),
                file_order_monotonic=bool(values[valid].is_monotonic_increasing)
                if int(valid.sum())
                else None,
                target_correlation=correlation,
                buckets=buckets,
                explanation=(
                    "Descriptive numeric correlation and equal-value-preserving "
                    "quantile buckets; ordering/label drift is not proven leakage or "
                    "model performance"
                ),
            )
        )
    pair = pd.DataFrame(views).dropna()
    rank_correlation = None
    if len(pair) >= 3 and all(pair[name].nunique() > 1 for name in pair):
        # Pearson over ranks avoids an optional SciPy dependency.
        rank_correlation = safe_float(pair[JOIN_KEY].rank().corr(pair[ELAPSED].rank()))
    return reviews, rank_correlation


def leakage_candidates(dataset, profile):
    findings = [warning for warning in profile.warnings if warning.code.startswith("leakage_")]
    findings.extend(
        [
            QualityWarning(
                code="ieee_identifier_direct_input",
                severity="WARNING",
                column=JOIN_KEY,
                explanation=(
                    "Unique transaction audit key; review numeric ordering artifacts "
                    "and exclude direct identity memorization"
                ),
                inferred=True,
            ),
            QualityWarning(
                code="ieee_elapsed_direct_input",
                severity="WARNING",
                column=ELAPSED,
                explanation=(
                    "Raw elapsed time can encode collection-period drift; use for "
                    "chronology and validate any direct predictive use across "
                    "held-out time"
                ),
                inferred=True,
            ),
        ]
    )
    elapsed, _, _, _ = numeric_view(dataset.frame[ELAPSED], dataset.config)
    negatives = int(elapsed.lt(0).sum())
    if negatives:
        findings.append(
            QualityWarning(
                code="ieee_negative_elapsed_time",
                severity="ERROR",
                column=ELAPSED,
                rows_affected=negatives,
                explanation="Negative elapsed values require source review",
                inferred=False,
            )
        )
    return findings


def cleaning_review(profile):
    missing = profile.quality.missing_values
    return CleaningReview(
        extreme_missingness_columns=[
            name for name, summary in missing.items() if summary.percentage >= 90
        ],
        moderate_missingness_columns=[
            name for name, summary in missing.items() if 20 <= summary.percentage < 90
        ],
        possible_missingness_indicators=[
            name for name, summary in missing.items() if summary.count
        ],
        recommendation=(
            "Retain all fields/rows in Phase 1.5. Review structural identity "
            "missingness, train-only numeric imputation and explicit "
            "categorical missing states plus candidate missingness indicators "
            "in Phase 2. No threshold-only dropping; test missingness drift "
            "and feature availability."
        ),
    )
