"Finite-only statistics, rank tests, categorical tests, FDR and Wilson estimation."

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.statistics import descriptive_statistics
from riskbyte_ml.data.values import safe_float
from riskbyte_ml.statistics.models import EffectSize, Prevalence, SummaryStatistics, TestOutcome

INDEPENDENCE = (
    "Independent observations are not established: repeated entities and attack episodes "
    "cluster rows. P-values are nominal exploratory evidence, not calibrated population inference."
)


def finite_values(values):
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numbers[np.isfinite(numbers)]


def summarize(values) -> SummaryStatistics:
    series = pd.Series(values)
    summary = descriptive_statistics(series, DatasetConfig())
    array = finite_values(series)
    kurtosis = None
    if len(array) >= 4 and np.ptp(array / (np.max(np.abs(array)) or 1)) > 0:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            kurtosis = safe_float(stats.kurtosis(array / np.max(np.abs(array)), bias=False))
    note = (
        "No finite observations"
        if not len(array)
        else "Constant observed values; standardized moments undefined"
        if np.unique(array).size == 1
        else "Strongly skewed; mean and median describe different aspects"
        if summary.skewness is not None and abs(summary.skewness) > 1
        else "Inspect quartiles and tails; shape alone does not establish normality"
    )
    return SummaryStatistics(**summary.model_dump(), kurtosis=kurtosis, distribution_note=note)


def cliffs_delta(first, second) -> float | None:
    """P(first > second)-P(first < second); O(n log n), no quadratic pair matrix."""
    x, y = finite_values(first), np.sort(finite_values(second))
    if not len(x) or not len(y):
        return None
    greater = np.searchsorted(y, x, side="left").sum(dtype=np.float64)
    smaller = (len(y) - np.searchsorted(y, x, side="right")).sum(dtype=np.float64)
    return float((greater - smaller) / (len(x) * len(y)))


def delta_effect(first, second) -> EffectSize:
    delta = cliffs_delta(first, second)
    magnitude = (
        "unavailable"
        if delta is None
        else "negligible"
        if abs(delta) < 0.147
        else "small"
        if abs(delta) < 0.33
        else "medium"
        if abs(delta) < 0.474
        else "large"
    )
    return EffectSize(
        method="Cliffs_delta",
        value=delta,
        magnitude=magnitude + "; conventional heuristic, not business importance",
        direction="positive means first group tends to have larger values; ties contribute zero",
    )


def mann_whitney(first, second) -> TestOutcome:
    x, y = finite_values(first), finite_values(second)
    counts = {"first": len(x), "second": len(y)}
    excluded = {"first": len(first) - len(x), "second": len(second) - len(y)}
    checks = [INDEPENDENCE, "Ordinal/numeric ranks; no equal-variance or normality requirement"]
    common = dict(
        group_counts=counts,
        excluded_counts=excluded,
        group_summaries={"first": summarize(first), "second": summarize(second)},
        assumption_checks=checks,
        effect_size=delta_effect(x, y),
    )
    if min(len(x), len(y)) < 2:
        return TestOutcome(
            status="UNAVAILABLE_WITH_REASON",
            reason="fewer than two finite rows in a group",
            method="Mann-Whitney U",
            **common,
        )
    ties = np.unique(np.concatenate([x, y])).size < len(x) + len(y)
    if np.unique(np.concatenate([x, y])).size == 1:
        return TestOutcome(
            status="AVAILABLE",
            method="Mann-Whitney U; constant pooled sample",
            statistic=float(len(x) * len(y) / 2),
            p_value=1.0,
            **common,
        )
    method = "exact" if min(len(x), len(y)) <= 8 and not ties else "asymptotic"
    # Small tied samples need permutation rather than an unreliable normal approximation.
    selected = (
        stats.PermutationMethod(n_resamples=9999, rng=np.random.default_rng(42))
        if min(len(x), len(y)) < 10 and ties
        else method
    )
    # Rank originals: scaling can underflow across extreme ranges and create artificial ties.
    result = stats.mannwhitneyu(x, y, alternative="two-sided", method=selected)
    checks.extend(
        [
            f"ties_present={ties}; finite-only group sizes={len(x)},{len(y)}",
            "Different distribution shapes prevent interpreting U solely as a median difference",
        ]
    )
    return TestOutcome(
        status="AVAILABLE",
        method="Mann-Whitney U; "
        + (
            "permutation seed=42, <=9999 resamples"
            if not isinstance(selected, str)
            else method + "; tie/continuity correction if asymptotic"
        ),
        statistic=safe_float(result.statistic),
        p_value=safe_float(result.pvalue),
        p_value_underflow=bool(result.pvalue == 0),
        **common,
    )


def categorical_test(table, *, group_counts=None, excluded_counts=None) -> TestOutcome:
    observed = np.asarray(table)
    if (
        observed.ndim != 2
        or min(observed.shape) < 2
        or not np.isfinite(observed).all()
        or (observed < 0).any()
        or (observed != np.floor(observed)).any()
    ):
        raise ValueError("Contingency table requires nonnegative integer counts and >=2x2")
    observed = observed.astype(np.int64)
    common = dict(
        group_counts=group_counts or {str(i): int(row.sum()) for i, row in enumerate(observed)},
        excluded_counts=excluded_counts or {},
        contingency_table=observed.tolist(),
        assumption_checks=[INDEPENDENCE],
    )
    if (observed.sum(axis=0) == 0).any() or (observed.sum(axis=1) == 0).any():
        return TestOutcome(
            status="UNAVAILABLE_WITH_REASON",
            reason="zero marginal count; no association test",
            method="categorical association",
            **common,
        )
    chi = stats.chi2_contingency(observed, correction=False)
    minimum = float(chi.expected_freq.min())
    common["assumption_checks"].append(f"minimum expected cell count={minimum:.6g}")
    v = float(np.sqrt(chi.statistic / (observed.sum() * min(np.array(observed.shape) - 1))))
    odds, odds_reason = None, None
    if observed.shape == (2, 2):
        a, b, c, d = (float(v) for v in observed.ravel())
        odds = a * d / (b * c) if b * c else None
        if odds is None:
            odds_reason = "zero denominator; unbounded/undefined odds ratio represented as null"
    effect = EffectSize(
        method="Cramers_V",
        value=v,
        magnitude="unsigned association strength; no universal practical threshold",
        direction="odds ratio uses table [fraud:new,seen; legitimate:new,seen] when 2x2",
        odds_ratio=odds,
        odds_ratio_reason=odds_reason,
    )
    if observed.shape == (2, 2) and minimum < 5:
        result = stats.fisher_exact(observed, alternative="two-sided")
        return TestOutcome(
            status="AVAILABLE",
            method="Fisher exact; two-sided",
            statistic=safe_float(result.statistic),
            p_value=safe_float(result.pvalue),
            effect_size=effect,
            **common,
        )
    if minimum < 5:
        return TestOutcome(
            status="UNAVAILABLE_WITH_REASON",
            reason="sparse non-2x2 table; asymptotic chi-square inappropriate",
            method="Chi-square independence",
            effect_size=effect,
            **common,
        )
    return TestOutcome(
        status="AVAILABLE",
        method="Chi-square independence; Pearson, no continuity correction",
        statistic=float(chi.statistic),
        p_value=float(chi.pvalue),
        p_value_underflow=bool(chi.pvalue == 0),
        effect_size=effect,
        **common,
    )


def benjamini_hochberg(p_values):
    """All prespecified tests remain in family; unavailable tests contribute p=1."""
    values = np.array([1.0 if p is None else p for p in p_values], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite in [0,1] or unavailable")
    if not len(values):
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate(
        (values[order] * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1].clip(0, 1)
    restored = np.empty(len(values))
    restored[order] = adjusted
    return [None if p is None else float(restored[i]) for i, p in enumerate(p_values)]


def prevalence_interval(fraud_count: int, rows: int, confidence=0.95) -> Prevalence:
    if rows < 0 or not 0 <= fraud_count <= rows or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts/confidence level")
    p, lower, upper = None, None, None
    if rows:
        p = fraud_count / rows
        z = float(stats.norm.ppf((1 + confidence) / 2))
        denominator = 1 + z * z / rows
        centre = (p + z * z / (2 * rows)) / denominator
        half = z * np.sqrt(p * (1 - p) / rows + z * z / (4 * rows * rows)) / denominator
        lower, upper = max(0.0, float(centre - half)), min(1.0, float(centre + half))
    return Prevalence(
        rows=rows,
        fraud_count=fraud_count,
        prevalence=p,
        confidence_level=confidence,
        lower=lower,
        upper=upper,
    )
