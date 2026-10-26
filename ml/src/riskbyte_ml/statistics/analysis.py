"Prespecified training hypotheses plus descriptive synthetic ground-truth exploration."

import numpy as np
import pandas as pd
from scipy import stats

from riskbyte_ml.statistics.methods import (
    benjamini_hochberg,
    categorical_test,
    delta_effect,
    mann_whitney,
    prevalence_interval,
    summarize,
)
from riskbyte_ml.statistics.models import (
    CorrelationPair,
    CorrelationReport,
    DescriptiveReport,
    GroupAnalysis,
    HardNegativeReport,
    Hypothesis,
    HypothesisReport,
    NoveltySummary,
    PrevalenceReport,
    ScenarioReport,
    TemporalBucket,
    TemporalReport,
)

NUMERICAL = (
    "TransactionAmt",
    "failed_attempts_before",
    "account_age_days",
    "identity_age_days",
    "amount_minus_prior_mean",
    "amount_ratio_to_prior_mean",
    "amount_zscore_prior",
    "transactions_prior_5m",
    "transactions_prior_30m",
    "transactions_prior_1h",
    "transactions_prior_24h",
    "time_since_previous_transaction",
)
NOVELTY = (
    "device_seen_before",
    "merchant_seen_before",
    "billing_region_seen_before",
    "product_seen_before",
)


def groups(target, evaluation):
    fraud = target.astype("string").eq("1").to_numpy(dtype=bool)
    hard = evaluation.hard_negative.to_numpy(dtype=bool)
    return {
        "all": np.ones(len(target), dtype=bool),
        "legitimate": ~fraud,
        "fraudulent": fraud,
        "normal_legitimate": ~fraud & ~hard,
        "hard_negative_legitimate": ~fraud & hard,
    }


def describe(features, masks):
    return DescriptiveReport(
        groups={
            group: {name: summarize(features.loc[mask, name]) for name in NUMERICAL}
            for group, mask in masks.items()
        },
        group_counts={group: int(mask.sum()) for group, mask in masks.items()},
        fraud_vs_legitimate_effects={
            name: delta_effect(
                features.loc[masks["fraudulent"], name], features.loc[masks["legitimate"], name]
            )
            for name in NUMERICAL
        },
    )


def novelty_summary(values):
    available = values.notna()
    new_count = int(values.loc[available].eq(False).sum())
    count = int(available.sum())
    return NoveltySummary(
        available_count=count,
        unavailable_count=len(values) - count,
        new_count=new_count,
        new_fraction=new_count / count if count else None,
    )


def group_analysis(features, mask, denominator):
    subset = features.loc[mask]
    return GroupAnalysis(
        count=len(subset),
        proportion=len(subset) / denominator if denominator else 0.0,
        numeric={name: summarize(subset[name]) for name in NUMERICAL},
        novelty={name: novelty_summary(subset[name]) for name in NOVELTY},
    )


def scenario_analysis(features, target, evaluation):
    fraud = target.astype("string").eq("1").to_numpy(dtype=bool)
    names = sorted(evaluation.loc[fraud, "fraud_scenario"].unique())
    return ScenarioReport(
        scenarios={
            str(name): group_analysis(
                features,
                fraud & evaluation.fraud_scenario.eq(name).to_numpy(dtype=bool),
                int(fraud.sum()),
            )
            for name in names
        }
    )


def hard_negative_analysis(features, target, evaluation):
    masks = groups(target, evaluation)
    hard = masks["hard_negative_legitimate"]
    names = sorted(evaluation.loc[hard, "hard_negative_type"].unique())
    return HardNegativeReport(
        groups={
            name: group_analysis(features, masks[name], len(features))
            for name in ("normal_legitimate", "hard_negative_legitimate", "fraudulent")
        },
        hard_negative_types={
            str(name): group_analysis(
                features,
                hard & evaluation.hard_negative_type.eq(name).to_numpy(dtype=bool),
                int(hard.sum()),
            )
            for name in names
        },
        fraud_vs_hard_negative_effects={
            name: delta_effect(features.loc[masks["fraudulent"], name], features.loc[hard, name])
            for name in NUMERICAL
        },
    )


def novelty_test(values, fraud):
    available = values.notna().to_numpy()
    new = values.eq(False).fillna(False).to_numpy(dtype=bool)
    table = [
        [int((fraud & available & new).sum()), int((fraud & available & ~new).sum())],
        [int((~fraud & available & new).sum()), int((~fraud & available & ~new).sum())],
    ]
    result = categorical_test(
        table,
        group_counts={"fraud": sum(table[0]), "legitimate": sum(table[1])},
        excluded_counts={
            "fraud": int((fraud & ~available).sum()),
            "legitimate": int((~fraud & ~available).sum()),
        },
    )
    return result.model_copy(
        update={
            "table_layout": "rows fraud,legitimate; columns new,seen; unavailable context excluded"
        }
    )


def hypotheses(features, target, evaluation):
    masks = groups(target, evaluation)
    fraud = masks["fraudulent"]
    specifications = [
        (
            "RQ1",
            "Do transaction amount distributions differ by fraud status?",
            "TransactionAmt",
            False,
            False,
        ),
        (
            "RQ2",
            "Are failed attempts associated with fraud status?",
            "failed_attempts_before",
            False,
            False,
        ),
        (
            "RQ3",
            "Does prior 30-minute transaction velocity differ by fraud status?",
            "transactions_prior_30m",
            False,
            False,
        ),
        (
            "RQ4",
            "Is new-device behaviour associated with fraud?",
            "device_seen_before",
            True,
            False,
        ),
        (
            "RQ5",
            "Is new-merchant behaviour associated with fraud?",
            "merchant_seen_before",
            True,
            False,
        ),
        (
            "RQ6",
            (
                "Do prior-mean amount ratios differ between fraud and hard-negative "
                "legitimate transactions?"
            ),
            "amount_ratio_to_prior_mean",
            False,
            True,
        ),
    ]
    results = []
    for identifier, question, feature, categorical, hard in specifications:
        outcome = (
            novelty_test(features[feature], fraud)
            if categorical
            else mann_whitney(
                features.loc[fraud, feature],
                features.loc[masks["hard_negative_legitimate"] if hard else ~fraud, feature],
            )
        )
        h0 = (
            "Novelty and fraud status are independent"
            if categorical
            else "The two feature distributions are equal; practical equivalence is not tested"
        )
        h1 = (
            "Novelty and fraud status are associated"
            if categorical
            else "The two feature distributions differ"
        )
        results.append(
            Hypothesis(
                id=identifier,
                question=question,
                feature=feature,
                h0=h0,
                h1=h1,
                alpha=0.05,
                test_selection_reasoning=(
                    (
                        "Binary context with observed-cell/expected-cell checks; Fisher exact "
                        "for sparse 2x2 tables"
                    )
                    if categorical
                    else (
                        "Skewed, discrete and unequal groups: two-sided rank test with ties; "
                        "no normal/equal variance claim"
                    )
                ),
                outcome=outcome,
                decision="UNAVAILABLE",
                practical_interpretation="Pending correction",
            )
        )
    adjusted = benjamini_hochberg([item.outcome.p_value for item in results])
    for result, p in zip(results, adjusted, strict=True):
        result.adjusted_p_value = p
        result.decision = (
            "UNAVAILABLE"
            if p is None
            else "REJECT_H0_NOMINALLY"
            if p < result.alpha
            else "DO_NOT_REJECT_H0"
        )
        result.practical_interpretation = (
            (
                "Synthetic-data finding: nominal evidence of association/distribution "
                "difference after FDR"
            )
            if result.decision == "REJECT_H0_NOMINALLY"
            else (
                "Synthetic-data finding: insufficient evidence; non-rejection does not "
                "establish equality"
            )
        ) + (
            ". Review effect magnitude, unequal shapes, missing-context selection "
            "and clustered observations; not a risk probability or causal claim."
        )
    return HypothesisReport(family_size=len(results), hypotheses=results)


def correlations(features):
    pairs = []
    for i, first in enumerate(NUMERICAL):
        for second in NUMERICAL[i + 1 :]:
            x = pd.to_numeric(features[first], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(features[second], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            reason, pearson, spearman = None, None, None
            if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
                reason = "insufficient paired observations or zero variance"
            else:
                pearson = float(
                    stats.pearsonr(
                        x / max(np.abs(x).max(), 1), y / max(np.abs(y).max(), 1)
                    ).statistic
                )
                spearman = float(stats.spearmanr(x, y).statistic)
            pairs.append(
                CorrelationPair(
                    first=first,
                    second=second,
                    pair_count=len(x),
                    pearson=pearson,
                    spearman=spearman,
                    reason=reason,
                )
            )
    return CorrelationReport(
        columns=list(NUMERICAL),
        pairs=pairs,
        selection_reason=(
            "Fixed 12 meaningful raw/history fields; identifiers, targets, "
            "evaluation metadata and calendar fields excluded. Pairwise finite "
            "observations; prefer Spearman for skew/discrete monotonic patterns."
        ),
    )


def prevalence_analysis(target, indexes):
    fraud = target.astype("string").eq("1")
    partitions = {"overall": prevalence_interval(int(fraud.sum()), len(target))}
    partitions.update(
        {
            name: prevalence_interval(int(fraud.iloc[rows].sum()), len(rows))
            for name, rows in indexes.items()
        }
    )
    return PrevalenceReport(
        partitions=partitions,
        interpretation=(
            "Synthetic-data finding: nominal 95% Wilson intervals describe a "
            "hypothetical repeated independent sample under this simulation, not "
            "banking populations. Generator fixes overall class counts; the "
            "realized full-sample proportion is exact. Entity/episode dependence "
            "invalidates an IID uncertainty claim; intervals are teaching "
            "approximations, not generator-parameter uncertainty."
        ),
    )


def temporal_analysis(features, frame, evaluation, split):
    fraud = frame.isFraud.astype("string").eq("1").to_numpy(dtype=bool)
    days = (pd.to_numeric(frame.TransactionDT).astype(float) // 86400).astype(int)
    buckets = []
    for day in range(int(days.min()), int(days.max()) + 1):
        mask = days.eq(day).to_numpy()
        counts = evaluation.loc[mask & fraud, "fraud_scenario"].value_counts()
        buckets.append(
            TemporalBucket(
                elapsed_day_index=day,
                prevalence=prevalence_interval(int((mask & fraud).sum()), int(mask.sum())),
                scenario_counts={str(k): int(v) for k, v in counts.items()},
            )
        )
    table = [
        [
            int(fraud[split.indexes[name]].sum()),
            len(split.indexes[name]) - int(fraud[split.indexes[name]].sum()),
        ]
        for name in ("train", "validation", "test")
    ]
    association = categorical_test(table).model_copy(
        update={"table_layout": "rows train,validation,test; columns fraud,legitimate"}
    )
    return TemporalReport(
        buckets=buckets,
        partition_group_summaries={
            name: {column: summarize(features.iloc[rows][column]) for column in NUMERICAL}
            for name, rows in split.indexes.items()
        },
        partition_fraud_association=association,
        interpretation=(
            "Synthetic-data finding: elapsed-day counts and changing "
            "prevalence/scenarios only; no calendar dates. Partition chi-square is "
            "supplementary exploratory, outside the six-question FDR family, and "
            "not a seventh confirmatory claim. Earlier/later periods are clustered "
            "and not randomized; aggregate holdout inspection must not drive "
            "model/threshold selection."
        ),
    )
