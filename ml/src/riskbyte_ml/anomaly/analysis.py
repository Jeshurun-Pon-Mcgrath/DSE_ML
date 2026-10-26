"""Post-fit validation analysis only; OR coverage is not deployed fusion logic."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from riskbyte_ml.datasets.synthetic_riskbyte.config import HARD_NEGATIVES, SCENARIOS
from riskbyte_ml.statistics.methods import cliffs_delta


def distribution(values):
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    return {
        "rows": len(array),
        "available": len(valid),
        "unavailable": len(array) - len(valid),
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(np.median(valid)) if len(valid) else None,
        "quantiles": {
            str(q): float(np.quantile(valid, q)) if len(valid) else None
            for q in [0.05, 0.25, 0.75, 0.95, 0.99]
        },
    }


def rates(mask, flags):
    count = int(np.count_nonzero(mask))
    flagged = int(np.count_nonzero(flags & mask))
    return {"count": count, "flagged": flagged, "rate": flagged / count if count else None}


def evaluate(labels, scores, threshold, metadata):
    y, values = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    available = np.isfinite(values)
    if (
        not set(np.unique(y)) <= {0, 1}
        or (values[available] < 0).any()
        or (values[available] > 1).any()
    ):
        raise ValueError("Binary labels and bounded deviation scores required")
    flags = available & (values >= threshold)
    hard = metadata.hard_negative.to_numpy(dtype=bool)
    tp, fp = int((flags & (y == 1)).sum()), int((flags & (y == 0)).sum())
    fn, tn = int(((~flags) & (y == 1)).sum()), int(((~flags) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(y[available])) == 2
    return {
        "rows": len(y),
        "score_available": int(available.sum()),
        "unavailable_fraud": int(((~available) & (y == 1)).sum()),
        "unavailable_policy": "no deviation flag; abstention is not a low-risk classification",
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "pr_auc": float(average_precision_score(y[available], values[available])) if both else None,
        "roc_auc": float(roc_auc_score(y[available], values[available])) if both else None,
        "auc_population": "finite scores only; flag rates retain every validation observation",
        "fpr": fp / int((y == 0).sum()) if (y == 0).any() else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "hard_negatives": rates(hard & (y == 0), flags),
        "hard_negative_types": {
            name: rates(hard & metadata.hard_negative_type.eq(name).to_numpy(dtype=bool), flags)
            for name in HARD_NEGATIVES
        },
        "scenario_recall": {
            name: rates((y == 1) & metadata.fraud_scenario.eq(name).to_numpy(dtype=bool), flags)
            for name in SCENARIOS
        },
        "score_semantics": "anomaly/deviation score; NOT fraud probability",
    }


def select_threshold(labels, scores, hard, config):
    """Maximize recall within explicit validation FPR caps; ties favor precision."""
    y, values, hard = np.asarray(labels), np.asarray(scores, float), np.asarray(hard, bool)
    if (
        hard.shape != y.shape
        or values.shape != y.shape
        or not len(y)
        or not set(np.unique(y)) <= {0, 1}
        or (values[np.isfinite(values)] < 0).any()
        or (values[np.isfinite(values)] > 1).any()
        or (hard & (y == 1)).any()
    ):
        raise ValueError("Aligned legitimate hard-negative mask required")
    valid = np.isfinite(values)
    order = np.flatnonzero(valid)[np.argsort(-values[valid], kind="stable")]
    candidates = np.unique(np.r_[values[valid], np.nextafter(1.0, np.inf)])[::-1]
    positions = np.searchsorted(-values[order], -candidates, side="right")
    tp = np.r_[0, np.cumsum(y[order])][positions]
    fp = positions - tp
    hfp = np.r_[0, np.cumsum(hard[order])][positions]
    fpr = fp / max(1, int((y == 0).sum()))
    hfpr = hfp / max(1, int(hard.sum()))
    feasible = np.flatnonzero(
        (fpr <= config.max_legitimate_rate) & (hfpr <= config.max_hard_negative_rate)
    )
    precision = np.divide(tp, positions, out=np.zeros(len(tp)), where=positions > 0)
    best = int(max(feasible, key=lambda i: (tp[i], precision[i], -fp[i], candidates[i])))
    points = np.unique(
        np.r_[np.linspace(0, len(candidates) - 1, min(201, len(candidates)), dtype=int), best]
    )
    return {
        "threshold": float(candidates[best]),
        "policy": "VALIDATION_RECALL_WITH_BUDGETS",
        "criterion": "maximize fraud recall; ties prefer precision, fewer FPs, higher threshold",
        "max_legitimate_rate": config.max_legitimate_rate,
        "max_hard_negative_rate": config.max_hard_negative_rate,
        "source": "validation only",
        "candidate_count": len(candidates),
        "points": [
            {
                "threshold": float(candidates[i]),
                "fraud_detected": int(tp[i]),
                "legitimate_rate": float(fpr[i]),
                "hard_negative_rate": float(hfpr[i]),
                "precision": float(precision[i]),
            }
            for i in points
        ],
    }


def complementarity(labels, supervised_flags, anomaly_flags):
    fraud = np.asarray(labels) == 1
    p, a = np.asarray(supervised_flags, bool), np.asarray(anomaly_flags, bool)
    return {
        "supervised_only": int((fraud & p & ~a).sum()),
        "anomaly_only": int((fraud & ~p & a).sum()),
        "both": int((fraud & p & a).sum()),
        "neither": int((fraud & ~p & ~a).sum()),
        "supervised_detected": int((fraud & p).sum()),
        "analytical_or_detected": int((fraud & (p | a)).sum()),
        "additional_fraud_recovered": int((fraud & ~p & a).sum()),
        "additional_false_positives": int((~fraud & ~p & a).sum()),
        "purpose": "analysis only; no operational OR rule or P/A/B fusion",
    }


def group_analysis(labels, metadata, anomaly, behaviour, confidence, at, bt):
    y, hard = np.asarray(labels), metadata.hard_negative.to_numpy(dtype=bool)
    masks = {
        "normal_legitimate": (y == 0) & ~hard,
        "hard_negative": (y == 0) & hard,
        "fraud": y == 1,
    }
    groups = {
        name: {
            "anomaly_score": distribution(anomaly[mask]),
            "behaviour_score": distribution(behaviour[mask]),
            "behaviour_confidence": distribution(confidence[mask]),
            "anomaly_flag_rate": rates(mask, anomaly >= at),
            "behaviour_flag_rate": rates(mask, np.isfinite(behaviour) & (behaviour >= bt)),
        }
        for name, mask in masks.items()
    }
    scenarios = {
        name: {
            "anomaly_score": distribution(anomaly[metadata.fraud_scenario.eq(name).to_numpy(bool)]),
            "behaviour_score": distribution(
                behaviour[metadata.fraud_scenario.eq(name).to_numpy(bool)]
            ),
        }
        for name in SCENARIOS
    }
    return {"groups": groups, "scenarios": scenarios}


def investigations(features, metadata, labels, anomaly, behaviour):
    y = np.asarray(labels)
    normal = (y == 0) & ~metadata.hard_negative.to_numpy(bool)
    columns = [
        "transactions_prior_5m",
        "transactions_prior_30m",
        "transactions_prior_1h",
        "transactions_prior_24h",
        "TransactionAmt",
        "amount_ratio_to_prior_mean",
        "time_since_previous_transaction",
        "is_international",
        "billing_region_seen_before",
        "device_seen_before",
        "merchant_seen_before",
    ]
    result = {}
    for name in ["VELOCITY_BURST", "LOW_AND_SLOW", "INTERNATIONAL_ANOMALY"]:
        scenario = metadata.fraud_scenario.eq(name).to_numpy(bool) & (y == 1)
        signals = {column: np.asarray(features[column], float) for column in columns}
        signals.update(anomaly_score=anomaly, behaviour_score=behaviour)
        result[name] = {}
        for column, values in signals.items():
            first, second = values[scenario], values[normal]
            first, second = first[np.isfinite(first)], second[np.isfinite(second)]
            result[name][column] = {
                "scenario": distribution(first),
                "normal_legitimate": distribution(second),
                "cliffs_delta_vs_normal": float(cliffs_delta(first, second))
                if len(first) and len(second)
                else None,
            }
    return result
