"""Correct imbalanced metrics, bounded curves, and evaluation-only subgroup summaries."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from riskbyte_ml.datasets.synthetic_riskbyte.config import HARD_NEGATIVES, SCENARIOS
from riskbyte_ml.models.supervised.schemas import (
    Confusion,
    CurvePoint,
    Curves,
    Evaluation,
    Metrics,
    ReliabilityBin,
    Subgroup,
)


def validate_predictions(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty, aligned one-dimensional labels and probabilities required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Binary labels and finite probabilities in [0,1] required")
    return y.astype(int), p


def metrics(labels, probabilities, threshold=0.5, *, probability=True):
    y, p = validate_predictions(labels, probabilities)
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1.000001:
        raise ValueError("Invalid operating threshold")
    predicted = p >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(y)) == 2
    return Metrics(
        rows=len(y),
        fraud_count=int(y.sum()),
        prevalence=float(y.mean()),
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        pr_auc=float(average_precision_score(y, p)) if both else None,
        roc_auc=float(roc_auc_score(y, p)) if both else None,
        brier_score=float(brier_score_loss(y, p)) if probability else None,
        accuracy=(tp + tn) / len(y),
        specificity=tn / (tn + fp) if tn + fp else None,
        fpr=fp / (tn + fp) if tn + fp else None,
        fnr=fn / (tp + fn) if tp + fn else None,
        confusion=Confusion(tp=tp, fp=fp, tn=tn, fn=fn),
        probability_note="P(fraud); empirical calibration requires review"
        if probability
        else "Hand-written rule score, not a calibrated fraud probability; Brier unavailable",
    )


def subgroup(mask, predicted):
    count = int(mask.sum())
    flagged = int(predicted[mask].sum())
    return Subgroup(count=count, flagged=flagged, rate=flagged / count if count else None)


def evaluate(labels, probabilities, threshold, metadata, seconds=None, *, probability=True):
    y, p = validate_predictions(labels, probabilities)
    if len(metadata) != len(y):
        raise ValueError("Evaluation-only metadata must align with predictions")
    hard = metadata.hard_negative.to_numpy(dtype=bool)
    if (hard & (y == 1)).any():
        raise ValueError("Hard negatives must be legitimate")
    predicted = p >= threshold
    return Evaluation(
        metrics=metrics(y, p, threshold, probability=probability),
        hard_negatives=subgroup(hard & (y == 0), predicted),
        hard_negative_types={
            name: subgroup(
                hard & metadata.hard_negative_type.eq(name).to_numpy(dtype=bool), predicted
            )
            for name in HARD_NEGATIVES
        },
        scenario_recall={
            name: subgroup(
                (y == 1) & metadata.fraud_scenario.eq(name).to_numpy(dtype=bool), predicted
            )
            for name in SCENARIOS
        },
        prediction_seconds=float(seconds) if seconds is not None else None,
        seconds_per_transaction=float(seconds / len(y)) if seconds is not None else None,
    )


def display_positions(size, limit=201):
    return np.unique(np.linspace(0, size - 1, min(size, limit), dtype=int)) if size else []


def curve_data(labels, probabilities):
    y, p = validate_predictions(labels, probabilities)
    pr, roc, counts = [], [], {"pr": 0, "roc": 0}
    if len(np.unique(y)) == 2:
        precision, recall, thresholds = precision_recall_curve(y, p)
        counts["pr"] = len(precision)
        pr = [
            CurvePoint(
                x=float(recall[i]),
                y=float(precision[i]),
                threshold=float(thresholds[i]) if i < len(thresholds) else None,
            )
            for i in display_positions(len(precision))
        ]
        fpr, tpr, thresholds = roc_curve(y, p)
        counts["roc"] = len(fpr)
        roc = [
            CurvePoint(
                x=float(fpr[i]),
                y=float(tpr[i]),
                threshold=float(thresholds[i]) if np.isfinite(thresholds[i]) else None,
            )
            for i in display_positions(len(fpr))
        ]
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (p >= low) & ((p < high) if index < 9 else (p <= high))
        bins.append(
            ReliabilityBin(
                lower=low,
                upper=high,
                count=int(mask.sum()),
                mean_probability=float(p[mask].mean()) if mask.any() else None,
                observed_prevalence=float(y[mask].mean()) if mask.any() else None,
            )
        )
    return Curves(pr=pr, roc=roc, reliability=bins, source_points=counts)
