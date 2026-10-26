"""Exact validation score candidates; explicit constrained policy criteria, no test input."""

import numpy as np

from riskbyte_ml.models.supervised.metrics import display_positions, validate_predictions
from riskbyte_ml.models.supervised.schemas import ThresholdAnalysis, ThresholdChoice, ThresholdPoint

CRITERIA = {
    "BALANCED": "Maximize F1 with FPR <=1% and hard-negative FPR <=10%; "
    "ties prefer recall, precision",
    "HIGH_RECALL": "Maximize recall with FPR <=2% and hard-negative FPR <=15%; "
    "ties prefer precision",
    "LOW_FALSE_POSITIVE": "Minimize FPR with recall >=60% and hard-negative FPR <=5%; "
    "ties prefer recall",
}


def threshold_analysis(labels, probabilities, hard_negative):
    y, p = validate_predictions(labels, probabilities)
    hard = np.asarray(hard_negative, dtype=bool)
    if hard.shape != y.shape or (hard & (y == 1)).any():
        raise ValueError("Aligned legitimate hard-negative mask required")
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    thresholds = np.unique(np.r_[p, 0.0, 0.5, 1.0, np.nextafter(1.0, np.inf)])[::-1]
    positions = np.searchsorted(-sorted_p, -thresholds, side="right")
    tp = np.r_[0, np.cumsum(y[order])][positions].astype(float)
    fp = positions - tp
    hfp = np.r_[0, np.cumsum(hard[order])][positions].astype(float)
    positives, negatives = int(y.sum()), int((y == 0).sum())
    precision = np.divide(tp, positions, out=np.zeros(len(tp)), where=positions != 0)
    recall = tp / positives if positives else np.zeros(len(tp))
    fpr = fp / negatives if negatives else np.zeros(len(tp))
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp)),
        where=precision + recall != 0,
    )
    hfpr = hfp / hard.sum() if hard.any() else np.zeros(len(tp))

    def point(i):
        return ThresholdPoint(
            threshold=float(thresholds[i]),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            fpr=float(fpr[i]),
            fnr=float(1 - recall[i]) if positives else 0.0,
            predicted_positive_count=int(positions[i]),
            hard_negative_fpr=float(hfpr[i]) if hard.any() else None,
        )

    policies, selected = {}, []
    for policy in CRITERIA:
        feasible = (
            ((fpr <= 0.01) & (hfpr <= 0.1))
            if policy == "BALANCED"
            else ((fpr <= 0.02) & (hfpr <= 0.15))
            if policy == "HIGH_RECALL"
            else ((recall >= 0.6) & (hfpr <= 0.05))
        )
        candidates = np.flatnonzero(feasible)
        met = bool(len(candidates))
        if not met:
            candidates = np.array([0])  # Explicit all-negative fallback, never claimed feasible.

        def rank(i, policy=policy):
            if policy == "BALANCED":
                return f1[i], recall[i], precision[i], thresholds[i]
            if policy == "HIGH_RECALL":
                return recall[i], precision[i], f1[i], thresholds[i]
            return -fpr[i], recall[i], precision[i], thresholds[i]

        chosen = int(max(candidates, key=rank))
        selected.append(chosen)
        policies[policy] = ThresholdChoice(
            policy=policy, chosen=point(chosen), constraints_met=met, criterion=CRITERIA[policy]
        )
    shown = np.unique(np.r_[display_positions(len(thresholds)), selected])
    return ThresholdAnalysis(
        candidate_count=len(thresholds), points=[point(int(i)) for i in shown], policies=policies
    )
