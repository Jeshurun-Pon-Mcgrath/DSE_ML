"""Bounded simplex study and exact validation thresholds under identical policy rules."""

import numpy as np
from sklearn.metrics import average_precision_score

from riskbyte_ml.abrf.contracts import P_THRESHOLD, Weights
from riskbyte_ml.datasets.synthetic_riskbyte.config import digest


def candidates(config):
    n = config.grid_divisions
    for i in range(1, n + 1):
        for j in range(n - i + 1):
            k = n - i - j
            weights = Weights(wp=i / n, wa=j / n, wb=k / n)
            variant = (
                "Supervised"
                if j == k == 0
                else "Supervised + Anomaly"
                if k == 0
                else ("Supervised + Behaviour" if j == 0 else "Full ABRF")
            )
            for curve in config.confidence_curves if k else ["linear"]:
                payload = {"weights": weights.model_dump(), "curve": curve, "variant": variant}
                yield {"candidate_id": digest(payload)[7:23], **payload}


def threshold_curve(labels, scores, metadata, config):
    y, scores = np.asarray(labels, int), np.asarray(scores, float)
    hard = metadata.hard_negative.to_numpy(bool)
    order = np.argsort(-scores, kind="stable")
    thresholds = np.unique(np.r_[scores, 0.0, 1.0, np.nextafter(1.0, np.inf)])[::-1]
    positions = np.searchsorted(-scores[order], -thresholds, side="right")
    tp = np.r_[0, np.cumsum(y[order])][positions]
    fp = positions - tp
    hfp = np.r_[0, np.cumsum(hard[order])][positions]
    precision = np.divide(tp, positions, out=np.zeros(len(tp)), where=positions > 0)
    recall = tp / max(1, int(y.sum()))
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp)),
        where=precision + recall > 0,
    )
    curve = {
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": np.full(len(tp), average_precision_score(y, scores)),
        "fpr": fp / max(1, int((y == 0).sum())),
        "fnr": 1 - recall,
        "hard_negative_fpr": hfp / max(1, int(hard.sum())),
        "fraud_detected": tp,
        "false_positives": fp,
        "false_negatives": y.sum() - tp,
        "hard_negative_false_positives": hfp,
    }
    protected = np.ones(len(tp), bool)
    for scenario in config.preserve_scenarios:
        mask = (y == 1) & metadata.fraud_scenario.eq(scenario).to_numpy(bool)
        detected = np.r_[0, np.cumsum(mask[order])][positions]
        required = int((mask & metadata.P.ge(P_THRESHOLD).to_numpy(bool)).sum())
        curve[scenario + "_detected"] = detected
        protected &= detected >= required
    choices = {}
    for name, policy in config.policies.items():
        feasible = np.flatnonzero(
            (curve["fpr"] <= policy.max_fpr)
            & (curve["hard_negative_fpr"] <= policy.max_hard_negative_fpr)
            & (recall >= policy.minimum_recall)
            & protected
        )
        if not len(feasible):
            choices[name] = None
            continue

        def rank(i, policy=policy):
            primary = (
                f1[i]
                if policy.objective == "maximize_f1"
                else (recall[i] if policy.objective == "maximize_recall" else -fp[i])
            )
            return primary, precision[i], recall[i], -hfp[i], thresholds[i]

        best = int(max(feasible, key=rank))
        choices[name] = {key: float(values[best]) for key, values in curve.items()}
    return curve, choices


def rank_candidate(record, policy):
    metric = record["metrics"]
    primary = (
        metric["f1"]
        if policy.objective == "maximize_f1"
        else (
            metric["recall"]
            if policy.objective == "maximize_recall"
            else -metric["confusion"]["fp"]
        )
    )
    return (
        primary,
        metric["pr_auc"],
        metric["precision"],
        metric["recall"],
        -metric["hard_negatives"]["rate"],
        -metric["fpr"],
        record["curve"] == "linear",
        record["weights"]["wp"],
        record["candidate_id"],
    )


def eligible_constraint(record, constraint):
    w = record["weights"]
    return (
        constraint == "unconstrained"
        or (constraint == "supervised_dominant" and w["wp"] >= max(w["wa"], w["wb"]))
        or (constraint == "supervised_floor_0.4" and w["wp"] >= 0.4)
    )
