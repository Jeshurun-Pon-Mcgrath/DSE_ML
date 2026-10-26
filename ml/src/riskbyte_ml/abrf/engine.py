"""Deterministic outcome-free confidence-aware fusion; no component fitting."""

import bisect

import numpy as np

from riskbyte_ml.abrf.contracts import (
    A_THRESHOLD,
    B_THRESHOLD,
    P_THRESHOLD,
    FusionConfig,
    FusionInput,
    FusionTrace,
    Weights,
)


def confidence_transform(confidence, curve):
    values = np.asarray(confidence, dtype=float)
    if curve == "linear":
        return values
    if curve in {"gated_0.5", "gated_0.75"}:
        floor = 0.5 if curve == "gated_0.5" else 0.75
        return np.where(values >= floor, values, 0.0)
    raise ValueError("Unknown confidence transform")


def fusion_arrays(p, a, b, confidence, weights: Weights, curve):
    """Array interface accepts only P/A/B/confidence, never a labelled dataframe."""
    p, a, b, c = [np.asarray(x, dtype=float) for x in [p, a, b, confidence]]
    if not (p.shape == a.shape == b.shape == c.shape) or p.ndim != 1:
        raise ValueError("Aligned one-dimensional component arrays required")
    if any(not np.isfinite(x).all() or ((x < 0) | (x > 1)).any() for x in [p, a, c]):
        raise ValueError("P/A/confidence must be finite and bounded")
    available = np.isfinite(b)
    if np.isinf(b).any() or ((b[available] < 0) | (b[available] > 1)).any():
        raise ValueError("B must be bounded or unavailable (NaN)")
    if (c[~available] != 0).any():
        raise ValueError("Unavailable behaviour requires zero confidence")
    effective_b = weights.wb * np.where(available, confidence_transform(c, curve), 0)
    denominator = weights.wp + weights.wa + effective_b
    effective = np.column_stack(
        [
            weights.wp / denominator,
            weights.wa / denominator,
            effective_b / denominator,
        ]
    )
    # This zero is only an arithmetic placeholder under exactly ZERO behavioural weight.
    contributions = effective * np.column_stack([p, a, np.where(available, b, 0)])
    return np.clip(contributions.sum(axis=1), 0, 1), effective, contributions


def reason_codes(record: FusionInput, config: FusionConfig):
    reasons = []
    if record.supervised_probability >= P_THRESHOLD:
        reasons.append("SUPERVISED_THRESHOLD_EXCEEDED")
    if record.anomaly_score >= A_THRESHOLD:
        reasons.append("HIGH_ANOMALY_SCORE")
    if record.behaviour_score is not None and record.behaviour_score >= B_THRESHOLD:
        reasons.append("HIGH_BEHAVIOURAL_DEVIATION")
    if (
        not record.behaviour_available
        or record.behaviour_confidence < config.limited_confidence_cutoff
    ):
        reasons.append("LIMITED_BEHAVIOURAL_HISTORY")
    mapping = {
        "device_novelty": "NOVEL_DEVICE",
        "merchant_novelty": "NEW_MERCHANT",
        "region_novelty": "NEW_REGION",
        "product_novelty": "NEW_PRODUCT",
        "transaction_velocity": "HIGH_TRANSACTION_VELOCITY",
        "long_window_velocity": "HIGH_TRANSACTION_VELOCITY",
        "amount_deviation": "ABNORMAL_AMOUNT",
        "amount_zscore": "ABNORMAL_AMOUNT",
        "failed_attempts": "ELEVATED_FAILED_ATTEMPTS",
        "short_time_gap": "SHORT_PRIOR_TIME_GAP",
    }
    for evidence in record.evidence:
        cutoff = 1.0 if evidence.signal.endswith("novelty") else config.reason_evidence_cutoff
        if evidence.available and evidence.score >= cutoff:
            reasons.append(mapping[evidence.signal])
    return list(dict.fromkeys(reasons))


def fuse(record: FusionInput, config: FusionConfig) -> FusionTrace:
    if not isinstance(record, FusionInput) or not isinstance(config, FusionConfig):
        raise TypeError("Validated live contracts required; ground truth belongs outside fusion")
    score, weights, contributions = fusion_arrays(
        [record.supervised_probability],
        [record.anomaly_score],
        [record.behaviour_score if record.behaviour_available else np.nan],
        [record.behaviour_confidence],
        config.weights,
        config.confidence_curve,
    )
    names = ["supervised", "anomaly", "behaviour"]
    levels = ["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"]
    value = float(score[0])
    return FusionTrace(
        transaction_id=record.transaction_id,
        version=config.version,
        supervised_probability=record.supervised_probability,
        anomaly_score=record.anomaly_score,
        behaviour_score=record.behaviour_score,
        behaviour_confidence=record.behaviour_confidence,
        behaviour_available=record.behaviour_available,
        base_weights=config.weights,
        effective_weights=dict(zip(names, map(float, weights[0]), strict=True)),
        weighted_contributions=dict(zip(names, map(float, contributions[0]), strict=True)),
        final_risk_score=value,
        risk_level=levels[bisect.bisect_right(config.risk_boundaries, value)],
        decision_threshold=config.decision_threshold,
        investigation_flag=value >= config.decision_threshold,
        reason_codes=reason_codes(record, config),
    )
