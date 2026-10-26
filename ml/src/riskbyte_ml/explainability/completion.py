"""Complete SHAP research without reselecting cases or changing non-SHAP evidence."""

import copy
import platform
from collections import Counter

import numpy as np
import scipy
import sklearn

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability.schemas import VERSION, ExplanationBundle
from riskbyte_ml.explainability.supervised import measure_latency
from riskbyte_ml.statistics.verification import file_hash, read_json

BASELINE_HASH = "sha256:da06df18ab0e1bfe305993bca40634658e21fe12ac0bf359458ac48ee63be775"
CASE_IDS = {
    "account_takeover_true_positive": "2845",
    "recovered_low_and_slow": "69117",
    "recovered_international_anomaly": "58423",
    "shopping_festival_false_positive": "14302",
    "normal_legitimate_low_risk": "86722",
    "remaining_fraud_false_negative": "10604",
}


def baseline_reports(directory):
    from riskbyte_ml.explainability.pipeline import verify_outputs

    if file_hash(directory / "analysis_bundle.json") != BASELINE_HASH:
        raise DatasetError("explanation_baseline_changed", "Original Phase 7 snapshot differs")
    bundle = ExplanationBundle.model_validate_json(
        (directory / "analysis_bundle.json").read_bytes()
    )
    if bundle.version != VERSION:
        raise DatasetError("explanation_baseline_changed", "Historical explainability-v1 required")
    verify_outputs(directory, bundle)
    reports = {
        name: read_json(directory / name)
        for name in [
            "case_studies.json",
            "hard_negative_explanations.json",
            "false_negative_explanations.json",
        ]
    }
    if {
        c["case"]: c["explanation"]["transaction_id"] for c in reports["case_studies.json"]["cases"]
    } != CASE_IDS:
        raise DatasetError("explanation_case_ids_changed", "Frozen validation cases differ")
    return reports


def non_shap_evidence(record):
    value = copy.deepcopy(record)
    explanation = value["explanation"]
    explanation.pop("version", None)
    for name in [
        "shap_status",
        "unavailable_reason",
        "base_value",
        "additivity_residual",
        "top_positive_features",
        "top_negative_features",
        "raw_tree_probability",
        "calibrated_supervised_probability",
    ]:
        explanation["supervised"].pop(name, None)
    return value


def verify_preserved(old_reports, studies, hard_report, fn_report):
    groups = [
        (old_reports["case_studies.json"]["cases"], studies),
        (old_reports["hard_negative_explanations.json"]["records"], hard_report["records"]),
        (old_reports["false_negative_explanations.json"]["records"], fn_report["records"]),
    ]
    for previous, current in groups:
        if len(previous) != len(current) or any(
            non_shap_evidence(old) != non_shap_evidence(new)
            for old, new in zip(previous, current, strict=True)
        ):
            raise DatasetError("explanation_evidence_changed", "Non-SHAP evidence changed")
    if len(hard_report["records"]) != 11 or len(fn_report["records"]) != 98:
        raise DatasetError(
            "explanation_error_population_changed", "Frozen error populations changed"
        )
    for name, report in [
        ("hard_negative_explanations.json", hard_report),
        ("false_negative_explanations.json", fn_report),
    ]:
        if old_reports[name]["summary"] != report["summary"]:
            raise DatasetError("explanation_evidence_changed", "Historical error summary changed")


def shap_summary(records):
    """Top-K recurrence is explicitly distinct from full global mean absolute SHAP."""
    groups = {"ALL": records}
    for record in records:
        name = (
            record["evaluation_only"]["fraud_scenario"]
            or (record["evaluation_only"]["hard_negative_type"])
        )
        groups.setdefault(name, []).append(record)
    result = {}
    for name, group in groups.items():
        row = {"records": len(group)}
        for direction in ["positive", "negative"]:
            counts, sums = Counter(), Counter()
            for record in group:
                features = record["explanation"]["supervised"]["top_" + direction + "_features"]
                for feature in features:
                    counts[feature["feature"]] += 1
                    sums[feature["feature"]] += abs(feature["shap_value"])
            row[direction] = [
                {
                    "feature": feature,
                    "top_k_occurrences": count,
                    "mean_absolute_shap_when_in_top_k": sums[feature] / count,
                }
                for feature, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:10]
            ]
        result[name] = row
    return {
        "scope": "Recurrence among recorded top-K factors; not full global importance",
        "groups": result,
    }


def benchmarks(inputs):
    return {
        "scope": "Cached TreeExplainer.values only; includes bounded densification, "
        "raw-tree prediction and independent additivity check; excludes other evidence",
        "explainer_initialization_seconds": inputs.shap.initialization_seconds,
        "cached_single": measure_latency(lambda: inputs.shap.values(inputs.matrix[:1]), repeats=5),
        "batch_10": measure_latency(lambda: inputs.shap.values(inputs.matrix[:10]), repeats=5),
        "batch_100": measure_latency(lambda: inputs.shap.values(inputs.matrix[:100]), repeats=5),
    }


def environment():
    import shap

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
    }


def additivity_examples(inputs, studies):
    examples = []
    for study in studies:
        transaction = study["explanation"]["transaction_id"]
        index = int(np.flatnonzero(inputs.scores.TransactionID.eq(transaction).to_numpy())[0])
        values, base, residual = inputs.shap.values(inputs.matrix[index : index + 1])
        examples.append(
            {
                "transaction_id": transaction,
                "expected_value": base,
                "sum_shap": float(values.sum()),
                "raw_tree_probability": float(inputs.raw_probability[index]),
                "calibrated_P_not_explained": float(inputs.scores.P.iloc[index]),
                "residual": float(residual[0]),
                "absolute_tolerance": 1e-7,
            }
        )
    return examples


def statistical_consistency(global_report, families, hard_report):
    ranks = {r["feature"]: r["rank"] for r in global_report["importance"]}
    family_ranks = {r["family"]: i + 1 for i, r in enumerate(families["importance"])}

    def rank(name):
        return ranks.get(name, len(ranks) + 1)

    velocity = (
        "ALIGNED"
        if family_ranks.get("velocity", 999) <= 3
        else ("PARTIALLY_ALIGNED" if family_ranks.get("relative_time", 999) <= 3 else "DIFFERENT")
    )
    device = (
        "ALIGNED"
        if rank("device_seen_before") <= 10
        else ("PARTIALLY_ALIGNED" if family_ranks.get("device", 999) <= 5 else "DIFFERENT")
    )
    return {
        "status": "AVAILABLE",
        "finding_label": "Synthetic-data finding",
        "scope": "Qualitative post-hoc comparison, not a new hypothesis test",
        "rubric": "Top-ten specific features / top-three temporal families / top-five device "
        "family are descriptive comparison cutoffs, not significance thresholds",
        "findings": [
            {
                "signal": "velocity",
                "relationship": velocity,
                "feature_rank": rank("transactions_prior_30m"),
                "explanation": "Compare direct rolling-window contribution with prior-event gap; "
                "a strong time-gap family is only partial alignment with velocity.",
            },
            {
                "signal": "failed_attempts",
                "relationship": "ALIGNED" if rank("failed_attempts_before") <= 10 else "DIFFERENT",
                "feature_rank": rank("failed_attempts_before"),
                "explanation": "Statistical association versus conditional fitted-tree reliance.",
            },
            {
                "signal": "device_novelty",
                "relationship": device,
                "feature_rank": rank("device_seen_before"),
                "explanation": "Device importance can include encoded context, not just novelty.",
            },
            {
                "signal": "merchant_novelty",
                "relationship": "ALIGNED" if rank("merchant_seen_before") > 10 else "DIFFERENT",
                "feature_rank": rank("merchant_seen_before"),
                "explanation": "Weak marginal association may align with low novelty importance; "
                "other merchant-family context is a separate signal.",
            },
            {
                "signal": "hard_negative_amount_overlap",
                "relationship": "PARTIALLY_ALIGNED",
                "hard_negative_ratio_in_top_positive": sum(
                    any(
                        f["feature"] == "amount_ratio_to_prior_mean"
                        for f in r["explanation"]["supervised"]["top_positive_features"]
                    )
                    for r in hard_report["records"]
                ),
                "explanation": "Attributions in selected errors describe classifier behaviour; "
                "they cannot establish group separability or reproduce marginal effects.",
            },
        ],
        "limitation": "Associations are marginal, SHAP is model-dependent; correlated features, "
        "class weighting, interactions and TRAIN/VALIDATION differences matter.",
    }
