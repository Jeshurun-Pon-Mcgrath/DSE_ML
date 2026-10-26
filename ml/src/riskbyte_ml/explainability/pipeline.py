"""Offline explanation artifacts; TRAIN path references and VALIDATION evidence only."""

import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from riskbyte_ml.anomaly.analysis import distribution
from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability.engine import ExplanationEngine
from riskbyte_ml.explainability.inputs import ExplanationInputs, verify_selection
from riskbyte_ml.explainability.schemas import (
    NON_CAUSAL,
    NOTICE,
    SHAP_VERSION,
    VERSION,
    ExplanationBundle,
)
from riskbyte_ml.explainability.supervised import measure_latency
from riskbyte_ml.preprocessing.pipeline import matrix_summary
from riskbyte_ml.statistics.verification import file_hash


def save_json(path, value):
    write_artifact(path, json.dumps(value, allow_nan=False, indent=2).encode(), overwrite=False)


def verify_outputs(directory, bundle):
    for name, expected in bundle.artifacts.items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise DatasetError(
                "explanation_artifact_changed", "Explanation snapshot integrity failed"
            )
    if bundle.version == SHAP_VERSION:
        from riskbyte_ml.explainability.freeze import verify_freeze

        verify_freeze(directory, bundle)


def case_selection(labels, metadata, scores):
    """Earliest eligible validation event; report absence without invented replacements."""
    y = np.asarray(labels)
    flag = scores.investigation_flag.to_numpy(bool)
    pflag = scores.supervised_prediction.to_numpy(bool)
    scenario = metadata.fraud_scenario
    hard = metadata.hard_negative.to_numpy(bool)
    types = metadata.hard_negative_type
    masks = {
        "account_takeover_true_positive": (y == 1) & flag & scenario.eq("ACCOUNT_TAKEOVER"),
        "recovered_low_and_slow": (y == 1) & flag & ~pflag & scenario.eq("LOW_AND_SLOW"),
        "recovered_international_anomaly": (y == 1)
        & flag
        & ~pflag
        & scenario.eq("INTERNATIONAL_ANOMALY"),
        "shopping_festival_false_positive": (y == 0) & flag & hard & types.eq("SHOPPING_FESTIVAL"),
        "normal_legitimate_low_risk": (y == 0) & ~hard & ~flag,
        "remaining_fraud_false_negative": (y == 1) & ~flag,
    }
    selected = {
        name: int(np.flatnonzero(np.asarray(mask, bool))[0]) if np.any(mask) else None
        for name, mask in masks.items()
    }
    eligible = np.flatnonzero(np.asarray(masks["normal_legitimate_low_risk"], bool))
    if len(eligible):
        selected["normal_legitimate_low_risk"] = int(
            eligible[np.argmin(scores.final_risk_score.to_numpy(float)[eligible])]
        )
    return selected


def evaluation_record(inputs, engine, position):
    explanation = engine.explain(position)
    meta = inputs.metadata.iloc[position]
    context = inputs.context.iloc[position]
    return {
        "finding_label": "Synthetic-data finding",
        "transaction_summary": {
            "transaction_amount": float(context.TransactionAmt),
            "elapsed_seconds": float(context.TransactionDT),
            "identity_context_available": (
                explanation.behaviour.confidence_explanation.current_identity_available
            ),
        },
        "evaluation_only": {
            "isFraud": int(inputs.labels[position]),
            "fraud_scenario": str(meta.fraud_scenario),
            "hard_negative": bool(meta.hard_negative),
            "hard_negative_type": str(meta.hard_negative_type),
        },
        "explanation": explanation.model_dump(mode="json"),
        "interpretation": (
            "Investigation flag on a synthetic transaction; review the ground truth separately."
            if explanation.fusion.investigation_flag
            else "No investigation flag; this is not proof of legitimacy."
        ),
    }


def error_summary(inputs, indexes):
    indexes = np.asarray(indexes, dtype=int)
    scores = inputs.scores.iloc[indexes]
    traced = inputs.fusion_scores.iloc[indexes]
    context = inputs.context.iloc[indexes]
    return {
        "count": len(indexes),
        "scenario_counts": dict(Counter(map(str, inputs.metadata.iloc[indexes].fraud_scenario))),
        "hard_negative_type_counts": dict(
            Counter(map(str, inputs.metadata.iloc[indexes].hard_negative_type))
        ),
        "P": distribution(scores.P),
        "A": distribution(scores.A),
        "B": distribution(scores.B),
        "confidence": distribution(scores.behaviour_confidence),
        "identity_unavailable_count": int(
            (context.device_id.isna() | context.device_id.eq("").fillna(False)).sum()
        ),
        "below_behaviour_confidence_gate": int(scores.behaviour_confidence.lt(0.5).sum()),
        "supervised_contribution": distribution(traced.contribution_supervised),
        "behaviour_contribution": distribution(traced.contribution_behaviour),
        "anomaly_contribution": distribution(traced.contribution_anomaly),
        "component_subtraction_diagnostic": {
            "supervised_term_necessary_for_threshold": int(
                (
                    traced.investigation_flag
                    & (
                        traced.final_risk_score - traced.contribution_supervised
                        < inputs.bundle.selected.decision_threshold
                    )
                ).sum()
            ),
            "behaviour_term_necessary_for_threshold": int(
                (
                    traced.investigation_flag
                    & (
                        traced.final_risk_score - traced.contribution_behaviour
                        < inputs.bundle.selected.decision_threshold
                    )
                ).sum()
            ),
            "scope": "Arithmetic subtraction at fixed effective weights, not causal attribution",
        },
    }


def global_importance(inputs, batch_size=256, output=None):
    if inputs.shap.status != "AVAILABLE":
        unavailable = {
            "status": inputs.shap.status,
            "reason": inputs.shap.reason,
            "finding_label": "Synthetic-data finding",
            "importance": [],
            "rows_explained": 0,
            "scope": "Underlying Decision Tree only; calibrated P is not SHAP-explained",
        }
        return unavailable, {**unavailable, "aggregation": "sum of mean absolute encoded SHAP"}
    total = np.zeros(len(inputs.names))
    complete = []
    base = None
    maximum_residual = 0.0
    for start in range(0, len(inputs.features), batch_size):
        values, base, residual = inputs.shap.values(inputs.matrix[start : start + batch_size])
        maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
        total += np.abs(values).sum(axis=0)
        if output is not None:
            complete.append(values)
    if output is not None:
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            shap_values=np.vstack(complete),
            base_value=np.array([base]),
            feature_names=np.array(inputs.names),
            transaction_ids=inputs.scores.TransactionID.astype(str).to_numpy(dtype=str),
        )
        write_artifact(output / "validation_shap_values.npz", stream.getvalue(), overwrite=False)
    average = total / len(inputs.features)
    rows = [
        {**metadata, "mean_absolute_shap": float(average[i])}
        for i, metadata in enumerate(inputs.feature_metadata)
    ]
    rows.sort(key=lambda r: (-r["mean_absolute_shap"], r["feature"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    families = {}
    for row in rows:
        key = row["feature_family"]
        families[key] = families.get(key, 0) + row["mean_absolute_shap"]
    grouped = [
        {"family": name, "summed_mean_absolute_shap": float(value)}
        for name, value in sorted(families.items(), key=lambda p: (-p[1], p[0]))
    ]
    for i, row in enumerate(grouped):
        row["rank"] = i + 1
    return (
        {
            "status": "AVAILABLE",
            "finding_label": "Synthetic-data finding",
            "importance": rows,
            "rows_explained": len(inputs.features),
            "population": "VALIDATION only",
            "scope": "Underlying Decision Tree only; calibrated P is not SHAP-explained",
            "expected_value": base,
            "maximum_additivity_residual": maximum_residual,
            "additivity_absolute_tolerance": 1e-7,
        },
        {
            "status": "AVAILABLE",
            "finding_label": "Synthetic-data finding",
            "importance": grouped,
            "aggregation": "sum of mean absolute encoded SHAP; not grouped signed SHAP",
            "limitation": "Larger one-hot families can accumulate more absolute importance",
        },
    )


def run(config, root: Path, output=None, top_k=5, snapshot_version=VERSION, baseline_dir=None):
    output = Path(output) if output else root / "explainability"
    bundle_path = output / "analysis_bundle.json"
    _, _, provenance = verify_selection(config, root)
    if bundle_path.exists():
        bundle = ExplanationBundle.model_validate_json(bundle_path.read_bytes())
        verify_outputs(output, bundle)
        if any(bundle.provenance.get(k) != v for k, v in provenance.items()):
            raise DatasetError("explanation_provenance_changed", "Explanation provenance differs")
        if bundle.version != snapshot_version:
            raise DatasetError(
                "explanation_snapshot_version_changed", "Use a distinct snapshot directory"
            )
        return bundle
    historical = None
    if snapshot_version == SHAP_VERSION:
        from riskbyte_ml.explainability.completion import baseline_reports

        historical = baseline_reports(
            Path(baseline_dir) if baseline_dir else root / "explainability"
        )
    start = perf_counter()
    inputs = ExplanationInputs(config, root)
    if snapshot_version == SHAP_VERSION and inputs.shap.status != "AVAILABLE":
        raise DatasetError(
            "shap_dependency_unavailable", "Genuine SHAP is required; generation stopped"
        )
    initialization = perf_counter() - start
    engine = ExplanationEngine(inputs, top_k=top_k, snapshot_version=snapshot_version)
    start = perf_counter()
    global_report, families = global_importance(inputs, output=output)
    global_seconds = perf_counter() - start
    if historical is not None:
        selected = {}
        for case in historical["case_studies.json"]["cases"]:
            matches = np.flatnonzero(
                inputs.scores.TransactionID.eq(case["explanation"]["transaction_id"]).to_numpy()
            )
            if len(matches) != 1:
                raise DatasetError("explanation_case_missing", "Frozen case is not in validation")
            selected[case["case"]] = int(matches[0])
    else:
        selected = case_selection(inputs.labels, inputs.metadata, inputs.fusion_scores)
    studies = []
    for name, position in selected.items():
        studies.append(
            {"case": name, "status": "AVAILABLE", **evaluation_record(inputs, engine, position)}
            if position is not None
            else {
                "case": name,
                "status": "UNAVAILABLE_WITH_REASON",
                "reason": "No eligible validation transaction",
            }
        )
    flag = inputs.fusion_scores.investigation_flag.to_numpy(bool)
    hard = inputs.metadata.hard_negative.to_numpy(bool)
    hn_indexes = np.flatnonzero(flag & hard & (inputs.labels == 0))
    fn_indexes = np.flatnonzero(~flag & (inputs.labels == 1))
    hard_report = {
        "finding_label": "Synthetic-data finding",
        "summary": error_summary(inputs, hn_indexes),
        "records": [evaluation_record(inputs, engine, int(i)) for i in hn_indexes],
    }
    fn_report = {
        "finding_label": "Synthetic-data finding",
        "summary": error_summary(inputs, fn_indexes),
        "records": [evaluation_record(inputs, engine, int(i)) for i in fn_indexes],
    }
    completion_metadata = {}
    if historical is not None:
        from riskbyte_ml.explainability.completion import (
            BASELINE_HASH,
            additivity_examples,
            benchmarks,
            environment,
            shap_summary,
            statistical_consistency,
            verify_preserved,
        )

        verify_preserved(historical, studies, hard_report, fn_report)
        hard_report["shap_summary"] = shap_summary(hard_report["records"])
        fn_report["shap_summary"] = shap_summary(fn_report["records"])
        completion_metadata = {
            "environment": environment(),
            "shap_benchmarks": benchmarks(inputs),
            "additivity_examples": additivity_examples(inputs, studies),
            "phase3_consistency": statistical_consistency(global_report, families, hard_report),
            "historical_snapshot_hash": BASELINE_HASH,
            "non_shap_evidence_preserved": True,
            "feature_count": len(inputs.names),
            "observed_shap_output_layout": inputs.shap.observed_layout,
            "model_output_semantics": "Positive-class raw tree predict_proba; not calibrated P",
            "expected_value_semantics": "Class-weighted TRAIN tree-path reference expectation",
        }
    latency = measure_latency(lambda: engine.explain(0))
    batch_count = min(100, len(inputs.features))
    batch = measure_latency(lambda: [engine.explain(i) for i in range(batch_count)], repeats=1)
    performance = {
        "initialization_seconds": initialization,
        "shap_explainer_initialization_seconds": inputs.shap.initialization_seconds,
        "single_explanation": latency,
        "batch_explanation": {**batch, "rows": batch_count},
        "latency_scope": "Explanation materialization from verified cached validation scores; "
        "includes local SHAP if available, local behaviour and exact fusion; not live prediction",
        "global_explanation_seconds": global_seconds,
        "validation_matrix": matrix_summary(inputs.matrix),
        "maximum_shap_dense_input_batch_bytes": 256 * len(inputs.names) * 8,
        "memory_scope": "Measured matrix allocations; process peak RSS is not measured",
    }
    metadata = {
        "version": snapshot_version,
        "explainability_version": snapshot_version,
        "created_timestamp": datetime.now(UTC).isoformat(),
        "finding_label": "Synthetic-data finding",
        "notice": NOTICE,
        "provenance": inputs.provenance,
        "shap_status": inputs.shap.status,
        "unavailable_reason": inputs.shap.reason,
        "shap_library_version": inputs.shap.library_version,
        "explainer_type": "TreeExplainer/tree_path_dependent/raw"
        if inputs.shap.explainer
        else None,
        "explained_model": "underlying_decision_tree",
        "calibrated_probability_shap_explained": False,
        "reference": {
            "method": "Frozen tree training-path counts; no explicit background sample required",
            "sample_size": 0,
            "seed": None,
            "source": "TRAIN tree fitting only",
            "dataset_fingerprint": inputs.provenance["dataset_fingerprint"],
        },
        "global_aggregation_population": "All 15000 validation rows if SHAP available",
        "feature_metadata_version": "frozen feature manifest plus explainability-v1 family mapping",
        "top_k": top_k,
        "anomaly_effective_weight": 0.0,
        "case_selection": "Earliest eligible event per prespecified validation category; "
        "normal legitimate case minimizes risk with chronological tie-breaking",
        "phase3_consistency": {
            "status": inputs.shap.status,
            "finding": "No SHAP/statistical consistency claim can be made until SHAP runs"
            if inputs.shap.explainer is None
            else "Review rankings against Phase 3; association is not attribution",
            "statistical_reference": "Phase 3: velocity/failed attempts/device novelty; "
            "weak merchant novelty and hard-negative amount overlap",
        },
        "phase6_consistency": "Behaviour added two validation fraud detections beyond the "
        "fair threshold-optimized P-only control; do not attribute all recall lift to behaviour.",
        "limitations": [
            NOTICE,
            NON_CAUSAL,
            "Synthetic identities are not real bank customers.",
            "Anomaly observable signals are not Isolation Forest feature attributions.",
            "Calibration is separate from the tree SHAP decomposition.",
        ],
        "performance": performance,
        **completion_metadata,
    }
    artifacts = {
        "global_feature_importance.json": global_report,
        "feature_family_importance.json": families,
        "case_studies.json": {"finding_label": "Synthetic-data finding", "cases": studies},
        "hard_negative_explanations.json": hard_report,
        "false_negative_explanations.json": fn_report,
        "explainer_metadata.json": metadata,
    }
    inputs.verify_unchanged()
    hashes = {}
    if inputs.shap.status == "AVAILABLE":
        hashes["validation_shap_values.npz"] = file_hash(output / "validation_shap_values.npz")
    for name, report in artifacts.items():
        save_json(output / name, report)
        hashes[name] = file_hash(output / name)
    bundle = ExplanationBundle(
        version=snapshot_version,
        generated_at=datetime.now(UTC).isoformat(),
        provenance=inputs.provenance,
        shap_status=inputs.shap.status,
        artifacts=hashes,
        performance=performance,
        isolation_audit=inputs.audit,
    )
    save_json(bundle_path, bundle.model_dump(mode="json"))
    if snapshot_version == SHAP_VERSION:
        from riskbyte_ml.explainability.freeze import freeze_snapshot

        freeze_snapshot(output, bundle)
    return bundle
