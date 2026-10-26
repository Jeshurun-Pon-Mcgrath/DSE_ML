"""Offline Phase 5 development; original temporal test records are inaccessible."""

import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskbyte_ml.anomaly.analysis import (
    complementarity,
    distribution,
    evaluate,
    group_analysis,
    investigations,
    select_threshold,
)
from riskbyte_ml.anomaly.development import DevelopmentInputs
from riskbyte_ml.anomaly.engine import (
    ANOMALY_VERSION,
    AnomalyConfig,
    IsolationEngine,
    feature_contract,
)
from riskbyte_ml.anomaly.schemas import Phase5Bundle
from riskbyte_ml.behavioural.deviation import (
    BEHAVIOUR_VERSION,
    SIGNALS,
    BehaviourConfig,
    DeviationEngine,
)
from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.models.supervised.artifacts import load_classifier
from riskbyte_ml.preprocessing.pipeline import load_preprocessor
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import file_hash, read_json, verify_sources

BASELINE_ID = "decision-tree-459fbcf71e8c2360"
BASELINE_HASH = "sha256:f0e081811268328766866cefcb983bc873bd30579125d1044005d5169c67854f"
BASELINE_THRESHOLD = 0.3665406264377367


def frozen_baseline(root, inputs):
    registry_path = root / "models" / "model_registry.json"
    before = file_hash(registry_path)
    registry = read_json(registry_path)
    selected = [m for m in registry["models"] if m["is_selected_supervised_model"]]
    if len(selected) != 1:
        raise DatasetError("anomaly_baseline_changed", "Exactly one frozen baseline required")
    record = selected[0]
    if (
        record["model_id"] != BASELINE_ID
        or record["artifact_hash"] != BASELINE_HASH
        or record["selected_threshold"] != BASELINE_THRESHOLD
        or record["dataset_fingerprint"] != FROZEN_FINGERPRINT
        or record["feature_version"] != FROZEN_FEATURE_VERSION
        or record["split_id"] != FROZEN_SPLIT_ID
    ):
        raise DatasetError("anomaly_baseline_changed", "Frozen Decision Tree differs")
    run = registry["provenance"]["run_id"][7:]
    filename = record["artifact_name"]
    if Path(filename).name != filename or not all(c in "0123456789abcdef" for c in run):
        raise DatasetError("anomaly_baseline_invalid", "Invalid baseline artifact reference")
    baseline = load_classifier(
        root / "models" / run / filename, BASELINE_HASH, FROZEN_FINGERPRINT, FROZEN_FEATURE_VERSION
    )
    family = inputs.manifest["families"]["tree_boosting"]
    processor = load_preprocessor(
        root / "preprocessing" / family["preprocessing_artifact"],
        FROZEN_FINGERPRINT,
        FROZEN_SPLIT_ID,
    )
    if (
        baseline.threshold != BASELINE_THRESHOLD
        or baseline.preprocessing_id != processor.preprocessing_id
        or baseline.feature_names != processor.feature_names
    ):
        raise DatasetError("anomaly_baseline_invalid", "Baseline preprocessing/threshold differs")
    probabilities = baseline.probability(processor.transform(inputs.features))
    if file_hash(registry_path) != before:
        raise DatasetError("anomaly_baseline_changed", "Registry changed during development")
    return probabilities, {
        "model_id": BASELINE_ID,
        "artifact_hash": BASELINE_HASH,
        "threshold": BASELINE_THRESHOLD,
        "registry_hash": before,
    }


def dump_json(path, value):
    write_artifact(path, json.dumps(value, indent=2, allow_nan=False).encode())


def save_object(path, value):
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    write_artifact(path, buffer.getvalue(), overwrite=False)
    return file_hash(path)


def run(config, root: Path, anomaly_config=None, behaviour_config=None):
    anomaly_config = anomaly_config or AnomalyConfig()
    behaviour_config = behaviour_config or BehaviourConfig()
    directory = root / "phase5"
    if (directory / "analysis_bundle.json").exists():
        verify_sources(config)
        from riskbyte_ml.models.supervised.inputs import verify_frozen_files

        verify_frozen_files(root)
        bundle = Phase5Bundle.model_validate_json((directory / "analysis_bundle.json").read_bytes())
        if bundle.configuration != {
            "anomaly": anomaly_config.model_dump(mode="json"),
            "behaviour": behaviour_config.model_dump(mode="json"),
        }:
            raise DatasetError("anomaly_run_frozen", "Completed Phase 5 configuration is frozen")
        for name, expected in bundle.artifact_hashes.items():
            if Path(name).name != name or file_hash(directory / name) != expected:
                raise DatasetError("anomaly_artifact_changed", "Phase 5 artifact checksum differs")
        if file_hash(root / "models" / "model_registry.json") != bundle.baseline["registry_hash"]:
            raise DatasetError("anomaly_baseline_changed", "Phase 4 registry changed")
        registry = read_json(root / "models" / "model_registry.json")
        classifier = (
            root / "models" / registry["provenance"]["run_id"][7:] / (BASELINE_ID + ".joblib")
        )
        if file_hash(classifier) != BASELINE_HASH:
            raise DatasetError("anomaly_baseline_changed", "Frozen classifier bytes changed")
        return bundle
    if directory.exists() and any(directory.iterdir()):
        raise DatasetError("anomaly_incomplete_run", "Incomplete Phase 5 outputs require review")
    inputs = DevelopmentInputs(config, root)
    count = inputs.train_count
    x_train, x_validation = inputs.features.iloc[:count], inputs.features.iloc[count:]
    y_train, y_validation = inputs.labels[:count], inputs.labels[count:]
    metadata = inputs.metadata.iloc[count:].reset_index(drop=True)
    hard = metadata.hard_negative.to_numpy(bool)
    probabilities, baseline = frozen_baseline(root, inputs)
    study, candidates = [], []
    for family in ("interpretable", "broad_numeric"):
        names = feature_contract(x_train, family)
        for population in ("legitimate_train", "mixed_train"):
            reference = x_train.loc[y_train == 0] if population == "legitimate_train" else x_train
            started = time.perf_counter()
            engine = IsolationEngine.fit(
                reference, names, anomaly_config, population=population, feature_family=family
            )
            seconds = time.perf_counter() - started
            raw, score = engine.score(x_validation)
            threshold = select_threshold(y_validation, score, hard, anomaly_config)
            evaluation = evaluate(y_validation, score, threshold["threshold"], metadata)
            contamination = []
            for value in anomaly_config.contamination_candidates:
                raw_threshold = (
                    0.5
                    if value == "auto"
                    else float(np.quantile(engine.normalization.reference, 1 - value))
                )
                native_score = engine.normalization.transform([raw_threshold])[0]
                native = evaluate(y_validation, score, float(native_score), metadata)
                # Native comparisons use actual raw offsets (not a tied percentile surrogate).
                flags = -raw > raw_threshold
                native["recall"] = float(flags[y_validation == 1].mean())
                native["fpr"] = float(flags[y_validation == 0].mean())
                native["hard_negatives"]["rate"] = float(flags[hard].mean()) if hard.any() else None
                contamination.append(
                    {
                        "setting": value,
                        "raw_anomaly_threshold": raw_threshold,
                        "recall": native["recall"],
                        "fpr": native["fpr"],
                        "hard_negative_rate": native["hard_negatives"]["rate"],
                    }
                )
            item = {
                "family": family,
                "population": population,
                "features": names,
                "raw_feature_count": len(names),
                "prepared_feature_count": 2 * len(names),
                "fit_rows": len(reference),
                "fit_seconds": seconds,
                "validation": evaluation,
                "threshold_analysis": threshold,
                "contamination_study": contamination,
            }
            study.append(item)
            candidates.append((engine, item, score))
    # Predeclared preference for normal-reference contract; mixed is a methodological comparator.
    selected_candidates = [c for c in candidates if c[0].population == "legitimate_train"]
    engine, selected, anomaly_validation = max(
        selected_candidates,
        key=lambda c: (
            c[1]["validation"]["pr_auc"],
            c[1]["validation"]["recall"],
            -c[1]["validation"]["hard_negatives"]["rate"],
        ),
    )
    feasible = [
        c
        for c in selected["contamination_study"]
        if c["fpr"] <= anomaly_config.max_legitimate_rate
        and (c["hard_negative_rate"] or 0) <= anomaly_config.max_hard_negative_rate
    ]
    contamination = (
        max(feasible, key=lambda c: (c["recall"], -c["fpr"]))
        if feasible
        else {
            "setting": "auto",
            "raw_anomaly_threshold": 0.5,
        }
    )
    engine.model.contamination = contamination["setting"]
    engine.model.offset_ = -contamination["raw_anomaly_threshold"]
    selected = {
        **selected,
        "dataset_fingerprint": FROZEN_FINGERPRINT,
        "feature_version": FROZEN_FEATURE_VERSION,
        "split_id": FROZEN_SPLIT_ID,
        "split_version": inputs.split.split_version,
        "model_version": ANOMALY_VERSION,
        "seed": anomaly_config.seed,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "parameters": {
            "n_estimators": anomaly_config.n_estimators,
            "max_samples": anomaly_config.max_samples,
            "n_jobs": anomaly_config.n_jobs,
        },
        "normalization_metadata": {
            "reference_rows": len(engine.normalization.reference),
            "quantiles": {
                str(q): float(np.quantile(engine.normalization.reference, q))
                for q in [0, 0.01, 0.5, 0.95, 0.99, 1]
            },
            "full_mapping": "sorted TRAIN reference stored in trusted model artifact",
        },
        "reference_matrix_bytes": len(engine.normalization.reference)
        * len(engine.preparation.names)
        * 2
        * np.dtype(np.float32).itemsize,
        "contamination": contamination["setting"],
        "selection": "highest validation average precision among legitimate-reference candidates; "
        "mixed-reference results retained separately",
        "normalization": "training-reference strict-left empirical CDF",
    }
    behaviour = DeviationEngine.fit(x_train, behaviour_config)
    b_train, conf_train, *_ = behaviour.score_arrays(x_train)
    b_validation, conf_validation, *_ = behaviour.score_arrays(x_validation)
    behaviour_threshold = select_threshold(y_validation, b_validation, hard, anomaly_config)
    behaviour_metrics = evaluate(
        y_validation, b_validation, behaviour_threshold["threshold"], metadata
    )
    original_train, anomaly_train = engine.score(x_train)
    original_validation, _ = engine.score(x_validation)
    at, bt = selected["threshold_analysis"]["threshold"], behaviour_threshold["threshold"]
    complementary = {
        "train_diagnostic": complementarity(
            y_train, probabilities[:count] >= BASELINE_THRESHOLD, anomaly_train >= at
        ),
        "validation": complementarity(
            y_validation, probabilities[count:] >= BASELINE_THRESHOLD, anomaly_validation >= at
        ),
        "supervised_validation": evaluate(
            y_validation, probabilities[count:], BASELINE_THRESHOLD, metadata
        ),
    }
    groups = group_analysis(
        y_validation, metadata, anomaly_validation, b_validation, conf_validation, at, bt
    )
    investigation = investigations(
        x_validation, metadata, y_validation, anomaly_validation, b_validation
    )
    # Aggregate hard-negative score distributions supplement binary flags.
    groups["hard_negative_types"] = {
        name: {
            "anomaly_score": distribution(
                anomaly_validation[metadata.hard_negative_type.eq(name).to_numpy(bool)]
            ),
            "behaviour_score": distribution(
                b_validation[metadata.hard_negative_type.eq(name).to_numpy(bool)]
            ),
        }
        for name in selected["validation"]["hard_negative_types"]
    }
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {
        "isolation_forest.joblib": save_object(directory / "isolation_forest.joblib", engine),
        "behavioural_engine.joblib": save_object(
            directory / "behavioural_engine.joblib", behaviour
        ),
    }
    # Reload checks compare scores on development data, never on original test.
    restored = joblib.load(directory / "isolation_forest.joblib")
    np.testing.assert_array_equal(restored.score(x_validation)[1], anomaly_validation)
    np.testing.assert_array_equal(
        joblib.load(directory / "behavioural_engine.joblib").score_arrays(x_validation)[0],
        b_validation,
    )
    audit = {
        **inputs.audit,
        "label_free_model_fit": True,
        "legitimate_reference_selection": "TRAIN labels only",
        "normalization_fit": "TRAIN only",
        "threshold_selection": "VALIDATION only",
        "frozen_phase4_unchanged": file_hash(root / "models" / "model_registry.json")
        == baseline["registry_hash"],
        "ablation_is_evaluation_only": True,
        "target_metadata_excluded": True,
        "fresh_holdout_generated": False,
        "serialization_reload_exact": True,
    }
    ablation = pd.DataFrame(
        {
            "partition": ["train"] * count + ["validation"] * len(metadata),
            "TransactionID": inputs.frame.TransactionID,
            "P": probabilities,
            "A": np.r_[anomaly_train, anomaly_validation],
            "raw_isolation_score": np.r_[original_train, original_validation],
            "B": np.r_[b_train, b_validation],
            "behaviour_confidence": np.r_[conf_train, conf_validation],
            "ground_truth": inputs.labels,
            "fraud_scenario": inputs.metadata.fraud_scenario,
            "hard_negative": inputs.metadata.hard_negative,
            "hard_negative_type": inputs.metadata.hard_negative_type,
        }
    )
    write_artifact(directory / "development_scores.csv", ablation.to_csv(index=False).encode())
    hashes["development_scores.csv"] = file_hash(directory / "development_scores.csv")
    bundle = Phase5Bundle(
        dataset_fingerprint=FROZEN_FINGERPRINT,
        feature_version=FROZEN_FEATURE_VERSION,
        split_id=FROZEN_SPLIT_ID,
        anomaly_version=ANOMALY_VERSION,
        behaviour_version=BEHAVIOUR_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        configuration={
            "anomaly": anomaly_config.model_dump(mode="json"),
            "behaviour": behaviour_config.model_dump(mode="json"),
        },
        baseline=baseline,
        selected=selected,
        study=study,
        anomaly_metrics=selected["validation"],
        behaviour_metrics={**behaviour_metrics, "threshold_analysis": behaviour_threshold},
        distributions=groups,
        complementarity=complementary,
        investigations=investigation,
        behavioural_components={
            "version": BEHAVIOUR_VERSION,
            "dataset_fingerprint": FROZEN_FINGERPRINT,
            "feature_version": FROZEN_FEATURE_VERSION,
            "split_id": FROZEN_SPLIT_ID,
            "sources": SIGNALS,
            "configuration": behaviour_config.model_dump(mode="json"),
            "normalization": "training-only strict-left CDF; binary novelty is 1-seen",
            "confidence": "n/(n+history_saturation) times available weight fraction",
            "missing": "exclude unavailable components from weighted mean; cold-start B=null",
            "international_novelty": "unavailable: no frozen prior international-history feature",
            "history_duration": "not used: frozen features lack prior history-span feature",
            "normalization_metadata": {
                name: {
                    "reference_rows": len(value.reference),
                    "quantiles": {
                        str(q): float(np.quantile(value.reference, q))
                        for q in [0, 0.5, 0.95, 0.99, 1]
                    },
                }
                for name, value in behaviour.normalizers.items()
            },
        },
        leakage_audit=audit,
        artifact_hashes=hashes,
    )
    for name, value in {
        "anomaly_metadata": selected,
        "behavioural_metadata": bundle.behavioural_components,
        "complementarity": complementary,
        "leakage_audit": audit,
    }.items():
        dump_json(directory / (name + ".json"), value)
    dump_json(directory / "analysis_bundle.json", bundle.model_dump(mode="json"))
    return bundle
