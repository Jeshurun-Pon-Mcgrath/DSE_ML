"""Train → earlier validation search/calibration → later validation selection → freeze → test."""

import importlib.metadata
import importlib.util
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn import config_context
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import digest
from riskbyte_ml.features.behaviour import finite_numeric
from riskbyte_ml.models.supervised import MODEL_VERSION, REGISTRY_VERSION
from riskbyte_ml.models.supervised.artifacts import (
    ClassifierBundle,
    load_classifier,
    save_classifier,
    verify_registry_artifacts,
)
from riskbyte_ml.models.supervised.calibration import select_calibration
from riskbyte_ml.models.supervised.estimators import (
    ALGORITHMS,
    benchmark_knn,
    make_estimator,
    positive_probability,
    search_parameters,
)
from riskbyte_ml.models.supervised.inputs import ExperimentInputs, verify_frozen_files
from riskbyte_ml.models.supervised.metrics import curve_data, evaluate, metrics
from riskbyte_ml.models.supervised.schemas import (
    CalibrationChoice,
    ErrorAnalysis,
    ErrorGroup,
    ExperimentConfig,
    FeatureImportance,
    ModelProvenance,
    ModelRecord,
    ModelRegistry,
    RandomComparison,
    SearchTrial,
    SelectionFreeze,
    Timing,
)
from riskbyte_ml.models.supervised.thresholds import threshold_analysis
from riskbyte_ml.preprocessing.pipeline import Preprocessor
from riskbyte_ml.statistics.verification import file_hash, verify_sources


def save_json(path, value, *, overwrite=True):
    content = (
        value.model_dump_json(indent=2)
        if hasattr(value, "model_dump_json")
        else json.dumps(value, indent=2, allow_nan=False)
    )
    write_artifact(path, content.encode(), overwrite=overwrite)


def dependency_versions():
    result = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def emit(event, **data):
    print(json.dumps({"event": event, **data}, allow_nan=False), flush=True)


def algorithm_family(algorithm):
    return "linear_distance" if algorithm in {"Logistic Regression", "KNN"} else "tree_boosting"


def ranking(records):
    def score(record):
        value = record.selection
        scenario_rates = [s.rate for s in value.scenario_recall.values() if s.rate is not None]
        return (
            value.metrics.pr_auc,
            value.metrics.recall,
            value.metrics.precision,
            value.metrics.f1,
            -(value.hard_negatives.rate or 0.0),
            min(scenario_rates, default=0.0),
            -(value.metrics.brier_score or 0.0),
            -record.timing.validation_prediction_seconds,
        )

    eligible = [
        r
        for r in records
        if r.algorithm != "Rule Baseline"
        and r.selection is not None
        and r.selection.metrics.pr_auc is not None
    ]
    return sorted(eligible, key=score, reverse=True)


def train_algorithm(
    algorithm, inputs, experiment, matrices, model_directory, evaluation_directory, run_id
):
    family = algorithm_family(algorithm)
    processor = inputs.processors[family]
    base = dict(
        algorithm=algorithm,
        version=MODEL_VERSION,
        dataset_fingerprint=inputs.split.dataset_fingerprint,
        feature_version=inputs.split.feature_pipeline_version,
        split_version=inputs.split.split_version,
        split_id=inputs.split.split_id,
        preprocessing_id=processor.preprocessing_id,
        threshold_policy=experiment.policy,
    )
    if algorithm == "XGBoost" and importlib.util.find_spec("xgboost") is None:
        return ModelRecord(
            model_id="xgboost-unavailable-" + run_id[7:19],
            status="DEPENDENCY_UNAVAILABLE",
            limitation="XGBoost package absent; network installation blocked in this "
            "environment. No XGBoost measurements are claimed.",
            class_imbalance_strategy="Planned comparison: scale_pos_weight=1 versus "
            "training negative/positive ratio",
            **base,
        ), None
    train, validation = matrices[family]
    fit_rows = np.arange(len(inputs.y_train))
    benchmark = None
    if algorithm == "KNN":
        fit_rows, benchmark = benchmark_knn(
            train,
            inputs.y_train,
            validation[inputs.calibration_positions],
            inputs.y_validation[inputs.calibration_positions],
            experiment,
        )
        emit("knn_benchmark", **benchmark.model_dump(mode="json"))
    trials = []
    best, best_parameters, best_fit, best_score = None, None, None, -1.0
    for number, parameters in enumerate(search_parameters(algorithm, inputs.y_train, experiment)):
        estimator = make_estimator(algorithm, parameters, experiment)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fit_options = (
                {
                    "eval_set": [
                        (
                            validation[inputs.calibration_positions],
                            inputs.y_validation[inputs.calibration_positions],
                        )
                    ],
                    "verbose": False,
                }
                if algorithm == "XGBoost"
                else {}
            )
            estimator.fit(train[fit_rows], inputs.y_train[fit_rows], **fit_options)
        fit_seconds = perf_counter() - start
        p = positive_probability(estimator, validation[inputs.calibration_positions])
        measured = metrics(inputs.y_validation[inputs.calibration_positions], p)
        convergence = [
            type(w.message).__name__ for w in caught if isinstance(w.message, ConvergenceWarning)
        ]
        trials.append(
            SearchTrial(
                parameters=parameters,
                fit_seconds=fit_seconds,
                calibration_block_metrics=measured,
                warnings=convergence,
            )
        )
        emit(
            "candidate_complete",
            algorithm=algorithm,
            candidate=number + 1,
            fit_seconds=fit_seconds,
            calibration_block_pr_auc=measured.pr_auc,
            converged=not bool(convergence),
        )
        if not convergence and measured.pr_auc is not None and measured.pr_auc > best_score:
            best, best_parameters, best_fit, best_score = (
                estimator,
                parameters,
                fit_seconds,
                measured.pr_auc,
            )
    if best is None:
        raise DatasetError(
            "model_training_failed", "No converged candidate with valid validation metrics"
        )
    start = perf_counter()
    raw_validation = positive_probability(best, validation)
    raw_seconds = perf_counter() - start
    calibrator, calibration_report, calibration_seconds = select_calibration(
        raw_validation[inputs.calibration_positions],
        inputs.y_validation[inputs.calibration_positions],
        raw_validation[inputs.selection_positions],
        inputs.y_validation[inputs.selection_positions],
    )
    start = perf_counter()
    calibrated = calibrator.transform(raw_validation)
    validation_seconds = raw_seconds + perf_counter() - start
    selection_metadata = inputs.validation_metadata.iloc[inputs.selection_positions].reset_index(
        drop=True
    )
    analysis = threshold_analysis(
        inputs.y_validation[inputs.selection_positions],
        calibrated[inputs.selection_positions],
        selection_metadata.hard_negative.to_numpy(dtype=bool),
    )
    threshold = analysis.policies[experiment.policy].chosen.threshold
    model_id = (
        algorithm.lower().replace(" ", "-")
        + "-"
        + digest(
            {
                "run": run_id,
                "algorithm": algorithm,
                "parameters": best_parameters,
                "calibration": calibration_report.method,
                "threshold": threshold,
            }
        )[7:23]
    )
    bundle = ClassifierBundle(
        best,
        calibrator,
        algorithm,
        model_id,
        threshold,
        processor.preprocessing_id,
        processor.feature_names,
        inputs.split.dataset_fingerprint,
        inputs.split.feature_pipeline_version,
    )
    artifact_name, artifact_hash = save_classifier(bundle, model_directory)
    loaded = load_classifier(
        model_directory / artifact_name,
        artifact_hash,
        inputs.split.dataset_fingerprint,
        inputs.split.feature_pipeline_version,
    )
    probe = validation[: min(100, validation.shape[0])]
    if not np.allclose(bundle.probability(probe), loaded.probability(probe), rtol=0, atol=0):
        raise DatasetError("model_reload_failed", "Serialization changed classifier probabilities")
    start = perf_counter()
    # A full KNN training-set prediction is unnecessary quadratic work; use its fit subset.
    diagnostic_rows = fit_rows if algorithm == "KNN" else np.arange(len(inputs.y_train))
    train_probability = bundle.probability(train[diagnostic_rows])
    train_prediction_seconds = perf_counter() - start
    importance = []
    if hasattr(best, "feature_importances_"):
        values = best.feature_importances_
        tags = {s.name: s.interpretability_class for s in processor.encoded_manifest()}
        importance = [
            FeatureImportance(feature=name, importance=float(v), interpretability_class=tags[name])
            for name, v in zip(processor.feature_names, values, strict=True)
        ]
        importance.sort(key=lambda entry: entry.importance, reverse=True)
    status = "COMPUTATIONALLY_CONSTRAINED" if benchmark and benchmark.constrained else "TRAINED"
    record = ModelRecord(
        model_id=model_id,
        status=status,
        trained_at=datetime.now(UTC),
        limitation=benchmark.reason if status == "COMPUTATIONALLY_CONSTRAINED" else None,
        parameters=best_parameters,
        class_imbalance_strategy=(
            "Representative stratified training sample; no class weights/oversampling"
            if algorithm == "KNN"
            else (
                f"scale_pos_weight={best_parameters['scale_pos_weight']:.6g}; "
                "ratio learned from training only"
            )
            if algorithm == "XGBoost"
            else (
                f"class_weight={best_parameters.get('class_weight')}; "
                "default and weighted candidates compared"
            )
        ),
        selected_threshold=threshold,
        calibration=calibration_report,
        train_metrics=metrics(inputs.y_train[diagnostic_rows], train_probability, threshold),
        validation=evaluate(
            inputs.y_validation,
            calibrated,
            threshold,
            inputs.validation_metadata,
            validation_seconds,
        ),
        selection=evaluate(
            inputs.y_validation[inputs.selection_positions],
            calibrated[inputs.selection_positions],
            threshold,
            selection_metadata,
            None,  # Reuses the measured full-validation prediction batch; no separate timing.
        ),
        default_threshold_metrics=metrics(inputs.y_validation, calibrated, 0.5),
        timing=Timing(
            search_training_seconds=sum(t.fit_seconds for t in trials),
            chosen_fit_seconds=best_fit,
            calibration_seconds=calibration_seconds,
            train_prediction_seconds=train_prediction_seconds,
            validation_prediction_seconds=validation_seconds,
        ),
        search_trials=trials,
        feature_importance=importance,
        threshold_analysis=analysis,
        knn_benchmark=benchmark,
        artifact_name=artifact_name,
        artifact_hash=artifact_hash,
        **base,
    )
    save_json(
        evaluation_directory / (model_id + "_curves.json"),
        curve_data(inputs.y_validation, calibrated),
    )
    record.validation_errors = error_analysis(
        inputs.features.iloc[inputs.validation_rows],
        inputs.y_validation,
        calibrated,
        threshold,
        inputs.validation_metadata,
        inputs.frame.iloc[inputs.validation_rows],
        float(
            np.quantile(
                finite_numeric(inputs.features.iloc[inputs.train_rows].TransactionAmt), 0.95
            )
        ),
    )
    save_json(model_directory / (model_id + ".json"), record)
    emit(
        "model_complete",
        algorithm=algorithm,
        validation_pr_auc=record.validation.metrics.pr_auc,
        selection_pr_auc=record.selection.metrics.pr_auc,
        threshold=threshold,
        calibration=calibration_report.method,
    )
    return record, bundle


def rule_scores(features):
    ratio = finite_numeric(features.amount_ratio_to_prior_mean)
    velocity = finite_numeric(features.transactions_prior_30m)
    novelty = features.device_seen_before.eq(False).fillna(False).to_numpy(dtype=bool) | (
        features.billing_region_seen_before.eq(False).fillna(False).to_numpy(dtype=bool)
    )
    return ((ratio >= 3).astype(float) + (velocity >= 3).astype(float) + novelty.astype(float)) / 3


def rule_record(inputs, experiment, run_id):
    start = perf_counter()
    p = rule_scores(inputs.features.iloc[inputs.validation_rows])
    seconds = perf_counter() - start
    threshold = 2 / 3
    record = ModelRecord(
        model_id="rule-baseline-" + run_id[7:23],
        algorithm="Rule Baseline",
        version=MODEL_VERSION,
        status="RULE_BASELINE",
        dataset_fingerprint=inputs.split.dataset_fingerprint,
        feature_version=inputs.split.feature_pipeline_version,
        split_version=inputs.split.split_version,
        split_id=inputs.split.split_id,
        parameters={"amount_ratio": 3.0, "velocity_30m": 3, "required_signals": 2},
        class_imbalance_strategy="None; fixed training-knowledge rules",
        selected_threshold=threshold,
        threshold_policy=experiment.policy,
        calibration=CalibrationChoice(
            method="rule_score",
            fitting_source="No fitting; rules prespecified",
            selection_source="Not tuned",
            candidates={},
            excluded_methods={},
            reasoning="Two of amount ratio >=3, prior 30m count >=3, new device/region; "
            "no labels in rule inputs",
        ),
        validation=evaluate(
            inputs.y_validation,
            p,
            threshold,
            inputs.validation_metadata,
            seconds,
            probability=False,
        ),
        selection=evaluate(
            inputs.y_validation[inputs.selection_positions],
            p[inputs.selection_positions],
            threshold,
            inputs.validation_metadata.iloc[inputs.selection_positions].reset_index(drop=True),
            None,
            probability=False,
        ),
    )
    return record


def error_analysis(features, labels, probabilities, threshold, metadata, raw, training_amount_q95):
    predicted = probabilities >= threshold
    groups = {}
    for name, mask in {
        "false_positives": (labels == 0) & predicted,
        "false_negatives": (labels == 1) & ~predicted,
    }.items():
        subset = features.iloc[np.flatnonzero(mask)]
        ev = metadata.iloc[np.flatnonzero(mask)]

        def median(column, subset=subset):
            values = finite_numeric(subset[column])
            values = values[np.isfinite(values)]
            return float(np.median(values)) if len(values) else None

        groups[name] = ErrorGroup(
            count=int(mask.sum()),
            hard_negative_count=int(ev.hard_negative.sum()),
            missing_identity_count=int(raw.device_id.iloc[np.flatnonzero(mask)].isna().sum()),
            median_amount=median("TransactionAmt"),
            median_amount_ratio=median("amount_ratio_to_prior_mean"),
            median_velocity_30m=median("transactions_prior_30m"),
            new_device_count=int(subset.device_seen_before.eq(False).sum()),
            new_merchant_count=int(subset.merchant_seen_before.eq(False).sum()),
            high_amount_count=int(
                (finite_numeric(subset.TransactionAmt) > training_amount_q95).sum()
            ),
            velocity_burst_count=int((finite_numeric(subset.transactions_prior_30m) >= 3).sum()),
            scenarios={
                str(k): int(v)
                for k, v in ev.loc[ev.fraud_scenario.ne(""), "fraud_scenario"]
                .value_counts()
                .items()
            },
            hard_negative_types={
                str(k): int(v)
                for k, v in ev.loc[ev.hard_negative, "hard_negative_type"].value_counts().items()
            },
        )
    return ErrorAnalysis(
        **groups,
        threshold_definitions="High amount exceeds training 95th percentile; velocity burst "
        "means strictly prior 30m count >=3; novelty unavailable is not "
        "new",
    )


def random_comparator(inputs, root, experiment, records, directory):
    split = inputs.random_split(root)
    results = []
    for record in ranking(records)[:2]:
        if record.status == "COMPUTATIONALLY_CONSTRAINED":
            continue
        family = algorithm_family(record.algorithm)
        processor = Preprocessor(inputs.processors[family].config)
        processor.fit_training(inputs.features, split, inputs.specs)
        processor.save(directory / "random_preprocessing")
        train = processor.transform(inputs.features.iloc[split.indexes["train"]])
        validation = processor.transform(inputs.features.iloc[split.indexes["validation"]])
        y_train = inputs.frame.isFraud.iloc[split.indexes["train"]].astype(int).to_numpy()
        y_validation = inputs.frame.isFraud.iloc[split.indexes["validation"]].astype(int).to_numpy()
        estimator = make_estimator(record.algorithm, record.parameters, experiment)
        if record.algorithm == "XGBoost":
            # Fixed chosen hyperparameters; no early-stop target access in comparator test data.
            estimator.set_params(early_stopping_rounds=None)
        start = perf_counter()
        estimator.fit(train, y_train)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        p = positive_probability(estimator, validation)
        seconds = perf_counter() - start
        metadata = inputs.metadata.iloc[split.indexes["validation"]].reset_index(drop=True)
        report = RandomComparison(
            algorithm=record.algorithm,
            split_id=split.split_id,
            validation=evaluate(y_validation, p, 0.5, metadata, seconds),
            temporal_selection_pr_auc=record.selection.metrics.pr_auc,
            preprocessing_id=processor.preprocessing_id,
            training_seconds=fit_seconds,
            note="Secondary raw-probability validation benchmark with fixed temporal-selected "
            "hyperparameters. Random test unscored. Globally prior unlabelled held-out "
            "context; cannot choose production model.",
        )
        results.append(report)
    return results


LIMITATIONS = [
    "All findings use synthetic academic data and cannot establish production banking performance.",
    "Scenario rules influence learnability; synthetic identities are "
    "not real customers and drift is approximate.",
    "Validation is subdivided chronologically: earlier "
    "search/calibration and later model/calibration/threshold "
    "selection. Entire-validation metrics include calibration-fitting "
    "rows and are not independent estimates.",
    "Later validation is reused for calibration method, threshold and "
    "algorithm selection, so selection metrics are optimistic; no "
    "nested temporal replication is claimed.",
    "Phase 3 already inspected aggregate test statistics; test "
    "remains excluded from Phase 4 fitting/selection, but was not "
    "unseen for all prior exploration.",
    "Repeated entities/episodes and small scenario counts limit "
    "uncertainty and independent generalization claims.",
    "KNN can retain only a stratified training subset; its comparison "
    "has a different fit budget. Memory figures describe retained "
    "CSR, not isolated peak RSS.",
    "Batch prediction timings are measured offline and include no "
    "HTTP/history/preprocessing service latency; no real-time "
    "readiness claim.",
    "Tree importances are exploratory correlated-feature allocations, "
    "not causal explanations; SHAP remains absent.",
    "Random comparator training reuses rows from different primary "
    "partitions after primary selection/test freeze; it cannot inform "
    "any primary decision.",
    "Missing identity evidence and contaminated prior attack history "
    "can impair detection. No target-aware history filtering or "
    "synthetic record deletion.",
    "No SMOTE, target encoding, Isolation Forest, SHAP, ABRF or "
    "public transaction prediction is implemented.",
]


def run_experiment(config, root: Path, experiment=None, output_root=None):
    experiment = experiment or ExperimentConfig()
    output = output_root or root
    # Verify sources and immutable files before returning even a cached completed run.
    verify_sources(config)
    manifest = verify_frozen_files(root)
    versions = dependency_versions()
    run_id = digest(
        {
            "dataset": manifest["dataset_fingerprint"],
            "manifest": file_hash(root / "features" / "feature_manifest.json"),
            "config": experiment.model_dump(mode="json"),
            "version": MODEL_VERSION,
            "dependencies": versions,
        }
    )
    registry_path = output / "models" / "model_registry.json"
    evaluation_directory = output / "evaluation"
    if registry_path.is_file():
        existing = ModelRegistry.model_validate_json(registry_path.read_bytes())
        if existing.provenance.run_id != run_id:
            raise DatasetError(
                "model_experiment_frozen",
                "Completed experiment differs; no automatic retraining or test reuse permitted",
            )
        verify_registry_artifacts(existing, output / "models")
        emit("completed_experiment_reused", run_id=run_id, test_reaccessed=False)
        return existing
    if (evaluation_directory / "selection_freeze.json").exists() or (
        evaluation_directory / "test_evaluation_started.json"
    ).exists():
        raise DatasetError(
            "model_experiment_interrupted",
            "Frozen/interrupted experiment cannot silently retrain or repeat test evaluation",
        )
    model_directory = output / "models" / run_id[7:]
    inputs = ExperimentInputs(config, root, experiment)
    matrices = {family: inputs.matrices(family) for family in inputs.processors}
    records, bundles = [], {}
    with (
        threadpool_limits(limits=experiment.n_jobs),
        config_context(working_memory=experiment.working_memory_mb),
    ):
        for algorithm in ALGORITHMS:
            record, bundle = train_algorithm(
                algorithm,
                inputs,
                experiment,
                matrices,
                model_directory,
                evaluation_directory,
                run_id,
            )
            records.append(record)
            if bundle:
                bundles[record.model_id] = bundle
        records.append(rule_record(inputs, experiment, run_id))
        rule = records[-1]
        save_json(
            evaluation_directory / (rule.model_id + "_curves.json"),
            curve_data(
                inputs.y_validation, rule_scores(inputs.features.iloc[inputs.validation_rows])
            ),
        )
        ordered = ranking(records)
        if not ordered:
            raise DatasetError(
                "model_selection_unavailable", "No valid supervised validation result"
            )
        selected = ordered[0]
        selected.is_selected_supervised_model = True
        rationale = (
            f"{selected.algorithm} selected by later-validation average precision "
            f"{selected.selection.metrics.pr_auc:.6g}; "
            "lexicographic tie-breaks: recall, precision, F1, hard-negative "
            "FPR, minimum scenario recall, Brier, batch prediction time. "
            "Accuracy/random comparator/test results are never selection criteria."
        )
        verify_sources(config)
        verify_frozen_files(root)
        freeze = SelectionFreeze(
            model_id=selected.model_id,
            threshold=selected.selected_threshold,
            calibration_method=selected.calibration.method,
            artifact_hash=selected.artifact_hash,
            run_id=run_id,
            frozen_at=datetime.now(UTC),
            ranking=[r.model_id for r in ordered],
            rationale=rationale,
        )
        save_json(evaluation_directory / "selection_freeze.json", freeze, overwrite=False)
        emit(
            "selection_frozen",
            algorithm=selected.algorithm,
            model_id=selected.model_id,
            threshold=freeze.threshold,
            selection_pr_auc=selected.selection.metrics.pr_auc,
        )
        test_features, test_labels, test_metadata, test_raw = inputs.gate.release(
            evaluation_directory, selected.model_id, selected.artifact_hash
        )
        matrix = inputs.processors[algorithm_family(selected.algorithm)].transform(test_features)
        start = perf_counter()
        probabilities = bundles[selected.model_id].probability(matrix)
        seconds = perf_counter() - start
        selected.test_metrics = evaluate(
            test_labels, probabilities, selected.selected_threshold, test_metadata, seconds
        )
        test_curves = curve_data(test_labels, probabilities)
        errors = error_analysis(
            test_features,
            test_labels,
            probabilities,
            selected.selected_threshold,
            test_metadata,
            test_raw,
            float(
                np.quantile(
                    finite_numeric(inputs.features.iloc[inputs.train_rows].TransactionAmt), 0.95
                )
            ),
        )
        save_json(
            evaluation_directory / "final_test_metrics.json",
            {
                "model_id": selected.model_id,
                "selection_hash": file_hash(evaluation_directory / "selection_freeze.json"),
                "evaluation": selected.test_metrics.model_dump(mode="json"),
                "curves": test_curves.model_dump(mode="json"),
                "errors": errors.model_dump(mode="json"),
                "test_gate_access_count": inputs.gate.access_count,
            },
            overwrite=False,
        )
        emit(
            "final_test_complete",
            model_id=selected.model_id,
            metrics=selected.test_metrics.metrics.model_dump(mode="json"),
            test_access_count=inputs.gate.access_count,
        )
        random = random_comparator(inputs, root, experiment, records, model_directory)
    verify_sources(config)
    verify_frozen_files(root)
    provenance = ModelProvenance(
        dataset_fingerprint=inputs.split.dataset_fingerprint,
        feature_version=inputs.split.feature_pipeline_version,
        split_id=inputs.split.split_id,
        feature_manifest_hash=file_hash(root / "features" / "feature_manifest.json"),
        split_file_hash=file_hash(root / "splits" / (inputs.split.split_id[7:] + ".json")),
        model_version=MODEL_VERSION,
        registry_version=REGISTRY_VERSION,
        run_id=run_id,
        generated_at=datetime.now(UTC),
        dependency_versions=versions,
    )
    registry = ModelRegistry(
        provenance=provenance,
        models=records,
        selection_freeze=freeze,
        partition_prevalence={
            name: summary.fraud_prevalence for name, summary in inputs.split.summaries.items()
        },
        partition_counts={
            name: summary.row_count for name, summary in inputs.split.summaries.items()
        },
        validation_blocks={
            "search_calibration": len(inputs.calibration_positions),
            "selection": len(inputs.selection_positions),
        },
        selected_test_curves=test_curves,
        selected_test_errors=errors,
        random_comparisons=random,
        leakage_audit={
            "preprocessing": "PASS: frozen bytes, training-index hash, ordered raw/encoded "
            "manifest, no temporal refit",
            "target_and_metadata": "PASS: source-lineage allowlist excludes "
            "target/scenario/hard-negative annotations",
            "behaviour": "PASS: existing strictly prior builder; current/equal/future "
            "event and target invariance tests retained",
            "split": "PASS: exact frozen temporal and secondary random split bytes preserved",
            "threshold_and_calibration": "PASS: validation blocks only; "
            "no test parameter selection",
            "final_test": "PASS: durable model/threshold/calibrator freeze before "
            f"one gate access ({inputs.gate.access_count})",
            "random_comparator": "PASS: post-freeze secondary experiment; "
            "never feeds primary selection",
        },
        limitations=LIMITATIONS,
    )
    for record in records:
        save_json(model_directory / (record.model_id + ".json"), record)
    save_json(evaluation_directory / "model_comparison.json", registry)
    from riskbyte_ml.models.supervised.report import generate_report

    generate_report(registry, output)
    # Atomically published complete registry is the API snapshot; no partial selection is exposed.
    save_json(registry_path, registry, overwrite=False)
    return registry
