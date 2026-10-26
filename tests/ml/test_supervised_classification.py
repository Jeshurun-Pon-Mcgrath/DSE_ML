"""Small synthetic fixtures, deterministic classifiers and model-phase isolation checks."""

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy import sparse

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.adapter import load_dataset
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig, SyntheticConfig
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate, save_generated
from riskbyte_ml.evaluation.splits import SplitConfig, make_split, save_split
from riskbyte_ml.features.behaviour import FEATURE_PIPELINE_VERSION, build_features
from riskbyte_ml.features.manifest import feature_manifest
from riskbyte_ml.models.supervised import inputs as input_module
from riskbyte_ml.models.supervised.artifacts import load_classifier
from riskbyte_ml.models.supervised.calibration import ProbabilityCalibrator, select_calibration
from riskbyte_ml.models.supervised.estimators import (
    benchmark_knn,
    make_estimator,
    positive_probability,
    representative_rows,
    search_parameters,
)
from riskbyte_ml.models.supervised.inputs import ExperimentInputs, TestGate, validate_feature_inputs
from riskbyte_ml.models.supervised.metrics import curve_data, evaluate, metrics
from riskbyte_ml.models.supervised.pipeline import ranking, run_experiment, save_json
from riskbyte_ml.models.supervised.schemas import ExperimentConfig, ModelRegistry, SelectionFreeze
from riskbyte_ml.models.supervised.thresholds import threshold_analysis
from riskbyte_ml.preprocessing.pipeline import PreprocessingConfig, Preprocessor
from riskbyte_ml.statistics import verification
from riskbyte_ml.statistics.verification import file_hash


def test_hand_calculated_metrics_and_areas():
    value = metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert value.confusion.model_dump() == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert (value.precision, value.recall, value.f1, value.fpr, value.fnr, value.specificity) == (
        0.5,
    ) * 6
    assert value.pr_auc == pytest.approx(5 / 6)
    assert value.roc_auc == 0.75
    assert value.brier_score == pytest.approx(0.255)


@pytest.mark.parametrize("p", [[np.nan], [np.inf], [-0.1], [1.1]])
def test_probability_bounds(p):
    with pytest.raises(ValueError):
        metrics([1], p)


def test_single_class_metrics_and_rule_scores_safe():
    result = metrics([0, 0], [0, 0.9], probability=False)
    assert result.pr_auc is None and result.roc_auc is None and result.fnr is None
    assert result.brier_score is None
    assert result.f1 == 0


def test_threshold_policies_and_ties():
    result = threshold_analysis([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], [False] * 4)
    chosen = result.policies["BALANCED"]
    assert chosen.chosen.threshold == 0.9
    assert chosen.chosen.precision == 1 and chosen.chosen.recall == 0.5
    assert chosen.chosen.fpr == 0 and chosen.constraints_met
    assert set(result.policies) == {"BALANCED", "HIGH_RECALL", "LOW_FALSE_POSITIVE"}
    tied = threshold_analysis([0, 1], [0.5, 0.5], [False, False])
    assert next(p for p in tied.points if p.threshold == 0.5).predicted_positive_count == 2


def test_threshold_constraints_hard_negatives_and_explicit_fallback():
    result = threshold_analysis([0, 1], [0.99, 0.1], [True, False])
    balanced = result.policies["BALANCED"].chosen
    assert balanced.predicted_positive_count == 0 and balanced.hard_negative_fpr == 0
    assert not result.policies["LOW_FALSE_POSITIVE"].constraints_met
    assert len(result.points) <= 204


def test_subgroup_counts_and_scenario_recall():
    import pandas as pd

    metadata = pd.DataFrame(
        {
            "hard_negative": [True, False, False, False],
            "hard_negative_type": ["LAPTOP_PURCHASE", "", "", ""],
            "fraud_scenario": ["", "", "AMOUNT_SPIKE", "LOW_AND_SLOW"],
        }
    )
    result = evaluate([0, 0, 1, 1], [0.8, 0.1, 0.9, 0.2], 0.5, metadata)
    assert result.hard_negatives.count == 1 and result.hard_negatives.rate == 1
    assert result.scenario_recall["AMOUNT_SPIKE"].rate == 1
    assert result.scenario_recall["LOW_AND_SLOW"].rate == 0
    assert result.scenario_recall["VELOCITY_BURST"].rate is None
    assert result.prediction_seconds is None


def test_curve_points_finite_bounded_and_reliability_counts():
    labels = np.tile([0, 1], 500)
    p = np.linspace(0, 1, 1000)
    curves = curve_data(labels, p)
    assert len(curves.pr) <= 201 and len(curves.roc) <= 201
    assert sum(b.count for b in curves.reliability) == len(labels)
    assert curves.roc[0].threshold is None
    assert "Infinity" not in curves.model_dump_json()


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        ("Logistic Regression", {"C": 1.0, "class_weight": "balanced"}),
        ("Decision Tree", {"max_depth": 5, "min_samples_leaf": 2}),
        ("KNN", {"n_neighbors": 3, "weights": "distance", "metric": "euclidean"}),
        ("Random Forest", {"n_estimators": 15, "max_depth": 5}),
    ],
)
def test_reproducible_training_and_probabilities(algorithm, parameters):
    rng = np.random.default_rng(42)
    x = sparse.csr_matrix(rng.normal(size=(80, 6)))
    y = (x.toarray()[:, 0] + x.toarray()[:, 1] > 0.5).astype(int)
    first = make_estimator(algorithm, parameters, ExperimentConfig()).fit(x, y)
    second = make_estimator(algorithm, parameters, ExperimentConfig()).fit(x, y)
    p = positive_probability(first, x)
    np.testing.assert_allclose(p, positive_probability(second, x), rtol=0, atol=0)
    assert ((p >= 0) & (p <= 1)).all()


def test_calibrator_train_validation_contract_and_bounds():
    p = np.linspace(0.02, 0.98, 200)
    y = (p > 0.8).astype(int)
    calibrator = ProbabilityCalibrator("sigmoid").fit(p, y)
    original = calibrator.estimator.coef_.copy()
    transformed = calibrator.transform(np.array([0, 0.5, 1]))
    np.testing.assert_array_equal(original, calibrator.estimator.coef_)
    assert ((transformed >= 0) & (transformed <= 1)).all()
    with pytest.raises(ValueError):
        calibrator.fit(p, 1 - y)
    _, selected, _ = select_calibration(p, y, p[::-1], y[::-1])
    assert "isotonic" in selected.excluded_methods
    assert selected.fitting_source.startswith("earlier")


def test_isotonic_available_only_with_sufficient_calibration_frauds():
    p = np.linspace(0, 1, 1200)
    y = (p > 0.8).astype(int)
    _, choice, _ = select_calibration(p, y, p, y)
    assert "isotonic" in choice.candidates
    assert choice.candidates[choice.method].brier_score <= choice.candidates["none"].brier_score


def test_search_contains_default_and_weighted_candidates():
    config = ExperimentConfig()
    y = np.r_[np.zeros(98, dtype=int), np.ones(2, dtype=int)]
    for algorithm in ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"):
        candidates = search_parameters(algorithm, y, config)
        assert len(candidates) == 4
        assert candidates == search_parameters(algorithm, y, config)
        if algorithm == "XGBoost":
            assert {c["scale_pos_weight"] for c in candidates} >= {1.0, 49.0}
        else:
            assert any(c["class_weight"] is None for c in candidates)
            assert any(c["class_weight"] is not None for c in candidates)


def test_knn_benchmark_fallback_and_memory():
    rng = np.random.default_rng(42)
    train = sparse.csr_matrix(rng.normal(size=(100, 4)))
    validation = sparse.csr_matrix(rng.normal(size=(40, 4)))
    labels, val_labels = np.tile([0, 0, 0, 1], 25), np.tile([0, 0, 0, 1], 10)
    config = ExperimentConfig(knn_sample_size=50, knn_benchmark_queries=20, knn_max_pairs=1)
    rows, report = benchmark_knn(train, labels, validation, val_labels, config)
    assert len(rows) == 50 and report.constrained
    assert report.matrix_memory_bytes > 0 and report.full_pair_count == 4000
    rows, report = benchmark_knn(
        train,
        labels,
        validation,
        val_labels,
        config.model_copy(update={"knn_max_pairs": 100_000, "knn_max_projected_seconds": 1e6}),
    )
    assert len(rows) == 100 and not report.constrained


def test_representative_sample_determinism_and_both_classes():
    labels = np.r_[np.zeros(980, dtype=int), np.ones(20, dtype=int)]
    rows = representative_rows(labels, 100, 42)
    np.testing.assert_array_equal(rows, representative_rows(labels, 100, 42))
    assert labels[rows].sum() == 2


@pytest.fixture(scope="module")
def frozen_model_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("SYNTHETIC_MODEL_TEST_ONLY")
    source, root = directory / "data", directory / "artifacts"
    generated = generate(
        GeneratorConfig(
            transaction_count=1200,
            entity_count=24,
            merchant_count=24,
            fraud_prevalence=0.15,
            duration_days=10,
        )
    )
    generation = save_generated(generated, source)
    config = SyntheticConfig(
        transaction_path=source / "train_transaction.csv",
        identity_path=source / "train_identity.csv",
        metadata_path=source / "generation_metadata.json",
    )
    dataset, _, _, _ = load_dataset(config)
    features = build_features(dataset.frame, horizon_seconds=10 * 86400)
    specs = feature_manifest(features, dataset.frame.columns)
    temporal = make_split(dataset.frame, dataset.fingerprint, SplitConfig(), generated.evaluation)
    random = make_split(
        dataset.frame,
        dataset.fingerprint,
        SplitConfig(strategy="random_stratified"),
        generated.evaluation,
    )
    temporal_path = save_split(temporal, root / "splits")
    random_path = save_split(random, root / "splits")
    families = {}
    hashes = {}
    for family in ("linear_distance", "tree_boosting"):
        processor = Preprocessor(
            PreprocessingConfig(
                family=family, scaling="robust" if family == "linear_distance" else "none"
            )
        )
        processor.fit_training(features, temporal, specs)
        path = processor.save(root / "preprocessing")
        families[family] = {
            "preprocessing_artifact": path.name,
            "preprocessing_id": processor.preprocessing_id,
            "encoded_feature_names": processor.feature_names,
        }
        hashes[family] = file_hash(path)
    manifest_path = root / "features" / "feature_manifest.json"
    save_json(
        manifest_path,
        {
            "dataset_fingerprint": dataset.fingerprint,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "split_id": temporal.split_id,
            "features": [s.model_dump(mode="json") for s in specs],
            "families": families,
        },
    )
    # Only test constants are scoped to this synthetic fixture. CLI exposes no freeze override.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(verification, "FROZEN_FINGERPRINT", dataset.fingerprint)
        patch.setattr(verification, "FROZEN_SPLIT_ID", temporal.split_id)
        patch.setattr(input_module, "FROZEN_SPLIT_ID", temporal.split_id)
        patch.setattr(input_module, "FROZEN_MANIFEST_HASH", file_hash(manifest_path))
        patch.setattr(input_module, "FROZEN_SPLIT_HASH", file_hash(temporal_path))
        patch.setattr(input_module, "FROZEN_PREPROCESSOR_HASHES", hashes)
        patch.setattr(input_module, "FROZEN_RANDOM_ID", random.split_id)
        patch.setattr(input_module, "FROZEN_RANDOM_HASH", file_hash(random_path))
        yield config, root, generation, features, specs, temporal


@pytest.fixture(scope="module")
def trained_model_run(frozen_model_inputs):
    config, root, *_ = frozen_model_inputs
    experiment = ExperimentConfig(
        search_candidates=2, knn_sample_size=100, knn_benchmark_queries=20, knn_max_pairs=1
    )
    registry = run_experiment(config, root, experiment)
    return registry, experiment


def test_feature_target_and_evaluation_metadata_excluded(frozen_model_inputs):
    _, _, _, features, specs, _ = frozen_model_inputs
    for name in (
        "isFraud",
        "fraud_scenario",
        "hard_negative",
        "hard_negative_type",
        "behavioural_entity_id",
    ):
        modified = features.copy()
        modified[name] = 0
        with pytest.raises(DatasetError):
            validate_feature_inputs(modified, specs)
    leaked_specs = [s.model_copy(update={"source_columns": ["isFraud"]}) for s in specs]
    with pytest.raises(DatasetError):
        validate_feature_inputs(features, leaked_specs)


@pytest.mark.parametrize("changed", ["dataset", "manifest", "split", "preprocessor"])
def test_frozen_input_tampering_aborts(frozen_model_inputs, tmp_path, changed):
    config, root, *_ = frozen_model_inputs
    copied = tmp_path / "isolated"
    shutil.copytree(root.parent, copied)
    config = config.model_copy(
        update={
            name: copied / "data" / getattr(config, name).name
            for name in ("transaction_path", "identity_path", "metadata_path")
        }
    )
    new_root = copied / "artifacts"
    path = (
        config.transaction_path
        if changed == "dataset"
        else (
            new_root / "features" / "feature_manifest.json"
            if changed == "manifest"
            else new_root / "splits" / (frozen_model_inputs[-1].split_id[7:] + ".json")
            if changed == "split"
            else next((new_root / "preprocessing").glob("*.joblib"))
        )
    )
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(DatasetError):
        run_experiment(config, new_root, output_root=tmp_path / "output")
    assert not (tmp_path / "output" / "evaluation" / "test_evaluation_started.json").exists()


def test_gate_requires_freeze_and_prevents_second_access(tmp_path):
    called = []
    gate = TestGate(lambda: called.append("TEST") or "released")
    with pytest.raises(DatasetError):
        gate.release(tmp_path, "model", "hash")
    assert not called
    from datetime import UTC, datetime

    freeze = SelectionFreeze(
        model_id="model",
        threshold=0.25,
        calibration_method="none",
        artifact_hash="hash",
        run_id="run",
        frozen_at=datetime.now(UTC),
        ranking=["model"],
        rationale="validation only",
    )
    save_json(tmp_path / "selection_freeze.json", freeze)
    assert gate.release(tmp_path, "model", "hash") == "released"
    with pytest.raises(DatasetError):
        gate.release(tmp_path, "model", "hash")
    second_process = TestGate(lambda: called.append("SECOND"))
    with pytest.raises(DatasetError):
        second_process.release(tmp_path, "model", "hash")
    assert called == ["TEST"]


def test_inputs_do_not_release_test_before_model_selection(frozen_model_inputs):
    config, root, *_ = frozen_model_inputs
    data = ExperimentInputs(config, root, ExperimentConfig())
    assert data.gate.access_count == 0
    assert len(data.y_train) == len(data.train_rows)
    assert set(data.train_rows).isdisjoint(data.split.indexes["test"])
    assert set(data.validation_rows).isdisjoint(data.split.indexes["test"])
    assert max(data.calibration_positions) < min(data.selection_positions)


def test_registry_selection_serialization_reuse_and_final_test_once(
    frozen_model_inputs, trained_model_run
):
    config, root, *_ = frozen_model_inputs
    registry, experiment = trained_model_run
    selected = [m for m in registry.models if m.is_selected_supervised_model]
    assert len(selected) == 1
    assert selected[0].test_metrics.metrics.rows == len(frozen_model_inputs[-1].indexes["test"])
    assert all(
        m.test_metrics is None for m in registry.models if not m.is_selected_supervised_model
    )
    loaded = ModelRegistry.model_validate_json(
        (root / "models" / "model_registry.json").read_bytes()
    )
    assert loaded == registry
    before = file_hash(root / "evaluation" / "test_evaluation_started.json")
    reused = run_experiment(config, root, experiment)
    assert reused == registry
    assert before == file_hash(root / "evaluation" / "test_evaluation_started.json")
    assert (
        json.loads((root / "evaluation" / "final_test_metrics.json").read_bytes())[
            "test_gate_access_count"
        ]
        == 1
    )
    assert registry.selection_freeze.ranking == [m.model_id for m in ranking(registry.models)]


def test_registry_rejects_multiple_selections_and_changed_threshold(trained_model_run):
    registry, _ = trained_model_run
    value = registry.model_dump(mode="json")
    extra = next(m for m in value["models"] if not m["is_selected_supervised_model"])
    extra["is_selected_supervised_model"] = True
    with pytest.raises(ValidationError):
        ModelRegistry.model_validate_json(json.dumps(value))
    value = registry.model_dump(mode="json")
    next(m for m in value["models"] if m["is_selected_supervised_model"])["selected_threshold"] = (
        0.123
    )
    with pytest.raises(ValidationError):
        ModelRegistry.model_validate_json(json.dumps(value))


def test_classifier_reload_hash_and_probability(frozen_model_inputs, trained_model_run, tmp_path):
    _, root, *_ = frozen_model_inputs
    registry, experiment = trained_model_run
    selected = next(m for m in registry.models if m.is_selected_supervised_model)
    path = root / "models" / registry.provenance.run_id[7:] / selected.artifact_name
    bundle = load_classifier(
        path,
        selected.artifact_hash,
        registry.provenance.dataset_fingerprint,
        registry.provenance.feature_version,
    )
    data = ExperimentInputs(frozen_model_inputs[0], root, experiment)
    processor = next(
        p for p in data.processors.values() if p.preprocessing_id == bundle.preprocessing_id
    )
    p = bundle.probability(processor.transform(data.features.iloc[data.validation_rows]))
    assert len(p) == len(data.validation_rows) and np.isfinite(p).all()
    tampered = tmp_path / "tampered.joblib"
    tampered.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(DatasetError):
        load_classifier(
            tampered,
            selected.artifact_hash,
            registry.provenance.dataset_fingerprint,
            registry.provenance.feature_version,
        )


def test_no_test_labels_can_change_selection_ranking(trained_model_run):
    registry, _ = trained_model_run
    changed = registry.model_copy(deep=True)
    for model in changed.models:
        model.test_metrics = None
    assert [m.model_id for m in ranking(changed.models)] == registry.selection_freeze.ranking


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "/comparison",
        "/selected",
        "/unknown/metrics",
        "/unknown/scenario-recall",
        "/unknown/threshold-analysis",
    ],
)
def test_model_api_unconfigured(endpoint):
    with TestClient(create_app(Settings(_env_file=None, dataset_config_path=None))) as client:
        result = client.get("/api/v1/models" + endpoint)
        assert result.status_code == 200
        assert result.json() == {"status": "dataset_not_configured"}


def test_model_api_ready_privacy_and_invalid_ids(
    frozen_model_inputs, trained_model_run, monkeypatch
):
    from riskbyte_api.services import models as service

    config, root, *_ = frozen_model_inputs
    registry, _ = trained_model_run
    monkeypatch.setattr(service, "FROZEN_FINGERPRINT", registry.provenance.dataset_fingerprint)
    monkeypatch.setattr(service, "FROZEN_SPLIT_ID", registry.provenance.split_id)
    path = root.parent / "adapter.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        dataset_config_path=path,
        dataset_adapter="synthetic_riskbyte",
        model_artifacts_dir=root / "models",
    )
    with TestClient(create_app(settings)) as client:
        selected_id = registry.selection_freeze.model_id
        for endpoint in (
            "",
            "/comparison",
            "/selected",
            f"/{selected_id}/metrics",
            f"/{selected_id}/scenario-recall",
            f"/{selected_id}/threshold-analysis",
        ):
            result = client.get("/api/v1/models" + endpoint)
            assert result.status_code == 200, result.text
            assert result.json()["status"] == "models_ready"
            assert str(root) not in result.text and "NaN" not in result.text
        assert client.get("/api/v1/models/not-registered/metrics").status_code == 404
        schema = client.get("/openapi.json").json()
        assert "ModelComparison" in schema["components"]["schemas"]
        assert "post" in schema["paths"]["/api/v1/predict"]


def test_xgboost_missing_dependency_is_explicit(monkeypatch):
    from riskbyte_ml.models.supervised import estimators

    monkeypatch.setattr(estimators.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(DatasetError) as exc:
        make_estimator("XGBoost", {}, ExperimentConfig())
    assert exc.value.code == "model_dependency_unavailable"


def test_report_ready_tables_and_svg_are_real_artifacts(frozen_model_inputs, trained_model_run):
    import csv
    import xml.etree.ElementTree as xml

    _, root, *_ = frozen_model_inputs
    registry, _ = trained_model_run
    directory = root / "evaluation"
    for name in ("model_comparison", "hard_negative_performance", "scenario_recall", "timing"):
        with (directory / (name + ".csv")).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) > 1
    for path in directory.glob("*.svg"):
        assert xml.parse(path).getroot().tag.endswith("svg")
    report = (directory / "SUPERVISED_MODEL_EVALUATION.md").read_text(encoding="utf-8")
    assert registry.provenance.dataset_fingerprint in report
    assert "Synthetic-data findings" in report and "Accuracy" in report


def test_changed_completed_experiment_cannot_reopen_test(frozen_model_inputs, trained_model_run):
    config, root, *_ = frozen_model_inputs
    _, experiment = trained_model_run
    with pytest.raises(DatasetError) as exc:
        run_experiment(config, root, experiment.model_copy(update={"policy": "HIGH_RECALL"}))
    assert exc.value.code == "model_experiment_frozen"


def test_registry_rejects_model_provenance_mismatch(trained_model_run):
    registry, _ = trained_model_run
    value = registry.model_dump(mode="json")
    value["models"][0]["dataset_fingerprint"] = "changed"
    with pytest.raises(ValidationError):
        ModelRegistry.model_validate_json(json.dumps(value))


@pytest.mark.parametrize("changed", ["classifier", "selection", "final_test"])
def test_completed_artifact_tampering_rejected_without_test_reuse(
    frozen_model_inputs, trained_model_run, tmp_path, changed
):
    config, root, *_ = frozen_model_inputs
    registry, experiment = trained_model_run
    destination = tmp_path / "isolated"
    shutil.copytree(root.parent, destination)
    config = config.model_copy(
        update={
            name: destination / "data" / getattr(config, name).name
            for name in ("transaction_path", "identity_path", "metadata_path")
        }
    )
    copied = destination / "artifacts"
    if changed == "classifier":
        selected = next(m for m in registry.models if m.is_selected_supervised_model)
        path = copied / "models" / registry.provenance.run_id[7:] / selected.artifact_name
        with path.open("ab") as stream:
            stream.write(b"tampered")
    else:
        path = (
            copied
            / "evaluation"
            / ("selection_freeze.json" if changed == "selection" else "final_test_metrics.json")
        )
        value = json.loads(path.read_bytes())
        if changed == "selection":
            value["threshold"] = 0.123
        else:
            value["evaluation"]["metrics"]["recall"] = 0.123
        path.write_text(json.dumps(value), encoding="utf-8")
    marker = file_hash(copied / "evaluation" / "test_evaluation_started.json")
    with pytest.raises(DatasetError):
        run_experiment(config, copied, experiment)
    assert marker == file_hash(copied / "evaluation" / "test_evaluation_started.json")
