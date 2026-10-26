"""Small synthetic Phase 7 fixtures; no frozen transaction/test records are read."""

import builtins
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_ml.abrf.contracts import A_THRESHOLD, P_THRESHOLD, FusionConfig, FusionInput, Weights
from riskbyte_ml.abrf.engine import fuse
from riskbyte_ml.behavioural.deviation import BehaviourConfig, DeviationEngine
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability import inputs as input_module
from riskbyte_ml.explainability.engine import (
    ExplanationEngine,
    confidence_explanation,
    formula_trace,
    ranked_components,
)
from riskbyte_ml.explainability.inputs import equal_scores, verify_selection
from riskbyte_ml.explainability.metadata import encoded_metadata, top_features
from riskbyte_ml.explainability.pipeline import (
    case_selection,
    global_importance,
    save_json,
    verify_outputs,
)
from riskbyte_ml.explainability.schemas import (
    NON_CAUSAL,
    AnomalyExplanation,
    ExplanationBundle,
    SupervisedExplanation,
    UnifiedExplanation,
)
from riskbyte_ml.explainability.supervised import TreeShap, measure_latency
from riskbyte_ml.statistics.verification import file_hash


@pytest.fixture
def configuration():
    return FusionConfig(
        weights=Weights(wp=0.9, wa=0, wb=0.1),
        confidence_curve="gated_0.5",
        policy="BALANCED",
        decision_threshold=0.29755749437222373,
        risk_boundaries=[0.01, 0.1, 0.6, 0.9],
    )


def fusion_record(p=0.3, a=0.99, b=0.8, c=0.6):
    return FusionInput(
        transaction_id="synthetic-fixture",
        supervised_probability=p,
        supervised_prediction=p >= P_THRESHOLD,
        anomaly_score=a,
        anomaly_prediction=a >= A_THRESHOLD,
        behaviour_score=b,
        behaviour_available=b is not None,
        behaviour_confidence=c,
    )


@pytest.mark.parametrize("confidence", [0, 0.49, 0.5, 0.8, 1])
def test_exact_formula_and_zero_anomaly(configuration, confidence):
    record = fusion_record(c=confidence)
    trace = fuse(record, configuration)
    formula = formula_trace(record, configuration)
    assert sum(formula.unnormalized_contributions.values()) / formula.denominator == pytest.approx(
        trace.final_risk_score, abs=1e-15
    )
    assert trace.weighted_contributions["anomaly"] == 0
    assert trace.effective_weights["anomaly"] == 0
    assert formula.transformed_confidence == (confidence if confidence >= 0.5 else 0)
    assert trace.investigation_flag == (trace.final_risk_score >= configuration.decision_threshold)


def test_unavailable_behaviour_formula(configuration):
    trace = fuse(fusion_record(b=None, c=0), configuration)
    assert trace.final_risk_score == 0.3
    assert trace.effective_weights == {"supervised": 1, "anomaly": 0, "behaviour": 0}


@pytest.fixture
def metadata():
    return [
        {
            "feature": name,
            "display_name": name,
            "source_feature": name,
            "interpretability": kind,
            "manifest_interpretability": kind,
            "source_columns": [name],
            "feature_family": name,
            "raw_or_derived": "raw",
        }
        for name, kind in [
            ("TransactionAmt", "INTERPRETABLE"),
            ("V7", "ANONYMIZED"),
            ("device_id", "PARTIALLY_INTERPRETABLE"),
        ]
    ]


def test_top_k_direction_and_determinism(metadata):
    p, n = top_features([5, 6, 7], [0.2, -0.1, 0.1], metadata, 1)
    assert [f.feature for f in p] == ["TransactionAmt"]
    assert [f.feature for f in n] == ["V7"]
    assert p[0].direction == "increases_classifier_risk"
    assert n[0].direction == "decreases_classifier_risk"
    assert [p, n] == top_features([5, 6, 7], [0.2, -0.1, 0.1], metadata, 1)
    assert "semantics are unavailable" in n[0].statement
    assert "suspicious card" not in n[0].statement


def test_zero_shap_has_no_direction(metadata):
    assert top_features([1, 2, 3], [0, 0, 0], metadata) == [[], []]


@pytest.mark.parametrize("k", [0, 21])
def test_top_k_invalid(metadata, k):
    with pytest.raises(ValueError):
        top_features([1, 2, 3], [0.1, 0.2, 0.3], metadata, k)


def test_feature_manifest_mapping_and_alignment():
    raw = {
        "name": "elapsed_day_index",
        "source_columns": ["TransactionDT"],
        "kind": "derived",
        "interpretability_class": "TEMPORAL",
        "transformation": "elapsed seconds only",
    }
    encoded = {**raw, "name": "elapsed_day_index__missing", "kind": "encoded"}
    manifest = {"features": [raw], "families": {"tree_boosting": {"feature_manifest": [encoded]}}}
    result = encoded_metadata(manifest, [encoded["name"]])[0]
    assert result["manifest_interpretability"] == "TEMPORAL"
    assert result["interpretability"] == "INTERPRETABLE"
    assert result["raw_or_derived"] == "derived"
    assert result["source_columns"] == ["TransactionDT"]
    with pytest.raises(ValueError):
        encoded_metadata(manifest, ["wrong_order"])


@pytest.mark.parametrize(
    "name",
    [
        "isFraud",
        "fraud_scenario",
        "hard_negative",
        "hard_negative_type",
        "behavioural_entity_id",
        "TransactionID",
    ],
)
def test_excluded_feature_metadata(name):
    spec = {"name": name, "kind": "raw", "interpretability_class": "INTERPRETABLE"}
    manifest = {"features": [spec], "families": {"tree_boosting": {"feature_manifest": [spec]}}}
    with pytest.raises(ValueError):
        encoded_metadata(manifest, [name])


@pytest.fixture
def tree():
    return DecisionTreeClassifier(max_depth=2, random_state=42).fit(
        np.array([[0, 0], [1, 0], [2, 0], [3, 0]]), [0, 0, 1, 1]
    )


def test_shap_missing_dependency_is_explicit(monkeypatch, tree):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "shap":
            raise ModuleNotFoundError("Synthetic blocked dependency fixture")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    adapter = TreeShap(tree, ["amount", "unused"])
    assert adapter.status == "UNAVAILABLE_WITH_REASON"
    assert adapter.reason == "shap_dependency_unavailable"
    assert adapter.values(csr_matrix([[3, 0]])) is None
    assert adapter.initialization_seconds is None


@pytest.mark.parametrize("format_", ["list", "array"])
def test_shap_output_shapes_additivity_and_stability(monkeypatch, tree, format_):
    """Adapter-only synthetic decomposition fixture, never used as application SHAP values."""

    class SyntheticExplainer:
        expected_value = [0.5, 0.5]

        def __init__(self, model, **kwargs):
            assert kwargs["feature_perturbation"] == "tree_path_dependent"
            assert kwargs["model_output"] == "raw"
            self.model = model

        def shap_values(self, matrix, **kwargs):
            positive = np.column_stack(
                [self.model.predict_proba(matrix)[:, 1] - 0.5, np.zeros(len(matrix))]
            )
            return (
                [-positive, positive] if format_ == "list" else np.stack([-positive, positive], 2)
            )

    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(TreeExplainer=SyntheticExplainer, __version__="synthetic-test-fixture"),
    )
    adapter = TreeShap(tree, ["amount", "unused"])
    values, base, residual = adapter.values(csr_matrix([[3, 0], [0, 0]]))
    assert values.shape == (2, 2)
    assert base == 0.5
    assert np.allclose(residual, 0)
    assert np.array_equal(values[:1], adapter.values(csr_matrix([[3, 0.1]]))[0])
    with pytest.raises(ValueError):
        adapter.values(csr_matrix([[3]]))


def test_genuine_tree_shap_when_dependency_available(tree):
    import shap

    assert shap.__version__ == "0.53.0"
    adapter = TreeShap(tree, ["amount", "unused"])
    matrix = csr_matrix([[3, 0], [0, 0]])
    values, base, residual = adapter.values(matrix)
    assert values.shape == (2, 2)
    assert np.allclose(base + values.sum(1), tree.predict_proba(matrix)[:, 1])
    assert np.allclose(residual, 0)


def test_tree_alignment(tree):
    with pytest.raises(ValueError):
        TreeShap(tree, ["one"])


@pytest.fixture
def engine_inputs(configuration):
    features = pd.DataFrame(
        {
            "behavioural_history_count": [0, 5, 50],
            "amount_ratio_to_prior_mean": [np.nan, 1, 5],
            "amount_zscore_prior": [np.nan, 0, 6],
            "transactions_prior_30m": [0, 1, 7],
            "transactions_prior_24h": [0, 2, 10],
            "time_since_previous_transaction": [np.nan, 500, 4],
            "failed_attempts_before": [0, 0, 3],
            "merchant_seen_before": [np.nan, 1, 0],
            "device_seen_before": [np.nan, 1, 0],
            "billing_region_seen_before": [np.nan, 1, 0],
            "product_seen_before": [np.nan, 1, 0],
            "merchant_novelty_available": [False, True, True],
            "device_novelty_available": [False, True, True],
        }
    )
    b_engine = DeviationEngine.fit(features)
    b, c, _, _, _ = b_engine.score_arrays(features)
    scores = pd.DataFrame(
        {
            "TransactionID": ["1", "2", "3"],
            "P": [0.1, 0.2, 0.5],
            "A": [0.1, 0.6, 0.99],
            "B": b,
            "behaviour_confidence": c,
        }
    )
    saved = []
    for _, row in scores.iterrows():
        trace = fuse(
            FusionInput(
                transaction_id=row.TransactionID,
                supervised_probability=row.P,
                supervised_prediction=row.P >= P_THRESHOLD,
                anomaly_score=row.A,
                anomaly_prediction=row.A >= A_THRESHOLD,
                behaviour_score=float(row.B) if np.isfinite(row.B) else None,
                behaviour_available=bool(np.isfinite(row.B)),
                behaviour_confidence=row.behaviour_confidence,
            ),
            configuration,
        )
        saved.append(
            {
                "final_risk_score": trace.final_risk_score,
                "risk_level": trace.risk_level,
                "investigation_flag": trace.investigation_flag,
                "reason_codes": ";".join(trace.reason_codes),
                **{"effective_weight_" + k: v for k, v in trace.effective_weights.items()},
                **{"contribution_" + k: v for k, v in trace.weighted_contributions.items()},
            }
        )
    return SimpleNamespace(
        features=features,
        context=pd.DataFrame({"device_id": [pd.NA, "known", "new"]}),
        behaviour=b_engine,
        scores=scores,
        fusion_scores=pd.DataFrame(saved),
        bundle=SimpleNamespace(selected=configuration),
        shap=SimpleNamespace(
            values=lambda x: None,
            status="UNAVAILABLE_WITH_REASON",
            reason="shap_dependency_unavailable",
        ),
        matrix=csr_matrix(np.zeros((3, 2))),
        feature_metadata=[],
        raw_probability=np.array([0.15, 0.25, 0.8]),
        classifier=SimpleNamespace(model_version="synthetic-fixture", model_id="synthetic-tree"),
        provenance={"calibration_version": "synthetic-sigmoid"},
        original_anomaly=np.array([-0.2] * 3),
        anomaly=SimpleNamespace(
            version="synthetic-if", preparation=SimpleNamespace(names=list(features.columns))
        ),
        labels=np.array([0, 0, 1]),
    )


def test_unified_explanation_determinism_and_calibration_distinction(engine_inputs):
    engine = ExplanationEngine(engine_inputs)
    explanation = engine.explain(2)
    assert explanation == engine.explain(2)
    assert explanation.supervised.raw_probability != explanation.supervised.calibrated_probability
    assert explanation.supervised.explained_model == "underlying_decision_tree"
    assert not explanation.supervised.calibrated_probability_shap_explained
    assert explanation.anomaly.diagnostic_only
    assert not explanation.anomaly.influenced_final_risk
    assert explanation.anomaly.effective_abrf_weight == 0
    assert UnifiedExplanation.model_validate_json(explanation.model_dump_json()) == explanation
    assert any(NON_CAUSAL == notice for notice in explanation.limitations)


@pytest.mark.parametrize("field", ["isFraud", "fraud_scenario", "hard_negative"])
def test_ground_truth_exclusion(engine_inputs, field):
    engine = ExplanationEngine(engine_inputs)
    explanation = engine.explain(2)
    engine_inputs.labels[:] = 0
    assert engine.explain(2) == explanation
    with pytest.raises(ValidationError):
        UnifiedExplanation.model_validate({**explanation.model_dump(), field: "not a live input"})


@pytest.mark.parametrize("position", [-1, 3])
def test_validation_only_bounds(engine_inputs, position):
    with pytest.raises(ValueError):
        ExplanationEngine(engine_inputs).explain(position)


def test_cold_start_and_identity_confidence(engine_inputs):
    first = ExplanationEngine(engine_inputs).explain(0)
    assert first.behaviour.behaviour_score is None
    assert first.behaviour.behaviour_confidence == 0
    assert first.behaviour.history_count == 0
    assert not first.behaviour.confidence_explanation.current_identity_available
    assert first.fusion.effective_weights["behaviour"] == 0
    assert "LIMITED_BEHAVIOURAL_HISTORY" in first.fusion.reason_codes


def test_component_ranking_and_confidence(engine_inputs):
    result = engine_inputs.behaviour.explain(engine_inputs.features, 2)
    components = ranked_components(result.components)
    assert sum(x.contribution for x in components) == pytest.approx(result.behaviour_score)
    assert components == sorted(components, key=lambda c: (-c.contribution, c.signal))
    confidence = confidence_explanation(
        result, engine_inputs.features.iloc[2], engine_inputs.context.iloc[2], BehaviourConfig()
    )
    assert confidence.prior_transaction_count == 50
    assert confidence.device_history_available
    assert confidence.merchant_history_available
    assert not confidence.history_duration_used


@pytest.mark.parametrize(
    "field", ["final_risk_score", "risk_level", "investigation_flag", "reason_codes"]
)
def test_fusion_inconsistency_fails(engine_inputs, field):
    engine_inputs.fusion_scores.loc[2, field] = {
        "final_risk_score": 0.01,
        "risk_level": "LOW",
        "investigation_flag": False,
        "reason_codes": "INVENTED_REASON",
    }[field]
    with pytest.raises((DatasetError, ValueError)):
        ExplanationEngine(engine_inputs).explain(2)


def test_unavailable_schema_rejects_fake_shap():
    with pytest.raises(ValidationError):
        SupervisedExplanation(
            model_version="fixture",
            model_id="fixture",
            raw_probability=0.3,
            calibrated_probability=0.2,
            calibration_version="fixture",
            threshold=0.4,
            prediction=False,
            shap_status="UNAVAILABLE_WITH_REASON",
            unavailable_reason="blocked",
            base_value=0.5,
        )


def test_anomaly_schema_cannot_claim_influence():
    with pytest.raises(ValidationError):
        AnomalyExplanation(
            version="fixture",
            score=0.99,
            original_model_score=-0.8,
            threshold=0.9,
            flag=True,
            effective_abrf_weight=0.1,
            influenced_final_risk=True,
            observable_signals=[],
        )


def test_global_unavailable_no_fake_importance(engine_inputs):
    reports = global_importance(engine_inputs)
    assert all(r["importance"] == [] for r in reports)
    assert all(r["status"] == "UNAVAILABLE_WITH_REASON" for r in reports)


def test_case_selection_deterministic_and_absence():
    meta = pd.DataFrame(
        {
            "fraud_scenario": ["ACCOUNT_TAKEOVER", "", ""],
            "hard_negative": [False] * 3,
            "hard_negative_type": [""] * 3,
        }
    )
    scores = pd.DataFrame(
        {
            "investigation_flag": [True, False, False],
            "supervised_prediction": [True, False, False],
            "final_risk_score": [0.9, 0.2, 0.1],
        }
    )
    selected = case_selection([1, 0, 0], meta, scores)
    assert selected == case_selection([1, 0, 0], meta, scores)
    assert selected["normal_legitimate_low_risk"] == 2
    assert selected["recovered_low_and_slow"] is None


def test_frozen_selection_mismatch_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(input_module, "verify_components", lambda *a: (None, None, {}))
    monkeypatch.setattr(input_module, "file_hash", lambda p: "sha256:changed")
    with pytest.raises(DatasetError, match="selection changed"):
        verify_selection(None, tmp_path)


def test_numeric_consistency_check():
    equal_scores([np.nan, 0.1], [np.nan, 0.1], "fixture")
    with pytest.raises(DatasetError):
        equal_scores([0.1], [0.11], "fixture")


def test_artifact_round_trip_exclusive_and_integrity(tmp_path):
    path = tmp_path / "fixture.json"
    save_json(path, {"actual": 0.3})
    bundle = SimpleNamespace(version="explainability-v1", artifacts={path.name: file_hash(path)})
    verify_outputs(tmp_path, bundle)
    with pytest.raises(DatasetError):
        save_json(path, {"actual": 0.5})
    path.write_text(json.dumps({"tampered": True}))
    with pytest.raises(DatasetError):
        verify_outputs(tmp_path, bundle)


def test_latency_utility_measures_calls():
    calls = []
    result = measure_latency(lambda: calls.append(True), repeats=3)
    assert len(calls) == 3
    assert result["minimum_seconds"] <= result["median_seconds"] <= result["maximum_seconds"]
    with pytest.raises(ValueError):
        measure_latency(lambda: None, 0)


@pytest.mark.parametrize("flag", [True, None])
def test_isolation_audit_required(flag):
    with pytest.raises(ValidationError):
        ExplanationBundle(
            generated_at="synthetic-fixture",
            provenance={},
            shap_status="UNAVAILABLE_WITH_REASON",
            artifacts={},
            performance={},
            isolation_audit={
                "original_temporal_test_accessed": flag,
                "future_holdout_generated": False,
            },
        )


@pytest.mark.parametrize(
    "endpoint",
    ["status", "global", "families", "case-studies", "validation-errors", "validation/1"],
)
def test_api_not_configured_and_openapi(endpoint, monkeypatch):
    monkeypatch.setattr(
        input_module, "DevelopmentInputs", lambda *a: pytest.fail("No HTTP scoring")
    )
    client = TestClient(create_app(Settings(_env_file=None)))
    response = client.get("/api/v1/explainability/" + endpoint)
    assert response.status_code == 200
    assert response.json()["status"] == "dataset_not_configured"
    assert (
        "/api/v1/explainability/validation/{transaction_id}"
        in client.get("/openapi.json").json()["paths"]
    )


def test_tampered_unified_schema_rejected(engine_inputs):
    explanation = ExplanationEngine(engine_inputs).explain(2).model_dump()
    bad = copy.deepcopy(explanation)
    bad["risk_summary"]["final_risk_score"] = 0.99
    with pytest.raises(ValidationError):
        UnifiedExplanation.model_validate(bad)


@pytest.fixture
def research_client(tmp_path, monkeypatch, engine_inputs):
    """Synthetic offline API snapshot; no real sources/model deserialization."""
    import riskbyte_api.services.explainability as service

    monkeypatch.setattr(service, "load_synthetic_config", lambda *a: None)
    monkeypatch.setattr(service, "verify_selection", lambda *a: (None, None, {}))
    record = {
        "status": "AVAILABLE",
        "case": "synthetic-fixture",
        "explanation": ExplanationEngine(engine_inputs).explain(2).model_dump(mode="json"),
    }
    reports = {
        "case_studies.json": {"cases": [record]},
        "hard_negative_explanations.json": {"summary": {}, "records": []},
        "false_negative_explanations.json": {"summary": {}, "records": []},
        "global_feature_importance.json": {"status": "UNAVAILABLE_WITH_REASON", "importance": []},
        "feature_family_importance.json": {"status": "UNAVAILABLE_WITH_REASON", "importance": []},
    }
    hashes = {}
    for name, value in reports.items():
        save_json(tmp_path / name, value)
        hashes[name] = file_hash(tmp_path / name)
    bundle = ExplanationBundle(
        generated_at="synthetic-fixture",
        provenance={},
        shap_status="UNAVAILABLE_WITH_REASON",
        artifacts=hashes,
        performance={},
        isolation_audit={
            "original_temporal_test_accessed": False,
            "future_holdout_generated": False,
        },
    )
    save_json(tmp_path / "analysis_bundle.json", bundle.model_dump(mode="json"))
    settings = Settings(
        _env_file=None,
        dataset_config_path="synthetic-fixture.json",
        dataset_adapter="synthetic_riskbyte",
        explainability_artifacts_dir=tmp_path,
    )
    return TestClient(create_app(settings)), tmp_path


@pytest.mark.parametrize(
    "endpoint",
    ["status", "global", "families", "case-studies", "validation-errors", "validation/3"],
)
def test_api_published_snapshot(research_client, endpoint):
    client, _ = research_client
    response = client.get("/api/v1/explainability/" + endpoint)
    assert response.status_code == 200
    assert response.json()["status"] == "explainability_partial"
    if endpoint == "validation/3":
        payload = response.json()["data"]
        UnifiedExplanation.model_validate(payload)
        assert not ({"isFraud", "fraud_scenario", "hard_negative"} & set(payload))


def test_api_no_live_scoring_and_missing_transaction(research_client):
    client, _ = research_client
    response = client.get("/api/v1/explainability/validation/not-precomputed")
    assert response.status_code == 404
    assert "No offline validation explanation" in response.text


def test_api_tampering_is_structured_error(research_client):
    client, directory = research_client
    (directory / "global_feature_importance.json").write_text("{}")
    response = client.get("/api/v1/explainability/global")
    assert response.status_code == 422
    assert "explanation_artifact_changed" in response.text


def test_api_not_generated(research_client):
    client, directory = research_client
    (directory / "analysis_bundle.json").unlink()
    response = client.get("/api/v1/explainability/status")
    assert response.status_code == 200
    assert response.json()["status"] == "explainability_not_generated"


@pytest.mark.parametrize("field", ["denominator", "transformed_confidence"])
def test_formula_schema_tampering_rejected(engine_inputs, field):
    payload = ExplanationEngine(engine_inputs).explain(2).model_dump()
    payload["formula_trace"][field] = 0.2
    with pytest.raises(ValidationError):
        UnifiedExplanation.model_validate(payload)


def test_explanation_latency_no_negative_repeats():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, -1)


def test_invalid_shap_additivity_aborts(monkeypatch, tree):
    class IncorrectSyntheticExplainer:
        expected_value = [0.5, 0.5]

        def __init__(self, *args, **kwargs):
            pass

        def shap_values(self, matrix, **kwargs):
            return np.zeros((len(matrix), matrix.shape[1], 2))

    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(
            TreeExplainer=IncorrectSyntheticExplainer, __version__="synthetic-invalid-fixture"
        ),
    )
    with pytest.raises(ValueError, match="raw tree probability"):
        TreeShap(tree, ["amount", "unused"]).values(csr_matrix([[3, 0]]))


def test_global_grouping_and_complete_values_serialization(tmp_path, metadata):
    """Only synthetic known attribution arrays are supplied to this aggregation unit test."""
    values = np.array([[0.2, -0.1, 0], [-0.4, 0.1, 0.3]])
    fixture = SimpleNamespace(
        names=[m["feature"] for m in metadata],
        feature_metadata=metadata,
        features=pd.DataFrame(index=range(2)),
        matrix=csr_matrix(np.ones((2, 3))),
        scores=pd.DataFrame({"TransactionID": ["fixture-1", "fixture-2"]}),
        shap=SimpleNamespace(status="AVAILABLE", values=lambda matrix: (values, 0.5, np.zeros(2))),
    )
    feature, grouped = global_importance(fixture, batch_size=2, output=tmp_path)
    assert feature["importance"][0]["mean_absolute_shap"] == pytest.approx(0.3)
    assert feature["importance"][0]["feature"] == "TransactionAmt"
    assert sum(r["summed_mean_absolute_shap"] for r in grouped["importance"]) == pytest.approx(0.55)
    with np.load(tmp_path / "validation_shap_values.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["shap_values"], values)
        assert saved["feature_names"].tolist() == fixture.names


def test_api_component_root_is_not_dataset_report_directory(research_client, monkeypatch):
    import riskbyte_api.services.explainability as service

    client, directory = research_client
    component_root = directory / "component-artifacts"
    client.app.state.settings.abrf_artifacts_dir = component_root / "abrf"
    client.app.state.settings.dataset_artifacts_dir = directory / "adapter-reports"

    def verify(config, root):
        assert root == component_root
        return None, None, {}

    monkeypatch.setattr(service, "verify_selection", verify)
    assert client.get("/api/v1/explainability/status").status_code == 200


def test_api_available_shap_snapshot(research_client):
    """Synthetic API payload verifies serving real-shaped positive/negative attributions."""
    from riskbyte_ml.explainability.schemas import SHAP_VERSION, FeatureExplanation

    client, directory = research_client
    cases_path = directory / "case_studies.json"
    cases = json.loads(cases_path.read_text())
    explanation = cases["cases"][0]["explanation"]
    explanation["version"] = SHAP_VERSION
    supervised = explanation["supervised"]
    supervised.update(
        shap_status="AVAILABLE", unavailable_reason=None, base_value=0.5, additivity_residual=0.0
    )
    feature = FeatureExplanation(
        feature="synthetic_amount",
        display_name="Synthetic fixture amount",
        feature_value=10.0,
        shap_value=0.3,
        direction="increases_classifier_risk",
        interpretability="INTERPRETABLE",
        manifest_interpretability="INTERPRETABLE",
        source_columns=["synthetic_amount"],
        feature_family="transaction_amount",
        raw_or_derived="raw",
        statement="Synthetic fixture",
    )
    supervised["top_positive_features"] = [feature.model_dump()]
    cases_path.write_text(json.dumps(cases))
    (directory / "global_feature_importance.json").write_text(
        json.dumps(
            {
                "status": "AVAILABLE",
                "importance": [{"feature": "synthetic_amount", "mean_absolute_shap": 0.3}],
            }
        )
    )
    bundle_path = directory / "analysis_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle.update(version=SHAP_VERSION, shap_status="AVAILABLE")
    from riskbyte_ml.explainability.completion import BASELINE_HASH
    from riskbyte_ml.explainability.freeze import freeze_snapshot

    meta_path = directory / "explainer_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "non_shap_evidence_preserved": True,
                "historical_snapshot_hash": BASELINE_HASH,
                "shap_library_version": "0.53.0",
                "explainer_type": "TreeExplainer/tree_path_dependent/raw",
                "explained_model": "underlying_decision_tree",
            }
        )
    )
    bundle["artifacts"][meta_path.name] = file_hash(meta_path)
    for name in bundle["artifacts"]:
        bundle["artifacts"][name] = file_hash(directory / name)
    bundle_path.write_text(json.dumps(bundle))
    freeze_snapshot(directory, ExplanationBundle.model_validate(bundle))
    global_response = client.get("/api/v1/explainability/global")
    assert global_response.json()["status"] == "explainability_ready"
    assert global_response.json()["data"]["importance"][0]["mean_absolute_shap"] == 0.3
    local = client.get("/api/v1/explainability/validation/3")
    assert local.status_code == 200
    assert local.json()["data"]["supervised"]["top_positive_features"][0]["shap_value"] == 0.3
    assert local.json()["data"]["anomaly"]["effective_abrf_weight"] == 0
    status = client.get("/api/v1/explainability/status").json()["data"]
    assert status["shap_version"] == "0.53.0"
    assert status["explained_model"] == "underlying_decision_tree"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_feature_explanation_numbers(metadata, value):
    with pytest.raises(ValidationError):
        top_features([value, 2, 3], [0.1, 0.2, -0.1], metadata)
