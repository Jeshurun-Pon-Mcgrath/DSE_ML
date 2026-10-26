"""Phase 7.1 synthetic adapter fixtures, without any real dataset/test partition I/O."""

import copy

import numpy as np
import pytest
import shap
from pydantic import ValidationError

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability.completion import (
    CASE_IDS,
    baseline_reports,
    non_shap_evidence,
    shap_summary,
    verify_preserved,
)
from riskbyte_ml.explainability.schemas import SHAP_VERSION, ExplanationBundle
from riskbyte_ml.explainability.supervised import normalize_output


@pytest.mark.parametrize("layout", ["list", "2d_scalar", "2d_vector", "3d"])
def test_output_compatibility(layout):
    values = np.array([[0.2, -0.1], [-0.2, 0.3]])
    output = (
        [-values, values]
        if layout == "list"
        else (np.stack([-values, values], 2) if layout == "3d" else values)
    )
    expected = 0.5 if layout == "2d_scalar" else [0.5, 0.5]
    actual, base = normalize_output(output, expected, (2, 2))
    assert np.array_equal(actual, values)
    assert base == 0.5


@pytest.mark.parametrize(
    "output,expected",
    [
        (np.zeros((2, 3)), [0.5, 0.5]),
        (np.full((2, 2), np.nan), [0.5, 0.5]),
        (np.zeros((2, 2, 1)), [0.5, 0.5]),
        (np.zeros((2, 2)), [np.inf]),
        (np.zeros((2, 2)), [0.1, 0.2, 0.3]),
    ],
)
def test_invalid_layout_or_expected_value(output, expected):
    with pytest.raises(ValueError):
        normalize_output(output, expected, (2, 2))


def test_genuine_installed_output_semantics():
    from sklearn.tree import DecisionTreeClassifier

    from riskbyte_ml.explainability.supervised import TreeShap

    model = DecisionTreeClassifier(random_state=42).fit(
        [[0, 0], [1, 0], [2, 0], [3, 0]], [0, 0, 1, 1]
    )
    assert shap.__version__ == "0.53.0"
    adapter = TreeShap(model, ["amount", "unused"])
    matrix = np.array([[0, 0], [3, 0]])
    raw = adapter.explainer.shap_values(matrix)
    assert np.asarray(raw).shape == (2, 2, 2)
    values, base, residual = adapter.values(matrix)
    assert adapter.observed_layout == "3D"
    assert np.allclose(values.sum(1) + base, model.predict_proba(matrix)[:, 1])
    assert np.max(np.abs(residual)) <= 1e-7


def test_completed_snapshot_requires_available_shap():
    with pytest.raises(ValidationError):
        ExplanationBundle(
            version=SHAP_VERSION,
            generated_at="synthetic-fixture",
            provenance={},
            shap_status="UNAVAILABLE_WITH_REASON",
            artifacts={},
            performance={},
            isolation_audit={
                "original_temporal_test_accessed": False,
                "future_holdout_generated": False,
            },
        )


def synthetic_record(id_, scenario="VELOCITY_BURST"):
    return {
        "evaluation_only": {"fraud_scenario": scenario, "hard_negative_type": ""},
        "explanation": {
            "transaction_id": id_,
            "version": SHAP_VERSION,
            "behaviour": {"score": 0.4},
            "anomaly": {"weight": 0},
            "supervised": {
                "raw_probability": 0.2,
                "shap_status": "AVAILABLE",
                "top_positive_features": [{"feature": "velocity", "shap_value": 0.1}],
                "top_negative_features": [{"feature": "amount", "shap_value": -0.3}],
            },
        },
    }


def test_metadata_only_merge_preserves_evidence():
    old = synthetic_record("synthetic-1")
    new = copy.deepcopy(old)
    new["explanation"]["supervised"]["shap_status"] = "UNAVAILABLE_WITH_REASON"
    new["explanation"]["supervised"]["top_positive_features"] = []
    assert non_shap_evidence(old) == non_shap_evidence(new)
    new["explanation"]["anomaly"]["weight"] = 0.1
    assert non_shap_evidence(old) != non_shap_evidence(new)


def test_false_negative_and_hard_negative_summaries():
    records = [synthetic_record("synthetic-1"), synthetic_record("synthetic-2", "LOW_AND_SLOW")]
    report = shap_summary(records)
    assert report["groups"]["ALL"]["positive"][0]["top_k_occurrences"] == 2
    assert report["groups"]["VELOCITY_BURST"]["negative"][0]["feature"] == "amount"
    assert report["groups"]["LOW_AND_SLOW"]["records"] == 1


def test_frozen_case_ids_and_baseline_hash(monkeypatch, tmp_path):
    import riskbyte_ml.explainability.completion as completion

    monkeypatch.setattr(completion, "file_hash", lambda p: "sha256:changed")
    with pytest.raises(DatasetError, match="snapshot differs"):
        baseline_reports(tmp_path)
    assert list(CASE_IDS.values()) == ["2845", "69117", "58423", "14302", "86722", "10604"]


def test_preservation_rejects_changed_cases():
    old = {
        "case_studies.json": {"cases": [synthetic_record("synthetic-1")]},
        "hard_negative_explanations.json": {"records": [], "summary": {}},
        "false_negative_explanations.json": {"records": [], "summary": {}},
    }
    with pytest.raises(DatasetError, match="evidence changed"):
        verify_preserved(
            old, [synthetic_record("different-case")], {"records": []}, {"records": []}
        )


def test_snapshot_freeze_serialization_and_tampering(tmp_path):
    import json

    from riskbyte_ml.explainability.completion import BASELINE_HASH
    from riskbyte_ml.explainability.freeze import freeze_snapshot, verify_freeze

    bundle = ExplanationBundle(
        version=SHAP_VERSION,
        generated_at="synthetic-fixture",
        provenance={},
        shap_status="AVAILABLE",
        artifacts={},
        performance={},
        isolation_audit={
            "original_temporal_test_accessed": False,
            "future_holdout_generated": False,
        },
    )
    (tmp_path / "analysis_bundle.json").write_text(bundle.model_dump_json())
    (tmp_path / "explainer_metadata.json").write_text(
        json.dumps(
            {
                "non_shap_evidence_preserved": True,
                "historical_snapshot_hash": BASELINE_HASH,
            }
        )
    )
    freeze_snapshot(tmp_path, bundle)
    verify_freeze(tmp_path, bundle)
    with pytest.raises(DatasetError):
        freeze_snapshot(tmp_path, bundle)
    path = tmp_path / "snapshot_freeze.json"
    contents = json.loads(path.read_text())
    contents["bundle_hash"] = "sha256:tampered"
    path.write_text(json.dumps(contents))
    with pytest.raises(DatasetError):
        verify_freeze(tmp_path, bundle)
