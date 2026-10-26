"""Synthetic, offline Phase 6 fixtures; never open original dataset/test partitions."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_ml.abrf import pipeline
from riskbyte_ml.abrf.contracts import (
    A_THRESHOLD,
    P_THRESHOLD,
    Evidence,
    FusionConfig,
    FusionInput,
    SearchConfig,
    Weights,
)
from riskbyte_ml.abrf.engine import fuse, fusion_arrays
from riskbyte_ml.abrf.inputs import safe_child
from riskbyte_ml.abrf.search import candidates, eligible_constraint, rank_candidate, threshold_curve
from riskbyte_ml.data.errors import DatasetError


def configuration(**updates):
    values = {
        "weights": Weights(wp=0.5, wa=0.2, wb=0.3),
        "confidence_curve": "linear",
        "policy": "BALANCED",
        "decision_threshold": 0.6,
        "risk_boundaries": [0.1, 0.3, 0.65, 0.9],
    }
    values.update(updates)
    return FusionConfig(**values)


def record(p=0.7, a=0.8, b=0.9, c=0.5, **extra):
    return FusionInput(
        transaction_id="synthetic-fixture",
        supervised_probability=p,
        supervised_prediction=bool(p >= P_THRESHOLD),
        anomaly_score=a,
        anomaly_prediction=bool(a >= A_THRESHOLD),
        behaviour_score=b,
        behaviour_available=b is not None,
        behaviour_confidence=c,
        **extra,
    )


@pytest.mark.parametrize(
    "weights",
    [
        (-0.1, 0.5, 0.6),
        (0.2, 0.2, 0.2),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (np.nan, 0.5, 0.5),
        (np.inf, 0.0, 0.0),
    ],
)
def test_invalid_weights(weights):
    with pytest.raises(ValidationError):
        Weights(wp=weights[0], wa=weights[1], wb=weights[2])


@pytest.mark.parametrize("w", [(0.5, 0.2, 0.3), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.8, 0.1)])
def test_weight_sums(w):
    weights = Weights(wp=w[0], wa=w[1], wb=w[2])
    assert sum(weights.model_dump().values()) == pytest.approx(1)


def test_formula_confidence_and_contribution_accounting():
    result = fuse(record(), configuration())
    assert result.final_risk_score == pytest.approx(
        (0.5 * 0.7 + 0.2 * 0.8 + 0.3 * 0.5 * 0.9) / (0.5 + 0.2 + 0.3 * 0.5)
    )
    assert sum(result.effective_weights.values()) == pytest.approx(1)
    assert sum(result.weighted_contributions.values()) == pytest.approx(result.final_risk_score)


@pytest.mark.parametrize("c", [0.0, 1.0])
def test_confidence_endpoints(c):
    result = fuse(record(c=c), configuration())
    assert result.effective_weights["behaviour"] == pytest.approx(0.3 * c / (0.7 + 0.3 * c))


def test_unavailable_behaviour_renormalizes_and_stays_null():
    result = fuse(record(b=None, c=0.0), configuration())
    assert result.behaviour_score is None
    assert result.effective_weights["behaviour"] == 0
    assert result.final_risk_score == pytest.approx((0.5 * 0.7 + 0.2 * 0.8) / 0.7)
    assert "LIMITED_BEHAVIOURAL_HISTORY" in result.reason_codes


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_all_score_endpoints(value):
    assert fuse(
        record(value, value, value, 1.0), configuration()
    ).final_risk_score == pytest.approx(value)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.inf, np.nan])
def test_score_bounds(value):
    with pytest.raises(ValidationError):
        record(p=value)
    with pytest.raises(ValueError):
        fusion_arrays([value], [0.5], [0.5], [0.5], Weights(wp=0.5, wa=0.2, wb=0.3), "linear")


def test_gate_boundary_and_zero_confidence():
    cfg = configuration(confidence_curve="gated_0.5")
    assert fuse(record(c=0.499), cfg).effective_weights["behaviour"] == 0
    assert fuse(record(c=0.5), cfg).effective_weights["behaviour"] > 0
    assert fuse(record(c=0.0), cfg).effective_weights["behaviour"] == 0


@pytest.mark.parametrize(
    "score,level",
    [
        (0.0, "LOW"),
        (0.1, "GUARDED"),
        (0.3, "ELEVATED"),
        (0.65, "HIGH"),
        (0.9, "CRITICAL"),
        (1.0, "CRITICAL"),
    ],
)
def test_risk_level_boundaries(score, level):
    assert fuse(record(score, score, score, 1.0), configuration()).risk_level == level


def test_threshold_inclusive_and_deterministic():
    cfg = configuration(weights=Weights(wp=1.0, wa=0.0, wb=0.0))
    result = fuse(record(p=0.6), cfg)
    assert result.investigation_flag
    assert not fuse(record(p=np.nextafter(0.6, 0.0)), cfg).investigation_flag
    assert fuse(record(p=0.6), cfg) == result


def test_reason_codes_require_support():
    result = fuse(
        record(
            p=0.2,
            a=0.2,
            b=0.2,
            evidence=[Evidence(signal="device_novelty", available=True, score=1.0)],
        ),
        configuration(),
    )
    assert result.reason_codes == ["NOVEL_DEVICE"]
    result = fuse(record(p=0.2, a=0.2, b=0.2), configuration())
    assert "NOVEL_DEVICE" not in result.reason_codes
    result = fuse(
        record(
            evidence=[
                Evidence(signal="transaction_velocity", available=True, score=0.96),
                Evidence(signal="merchant_novelty", available=False),
            ]
        ),
        configuration(),
    )
    assert "HIGH_TRANSACTION_VELOCITY" in result.reason_codes
    assert "NEW_MERCHANT" not in result.reason_codes


@pytest.mark.parametrize("field", ["isFraud", "fraud_scenario", "hard_negative", "ground_truth"])
def test_live_metadata_exclusion(field):
    with pytest.raises(ValidationError):
        record(**{field: 1})


def test_no_target_use_and_configuration_reload():
    cfg = configuration()
    restored = FusionConfig.model_validate_json(cfg.model_dump_json())
    assert fuse(record(), restored) == fuse(record(), cfg)
    with pytest.raises(TypeError):
        fuse({**record().model_dump(), "isFraud": 1}, cfg)


def test_flags_and_availability_must_match():
    with pytest.raises(ValidationError):
        FusionInput.model_validate({**record().model_dump(), "supervised_prediction": False})
    with pytest.raises(ValidationError):
        record(b=None, c=0.5)
    with pytest.raises(ValidationError):
        FusionInput.model_validate({**record().model_dump(), "behaviour_available": False})


def test_ablation_generation_and_constraints():
    values = list(candidates(SearchConfig()))
    assert len(values) == 145
    assert len({c["candidate_id"] for c in values}) == 145
    assert {c["variant"] for c in values} == {
        "Supervised",
        "Supervised + Anomaly",
        "Supervised + Behaviour",
        "Full ABRF",
    }
    assert eligible_constraint(
        {"weights": {"wp": 0.6, "wa": 0.2, "wb": 0.2}}, "supervised_dominant"
    )
    assert not eligible_constraint(
        {"weights": {"wp": 0.2, "wa": 0.6, "wb": 0.2}}, "supervised_dominant"
    )


@pytest.fixture
def synthetic_scores():
    rng = np.random.default_rng(42)
    n = 180
    y = (np.arange(n) % 10 == 0).astype(int)
    p = rng.uniform(0.005, 0.25, n)
    p[y == 1] = rng.uniform(0.65, 0.95, int(y.sum()))
    return pd.DataFrame(
        {
            "partition": ["train"] * 120 + ["validation"] * 60,
            "TransactionID": [str(i) for i in range(n)],
            "P": p,
            "A": rng.uniform(0, 1, n),
            "B": rng.uniform(0, 1, n),
            "behaviour_confidence": rng.uniform(0.2, 1, n),
            "ground_truth": y,
            "hard_negative": (np.arange(n) % 7 == 0) & (y == 0),
            "hard_negative_type": np.where(
                (np.arange(n) % 7 == 0) & (y == 0), "LAPTOP_PURCHASE", ""
            ),
            "fraud_scenario": np.where(y == 1, "AMOUNT_SPIKE", ""),
        }
    )


def test_false_positive_budget_and_protection(synthetic_scores):
    frame = synthetic_scores
    cfg = SearchConfig(preserve_scenarios=[])
    curve, choices = threshold_curve(frame.ground_truth, frame.P, frame, cfg)
    for name, point in choices.items():
        assert point["fpr"] <= cfg.policies[name].max_fpr
        assert point["hard_negative_fpr"] <= cfg.policies[name].max_hard_negative_fpr
    assert len(curve["threshold"]) > 100


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch, synthetic_scores):
    # All production readers are replaced with this explicitly synthetic test fixture.
    provenance = {
        "dataset_fingerprint": "synthetic-test-only",
        "fusion_input_version": "fusion-input-v1",
    }
    monkeypatch.setattr(pipeline, "verify_components", lambda *_: (None, None, provenance))
    monkeypatch.setattr(pipeline, "load_inputs", lambda *_: (synthetic_scores, None, provenance))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Phase 6 may not load models/original transactions/generate future data")

    monkeypatch.setattr("joblib.load", forbidden)
    monkeypatch.setattr("riskbyte_ml.anomaly.development.prefix_csv", forbidden)
    monkeypatch.setattr("riskbyte_ml.datasets.synthetic_riskbyte.generator.generate", forbidden)
    search = SearchConfig(
        grid_divisions=5, preserve_scenarios=[], confidence_curves=["linear", "gated_0.5"]
    )
    bundle, directory = pipeline.run(None, tmp_path, search)
    return bundle, directory, search, provenance


def test_small_pipeline_unique_selection_and_test_isolation(completed_fixture):
    bundle, directory, _, _ = completed_fixture
    rows = json.loads((directory / "candidates.json").read_text())
    assert sum(r["selected"] for r in rows) == 1
    assert bundle.leakage_audit["original_temporal_test_accessed"] is False
    assert bundle.future_holdout_status == "PREPARATION_ONLY_NOT_GENERATED"
    saved = pd.read_csv(
        directory / "development_fusion_scores.csv.gz", float_precision="round_trip"
    )
    assert set(saved.partition) == {"train", "validation"}
    assert {
        "supervised_prediction",
        "anomaly_prediction",
        "behaviour_available",
        "risk_level",
        "reason_codes",
    } <= set(saved)
    assert len(bundle.ablation) == 6


def test_ranking_and_cached_execution(completed_fixture):
    bundle, directory, search, provenance = completed_fixture
    loaded, _ = pipeline.run(None, directory.parents[1], search)
    assert loaded == bundle
    rows = json.loads((directory / "candidates.json").read_text())
    eligible = [r for r in rows if r["status"] == "AVAILABLE" and r["policy"] == "BALANCED"]
    assert max(eligible, key=lambda r: rank_candidate(r, search.policies["BALANCED"]))["selected"]
    pipeline.verify_completed(directory, bundle, provenance)


def test_artifact_tampering_detected(completed_fixture):
    bundle, directory, _, provenance = completed_fixture
    with (directory / "selected_config.json").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(DatasetError):
        pipeline.verify_completed(directory, bundle, provenance)


def test_changed_search_cannot_replace_frozen_selection(completed_fixture):
    _, directory, _, _ = completed_fixture
    with pytest.raises(DatasetError):
        pipeline.run(
            None, directory.parents[1], SearchConfig(grid_divisions=6, preserve_scenarios=[])
        )


def test_input_failure_before_any_score_csv_decoding(monkeypatch, tmp_path):
    from riskbyte_ml.abrf import inputs

    def rejected(*args):
        raise DatasetError("frozen_mismatch", "Fixture mismatch")

    monkeypatch.setattr(inputs, "verify_components", rejected)
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: pytest.fail("Decoding after mismatch"))
    with pytest.raises(DatasetError):
        inputs.load_inputs(None, tmp_path)
    with pytest.raises(DatasetError):
        safe_child(tmp_path, "../escape")


def test_nullable_cold_start_csv_projection(monkeypatch, tmp_path, synthetic_scores):
    from types import SimpleNamespace

    from riskbyte_ml.abrf import inputs

    frame = synthetic_scores.copy()
    frame.loc[0, "B"] = np.nan
    frame.loc[0, "behaviour_confidence"] = 0.0
    frame.loc[1, "B"] = np.nextafter(0.482367645255099, np.inf)
    path = tmp_path / "phase5" / "development_scores.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    split = SimpleNamespace(
        indexes={"train": list(range(120)), "validation": list(range(120, 180))},
        summaries={
            name: SimpleNamespace(row_count=len(group), fraud_count=int(group.ground_truth.sum()))
            for name, group in frame.groupby("partition")
        },
    )
    monkeypatch.setattr(inputs, "verify_components", lambda *_: (None, split, {}))
    loaded, _, _ = inputs.load_inputs(None, tmp_path)
    assert loaded.B.dtype == np.dtype(float)
    assert np.isnan(loaded.B.iloc[0])
    assert loaded.B.iloc[1] == frame.B.iloc[1]
    assert not loaded.B.ge(0.5).to_numpy(bool)[0]
    score, _, _ = fusion_arrays(
        loaded.P,
        loaded.A,
        loaded.B,
        loaded.behaviour_confidence,
        Weights(wp=0.5, wa=0.2, wb=0.3),
        "linear",
    )
    assert np.isfinite(score).all()


@pytest.mark.parametrize(
    "route", ["status", "config", "validation", "ablation", "scenarios", "hard-negatives"]
)
def test_api_unconfigured(route):
    with TestClient(create_app(Settings(_env_file=None, environment="test"))) as client:
        response = client.get("/api/v1/abrf/" + route)
    assert response.status_code == 200
    assert response.json()["status"] == "dataset_not_configured"


def test_api_serves_typed_fixture_and_rejects_corruption(completed_fixture, monkeypatch):
    bundle, directory, _, provenance = completed_fixture
    from riskbyte_api.services import abrf as service

    monkeypatch.setattr(service, "load_synthetic_config", lambda *_: None)
    monkeypatch.setattr(service, "verify_components", lambda *_: (None, None, provenance))
    settings = Settings(
        _env_file=None,
        environment="test",
        dataset_adapter="synthetic_riskbyte",
        dataset_config_path=directory / "synthetic-test.json",
        abrf_artifacts_dir=directory.parent,
    )
    with TestClient(create_app(settings)) as client:
        for route in ["status", "config", "validation", "ablation", "scenarios", "hard-negatives"]:
            response = client.get("/api/v1/abrf/" + route)
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "abrf_ready"
            assert str(directory) not in response.text
        (directory / "selected_config.json").write_text("{}")
        assert client.get("/api/v1/abrf/config").status_code == 422
