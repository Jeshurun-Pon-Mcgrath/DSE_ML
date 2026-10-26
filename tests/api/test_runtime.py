"""Small synthetic API/SQLite fixtures; no final-holdout rows or labels are read."""

from threading import RLock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from riskbyte_api.core.config import Settings
from riskbyte_api.db.base import Base
from riskbyte_api.db.models import AuditEvent, Prediction
from riskbyte_api.main import create_app
from riskbyte_api.services.runtime import RuntimeService
from riskbyte_ml.abrf.contracts import A_THRESHOLD, P_THRESHOLD, FusionConfig, FusionInput, Weights
from riskbyte_ml.abrf.engine import fuse
from riskbyte_ml.explainability.engine import formula_trace
from riskbyte_ml.explainability.schemas import (
    SHAP_VERSION,
    AnomalyExplanation,
    BehaviourExplanation,
    ConfidenceExplanation,
    SupervisedExplanation,
    UnifiedExplanation,
)


class SyntheticSystem:
    """Explicit deterministic fixture, not a trained RiskByte model or report metric."""

    fusion_config = FusionConfig(
        weights=Weights(wp=0.9, wa=0, wb=0.1),
        confidence_curve="gated_0.5",
        policy="BALANCED",
        decision_threshold=0.29755749437222373,
        risk_boundaries=[0.01, 0.1, 0.6, 0.9],
    )
    classifier = SimpleNamespace(model_id="synthetic-test-model", model_version="fixture-v1")
    provenance = {"feature_version": "synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1"}
    preprocessor = SimpleNamespace(columns=[])

    def score(self, features):
        return {"P": np.where(features.TransactionAmt > 500, 0.8, 0.1)}

    def explain(self, tx, features, cached_scores=None, position=0):
        p = float(self.score(features)["P"][0])
        record = FusionInput(
            transaction_id=str(tx.TransactionID),
            supervised_probability=p,
            supervised_prediction=p >= P_THRESHOLD,
            anomaly_score=0.99,
            anomaly_prediction=True,
            behaviour_score=None,
            behaviour_available=False,
            behaviour_confidence=0.0,
        )
        trace = fuse(record, self.fusion_config)
        return UnifiedExplanation(
            version=SHAP_VERSION,
            transaction_id=record.transaction_id,
            risk_summary={
                "final_risk_score": trace.final_risk_score,
                "risk_level": trace.risk_level,
                "investigation_flag": trace.investigation_flag,
            },
            supervised=SupervisedExplanation(
                model_id="synthetic-test-model",
                model_version="fixture-v1",
                raw_probability=p,
                calibrated_probability=p,
                calibration_version="fixture-none",
                threshold=P_THRESHOLD,
                prediction=record.supervised_prediction,
                shap_status="AVAILABLE",
                base_value=p,
                additivity_residual=0.0,
            ),
            behaviour=BehaviourExplanation(
                behaviour_score=None,
                behaviour_confidence=0.0,
                history_count=int(features.behavioural_history_count.iloc[0]),
                history_available=False,
                available_weight_fraction=0.0,
                components=[],
                version="fixture-v1",
                confidence_explanation=ConfidenceExplanation(
                    prior_transaction_count=0,
                    history_saturation=20.0,
                    available_weight_fraction=0.0,
                    merchant_history_available=False,
                    device_history_available=False,
                    current_identity_available=False,
                    supporting_measurements=["Synthetic fixture"],
                ),
            ),
            anomaly=AnomalyExplanation(
                version="fixture-v1",
                score=0.99,
                original_model_score=-0.8,
                threshold=A_THRESHOLD,
                flag=True,
                effective_abrf_weight=0.0,
                observable_signals=[],
            ),
            fusion=trace,
            formula_trace=formula_trace(record, self.fusion_config),
        )


@pytest.fixture
def runtime_client():
    settings = Settings(
        _env_file=None, inference_enabled=False, max_batch_size=2, max_request_bytes=2000
    )
    service = RuntimeService.__new__(RuntimeService)
    service.settings = settings
    service.lock = RLock()
    service.system = SyntheticSystem()
    service.engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(service.engine)
    service.history = pd.DataFrame(
        columns=["TransactionID", "TransactionDT", "TransactionAmt", "behavioural_entity_id"]
    )
    service.watermark = 0.0
    service.manifest = {"status": "FINAL_EVALUATED_LOCKED"}
    service.final_reports = {
        "final": {"finding_label": "Synthetic fixture", "transaction_count": 10},
        "scenarios": {"fixture": {"count": 1}},
        "hard-negatives": {"fixture": {"count": 1}},
    }
    app = create_app(settings)
    app.state.runtime = service
    yield TestClient(app), service
    service.engine.dispose()


def payload(id_="fixture-1", time=1, amount=100):
    return {
        "TransactionID": id_,
        "TransactionDT": time,
        "TransactionAmt": amount,
        "behavioural_entity_id": "fixture-entity",
    }


def test_single_prediction_bounds_and_zero_anomaly(runtime_client):
    client, service = runtime_client
    response = client.post("/api/v1/predict", json=payload())
    assert response.status_code == 200, response.text
    data = response.json()
    assert 0 <= data["risk"]["score"] <= 1
    assert data["anomaly"]["abrf_weight"] == 0
    assert data["abrf"]["contributions"]["anomaly"] == 0
    with Session(service.engine) as s:
        assert s.scalars(select(AuditEvent)).first().action == "prediction_generated"


def test_batch_and_limits(runtime_client):
    client, _ = runtime_client
    response = client.post(
        "/api/v1/predict/batch",
        json={"transactions": [payload("fixture-a"), payload("fixture-b", amount=1000)]},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()) == 2
    assert (
        client.post(
            "/api/v1/predict/batch", json={"transactions": [payload(str(i), 2) for i in range(3)]}
        ).status_code
        == 413
    )


@pytest.mark.parametrize(
    "extra", [{"isFraud": 1}, {"fraud_scenario": "fake"}, {"TransactionAmt": -1}]
)
def test_invalid_request_rejected(runtime_client, extra):
    client, _ = runtime_client
    assert client.post("/api/v1/predict", json={**payload(), **extra}).status_code == 422


def test_duplicate_and_out_of_order(runtime_client):
    client, service = runtime_client
    service.historical_id_limit = 100000
    assert client.post("/api/v1/predict", json=payload("42", time=10)).status_code == 409
    assert client.post("/api/v1/predict", json=payload(time=10)).status_code == 200
    assert client.post("/api/v1/predict", json=payload(time=10)).status_code == 409
    assert client.post("/api/v1/predict", json=payload("earlier", 9)).status_code == 409


def test_listing_filters_pagination_and_detail(runtime_client):
    client, _ = runtime_client
    client.post(
        "/api/v1/predict/batch",
        json={"transactions": [payload("low"), payload("high", amount=1000)]},
    )
    listing = client.get("/api/v1/transactions", params={"page_size": 1, "sort": "risk"}).json()
    assert listing["total"] == 2 and len(listing["items"]) == 1
    filtered = client.get(
        "/api/v1/transactions", params={"risk_level": "HIGH", "investigation_required": True}
    ).json()
    assert filtered["total"] == 1
    assert client.get("/api/v1/transactions/high").status_code == 200
    assert client.get("/api/v1/transactions", params={"page": 0}).status_code == 422


def test_explanation_dashboard_evaluation_and_health(runtime_client):
    client, _ = runtime_client
    client.post("/api/v1/predict", json=payload())
    e = client.get("/api/v1/transactions/fixture-1/explanation")
    assert e.status_code == 200
    assert e.json()["supervised"]["explained_model"] == "underlying_decision_tree"
    assert e.json()["fusion"]["weighted_contributions"]["anomaly"] == 0
    assert client.get("/api/v1/dashboard/summary").json()["transaction_count"] == 1
    for path in ["final", "scenarios", "hard-negatives"]:
        assert client.get("/api/v1/evaluation/" + path).status_code == 200
    assert client.get("/api/v1/system/health").json()["inference_ready"]
    assert client.get("/api/v1/system/model").json()["dataset_type"] == "synthetic"


def test_review_state_does_not_alter_scores(runtime_client):
    client, service = runtime_client
    prediction = client.post("/api/v1/predict", json=payload()).json()
    for state in ["UNDER_REVIEW", "CLEARED", "ESCALATED", "UNREVIEWED"]:
        response = client.patch(
            "/api/v1/transactions/fixture-1/investigation", json={"state": state}
        )
        assert response.status_code == 200
        assert (
            client.get("/api/v1/transactions/fixture-1").json()["prediction"]["risk"]
            == prediction["risk"]
        )
    assert (
        client.patch(
            "/api/v1/transactions/fixture-1/investigation", json={"state": "BAD"}
        ).status_code
        == 422
    )
    with Session(service.engine) as s:
        assert s.get(Prediction, "fixture-1").risk_score == prediction["risk"]["score"]


def test_read_only_audit_history_is_paginated_and_redacted(runtime_client):
    client, service = runtime_client
    client.post("/api/v1/predict", json=payload("audit-fixture"))
    with Session(service.engine) as session:
        session.add(
            AuditEvent(
                transaction_id="audit-fixture",
                action="prediction_generated",
                safe_metadata={
                    "state": "UNREVIEWED",
                    "credential": "fixture-secret",
                    "amount": 123,
                },
            )
        )
        session.commit()
    response = client.get(
        "/api/v1/audit", params={"transaction_id": "audit-fixture", "page_size": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 1
    assert data["items"][0]["metadata"] == {"state": "UNREVIEWED"}
    assert "fixture-secret" not in response.text
    assert data["items"][0]["timestamp"].endswith("+00:00")
    assert client.get("/api/v1/audit?page_size=101").status_code == 422


def test_body_limit_and_write_key(runtime_client):
    client, service = runtime_client
    assert (
        client.post(
            "/api/v1/predict", content=b"x" * 2001, headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    from pydantic import SecretStr

    service.settings.write_api_key = SecretStr("synthetic-test-key")
    assert client.post("/api/v1/predict", json=payload()).status_code == 403
    assert (
        client.post(
            "/api/v1/predict", json=payload(), headers={"X-Api-Key": "synthetic-test-key"}
        ).status_code
        == 200
    )


def test_artifacts_load_once_in_lifespan(monkeypatch):
    import riskbyte_api.services.runtime as module

    calls = []

    def loaded(settings):
        calls.append(True)
        return SimpleNamespace(engine=SimpleNamespace(dispose=lambda: None))

    monkeypatch.setattr(module, "RuntimeService", loaded)
    app = create_app(Settings(_env_file=None, dataset_config_path="synthetic-test-only.json"))
    with TestClient(app) as client:
        assert client.get("/api/v1/system/health").status_code == 200
        assert client.get("/api/v1/system/health").status_code == 200
    assert len(calls) == 1
