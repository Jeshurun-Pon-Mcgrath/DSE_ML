import json
import logging
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from riskbyte_api.core.config import Settings
from riskbyte_api.core.logging import JsonFormatter
from riskbyte_api.db.base import Base
from riskbyte_api.db.session import create_database_engine
from riskbyte_api.main import create_app


def test_health_has_version_and_unique_server_request_ids(client: TestClient):
    first = client.get("/health", headers={"X-Request-ID": "untrusted"})
    second = client.get("/health")
    assert first.status_code == 200
    assert first.json() == {"status": "ok", "service": "riskbyte-api", "version": "0.1.0"}
    assert UUID(first.headers["X-Request-ID"])
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_health_and_data_routes_in_openapi_and_runtime_tables(client: TestClient):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/health",
        "/api/v1/predict",
        "/api/v1/predict/batch",
        "/api/v1/transactions",
        "/api/v1/transactions/{transaction_id}",
        "/api/v1/transactions/{transaction_id}/explanation",
        "/api/v1/transactions/{transaction_id}/investigation",
        "/api/v1/dashboard/summary",
        "/api/v1/audit",
        "/api/v1/evaluation/final",
        "/api/v1/evaluation/scenarios",
        "/api/v1/evaluation/hard-negatives",
        "/api/v1/system/model",
        "/api/v1/system/health",
        "/api/v1/models",
        "/api/v1/models/comparison",
        "/api/v1/models/selected",
        "/api/v1/models/{model_id}/metrics",
        "/api/v1/models/{model_id}/scenario-recall",
        "/api/v1/models/{model_id}/threshold-analysis",
        "/api/v1/anomaly/status",
        "/api/v1/anomaly/metrics",
        "/api/v1/anomaly/scenarios",
        "/api/v1/anomaly/hard-negatives",
        "/api/v1/behaviour/summary",
        "/api/v1/behaviour/components",
        "/api/v1/behaviour/validation-analysis",
        "/api/v1/abrf/status",
        "/api/v1/abrf/config",
        "/api/v1/abrf/validation",
        "/api/v1/abrf/ablation",
        "/api/v1/abrf/scenarios",
        "/api/v1/abrf/hard-negatives",
        "/api/v1/explainability/status",
        "/api/v1/explainability/global",
        "/api/v1/explainability/families",
        "/api/v1/explainability/case-studies",
        "/api/v1/explainability/validation-errors",
        "/api/v1/explainability/validation/{transaction_id}",
        *(
            f"/api/v1/dataset/{endpoint}"
            for endpoint in (
                "status",
                "profile",
                "quality",
                "statistics",
                "eda",
                "schema",
                "integration",
            )
        ),
        *(
            f"/api/v1/statistics/{endpoint}"
            for endpoint in (
                "summary",
                "hypotheses",
                "correlations",
                "prevalence",
                "scenarios",
                "hard-negatives",
                "temporal",
            )
        ),
    }
    assert "get" in schema["paths"]["/health"]
    assert set(Base.metadata.tables) == {
        "transactions",
        "predictions",
        "investigations",
        "audit_events",
    }


@pytest.mark.parametrize("method,path,status", [("get", "/missing", 404), ("post", "/health", 405)])
def test_http_errors_have_consistent_envelopes(
    client: TestClient, method: str, path: str, status: int
):
    response = getattr(client, method)(path)
    assert response.status_code == status
    body = response.json()["error"]
    assert body["code"] == f"http_{status}"
    assert body["request_id"] == response.headers["X-Request-ID"]
    if status == 405:
        assert response.headers["allow"] == "GET"


def test_unhandled_errors_are_sanitized(app: FastAPI, client: TestClient):
    @app.get("/test-error")
    def fail():
        raise RuntimeError("sensitive internal payload")

    response = client.get("/test-error")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    assert "sensitive" not in response.text


def test_http_exception_details_are_not_leaked(app: FastAPI, client: TestClient):
    @app.get("/test-http-error")
    def fail():
        raise HTTPException(status_code=403, detail="secret operational detail")

    response = client.get("/test-http-error")
    assert response.status_code == 403
    assert response.json()["error"]["message"] == "Forbidden"
    assert "secret" not in response.text


class TestPayload(BaseModel):
    __test__ = False
    amount: int


def test_validation_response_omits_input_values(app: FastAPI, client: TestClient):
    @app.post("/test-validation")
    def validate(payload: TestPayload):
        return payload

    response = client.post("/test-validation", json={"amount": "sensitive-input"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["issues"] == [{"location": ["body", "amount"], "code": "int_parsing"}]
    assert "sensitive-input" not in response.text


def test_cors_only_allows_configured_origin(client: TestClient):
    allowed = client.get("/health", headers={"Origin": "http://localhost:5173"})
    denied = client.get("/health", headers={"Origin": "https://untrusted.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "X-Request-ID" in allowed.headers["access-control-expose-headers"]
    assert "access-control-allow-origin" not in denied.headers
    preflight = client.options(
        "/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 200


@pytest.mark.parametrize("origin", ["*", "ftp://localhost", "http://localhost:5173/path"])
def test_settings_reject_invalid_cors(origin: str):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_origins=[origin])


def test_settings_validate_env_and_keep_database_secrets_hidden(monkeypatch):
    monkeypatch.setenv("RISKBYTE_LOG_LEVEL", "INVALID")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("RISKBYTE_LOG_LEVEL", "WARNING")
    config = Settings(
        _env_file=None, database_url="postgresql+psycopg://unit:synthetic@localhost/riskbyte_test"
    )
    assert config.log_level == "WARNING"
    assert "synthetic" not in repr(config)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="sqlite:///local.db")


def test_database_is_optional_and_no_engine_is_initialized(app: FastAPI):
    assert not hasattr(app.state, "database_engine")
    with pytest.raises(RuntimeError, match="Database configuration is required"):
        create_database_engine(Settings(_env_file=None, database_url=None))


def test_production_disables_openapi_and_debug():
    app = create_app(Settings(_env_file=None, environment="production"))
    assert app.debug is False
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/health").status_code == 200


def test_logging_emits_valid_json_without_exception_payload():
    record = logging.LogRecord("riskbyte", logging.ERROR, __file__, 1, "Request failed", (), None)
    record.request_id = "test-correlation"
    record.error_type = "RuntimeError"
    try:
        raise RuntimeError("sensitive exception data")
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()
    result = JsonFormatter().format(record)
    data = json.loads(result)
    assert data["request_id"] == "test-correlation"
    assert data["level"] == "ERROR"
    assert data["error_type"] == "RuntimeError"
    assert "sensitive" not in result
