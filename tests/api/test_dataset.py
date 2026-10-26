import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_api.schemas.dataset import DatasetResponse
from riskbyte_api.services.dataset import DatasetService
from riskbyte_ml.data.models import NumericStatistics

ENDPOINTS = ("status", "profile", "quality", "statistics", "eda", "schema")


def make_settings(config, tmp_path):
    configuration = tmp_path / "dataset.json"
    configuration.write_text(config.model_dump_json(), encoding="utf-8")
    return Settings(
        _env_file=None,
        environment="test",
        dataset_config_path=configuration,
        dataset_artifacts_dir=tmp_path / "artifacts",
    )


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_dataset_not_configured_is_typed_200(client, endpoint):
    result = client.get("/api/v1/dataset/" + endpoint)
    assert result.status_code == 200
    assert result.json() == {"status": "dataset_not_configured"}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_configured_endpoints_and_private_paths(synthetic_config, tmp_path, endpoint):
    app = create_app(make_settings(synthetic_config, tmp_path))
    with TestClient(app) as client:
        result = client.get("/api/v1/dataset/" + endpoint)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "dataset_ready"
    assert "source_path" not in result.text
    assert str(synthetic_config.source_path) not in result.text
    assert str(tmp_path) not in result.text
    assert result.headers["X-Request-ID"]
    json.loads(result.text, parse_constant=lambda value: pytest.fail(value))


def test_profile_quality_and_schema_payloads(synthetic_config, tmp_path):
    with TestClient(create_app(make_settings(synthetic_config, tmp_path))) as client:
        profile = client.get("/api/v1/dataset/profile").json()["data"]
        assert profile["row_count"] == 12
        assert profile["column_count"] == 7
        assert profile["fingerprint"].startswith("sha256:")
        assert client.get("/api/v1/dataset/quality").json()["data"]["duplicates"]["extra_rows"] == 1
        assert client.get("/api/v1/dataset/schema").json()["data"]["target"]["column"] == "label"
        assert client.get("/api/v1/dataset/statistics").json()["data"]["value"]["count"] == 12
        eda = client.get("/api/v1/dataset/eda").json()["data"]
        assert eda["fraud_by_hour"]["status"] == "AVAILABLE"
    assert (tmp_path / "artifacts" / "data_profile.json").exists()


@pytest.mark.parametrize(
    "content,code",
    [
        ("", "dataset_empty"),
        ("x,x\n1,2\n", "dataset_duplicate_columns"),
        ('x\n"bad', "dataset_csv_malformed"),
    ],
)
def test_invalid_datasets_use_structured_errors(csv_factory, tmp_path, content, code):
    config = csv_factory(content)
    with TestClient(create_app(make_settings(config, tmp_path))) as client:
        result = client.get("/api/v1/dataset/profile")
    assert result.status_code == 422
    assert result.json()["error"]["code"] == code
    assert result.json()["error"]["request_id"] == result.headers["X-Request-ID"]
    assert str(tmp_path) not in result.text


def test_missing_file_does_not_break_liveness(synthetic_config, tmp_path):
    config = synthetic_config.model_copy(update={"source_path": tmp_path / "private.csv"})
    with TestClient(create_app(make_settings(config, tmp_path))) as client:
        assert client.get("/health").status_code == 200
        result = client.get("/api/v1/dataset/status")
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "dataset_file_missing"
    assert str(tmp_path) not in result.text


def test_target_validation_failure_has_reviewable_quality_report(csv_factory, tmp_path):
    config = csv_factory(
        "answer\nnormal\nunsupported\n",
        mode="explicit",
        target_column="answer",
        positive_class="fraud",
        negative_class="normal",
    )
    with TestClient(create_app(make_settings(config, tmp_path))) as client:
        result = client.get("/api/v1/dataset/quality")
        assert result.status_code == 200
        assert result.json()["status"] == "dataset_validation_failed"
        assert result.json()["data"]["validation_status"] == "FAILED"
        assert result.json()["data"]["target"]["unsupported_count"] == 1
        eda = client.get("/api/v1/dataset/eda").json()["data"]
        assert eda["class_distribution"]["status"] == "UNAVAILABLE_WITH_REASON"


def test_strict_response_rejects_raw_invalid_and_extra_values():
    with pytest.raises(ValidationError):
        DatasetResponse[NumericStatistics](
            status="dataset_not_configured",
            data=NumericStatistics(count=1, missing_count=0, invalid_count=0, infinite_count=0),
        )
    with pytest.raises(ValidationError):
        NumericStatistics(
            count=1, missing_count=0, invalid_count=0, infinite_count=0, mean=float("inf")
        )
    with pytest.raises(ValidationError):
        DatasetResponse(status="dataset_not_configured", unexpected="value")


def test_openapi_includes_typed_dataset_schemas(client):
    schema = client.get("/openapi.json").json()
    for endpoint in ENDPOINTS:
        response = schema["paths"]["/api/v1/dataset/" + endpoint]["get"]["responses"]["200"]
        assert "$ref" in response["content"]["application/json"]["schema"]
    assert "source_path" not in json.dumps(schema["components"]["schemas"])


def test_concurrent_requests_share_snapshot_and_source_change_reloads(csv_factory, tmp_path):
    config = csv_factory("x\n1\n2\n3\n")
    service = DatasetService(make_settings(config, tmp_path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(lambda _: service.snapshot(), range(4)))
    assert all(report is reports[0] for report in reports)
    config.source_path.write_text("x\n1\n2\n3\n4\n", encoding="utf-8")
    updated = service.snapshot()
    assert updated.row_count == 4
    assert updated.fingerprint != reports[0].fingerprint
