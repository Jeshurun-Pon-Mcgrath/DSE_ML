"""IEEE-CIS integration API tests use synthetic two-table fixtures only."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_ml.datasets.ieee_cis.config import IEEEConfig


def settings(tmp_path, missing=False):
    fixture = Path(__file__).parents[1] / "fixtures"
    tx, identity = tmp_path / "tx.csv", tmp_path / "identity.csv"
    tx.write_bytes((fixture / "synthetic_ieee_transaction.csv").read_bytes())
    if not missing:
        identity.write_bytes((fixture / "synthetic_ieee_identity.csv").read_bytes())
    config = tmp_path / "ieee.json"
    config.write_text(IEEEConfig(transaction_path=tx, identity_path=identity).model_dump_json())
    return Settings(
        _env_file=None,
        environment="test",
        dataset_adapter="ieee_cis",
        dataset_config_path=config,
        dataset_artifacts_dir=tmp_path / "reports",
    )


def test_metadata_endpoint_joined_paths_private(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        result = client.get("/api/v1/dataset/integration")
        assert result.status_code == 200, result.text
        assert str(tmp_path) not in result.text
        assert "transaction_path" not in result.text
        report = result.json()["data"]["data"]
        assert report["status"] == "joined"
        assert report["join_audit"]["identity_coverage_percentage"] == 62.5
        assert report["recommended_behavioural_groups"] == []
        profile = client.get("/api/v1/dataset/profile").json()["data"]
        assert profile["fingerprint"] == report["fingerprint"]
        assert profile["row_count"] == 8
        assert "source_path" not in profile
        schema = client.get("/openapi.json").json()
        assert "source_path" not in json.dumps(schema["components"]["schemas"])


def test_missing_ieee_source_leaves_health_and_app_available(tmp_path):
    with TestClient(create_app(settings(tmp_path, missing=True))) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/api/v1/dataset/status")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "ieee_cis_files_missing"
        assert str(tmp_path) not in response.text
        assert not (tmp_path / "reports").exists()


def test_unconfigured_integration_remains_typed(client):
    response = client.get("/api/v1/dataset/integration")
    assert response.status_code == 200
    assert response.json() == {"status": "dataset_not_configured"}


def test_identity_content_change_invalidates_cached_snapshot(tmp_path):
    config = settings(tmp_path)
    with TestClient(create_app(config)) as client:
        first = client.get("/api/v1/dataset/profile").json()["data"]["fingerprint"]
        identity = tmp_path / "identity.csv"
        identity.write_text(
            identity.read_text().replace("synthetic-device-A", "changed-long-device")
        )
        second = client.get("/api/v1/dataset/profile").json()["data"]["fingerprint"]
        assert first != second


def test_generic_csv_exposes_unavailable_integration(synthetic_config, tmp_path):
    config_path = tmp_path / "generic.json"
    config_path.write_text(synthetic_config.model_dump_json())
    config = Settings(
        _env_file=None,
        environment="test",
        dataset_adapter="generic_csv",
        dataset_config_path=config_path,
        dataset_artifacts_dir=tmp_path / "reports",
    )
    with TestClient(create_app(config)) as client:
        response = client.get("/api/v1/dataset/integration")
        assert response.status_code == 200
        analysis = response.json()["data"]
        assert analysis["status"] == "UNAVAILABLE_WITH_REASON"
        assert analysis["reason"] == "dataset_adapter_is_not_ieee_cis"
