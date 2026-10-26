"""Synthetic metadata uses the existing API and exposes no generator row labels."""

from fastapi.testclient import TestClient

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig, SyntheticConfig
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate, save_generated


def test_synthetic_api(tmp_path):
    data = tmp_path / "data"
    save_generated(
        generate(GeneratorConfig(transaction_count=200, entity_count=10, merchant_count=12)), data
    )
    config = SyntheticConfig(
        transaction_path=data / "train_transaction.csv",
        identity_path=data / "train_identity.csv",
        metadata_path=data / "generation_metadata.json",
    )
    config_path = tmp_path / "adapter.json"
    config_path.write_text(config.model_dump_json())
    app = create_app(
        Settings(
            _env_file=None,
            environment="test",
            dataset_adapter="synthetic_riskbyte",
            dataset_config_path=config_path,
            dataset_artifacts_dir=tmp_path / "artifacts",
        )
    )
    with TestClient(app) as client:
        for endpoint in [
            "status",
            "profile",
            "quality",
            "schema",
            "eda",
            "statistics",
            "integration",
        ]:
            response = client.get("/api/v1/dataset/" + endpoint)
            assert response.status_code == 200
            assert response.json()["status"] == "dataset_ready"
            assert str(tmp_path) not in response.text
        metadata = client.get("/api/v1/dataset/integration").json()["data"]["data"]
        assert metadata["dataset_type"] == "synthetic"
        assert "synthetically generated" in metadata["notice"]
        assert "fraud_scenario" not in metadata["feature_classes"]
        assert "hard_negative" not in metadata["feature_classes"]
        assert client.get("/health").status_code == 200


def test_missing_synthetic_does_not_break_health(tmp_path):
    config_path = tmp_path / "adapter.json"
    config_path.write_text(
        SyntheticConfig(transaction_path=tmp_path / "absent.csv").model_dump_json()
    )
    with TestClient(
        create_app(
            Settings(
                _env_file=None,
                environment="test",
                dataset_adapter="synthetic_riskbyte",
                dataset_config_path=config_path,
            )
        )
    ) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/api/v1/dataset/profile")
        assert response.status_code == 422
        assert "synthetic_files_missing" in response.text


def test_unconfigured_synthetic_state():
    with TestClient(
        create_app(
            Settings(_env_file=None, environment="test", dataset_adapter="synthetic_riskbyte")
        )
    ) as client:
        assert client.get("/api/v1/dataset/integration").json() == {
            "status": "dataset_not_configured"
        }
