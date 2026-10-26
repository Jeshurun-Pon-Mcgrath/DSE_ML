import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app


@pytest.fixture
def app() -> FastAPI:
    return create_app(
        Settings(_env_file=None, environment="test", database_url=None, dataset_config_path=None)
    )


@pytest.fixture
def client(app: FastAPI):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
