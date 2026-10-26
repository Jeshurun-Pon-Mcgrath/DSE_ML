"""Typed, validated server configuration loaded at application creation."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RISKBYTE_",
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # The root .env also contains frontend configuration.
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]
    database_url: SecretStr | None = None
    dataset_config_path: Path | None = None
    dataset_adapter: Literal["generic_csv", "ieee_cis", "synthetic_riskbyte"] = "generic_csv"
    dataset_artifacts_dir: Path = REPOSITORY_ROOT / "ml" / "artifacts"
    statistics_artifacts_dir: Path = REPOSITORY_ROOT / "ml" / "artifacts" / "statistics"
    model_artifacts_dir: Path = REPOSITORY_ROOT / "ml" / "artifacts" / "models"
    phase5_artifacts_dir: Path = REPOSITORY_ROOT / "ml" / "artifacts" / "phase5"
    abrf_artifacts_dir: Path = REPOSITORY_ROOT / "ml" / "artifacts" / "abrf"
    explainability_artifacts_dir: Path = REPOSITORY_ROOT / "ml" / "artifacts" / "explainability"
    inference_enabled: bool = True
    max_batch_size: int = 500
    max_request_bytes: int = 2_000_000
    development_database_path: Path = REPOSITORY_ROOT / "ml" / "artifacts" / "api.sqlite3"
    write_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def production_write_access(self):
        if (
            self.environment == "production"
            and self.dataset_config_path
            and self.inference_enabled
            and (not self.write_api_key or len(self.write_api_key.get_secret_value()) < 32)
        ):
            raise ValueError("Production inference requires an operator-provided write API key")
        return self

    @field_validator("max_batch_size")
    @classmethod
    def valid_batch(cls, value):
        if not 1 <= value <= 500:
            raise ValueError("Batch limit must be between 1 and 500")
        return value

    @field_validator("max_request_bytes")
    @classmethod
    def valid_request_limit(cls, value):
        if not 1024 <= value <= 10_000_000:
            raise ValueError("Bounded request size between 1KB and 10MB required")
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_origins(cls, values: list[str]) -> list[str]:
        for value in values:
            url = urlsplit(value)
            if (
                url.scheme not in {"http", "https"}
                or not url.netloc
                or url.username
                or url.password
                or url.path
                or url.query
                or url.fragment
            ):
                raise ValueError("CORS origins must be explicit HTTP(S) origins")
        return values

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            from sqlalchemy.engine import make_url

            try:
                url = make_url(value.get_secret_value())
                if url.drivername != "postgresql+psycopg" or not url.host or not url.database:
                    raise ValueError
            except Exception:
                raise ValueError("Use a valid postgresql+psycopg database URL") from None
        return value
