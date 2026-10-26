"""Versioned synthetic semantics and explicit operator choices."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte import DATASET_NAME

PRESETS = {"small": 10_000, "medium": 100_000, "large": 500_000}
SCENARIOS = (
    "AMOUNT_SPIKE",
    "VELOCITY_BURST",
    "NEW_DEVICE_HIGH_AMOUNT",
    "REGION_CHANGE",
    "NEW_MERCHANT_AMOUNT_DEVIATION",
    "REPEATED_FAILED_ATTEMPTS",
    "ACCOUNT_TAKEOVER",
    "LOW_AND_SLOW",
    "INTERNATIONAL_ANOMALY",
    "MIXED_MODE",
)
HARD_NEGATIVES = (
    "LAPTOP_PURCHASE",
    "TRAVEL_PURCHASE",
    "MEDICAL_EXPENSE",
    "NEW_LEGITIMATE_DEVICE",
    "NEW_MERCHANT_PURCHASE",
    "SHOPPING_FESTIVAL",
)
NUMERIC = (
    "TransactionAmt",
    "failed_attempts_before",
    "account_age_days",
    "is_international",
    "is_card_present",
    "identity_age_days",
)
CATEGORICAL = (
    "merchant_id",
    "merchant_category",
    "transaction_type",
    "product_code",
    "card_type",
    "card_network",
    "billing_region",
    "shipping_region",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "device_id",
    "os_family",
    "browser_family",
    "identity_region",
)
EXCLUDED = {
    "TransactionID",
    "TransactionDT",
    "behavioural_entity_id",
    "isFraud",
    "fraud_scenario",
    "hard_negative",
    "hard_negative_type",
}


def digest(value) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    preset: Literal["small", "medium", "large"] = "medium"
    transaction_count: int | None = Field(default=None, ge=100, le=1_000_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    entity_count: int = Field(default=2_000, ge=2)
    merchant_count: int = Field(default=300, ge=12)
    fraud_prevalence: float = Field(default=0.02, gt=0, lt=0.2, allow_inf_nan=False)
    hard_negative_rate: float = Field(default=0.05, gt=0, lt=0.3, allow_inf_nan=False)
    identity_coverage: float = Field(default=0.7, gt=0, lt=1, allow_inf_nan=False)
    duration_days: int = Field(default=90, ge=1)
    anonymized_feature_count: int = Field(default=0, ge=0, le=20)
    scenarios: tuple[str, ...] = SCENARIOS
    missingness: dict[str, float] = Field(
        default_factory=lambda: {
            "DeviceInfo": 0.06,
            "P_emaildomain": 0.03,
            "R_emaildomain": 0.03,
            "shipping_region": 0.02,
            "identity_region": 0.04,
        }
    )

    @property
    def rows(self):
        return self.transaction_count or PRESETS[self.preset]

    @model_validator(mode="after")
    def validate_policy(self):
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("At least one distinct scenario required")
        if not set(self.scenarios) <= set(SCENARIOS):
            raise ValueError("Unknown fraud scenario")
        if self.entity_count > self.rows // 4:
            raise ValueError("Allow at least four transactions per configured entity on average")
        allowed = set(CATEGORICAL) - {"merchant_id", "device_id"}
        if any(k not in allowed or not 0 <= v < 1 for k, v in self.missingness.items()):
            raise ValueError("Missingness requires supported context fields and rates in [0,1)")
        if round(self.rows * self.fraud_prevalence) < 1:
            raise ValueError("Configuration must generate at least one fraudulent transaction")
        return self


def load_generator_config(path: Path) -> GeneratorConfig:
    try:
        if path.stat().st_size > 65_536:
            raise ValueError
        return GeneratorConfig.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise DatasetError(
            "synthetic_config_invalid", "Synthetic configuration cannot be loaded"
        ) from None


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    transaction_path: Path = Path("../data/synthetic_riskbyte/train_transaction.csv")
    identity_path: Path = Path("../data/synthetic_riskbyte/train_identity.csv")
    metadata_path: Path = Path("../data/synthetic_riskbyte/generation_metadata.json")
    phase2_summary_path: Path | None = Path("../artifacts/phase2_summary.json")
    encoding: str = "utf-8"
    chunk_rows: int = Field(default=10_000, ge=1, le=100_000)
    max_retained_memory_bytes: int = Field(default=2_000_000_000, ge=1)

    def table_config(self, kind):
        return DatasetConfig(
            name=DATASET_NAME,
            mode="explicit",
            encoding=self.encoding,
            source_path=self.transaction_path if kind == "transaction" else self.identity_path,
            categorical_columns=CATEGORICAL,
            max_bytes=1_000_000_000,
        )

    def analysis_config(self, columns):
        return DatasetConfig(
            name=DATASET_NAME,
            source_path=self.transaction_path,
            mode="explicit",
            target_column="isFraud",
            positive_class="1",
            negative_class="0",
            timestamp_column="TransactionDT",
            timestamp_kind="elapsed",
            timestamp_unit="s",
            amount_column="TransactionAmt",
            identifier_columns=("TransactionID", "behavioural_entity_id", "device_id"),
            unique_identifier_columns=("TransactionID",),
            numeric_columns=tuple(n for n in (*NUMERIC, "TransactionDT") if n in columns),
            categorical_columns=tuple(n for n in CATEGORICAL if n in columns),
            max_bytes=1_000_000_000,
        )


def load_synthetic_config(path: Path) -> SyntheticConfig:
    try:
        if path.stat().st_size > 65_536:
            raise ValueError
        config = SyntheticConfig.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise DatasetError(
            "synthetic_config_invalid", "Synthetic adapter configuration cannot be loaded"
        ) from None
    return config.model_copy(
        update={
            name: (path.parent / getattr(config, name)).resolve()
            for name in (
                "transaction_path",
                "identity_path",
                "metadata_path",
                "phase2_summary_path",
            )
            if getattr(config, name) is not None and not getattr(config, name).is_absolute()
        }
    )
