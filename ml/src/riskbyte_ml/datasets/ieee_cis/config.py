"""Central IEEE-CIS semantics and operator-controlled resource limits."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.errors import DatasetError

JOIN_KEY = "TransactionID"
TARGET = "isFraud"
ELAPSED = "TransactionDT"
AMOUNT = "TransactionAmt"
ADAPTER_VERSION = "ieee-cis-left-join-v1"
TRANSACTION_CATEGORIES = (
    "ProductCD",
    *(f"card{i}" for i in range(1, 7)),
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)
IDENTITY_CATEGORIES = ("DeviceType", "DeviceInfo", *(f"id_{i}" for i in range(12, 39)))
ENTITY_CANDIDATES = (
    ("card1", "card2", "card3", "card4", "card5", "card6"),
    ("card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"),
    ("card1", "card2", "addr1", "addr2", "P_emaildomain"),
    ("card1", "card2", "addr1", "addr2", "DeviceInfo"),
)


def is_anonymized_engineered(name: str) -> bool:
    match = re.fullmatch(r"V([1-9]\d*)", name)
    return bool(match and 1 <= int(match[1]) <= 339)


class IEEEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    transaction_path: Path = Path("../data/ieee_cis/train_transaction.csv")
    identity_path: Path = Path("../data/ieee_cis/train_identity.csv")
    encoding: str = "utf-8-sig"
    max_bytes_per_file: int = Field(default=1_000_000_000, ge=1)
    max_rows_per_file: int = Field(default=1_000_000, ge=1)
    max_columns_per_file: int = Field(default=500, ge=1)
    chunk_rows: int = Field(default=10_000, ge=1, le=100_000)
    max_retained_memory_bytes: int = Field(default=3_000_000_000, ge=1)
    optimize_categories: bool = True
    entity_candidates: tuple[tuple[str, ...], ...] = ENTITY_CANDIDATES
    min_entity_count_for_recommendation: int = Field(default=20, ge=2)
    min_repeated_entity_percentage: float = Field(default=20, ge=0, le=100, allow_inf_nan=False)
    min_complete_row_percentage: float = Field(default=50, ge=0, le=100, allow_inf_nan=False)

    @field_validator("entity_candidates")
    @classmethod
    def validate_candidates(cls, candidates):
        allowed = {
            *(f"card{i}" for i in range(1, 7)),
            "addr1",
            "addr2",
            "P_emaildomain",
            "DeviceInfo",
        }
        if len(candidates) > 16:
            raise ValueError("At most 16 proxy candidates may be studied")
        for fields in candidates:
            if not fields or len(set(fields)) != len(fields) or not set(fields) <= allowed:
                raise ValueError("Entity candidates require distinct approved proxy fields")
        if len(set(candidates)) != len(candidates):
            raise ValueError("Entity candidate definitions must be distinct")
        return candidates

    def table_config(self, kind: str) -> DatasetConfig:
        return DatasetConfig(
            name="IEEE-CIS training (joined)",
            source_path=self.transaction_path if kind == "transaction" else self.identity_path,
            mode="explicit",
            encoding=self.encoding,
            categorical_columns=TRANSACTION_CATEGORIES
            if kind == "transaction"
            else IDENTITY_CATEGORIES,
            max_bytes=self.max_bytes_per_file,
            max_rows=self.max_rows_per_file,
            max_columns=self.max_columns_per_file,
        )

    def analysis_config(self, columns) -> DatasetConfig:
        base = self.table_config("transaction")
        return DatasetConfig.model_validate(
            {
                **base.model_dump(),
                "target_column": TARGET,
                "positive_class": "1",
                "negative_class": "0",
                "identifier_columns": (JOIN_KEY,),
                "unique_identifier_columns": (JOIN_KEY,),
                "timestamp_column": ELAPSED,
                "timestamp_kind": "elapsed",
                "timestamp_unit": "s",
                "amount_column": AMOUNT,
                "numeric_columns": tuple(name for name in (AMOUNT, ELAPSED) if name in columns),
                "categorical_columns": tuple(
                    name
                    for name in (*TRANSACTION_CATEGORIES, *IDENTITY_CATEGORIES)
                    if name in columns
                ),
            }
        )


def load_ieee_config(path: Path) -> IEEEConfig:
    try:
        with path.open("rb") as stream:
            content = stream.read(65_537)
        if len(content) > 65_536:
            raise ValueError
        config = IEEEConfig.model_validate_json(content)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError):
        raise DatasetError(
            "ieee_cis_config_invalid", "IEEE-CIS configuration cannot be loaded"
        ) from None
    return config.model_copy(
        update={
            name: (path.parent / getattr(config, name)).resolve()
            for name in ("transaction_path", "identity_path")
            if not getattr(config, name).is_absolute()
        }
    )


def verify_sources(config: IEEEConfig) -> None:
    missing = [
        label
        for label, path in (
            ("train_transaction.csv", config.transaction_path),
            ("train_identity.csv", config.identity_path),
        )
        if not path.is_file()
    ]
    if missing:
        raise DatasetError(
            "ieee_cis_files_missing",
            "IEEE-CIS dataset configuration error: missing " + ", ".join(missing),
        )
