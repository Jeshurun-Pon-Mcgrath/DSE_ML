"""Explicit ingestion and analysis policy, separate from inferred schema candidates."""

import json
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from riskbyte_ml.data.errors import DatasetError


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(default="configured_dataset", min_length=1, max_length=128)
    source_path: Path | None = None
    mode: Literal["inspection", "explicit"] = "inspection"
    encoding: str = "utf-8-sig"
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    na_values: tuple[str, ...] = ()
    target_column: str | None = None
    positive_class: str | None = None
    negative_class: str | None = None
    timestamp_column: str | None = None
    timestamp_format: str = "ISO8601"
    timestamp_kind: Literal["absolute", "elapsed"] = "absolute"
    timestamp_unit: Literal["D", "h", "m", "s", "ms", "us", "ns"] | None = None
    timestamp_origin: str | None = None
    timezone: str = "UTC"
    identifier_columns: tuple[str, ...] = ()
    unique_identifier_columns: tuple[str, ...] = ()
    amount_column: str | None = None
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    max_bytes: int = Field(default=256_000_000, ge=1)
    max_rows: int = Field(default=1_000_000, ge=1)
    max_columns: int = Field(default=500, ge=1)
    near_constant_threshold: float = Field(default=0.98, gt=0, le=1, allow_inf_nan=False)
    high_cardinality_threshold: int = Field(default=100, ge=2)
    unique_ratio_threshold: float = Field(default=0.98, gt=0, le=1, allow_inf_nan=False)
    heuristic_min_rows: int = Field(default=20, ge=2)
    leakage_correlation_threshold: float = Field(default=0.995, gt=0, le=1, allow_inf_nan=False)
    imbalance_warning_ratio: float = Field(default=20, ge=1, allow_inf_nan=False)
    iqr_multiplier: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    zscore_threshold: float = Field(default=3, gt=0, allow_inf_nan=False)
    percentiles: tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99)
    frequency_limit: int = Field(default=20, ge=1, le=100)
    correlation_column_limit: int = Field(default=40, ge=2, le=100)

    @field_validator("delimiter")
    @classmethod
    def valid_delimiter(cls, value: str) -> str:
        if value in {"\r", "\n", "\x00"}:
            raise ValueError("CSV delimiters cannot be line breaks or null characters")
        return value

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError:
            raise ValueError("Use an IANA timezone") from None
        return value

    @field_validator("percentiles")
    @classmethod
    def valid_percentiles(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not 0 <= value <= 100 for value in values):
            raise ValueError("Percentiles must be finite values between 0 and 100")
        return tuple(sorted(set(values)))

    @model_validator(mode="after")
    def validate_roles(self):
        roles = (
            self.target_column,
            self.timestamp_column,
            self.amount_column,
            self.identifier_columns,
            self.numeric_columns,
            self.categorical_columns,
        )
        if self.mode == "inspection" and any(roles):
            raise ValueError("Role selections require explicit mode")
        if set(self.numeric_columns) & set(self.categorical_columns):
            raise ValueError("Numeric and categorical roles cannot overlap")
        if self.target_column is not None:
            if self.positive_class is None or self.negative_class is None:
                raise ValueError("Configure both positive and negative target classes explicitly")
            if self.positive_class == self.negative_class:
                raise ValueError("Target classes must differ")
            if any(
                value in self.na_values or not value.strip()
                for value in (self.positive_class, self.negative_class)
            ):
                raise ValueError("Target labels cannot be missing-value tokens")
        elif self.positive_class is not None or self.negative_class is not None:
            raise ValueError("Target labels require a configured target column")
        if not set(self.unique_identifier_columns) <= set(self.identifier_columns):
            raise ValueError("Unique identifiers must be selected identifier columns")
        if self.timestamp_kind == "elapsed" and self.timestamp_unit is None:
            raise ValueError("Elapsed timestamps require an explicit unit")
        if self.timestamp_kind == "absolute" and self.timestamp_unit is not None:
            if self.timestamp_origin is None:
                raise ValueError("Numeric absolute timestamps require an explicit origin")
        return self


def load_config(path: Path) -> DatasetConfig:
    """Resolve source paths relative to the JSON configuration, never a request parameter."""
    try:
        with path.open("rb") as stream:
            content = stream.read(65_537)
        if len(content) > 65_536:
            raise DatasetError("dataset_config_invalid", "Dataset configuration is too large")
        config = DatasetConfig.model_validate_json(content)
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError):
        raise DatasetError(
            "dataset_config_invalid", "Dataset configuration cannot be loaded"
        ) from None
    if config.source_path is not None and not config.source_path.is_absolute():
        config = config.model_copy(
            update={"source_path": (path.parent / config.source_path).resolve()}
        )
    return config
