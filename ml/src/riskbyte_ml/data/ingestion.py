"""Bounded, strict CSV ingestion that preserves every parsed field as a string."""

import codecs
import csv
import hashlib
import io
from dataclasses import dataclass

import pandas as pd

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.errors import DatasetError


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    config: DatasetConfig
    fingerprint: str


def load_csv(config: DatasetConfig) -> LoadedDataset:
    if config.source_path is None:
        raise DatasetError("dataset_not_configured", "No dataset is configured")
    try:
        codecs.lookup(config.encoding)
    except LookupError:
        raise DatasetError("dataset_encoding_unsupported", "CSV encoding is unsupported") from None
    try:
        if config.source_path.exists() and not config.source_path.is_file():
            raise DatasetError("dataset_file_unreadable", "Dataset must be a regular file")
        with config.source_path.open("rb") as stream:
            if not config.source_path.is_file():
                raise DatasetError("dataset_file_unreadable", "Dataset must be a regular file")
            content = stream.read(config.max_bytes + 1)
    except FileNotFoundError:
        raise DatasetError(
            "dataset_file_missing", "Configured dataset file does not exist"
        ) from None
    except OSError:
        raise DatasetError("dataset_file_unreadable", "Configured dataset cannot be read") from None
    if len(content) > config.max_bytes:
        raise DatasetError("dataset_limit_exceeded", "Dataset exceeds the configured byte limit")
    try:
        text = content.decode(config.encoding, errors="strict")
    except LookupError:
        raise DatasetError(
            "dataset_encoding_unsupported", "CSV requires a supported text encoding"
        ) from None
    except UnicodeError:
        raise DatasetError(
            "dataset_encoding_invalid", "CSV cannot be decoded with its encoding"
        ) from None
    if not text.strip():
        raise DatasetError("dataset_empty", "Dataset contains no records")
    if "\x00" in text:
        raise DatasetError("dataset_csv_malformed", "CSV contains unsupported null characters")
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=config.delimiter, strict=True)
        columns = next(reader)
        if not columns or any(not column.strip() for column in columns):
            raise DatasetError("dataset_columns_invalid", "CSV requires non-empty column names")
        if len(columns) != len(set(columns)):
            raise DatasetError("dataset_duplicate_columns", "CSV contains duplicate column names")
        if len(columns) > config.max_columns:
            raise DatasetError("dataset_limit_exceeded", "Dataset exceeds the column limit")
        rows = []
        for row in reader:
            if len(row) != len(columns):
                raise DatasetError(
                    "dataset_csv_malformed", "CSV row width does not match its header"
                )
            rows.append(row)
            if len(rows) > config.max_rows:
                raise DatasetError("dataset_limit_exceeded", "Dataset exceeds the row limit")
    except (csv.Error, StopIteration):
        raise DatasetError("dataset_csv_malformed", "CSV has invalid structure") from None
    if not rows:
        raise DatasetError("dataset_empty", "Dataset contains a header but no records")
    if config.target_column is not None and config.target_column not in columns:
        raise DatasetError("dataset_target_missing", "Configured target column is absent")
    frame = pd.DataFrame(rows, columns=columns, dtype="string")
    return LoadedDataset(frame, config, "sha256:" + hashlib.sha256(content).hexdigest())
