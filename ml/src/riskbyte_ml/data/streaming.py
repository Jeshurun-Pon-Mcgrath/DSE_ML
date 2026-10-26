"""Bounded CSV batches, content hashing and reversible categorical storage optimization."""

import codecs
import csv
import hashlib
import io
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import union_categoricals

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import LoadedDataset


class HashingReader(io.RawIOBase):
    def __init__(self, stream, limit: int):
        self.stream, self.limit = stream, limit
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        size = self.stream.readinto(buffer)
        if size:
            self.bytes_read += size
            if self.bytes_read > self.limit:
                raise DatasetError("dataset_limit_exceeded", "CSV exceeds its byte limit")
            self.digest.update(memoryview(buffer)[:size])
        return size


@dataclass(frozen=True)
class StreamedTable:
    dataset: LoadedDataset
    memory_before_bytes: int
    memory_after_bytes: int
    optimized_columns: tuple[str, ...]


def optimize_categories(frame: pd.DataFrame, columns) -> tuple[int, int, tuple[str, ...]]:
    """Never downcast numbers; store exact strings categorically only when smaller."""
    before = int(frame.memory_usage(index=True, deep=True).sum())
    changed = []
    for name in columns:
        if name not in frame.columns:
            continue
        candidate = frame[name].astype("category")
        if candidate.memory_usage(deep=True) < frame[name].memory_usage(deep=True):
            frame[name] = candidate
            changed.append(name)
    return before, int(frame.memory_usage(index=True, deep=True).sum()), tuple(changed)


def load_streamed_csv(
    config: DatasetConfig,
    *,
    chunk_rows: int,
    max_memory_bytes: int,
    optimize_all_strings: bool = False,
) -> StreamedTable:
    """Parse strict full-width records in batches. Final frame remains in memory."""
    path = config.source_path
    chunks, batch, row_count, retained, unoptimized = [], [], 0, 0, 0
    try:
        codecs.lookup(config.encoding)
        if path is None or not path.is_file():
            raise DatasetError("dataset_file_missing", "Configured CSV regular file is missing")
        initial = path.stat()
        with path.open("rb") as raw:
            hashing = HashingReader(raw, config.max_bytes)
            with io.TextIOWrapper(io.BufferedReader(hashing), encoding=config.encoding) as text:
                reader = csv.reader(text, delimiter=config.delimiter, strict=True)
                columns = next(reader, None)
                if columns is None:
                    raise DatasetError("dataset_empty", "CSV contains no records")
                if not columns or any(not name.strip() for name in columns):
                    raise DatasetError("dataset_columns_invalid", "CSV has empty column names")
                if len(columns) != len(set(columns)):
                    raise DatasetError("dataset_duplicate_columns", "CSV has duplicate headers")
                if len(columns) > config.max_columns:
                    raise DatasetError("dataset_limit_exceeded", "CSV exceeds its column limit")

                def append_batch():
                    nonlocal retained, unoptimized
                    chunk = pd.DataFrame(batch, columns=columns, dtype="string")
                    unoptimized += int(chunk.memory_usage(index=True, deep=True).sum())
                    storage_columns = (
                        columns if optimize_all_strings else config.categorical_columns
                    )
                    _, optimized_size, _ = optimize_categories(chunk, storage_columns)
                    retained += optimized_size
                    if retained > max_memory_bytes:
                        raise DatasetError(
                            "dataset_memory_limit", "Retained CSV data exceeds the memory budget"
                        )
                    chunks.append(chunk)
                    batch.clear()

                for row in reader:
                    if len(row) != len(columns) or any("\x00" in field for field in row):
                        raise DatasetError("dataset_csv_malformed", "CSV has an invalid record")
                    row_count += 1
                    if row_count > config.max_rows:
                        raise DatasetError("dataset_limit_exceeded", "CSV exceeds its row limit")
                    batch.append(row)
                    if len(batch) >= chunk_rows:
                        append_batch()
                if batch:
                    append_batch()
        final = path.stat()
        if (initial.st_size, initial.st_mtime_ns, initial.st_ino) != (
            final.st_size,
            final.st_mtime_ns,
            final.st_ino,
        ):
            raise DatasetError("dataset_source_changed", "CSV changed while being read", 409)
    except LookupError:
        raise DatasetError("dataset_encoding_unsupported", "CSV requires a text encoding") from None
    except UnicodeError:
        raise DatasetError("dataset_encoding_invalid", "CSV cannot be decoded") from None
    except csv.Error:
        raise DatasetError("dataset_csv_malformed", "CSV has invalid structure") from None
    except OSError:
        raise DatasetError("dataset_file_unreadable", "Configured CSV cannot be read") from None
    if not chunks:
        raise DatasetError("dataset_empty", "CSV contains a header but no records")
    if len(chunks) == 1:
        frame = chunks[0]
    else:
        # Union dictionaries per column; normal concat can expand differing categories to objects.
        combined = {}
        for name in columns:
            parts = [chunk[name] for chunk in chunks]
            if all(isinstance(part.dtype, pd.CategoricalDtype) for part in parts):
                combined[name] = union_categoricals(parts, sort_categories=True, ignore_order=True)
            else:
                combined[name] = pd.concat(
                    [part.astype("string") for part in parts], ignore_index=True
                )
        frame = pd.DataFrame(combined, copy=False)
    chunks.clear()
    storage_columns = columns if optimize_all_strings else config.categorical_columns
    _, after, _ = optimize_categories(frame, storage_columns)
    if after > max_memory_bytes:
        raise DatasetError("dataset_memory_limit", "Combined CSV exceeds retained-memory budget")
    optimized = tuple(
        name for name in columns if isinstance(frame[name].dtype, pd.CategoricalDtype)
    )
    return StreamedTable(
        LoadedDataset(frame, config, "sha256:" + hashing.digest.hexdigest()),
        unoptimized,
        after,
        optimized,
    )
