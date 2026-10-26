"""Frozen, train/validation-only projection. No test frame or test accessor exists."""

import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import HARD_NEGATIVES, SCENARIOS
from riskbyte_ml.datasets.two_table import audited_left_join
from riskbyte_ml.features.behaviour import build_features
from riskbyte_ml.features.manifest import FeatureSpec
from riskbyte_ml.models.supervised.inputs import validate_feature_inputs, verify_frozen_files
from riskbyte_ml.statistics.verification import frozen_artifacts, verify_sources


def prefix_csv(path: Path, count: int, encoding="utf-8"):
    """Decode exactly count records, without asking the CSV reader for another record."""
    with path.open("rb") as stream:
        # Decode a physical line only when csv.reader requests it. TextIOWrapper
        # otherwise decodes a buffered block extending into the protected partition.
        lines = (line.decode(encoding) for line in iter(stream.readline, b""))
        reader = csv.reader(lines, strict=True)
        header = next(reader)
        rows = list(itertools.islice(reader, count))
    if len(rows) != count or len(header) != len(set(header)):
        raise ValueError("CSV prefix length/header invalid")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Malformed CSV prefix")
    return pd.DataFrame(rows, columns=header, dtype="string")


def identity_projection(path: Path, allowed: set[str], encoding="utf-8"):
    """Project by opaque join key BEFORE decoding fields of excluded records.

    This is deliberately restricted to frozen generator v1's single-line records and
    numeric unquoted first-column keys. It is not a general CSV ingestion replacement.
    Excluded identity fields are never CSV-parsed or materialized.
    """
    with path.open("rb") as stream:
        header = next(csv.reader([stream.readline().decode(encoding)], strict=True))
        if header[0] != "TransactionID":
            raise ValueError("Expected leading numeric join key")
        rows = []
        for line in stream:
            key = line.partition(b",")[0]
            if not key.isdigit():
                raise ValueError("Frozen generator identity record contract changed")
            if key.decode("ascii") in allowed:
                row = next(csv.reader([line.decode(encoding)], strict=True))
                if len(row) != len(header):
                    raise ValueError("Malformed selected identity record")
                rows.append(row)
    return pd.DataFrame(rows, columns=header, dtype="string")


class DevelopmentInputs:
    """Checksum opaque full sources; decode only the frozen development prefix."""

    def __init__(self, config, root):
        self.generation = verify_sources(config)
        manifest = verify_frozen_files(root)
        self.split = frozen_artifacts(root)
        train, validation = self.split.indexes["train"], self.split.indexes["validation"]
        count = len(train) + len(validation)
        all_indexes = [i for indexes in self.split.indexes.values() for i in indexes]
        if (
            train != list(range(len(train)))
            or validation != list(range(len(train), count))
            or sorted(all_indexes) != list(range(self.generation["transaction_count"]))
        ):
            raise DatasetError("anomaly_split_invalid", "Frozen temporal prefix is invalid")
        try:
            transaction = prefix_csv(config.transaction_path, count, config.encoding)
            identity = identity_projection(
                config.identity_path, set(transaction.TransactionID), config.encoding
            )
            self.frame, self.join_audit = audited_left_join(
                transaction,
                identity,
                config.table_config("transaction"),
                key="TransactionID",
                target="isFraud",
            )
            self.metadata = prefix_csv(
                config.transaction_path.parent / "evaluation_metadata.csv", count, config.encoding
            )
            if not self.metadata.TransactionID.equals(transaction.TransactionID):
                raise ValueError("Evaluation row order differs")
            if not transaction.isFraud.isin(["0", "1"]).all():
                raise ValueError("Invalid target")
            self.labels = transaction.isFraud.astype(int).to_numpy()
            if not self.metadata.hard_negative.isin(["True", "False"]).all():
                raise ValueError("Invalid evaluation flag")
            self.metadata["hard_negative"] = self.metadata.hard_negative.eq("True")
            hard = self.metadata.hard_negative.to_numpy(dtype=bool)
            scenario = self.metadata.fraud_scenario
            if (
                (hard & (self.labels == 1)).any()
                or not scenario[self.labels == 1].isin(SCENARIOS).all()
                or not self.metadata.hard_negative_type[hard].isin(HARD_NEGATIVES).all()
            ):
                raise ValueError("Invalid evaluation metadata")
            times = transaction.TransactionDT.astype(float).to_numpy()
            if not np.isfinite(times).all() or (np.diff(times) < 0).any():
                raise ValueError("Invalid chronology")
            for name, indexes in (("train", train), ("validation", validation)):
                summary = self.split.summaries[name]
                if (
                    int(self.labels[indexes].sum()) != summary.fraud_count
                    or times[indexes[0]] != summary.elapsed_min
                    or times[indexes[-1]] != summary.elapsed_max
                    or int(hard[indexes].sum()) != summary.hard_negative_count
                ):
                    raise ValueError("Development partition integrity differs")
            self.features = build_features(
                self.frame, horizon_seconds=self.generation["config"]["duration_days"] * 86400
            )
            specs = [FeatureSpec.model_validate(s) for s in manifest["features"]]
            validate_feature_inputs(self.features, specs)
        except (OSError, ValueError, KeyError, csv.Error, UnicodeError):
            raise DatasetError(
                "anomaly_development_invalid", "Frozen development projection failed validation"
            ) from None
        # Recheck bytes against mutation during projection; still no decoded test records.
        if verify_sources(config) != self.generation:
            raise DatasetError("anomaly_source_changed", "Sources changed during projection")
        self.train_count = len(train)
        self.manifest = manifest
        self.audit = {
            "decoded_transaction_rows": count,
            "decoded_evaluation_rows": count,
            "decoded_identity_rows": len(identity),
            "original_temporal_test_accessed": False,
            "test_rows_decoded": 0,
            "test_rows_scored": 0,
            "full_source_io": "opaque checksum bytes and identity join-key allowlist only",
            "strict_prior": "same frozen feature builder, development prefix only",
        }
