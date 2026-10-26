"""Frozen-source verification and a one-use final-test gate after durable selection."""

import json
from pathlib import Path

import numpy as np

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import EXCLUDED, digest
from riskbyte_ml.evaluation.splits import SplitArtifact, validate_partition
from riskbyte_ml.features.behaviour import build_features
from riskbyte_ml.features.manifest import FeatureSpec
from riskbyte_ml.models.supervised import (
    FROZEN_MANIFEST_HASH,
    FROZEN_PREPROCESSOR_HASHES,
    FROZEN_RANDOM_HASH,
    FROZEN_RANDOM_ID,
    FROZEN_SPLIT_HASH,
)
from riskbyte_ml.models.supervised.schemas import SelectionFreeze
from riskbyte_ml.preprocessing.pipeline import load_preprocessor
from riskbyte_ml.statistics import FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import file_hash, load_verified, read_json


def verify_frozen_files(root):
    paths = {
        root / "features" / "feature_manifest.json": FROZEN_MANIFEST_HASH,
        root / "splits" / (FROZEN_SPLIT_ID[7:] + ".json"): FROZEN_SPLIT_HASH,
    }
    for path, expected in paths.items():
        if file_hash(path) != expected:
            raise DatasetError(
                "model_frozen_artifact_changed",
                "Frozen feature manifest or temporal split contents changed; training aborted",
            )
    manifest = read_json(root / "features" / "feature_manifest.json")
    for family, expected in FROZEN_PREPROCESSOR_HASHES.items():
        filename = manifest["families"][family]["preprocessing_artifact"]
        if (
            Path(filename).name != filename
            or file_hash(root / "preprocessing" / filename) != expected
        ):
            raise DatasetError(
                "model_preprocessing_changed",
                "Frozen preprocessing differs from the recorded snapshot",
            )
    return manifest


def validate_feature_inputs(features, specs):
    if set(features) & EXCLUDED or list(features) != [s.name for s in specs]:
        raise DatasetError(
            "model_feature_leakage",
            "Predictive columns differ from the frozen approved feature list",
        )
    if any(
        s.leakage_status != "APPROVED_PRE_OUTCOME"
        or set(s.source_columns)
        & {"isFraud", "fraud_scenario", "hard_negative", "hard_negative_type"}
        for s in specs
    ):
        raise DatasetError(
            "model_feature_leakage", "Outcome/evaluation metadata cannot enter predictive lineage"
        )


class TestGate:
    "One access per output run, after selection has been atomically"

    "and exclusively frozen."

    __test__ = False

    def __init__(self, loader):
        self.loader = loader
        self.access_count = 0

    def release(self, directory: Path, model_id: str, artifact_hash: str):
        try:
            freeze = SelectionFreeze.model_validate_json(
                (directory / "selection_freeze.json").read_bytes()
            )
        except (OSError, ValueError):
            raise DatasetError(
                "model_test_locked", "Test evaluation requires durable frozen selection"
            ) from None
        if (
            freeze.model_id != model_id
            or freeze.artifact_hash != artifact_hash
            or self.access_count
        ):
            raise DatasetError(
                "model_test_locked", "Test selection mismatch or test already accessed"
            )
        # Exclusive marker prevents accidental repeat access even after an interrupted process.
        write_artifact(
            directory / "test_evaluation_started.json",
            json.dumps(
                {
                    "model_id": model_id,
                    "selection_hash": file_hash(directory / "selection_freeze.json"),
                }
            ).encode(),
            overwrite=False,
        )
        self.access_count += 1
        return self.loader()


class ExperimentInputs:
    def __init__(self, config, root, experiment):
        self.frozen_manifest = verify_frozen_files(root)
        self.frame, self.metadata, self.split, self.generation = load_verified(config, root)
        self.features = build_features(
            self.frame, horizon_seconds=self.generation["config"]["duration_days"] * 86400
        )
        self.specs = [FeatureSpec.model_validate(s) for s in self.frozen_manifest["features"]]
        validate_feature_inputs(self.features, self.specs)
        self.processors = {}
        for family, recorded in self.frozen_manifest["families"].items():
            value = load_preprocessor(
                root / "preprocessing" / recorded["preprocessing_artifact"],
                self.split.dataset_fingerprint,
                self.split.split_id,
            )
            if (
                value.preprocessing_id != recorded["preprocessing_id"]
                or value.train_indexes_fingerprint != digest(self.split.indexes["train"])
                or value.columns != list(self.features)
                or value.feature_names != recorded["encoded_feature_names"]
                or value.train_count != len(self.split.indexes["train"])
            ):
                raise DatasetError(
                    "model_preprocessing_invalid",
                    "Preprocessing training/feature provenance mismatch",
                )
            self.processors[family] = value
        self.train_rows = self.split.indexes["train"]
        self.validation_rows = self.split.indexes["validation"]
        boundary = int(len(self.validation_rows) * experiment.calibration_fraction)
        times = self.frame.TransactionDT.astype(float).to_numpy()
        while 0 < boundary < len(self.validation_rows) and (
            times[self.validation_rows[boundary]] == times[self.validation_rows[boundary - 1]]
        ):
            boundary += 1
        if not 0 < boundary < len(self.validation_rows):
            raise DatasetError(
                "model_validation_blocks_invalid", "Validation timing ties produced an empty block"
            )
        self.calibration_positions = np.arange(boundary)
        self.selection_positions = np.arange(boundary, len(self.validation_rows))
        self.y_train = self._labels(self.train_rows)
        self.y_validation = self._labels(self.validation_rows)
        for rows in (self.calibration_positions, self.selection_positions):
            if len(np.unique(self.y_validation[rows])) != 2:
                raise DatasetError(
                    "model_validation_blocks_invalid",
                    "Both classes required in each validation block",
                )
        self.validation_metadata = self.metadata.iloc[self.validation_rows].reset_index(drop=True)
        self.gate = TestGate(self._test_data)

    def _labels(self, rows):
        return self.frame.isFraud.iloc[rows].astype(int).to_numpy()

    def matrices(self, family):
        processor = self.processors[family]
        return (
            processor.transform(self.features.iloc[self.train_rows]),
            processor.transform(self.features.iloc[self.validation_rows]),
        )

    def _test_data(self):
        rows = self.split.indexes["test"]
        return (
            self.features.iloc[rows],
            self._labels(rows),
            self.metadata.iloc[rows].reset_index(drop=True),
            self.frame.iloc[rows],
        )

    def random_split(self, root):
        path = root / "splits" / (FROZEN_RANDOM_ID[7:] + ".json")
        if file_hash(path) != FROZEN_RANDOM_HASH:
            raise DatasetError(
                "model_random_split_changed",
                "Saved random comparator split differs from the frozen snapshot",
            )
        split = SplitArtifact.model_validate_json(path.read_bytes())
        validate_partition(split, self.frame)
        if (
            split.split_id != FROZEN_RANDOM_ID
            or split.dataset_fingerprint != self.split.dataset_fingerprint
        ):
            raise DatasetError(
                "model_random_split_invalid", "Saved random comparator split is incompatible"
            )
        return split
