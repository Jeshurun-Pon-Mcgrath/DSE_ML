"""Immutable trusted classifier bundles; API uses JSON and never unpickles models."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.models.supervised import MODEL_VERSION
from riskbyte_ml.models.supervised.estimators import positive_probability
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT
from riskbyte_ml.statistics.verification import file_hash


@dataclass(frozen=True)
class ClassifierBundle:
    estimator: object
    calibrator: object
    algorithm: str
    model_id: str
    threshold: float
    preprocessing_id: str
    feature_names: list[str]
    dataset_fingerprint: str = FROZEN_FINGERPRINT
    feature_version: str = FROZEN_FEATURE_VERSION
    model_version: str = MODEL_VERSION

    def probability(self, matrix):
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("Encoded features do not match classifier schema")
        return self.calibrator.transform(positive_probability(self.estimator, matrix))


def save_classifier(bundle, directory):
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    path = directory / (bundle.model_id + ".joblib")
    write_artifact(path, buffer.getvalue(), overwrite=False)
    return path.name, file_hash(path)


def load_classifier(path: Path, expected_hash: str, fingerprint: str, feature_version: str):
    if file_hash(path) != expected_hash:
        raise DatasetError(
            "model_artifact_hash_mismatch", "Classifier bytes differ from its registered hash"
        )
    # Executable serialization: only locally generated, operator-trusted paths are supported.
    bundle = joblib.load(path)
    if (
        not isinstance(bundle, ClassifierBundle)
        or bundle.dataset_fingerprint != fingerprint
        or bundle.feature_version != feature_version
        or bundle.model_version != MODEL_VERSION
    ):
        raise DatasetError("model_artifact_incompatible", "Classifier provenance is incompatible")
    return bundle


def verify_registry_artifacts(registry, model_directory: Path):
    """Verify completed journals and classifier bytes without executing serialization."""
    from riskbyte_ml.models.supervised.schemas import Evaluation, SelectionFreeze

    directory = model_directory.parent / "evaluation"
    try:
        freeze = SelectionFreeze.model_validate_json(
            (directory / "selection_freeze.json").read_bytes()
        )
        started = json.loads((directory / "test_evaluation_started.json").read_bytes())
        final = json.loads((directory / "final_test_metrics.json").read_bytes())
        expected_hash = file_hash(directory / "selection_freeze.json")
        selected = next(m for m in registry.models if m.is_selected_supervised_model)
        if (
            freeze != registry.selection_freeze
            or started["model_id"] != selected.model_id
            or started["selection_hash"] != expected_hash
            or final["selection_hash"] != expected_hash
            or final["model_id"] != selected.model_id
            or final["test_gate_access_count"] != 1
            or Evaluation.model_validate(final["evaluation"]) != selected.test_metrics
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, StopIteration):
        raise DatasetError(
            "model_journal_invalid", "Frozen selection/final-test journals are inconsistent"
        ) from None
    for record in registry.models:
        if record.artifact_name is None:
            continue
        if Path(record.artifact_name).name != record.artifact_name:
            raise DatasetError("model_artifact_invalid", "Invalid registered classifier filename")
        path = model_directory / registry.provenance.run_id[7:] / record.artifact_name
        if file_hash(path) != record.artifact_hash:
            raise DatasetError(
                "model_artifact_hash_mismatch", "Classifier differs from registered bytes"
            )
