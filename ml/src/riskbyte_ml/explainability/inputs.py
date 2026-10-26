"""Pinned Phase 4/5/6 provenance and a development-only scoring projection."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskbyte_ml.abrf.contracts import P_THRESHOLD, ABRFBundle
from riskbyte_ml.abrf.engine import fusion_arrays
from riskbyte_ml.abrf.inputs import load_inputs, verify_components
from riskbyte_ml.abrf.pipeline import verify_completed
from riskbyte_ml.anomaly.development import DevelopmentInputs
from riskbyte_ml.anomaly.pipeline import BASELINE_HASH, BASELINE_ID
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability.metadata import encoded_metadata
from riskbyte_ml.explainability.supervised import TreeShap
from riskbyte_ml.models.supervised.artifacts import load_classifier
from riskbyte_ml.models.supervised.estimators import positive_probability
from riskbyte_ml.preprocessing.pipeline import load_preprocessor
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import file_hash, read_json

ACTIVE_HASH = "sha256:223b8f52c3d791f31014baa08181a5d7ade40999c9287aad2156606382efb1a2"
ABRF_HASH = "sha256:0cdc5286a11e532730a8a1fc31294aeb20c893f23a00289a6ad971e59ad5df46"
ABRF_RUN = "c8b265bcaa4c91db3a41a4ee8f708281ee4fe224bcf8ad369a7121922bb057b0"


def verify_selection(config, root):
    _, _, provenance = verify_components(config, root)
    active_path = root / "abrf" / "active.json"
    directory = root / "abrf" / ABRF_RUN
    if file_hash(active_path) != ACTIVE_HASH or (
        file_hash(directory / "analysis_bundle.json") != ABRF_HASH
    ):
        raise DatasetError("explanation_frozen_selection_changed", "Frozen ABRF selection changed")
    active = read_json(active_path)
    if active != {"run_id": ABRF_RUN, "bundle_hash": ABRF_HASH}:
        raise DatasetError("explanation_frozen_selection_changed", "Active selection differs")
    bundle = ABRFBundle.model_validate_json((directory / "analysis_bundle.json").read_bytes())
    verify_completed(directory, bundle, provenance)
    selected = bundle.selected
    if (
        selected.weights.model_dump() != {"wp": 0.9, "wa": 0.0, "wb": 0.1}
        or selected.confidence_curve != "gated_0.5"
        or selected.decision_threshold != 0.29755749437222373
        or selected.version != "abrf-v1"
    ):
        raise DatasetError("explanation_frozen_selection_changed", "Frozen fusion contract differs")
    provenance = {
        **provenance,
        "abrf_version": selected.version,
        "abrf_active_hash": ACTIVE_HASH,
        "abrf_bundle_hash": ABRF_HASH,
    }
    return bundle, directory, provenance


def equal_scores(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True):
        raise DatasetError(
            "explanation_inconsistent", f"Frozen {label} output cannot be reproduced"
        )


class ExplanationInputs:
    """Reproduce only VALIDATION scores; no original-test accessor or fitting path."""

    def __init__(self, config, root: Path):
        self.bundle, self.abrf_directory, self.provenance = verify_selection(config, root)
        certified, _, _ = load_inputs(config, root)
        development = DevelopmentInputs(config, root)
        n = development.train_count
        self.features = development.features.iloc[n:].reset_index(drop=True)
        self.context = development.frame.iloc[n:].reset_index(drop=True)
        self.metadata = development.metadata.iloc[n:].reset_index(drop=True)
        self.labels = development.labels[n:]
        self.scores = certified.iloc[n:].reset_index(drop=True)
        if not self.context.TransactionID.equals(self.scores.TransactionID):
            raise DatasetError("explanation_row_alignment", "Validation event alignment failed")
        family = development.manifest["families"]["tree_boosting"]
        self.preprocessor = load_preprocessor(
            root / "preprocessing" / family["preprocessing_artifact"],
            FROZEN_FINGERPRINT,
            FROZEN_SPLIT_ID,
        )
        registry = read_json(root / "models" / "model_registry.json")
        model_path = (
            root / "models" / registry["provenance"]["run_id"][7:] / (BASELINE_ID + ".joblib")
        )
        self.classifier = load_classifier(
            model_path, BASELINE_HASH, FROZEN_FINGERPRINT, FROZEN_FEATURE_VERSION
        )
        self.names = self.classifier.feature_names
        self.feature_metadata = encoded_metadata(development.manifest, self.names)
        if (
            self.classifier.preprocessing_id != family["preprocessing_id"]
            or self.names != self.preprocessor.feature_names
            or self.classifier.threshold != P_THRESHOLD
            or self.classifier.model_id != BASELINE_ID
        ):
            raise DatasetError(
                "explanation_feature_mismatch", "Frozen preprocessing identity differs"
            )
        calibration = self.classifier.calibrator
        if not calibration.fitted or calibration.method != "sigmoid":
            raise DatasetError("explanation_calibration_mismatch", "Frozen calibration differs")
        self.provenance["calibration_version"] = "supervised-classification-v1/sigmoid"
        self.provenance["calibration_integrity_hash"] = BASELINE_HASH
        self.provenance["calibration_version_semantics"] = (
            "Derived provenance label; sigmoid calibrator is embedded in the frozen model bytes"
        )
        self.matrix = self.preprocessor.transform(self.features)
        self.raw_probability = positive_probability(self.classifier.estimator, self.matrix)
        equal_scores(
            self.classifier.probability(self.matrix), self.scores.P, "calibrated classifier"
        )
        # Only deserialize executable artifacts after pinned checksums have succeeded.
        self.anomaly = joblib.load(root / "phase5" / "isolation_forest.joblib")
        self.behaviour = joblib.load(root / "phase5" / "behavioural_engine.joblib")
        for engine, expected in [
            (self.anomaly, "anomaly-detection-v1"),
            (self.behaviour, "behavioural-deviation-v1"),
        ]:
            if (
                engine.version != expected
                or engine.dataset_fingerprint != FROZEN_FINGERPRINT
                or engine.feature_version != FROZEN_FEATURE_VERSION
                or engine.split_id != FROZEN_SPLIT_ID
            ):
                raise DatasetError("explanation_engine_mismatch", "Frozen engine identity differs")
        self.original_anomaly, a = self.anomaly.score(self.features)
        b, c, _, _, _ = self.behaviour.score_arrays(self.features)
        for actual, expected, name in [
            (a, self.scores.A, "anomaly"),
            (b, self.scores.B, "behaviour"),
            (c, self.scores.behaviour_confidence, "confidence"),
        ]:
            equal_scores(actual, expected, name)
        self.fusion_scores = (
            pd.read_csv(
                self.abrf_directory / "development_fusion_scores.csv.gz",
                float_precision="round_trip",
                keep_default_na=False,
                dtype={"TransactionID": "string", "B": "string"},
            )
            .iloc[n:]
            .reset_index(drop=True)
        )
        if not self.fusion_scores.TransactionID.equals(self.scores.TransactionID):
            raise DatasetError("explanation_row_alignment", "Fusion event alignment failed")
        selected = self.bundle.selected
        r, weights, contributions = fusion_arrays(
            self.scores.P,
            self.scores.A,
            self.scores.B,
            self.scores.behaviour_confidence,
            selected.weights,
            selected.confidence_curve,
        )
        equal_scores(r, self.fusion_scores.final_risk_score, "validation fusion")
        expected_levels = np.array(["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"])[
            np.searchsorted(selected.risk_boundaries, r, side="right")
        ]
        if not np.array_equal(expected_levels, self.fusion_scores.risk_level) or not np.array_equal(
            r >= selected.decision_threshold, self.fusion_scores.investigation_flag
        ):
            raise DatasetError("explanation_inconsistent", "Frozen risk bands/decisions differ")
        for index, name in enumerate(["supervised", "anomaly", "behaviour"]):
            equal_scores(weights[:, index], self.fusion_scores["effective_weight_" + name], name)
            equal_scores(contributions[:, index], self.fusion_scores["contribution_" + name], name)
        self.shap = TreeShap(self.classifier.estimator, self.names)
        self.audit = {
            **development.audit,
            "validation_rows_scored": len(self.features),
            "test_rows_explained": 0,
            "future_holdout_generated": False,
            "component_models_changed": False,
        }
        self.config, self.root = config, root

    def verify_unchanged(self):
        _, _, after = verify_selection(self.config, self.root)
        if any(self.provenance[key] != value for key, value in after.items()):
            raise DatasetError(
                "explanation_sources_changed", "Frozen inputs changed during execution"
            )
