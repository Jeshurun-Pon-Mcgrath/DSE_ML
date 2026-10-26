"""Bind SHAP-complete publication to its exact bundle and preserved historical snapshot."""

import json
from datetime import UTC, datetime

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability.completion import BASELINE_HASH, CASE_IDS
from riskbyte_ml.explainability.schemas import SHAP_VERSION
from riskbyte_ml.statistics.verification import file_hash, read_json


def freeze_snapshot(directory, bundle):
    if bundle.version != SHAP_VERSION or bundle.shap_status != "AVAILABLE":
        raise DatasetError("explanation_not_complete", "Only genuine SHAP snapshots can be frozen")
    metadata = read_json(directory / "explainer_metadata.json")
    if (
        not metadata["non_shap_evidence_preserved"]
        or metadata["historical_snapshot_hash"] != BASELINE_HASH
    ):
        raise DatasetError(
            "explanation_not_complete", "Historical evidence preservation is required"
        )
    value = {
        "version": SHAP_VERSION,
        "status": "FROZEN_SHAP_COMPLETE",
        "created_timestamp": datetime.now(UTC).isoformat(),
        "bundle_hash": file_hash(directory / "analysis_bundle.json"),
        "artifact_hashes": bundle.artifacts,
        "historical_snapshot_hash": BASELINE_HASH,
        "case_ids": CASE_IDS,
        "provenance": bundle.provenance,
        "original_temporal_test_accessed": False,
        "future_holdout_status": "PREPARATION_ONLY_NOT_GENERATED",
    }
    write_artifact(
        directory / "snapshot_freeze.json", json.dumps(value, indent=2).encode(), overwrite=False
    )


def verify_freeze(directory, bundle):
    value = read_json(directory / "snapshot_freeze.json", 100_000)
    if (
        value["version"] != SHAP_VERSION
        or value["status"] != "FROZEN_SHAP_COMPLETE"
        or value["bundle_hash"] != file_hash(directory / "analysis_bundle.json")
        or value["artifact_hashes"] != bundle.artifacts
        or value["historical_snapshot_hash"] != BASELINE_HASH
        or value["case_ids"] != CASE_IDS
        or value["provenance"] != bundle.provenance
        or value["original_temporal_test_accessed"] is not False
        or value["future_holdout_status"] != "PREPARATION_ONLY_NOT_GENERATED"
    ):
        raise DatasetError("explanation_freeze_changed", "Frozen SHAP publication integrity failed")
