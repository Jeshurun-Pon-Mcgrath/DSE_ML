"""Verify frozen inputs without loading models or accessing original test records."""

from pathlib import Path

import numpy as np
import pandas as pd

from riskbyte_ml.abrf.contracts import A_THRESHOLD, B_THRESHOLD, P_THRESHOLD
from riskbyte_ml.anomaly.pipeline import BASELINE_HASH, BASELINE_ID
from riskbyte_ml.anomaly.schemas import Phase5Bundle
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.models.supervised.inputs import verify_frozen_files
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import (
    file_hash,
    frozen_artifacts,
    read_json,
    verify_sources,
)

FROZEN_PHASE5_HASH = "sha256:0b317ec40e1d7e9d1a9ca8a44a9032b958a3ec2dbdaa317199d8d076b2d356de"


def safe_child(directory: Path, name: str):
    path = directory / name
    if (
        path.resolve().parent != directory.resolve()
        and directory.resolve() not in path.resolve().parents
    ):
        raise DatasetError("abrf_artifact_invalid", "Unsafe artifact reference")
    if path.is_symlink():
        raise DatasetError("abrf_artifact_invalid", "Linked artifacts are unsupported")
    return path


def verify_components(config, root):
    verify_sources(config)  # Opaque content hashes only; no original transaction CSV parser.
    verify_frozen_files(root)
    split = frozen_artifacts(root)
    path = root / "phase5" / "analysis_bundle.json"
    if file_hash(path) != FROZEN_PHASE5_HASH:
        raise DatasetError("abrf_frozen_component_changed", "Frozen Phase 5 snapshot changed")
    bundle = Phase5Bundle.model_validate_json(path.read_bytes())
    for name, expected in bundle.artifact_hashes.items():
        if Path(name).name != name or file_hash(safe_child(root / "phase5", name)) != expected:
            raise DatasetError("abrf_frozen_component_changed", "Frozen Phase 5 artifact changed")
    registry_path = root / "models" / "model_registry.json"
    if file_hash(registry_path) != bundle.baseline["registry_hash"]:
        raise DatasetError("abrf_frozen_component_changed", "Phase 4 registry changed")
    registry = read_json(registry_path)
    selected = [record for record in registry["models"] if record["is_selected_supervised_model"]]
    if len(selected) != 1:
        raise DatasetError(
            "abrf_baseline_invalid", "Exactly one frozen supervised component required"
        )
    record = selected[0]
    if (
        record["model_id"] != BASELINE_ID
        or record["artifact_hash"] != BASELINE_HASH
        or record["selected_threshold"] != P_THRESHOLD
        or record["version"] != "supervised-classification-v1"
        or record["feature_version"] != FROZEN_FEATURE_VERSION
        or record["split_id"] != FROZEN_SPLIT_ID
        or bundle.anomaly_metrics["threshold"] != A_THRESHOLD
        or bundle.behaviour_metrics["threshold"] != B_THRESHOLD
        or bundle.selected["normalization"] != "training-reference strict-left empirical CDF"
    ):
        raise DatasetError("abrf_component_contract_changed", "Frozen component contract differs")
    run = registry["provenance"]["run_id"][7:]
    if len(run) != 64 or not all(c in "0123456789abcdef" for c in run):
        raise DatasetError("abrf_baseline_invalid", "Invalid registered run identity")
    if file_hash(root / "models" / run / (BASELINE_ID + ".joblib")) != BASELINE_HASH:
        raise DatasetError("abrf_frozen_component_changed", "Frozen Decision Tree bytes changed")
    provenance = {
        "dataset_fingerprint": FROZEN_FINGERPRINT,
        "feature_version": FROZEN_FEATURE_VERSION,
        "split_id": FROZEN_SPLIT_ID,
        "split_version": split.split_version,
        "supervised_version": record["version"],
        "supervised_model_id": BASELINE_ID,
        "supervised_threshold": P_THRESHOLD,
        "anomaly_version": bundle.anomaly_version,
        "anomaly_threshold": A_THRESHOLD,
        "behaviour_version": bundle.behaviour_version,
        "behaviour_threshold": B_THRESHOLD,
        "phase5_bundle_hash": FROZEN_PHASE5_HASH,
        "phase5_artifact_hashes": bundle.artifact_hashes,
        "supervised_registry_hash": bundle.baseline["registry_hash"],
        "supervised_artifact_hash": BASELINE_HASH,
        "fusion_input_version": "fusion-input-v1",
    }
    return bundle, split, provenance


def load_inputs(config, root):
    bundle, split, provenance = verify_components(config, root)
    # Hash is pinned and verified BEFORE decoding this train/validation-only sidecar.
    frame = pd.read_csv(
        root / "phase5" / "development_scores.csv",
        float_precision="round_trip",
        keep_default_na=False,
        dtype={
            "TransactionID": "string",
            "fraud_scenario": "string",
            "hard_negative_type": "string",
            "B": "string",
        },
        low_memory=False,
    )
    required = {
        "partition",
        "TransactionID",
        "P",
        "A",
        "B",
        "behaviour_confidence",
        "ground_truth",
        "fraud_scenario",
        "hard_negative",
        "hard_negative_type",
    }
    if not required <= set(frame) or set(frame.partition) != {"train", "validation"}:
        raise DatasetError(
            "abrf_partition_invalid", "Only TRAIN/VALIDATION artifacts may be decoded"
        )
    if frame.TransactionID.duplicated().any():
        raise DatasetError("abrf_partition_invalid", "Development event IDs must be unique")
    for name in ["P", "A", "B", "behaviour_confidence"]:
        if name == "B":
            # Python float parsing preserves the exact CSV round trip for nullable B;
            # pd.to_numeric's string parser can shift a saved threshold by one ULP.
            frame[name] = (
                frame[name].map(lambda value: float(value) if value else np.nan).astype(float)
            )
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
    y = frame.ground_truth.to_numpy()
    if not np.isin(y, [0, 1]).all() or frame.hard_negative.dtype != bool:
        raise DatasetError("abrf_target_invalid", "Binary evaluation labels/flags required")
    if (frame.hard_negative.to_numpy() & (y == 1)).any():
        raise DatasetError("abrf_target_invalid", "Hard negatives must be legitimate")
    train = len(split.indexes["train"])
    count = train + len(split.indexes["validation"])
    if len(frame) != count or frame.partition.tolist() != ["train"] * train + ["validation"] * (
        count - train
    ):
        raise DatasetError(
            "abrf_partition_invalid", "Development rows differ from frozen temporal split"
        )
    for name, group in frame.groupby("partition", sort=False):
        if (
            len(group) != split.summaries[name].row_count
            or int(group.ground_truth.sum()) != split.summaries[name].fraud_count
        ):
            raise DatasetError("abrf_partition_invalid", "Frozen partition label integrity failed")
    return frame, bundle, provenance
