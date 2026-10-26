"Fail closed on changed source contents, feature semantics, splits or ground truth."

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte import GENERATOR_VERSION
from riskbyte_ml.datasets.synthetic_riskbyte.adapter import load_dataset
from riskbyte_ml.datasets.synthetic_riskbyte.config import (
    HARD_NEGATIVES,
    SCENARIOS,
    GeneratorConfig,
    digest,
)
from riskbyte_ml.evaluation.splits import SplitArtifact, validate_partition
from riskbyte_ml.features.behaviour import FEATURE_PIPELINE_VERSION
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID


def file_hash(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            result = hashlib.file_digest(stream, "sha256")
        return "sha256:" + result.hexdigest()
    except OSError:
        raise DatasetError(
            "statistics_source_missing", "Required statistical source is absent"
        ) from None


def read_json(path: Path, limit=25_000_000):
    try:
        if path.stat().st_size > limit:
            raise ValueError
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise DatasetError(
            "statistics_artifact_invalid", "Statistical provenance artifact is absent or invalid"
        ) from None


def verify_fingerprint(fingerprint: str):
    if fingerprint != FROZEN_FINGERPRINT:
        raise DatasetError(
            "statistics_fingerprint_mismatch",
            "Dataset differs from the frozen Phase 3 fingerprint; analysis aborted",
        )


def verify_sources(config):
    manifest = read_json(config.metadata_path, 1_000_000)
    try:
        GeneratorConfig.model_validate(manifest["config"])
        sources = {
            "transaction": file_hash(config.transaction_path),
            "identity": file_hash(config.identity_path),
        }
        fingerprint = digest(
            {
                "sources": sources,
                "config": manifest["config"],
                "generator_version": manifest["generator_version"],
            }
        )
        verify_fingerprint(fingerprint)
        if (
            manifest["fingerprint"] != fingerprint
            or manifest["source_fingerprints"] != sources
            or manifest["generator_version"] != GENERATOR_VERSION
            or manifest["dataset_type"] != "SYNTHETIC"
        ):
            raise ValueError
        if (
            file_hash(config.transaction_path.parent / "evaluation_metadata.csv")
            != manifest["evaluation_fingerprint"]
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise DatasetError(
            "statistics_provenance_invalid", "Synthetic source or evaluation provenance is invalid"
        ) from None
    return manifest


def frozen_artifacts(directory: Path):
    try:
        manifest = read_json(directory / "features" / "feature_manifest.json")
        verify_fingerprint(manifest["dataset_fingerprint"])
        if (
            FEATURE_PIPELINE_VERSION != FROZEN_FEATURE_VERSION
            or manifest["feature_pipeline_version"] != FROZEN_FEATURE_VERSION
            or manifest["split_id"] != FROZEN_SPLIT_ID
        ):
            raise ValueError
        split = SplitArtifact.model_validate(
            read_json(directory / "splits" / (FROZEN_SPLIT_ID[7:] + ".json"))
        )
        verify_fingerprint(split.dataset_fingerprint)
        computed = digest(
            {
                "fingerprint": split.dataset_fingerprint,
                "policy": split.config.model_dump(mode="json"),
                "version": split.split_version,
                "features": split.feature_pipeline_version,
                "indexes": split.indexes,
            }
        )
        if (
            split.split_id != computed
            or split.split_id != FROZEN_SPLIT_ID
            or split.split_strategy != "temporal"
            or split.config.strategy != "temporal"
            or split.feature_pipeline_version != FROZEN_FEATURE_VERSION
        ):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise DatasetError(
            "statistics_frozen_artifact_mismatch",
            "Frozen feature version or temporal split is invalid",
        ) from None
    return split


def load_verified(config, directory: Path):
    manifest = verify_sources(config)
    split = frozen_artifacts(directory)
    dataset, _, _, loaded_manifest = load_dataset(config)
    verify_fingerprint(dataset.fingerprint)
    if loaded_manifest != manifest:
        raise DatasetError("statistics_source_changed", "Sources changed during verification")
    frame = dataset.frame
    target = frame.isFraud.astype("string")
    if (
        target.isna().any()
        or set(target.unique()) != {"0", "1"}
        or int(target.eq("1").sum()) != manifest["fraud_count"]
    ):
        raise DatasetError("statistics_target_invalid", "Frozen target integrity failed")
    try:
        validate_partition(split, frame)
        time = pd.to_numeric(frame.TransactionDT, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(time).all() or (time < 0).any():
            raise ValueError
        previous = -1.0
        for name in ("train", "validation", "test"):
            rows = split.indexes[name]
            times = time[rows]
            recorded = split.summaries[name]
            if (np.diff(times) < 0).any() or times.min() <= previous:
                raise ValueError
            if recorded.row_count != len(rows) or recorded.fraud_count != int(
                target.iloc[rows].eq("1").sum()
            ):
                raise ValueError
            previous = times.max()
    except (ValueError, KeyError):
        raise DatasetError(
            "statistics_split_invalid",
            "Frozen temporal row order, partition or target counts are invalid",
        ) from None
    evaluation_path = config.transaction_path.parent / "evaluation_metadata.csv"
    evaluation = pd.read_csv(evaluation_path, dtype="string", keep_default_na=False)
    if (
        evaluation.columns.tolist()
        != ["TransactionID", "fraud_scenario", "hard_negative", "hard_negative_type"]
        or evaluation.TransactionID.tolist() != frame.TransactionID.astype("string").tolist()
        or not evaluation.hard_negative.isin(["True", "False"]).all()
    ):
        raise DatasetError(
            "statistics_evaluation_invalid", "Evaluation-only metadata integrity failed"
        )
    evaluation["hard_negative"] = evaluation.hard_negative.eq("True")
    fraud = target.eq("1").to_numpy()
    hard = evaluation.hard_negative.to_numpy(dtype=bool)
    if (
        (hard & fraud).any()
        or not evaluation.loc[fraud, "fraud_scenario"].isin(SCENARIOS).all()
        or evaluation.loc[~fraud, "fraud_scenario"].ne("").any()
        or not evaluation.loc[hard, "hard_negative_type"].isin(HARD_NEGATIVES).all()
        or evaluation.loc[~hard, "hard_negative_type"].ne("").any()
    ):
        raise DatasetError(
            "statistics_evaluation_invalid", "Scenario or hard-negative metadata contradicts target"
        )
    # Detect changes after CSV parsing as well as before it.
    verify_sources(config)
    return frame, evaluation, split, manifest
