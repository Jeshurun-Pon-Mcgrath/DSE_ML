"""Explicit offline simulation → profile → split → features → train-only preprocessing."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.pipeline import profile_loaded, save_reports
from riskbyte_ml.datasets.synthetic_riskbyte import DATASET_NAME, NOTICE
from riskbyte_ml.datasets.synthetic_riskbyte.adapter import load_dataset
from riskbyte_ml.datasets.synthetic_riskbyte.config import SyntheticConfig
from riskbyte_ml.datasets.synthetic_riskbyte.models import SyntheticMetadata
from riskbyte_ml.evaluation.splits import SplitConfig, make_split, save_split
from riskbyte_ml.features.behaviour import FEATURE_PIPELINE_VERSION, build_features
from riskbyte_ml.features.manifest import column_catalog, feature_manifest
from riskbyte_ml.preprocessing.pipeline import (
    PreprocessingConfig,
    Preprocessor,
    load_preprocessor,
    matrix_summary,
)


def save_json(path, value):
    write_artifact(path, json.dumps(value, indent=2, allow_nan=False).encode())


def metadata(dataset, audit, memory, manifest):
    frame = dataset.frame
    counts = frame.groupby("behavioural_entity_id", observed=True).size()
    prior = bool((counts > 1).any())
    requirements = {
        "amount_deviation": "TransactionAmt",
        "transaction_velocity": "TransactionDT",
        "time_since_previous_transaction": "TransactionDT",
        "merchant_novelty": "merchant_id",
        "device_novelty": "device_id",
        "region_novelty": "billing_region",
        "product_novelty": "product_code",
        "failed_attempt_signal": "failed_attempts_before",
        "repeated_entity_history": "behavioural_entity_id",
    }
    availability = {
        name: "SUPPORTED"
        if prior and field in frame and frame[field].notna().any()
        else "UNSUPPORTED"
        for name, field in requirements.items()
    }
    return SyntheticMetadata(
        dataset_name=DATASET_NAME,
        notice=NOTICE,
        generator_version=manifest["generator_version"],
        seed=manifest["seed"],
        dataset_fingerprint=dataset.fingerprint,
        transaction_rows=audit.transaction_rows_before,
        identity_rows=audit.identity_rows_before,
        joined_rows=len(frame),
        fraud_count=int((frame.isFraud.astype("string") == "1").sum()),
        fraud_prevalence=float((frame.isFraud.astype("string") == "1").mean()),
        identity_coverage=audit.matched_identity_rows / len(frame),
        entity_count=int(frame.behavioural_entity_id.nunique()),
        merchant_count=int(frame.merchant_id.nunique()),
        device_count=int(frame.device_id.nunique()),
        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
        behavioural_feature_availability=availability,
        feature_classes=column_catalog(frame.columns),
        join_audit=audit,
        memory=memory,
    )


def integrate(config: SyntheticConfig):
    dataset, audit, memory, manifest = load_dataset(config)
    meta = metadata(dataset, audit, memory, manifest)
    # Only bind completed split metadata to the exact current dataset/version.
    path = config.phase2_summary_path
    if path is not None and path.is_file():
        try:
            if path.stat().st_size > 2_000_000:
                raise ValueError
            recorded = SyntheticMetadata.model_validate(json.loads(path.read_bytes())["dataset"])
            if (
                recorded.dataset_fingerprint == dataset.fingerprint
                and recorded.feature_pipeline_version == FEATURE_PIPELINE_VERSION
            ):
                meta = meta.model_copy(update={"split_strategy": recorded.split_strategy})
        except (OSError, ValueError, KeyError, TypeError):
            raise DatasetError(
                "synthetic_summary_invalid", "Synthetic preprocessing summary is invalid"
            ) from None
    return profile_loaded(dataset), meta


def save_integration(profile, report, directory, config):
    outputs = [
        directory / name
        for name in (
            "data_profile.json",
            "data_quality_report.md",
            "synthetic_integration.json",
            "join_audit.json",
        )
    ]
    if any(
        p.resolve()
        in {
            config.transaction_path.resolve(),
            config.identity_path.resolve(),
            config.metadata_path.resolve(),
        }
        for p in outputs
    ):
        raise DatasetError("dataset_source_overwrite_forbidden", "Reports cannot overwrite sources")
    save_reports(profile, directory)
    write_artifact(outputs[2], report.model_dump_json(indent=2).encode())
    write_artifact(outputs[3], report.join_audit.model_dump_json(indent=2).encode())


def run_phase2(config: SyntheticConfig, directory: Path, *, split_config=None):
    split_config = split_config or SplitConfig()
    dataset, audit, memory, manifest = load_dataset(config)
    profile = profile_loaded(dataset)
    meta = metadata(dataset, audit, memory, manifest)
    save_integration(profile, meta, directory / "synthetic_riskbyte", config)
    save_json(directory / "synthetic_dataset_manifest.json", manifest)
    # Explicit derived views for offline intelligence; source files remain untouched.
    save_json(
        directory / "synthetic_riskbyte" / "statistics.json",
        {k: v.model_dump(mode="json") for k, v in profile.descriptive_statistics.items()},
    )
    save_json(
        directory / "synthetic_riskbyte" / "schema.json",
        profile.schema_metadata.model_dump(mode="json"),
    )
    save_json(directory / "synthetic_riskbyte" / "eda.json", profile.eda.model_dump(mode="json"))
    if profile.quality.validation_status != "PASSED":
        raise DatasetError(
            "synthetic_validation_failed",
            "Generated dataset failed existing quality checks; inspect the report",
        )
    ev_path = config.transaction_path.parent / "evaluation_metadata.csv"
    ev_bytes = ev_path.read_bytes()
    if "sha256:" + hashlib.sha256(ev_bytes).hexdigest() != manifest["evaluation_fingerprint"]:
        raise DatasetError(
            "synthetic_evaluation_mismatch",
            "Evaluation-only sidecar does not match generated contents",
        )
    evaluation = pd.read_csv(ev_path, dtype={"TransactionID": "string"}, keep_default_na=False)
    if evaluation.TransactionID.tolist() != dataset.frame.TransactionID.astype("string").tolist():
        raise DatasetError(
            "synthetic_evaluation_mismatch", "Evaluation-only sidecar row order mismatch"
        )
    evaluation.hard_negative = evaluation.hard_negative.astype(str).str.lower().eq("true")
    features = build_features(
        dataset.frame, horizon_seconds=manifest["config"]["duration_days"] * 86400
    )
    specs = feature_manifest(features, dataset.frame.columns)
    splits = {
        strategy: make_split(
            dataset.frame,
            dataset.fingerprint,
            split_config.model_copy(update={"strategy": strategy}),
            evaluation,
        )
        for strategy in ("temporal", "random_stratified")
    }
    paths = {
        strategy: str(save_split(split, directory / "splits").name)
        for strategy, split in splits.items()
    }
    split = splits[split_config.strategy]
    if any(
        summary.fraud_count == 0 or summary.fraud_count == summary.row_count
        for summary in split.summaries.values()
    ):
        raise DatasetError(
            "synthetic_split_class_unavailable",
            "Primary split requires both classes in every partition",
        )
    families = {}
    for family in ("linear_distance", "tree_boosting"):
        processor = Preprocessor(
            PreprocessingConfig(
                family=family, scaling="robust" if family == "linear_distance" else "none"
            )
        )
        processor.fit_training(features, split, specs)
        artifact = processor.save(directory / "preprocessing")
        reloaded = load_preprocessor(artifact, dataset.fingerprint, split.split_id)
        summaries = {}
        for name, rows in split.indexes.items():
            subset = features.iloc[rows]
            matrix = processor.transform(subset)
            restored = reloaded.transform(subset)
            if matrix.shape != restored.shape or (matrix != restored).nnz:
                raise ValueError("Serialization changed transformed matrices")
            summaries[name] = matrix_summary(matrix)
        families[family] = {
            "preprocessing_artifact": artifact.name,
            "preprocessing_id": processor.preprocessing_id,
            "scaling": processor.config.scaling,
            "frequency_columns": processor.frequency,
            "onehot_columns": processor.onehot,
            "encoded_feature_names": processor.feature_names,
            "matrices": summaries,
            "serialization_reload_verified": True,
            "feature_manifest": [s.model_dump(mode="json") for s in processor.encoded_manifest()],
        }
    raw = sum(s.kind == "raw" for s in specs)
    feature_report = {
        "dataset_fingerprint": dataset.fingerprint,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "generator_version": manifest["generator_version"],
        "generator_schema": "synthetic-riskbyte-v1",
        "history_key": "behavioural_entity_id",
        "history_policy": "strict event-time prior; equal-time batch; no targets",
        "relative_time_horizon_seconds": manifest["config"]["duration_days"] * 86400,
        "raw_column_count": len(dataset.frame.columns),
        "raw_predictive_feature_count": raw,
        "derived_feature_count": len(specs) - raw,
        "unencoded_feature_count": len(specs),
        "feature_frame_memory_bytes": int(features.memory_usage(index=True, deep=True).sum()),
        "features": [s.model_dump(mode="json") for s in specs],
        "column_catalog": column_catalog([*dataset.frame.columns, *evaluation.columns]),
        "families": families,
        "split_id": split.split_id,
    }
    save_json(directory / "features" / "feature_manifest.json", feature_report)
    elapsed = pd.to_numeric(dataset.frame.TransactionDT).astype(float)
    target = dataset.frame.isFraud.astype("string").eq("1")
    relative = pd.DataFrame(
        {"elapsed_day_index": (elapsed // 86400).astype(int), "fraud": target.astype(int)}
    )
    relative_summary = (
        relative.groupby("elapsed_day_index")
        .fraud.agg(["size", "sum", "mean"])
        .reset_index()
        .to_dict("records")
    )
    save_json(
        directory / "synthetic_riskbyte" / "synthetic_eda.json",
        {
            "observation_type": "SYNTHETIC",
            "relative_time_distribution": relative_summary,
            "fraud_scenarios": manifest["scenario_distribution"],
            "hard_negatives": manifest["hard_negative_distribution"],
            "identity_coverage_by_class": {
                label: float(dataset.frame.loc[target == positive, "device_id"].notna().mean())
                for label, positive in [("fraud", True), ("legitimate", False)]
            },
        },
    )
    meta = meta.model_copy(update={"split_strategy": split_config.strategy})
    summary = {
        "dataset": meta.model_dump(mode="json"),
        "manifest": manifest,
        "split_strategy": split_config.strategy,
        "split_artifacts": paths,
        "splits": {k: v.model_dump(mode="json", exclude={"indexes"}) for k, v in splits.items()},
        "features": {
            k: v
            for k, v in feature_report.items()
            if k not in {"features", "column_catalog", "families"}
        },
        "families": {
            k: {
                n: x for n, x in v.items() if n not in {"feature_manifest", "encoded_feature_names"}
            }
            for k, v in families.items()
        },
        "quality_warning_count": len(profile.warnings),
        "warnings": [w.model_dump(mode="json") for w in profile.warnings],
    }
    save_json(directory / "phase2_summary.json", summary)
    write_artifact(
        directory / "synthetic_riskbyte" / "behavioural_feasibility.md",
        (
            "# Synthetic behavioural feasibility\n\n"
            + NOTICE
            + "\n\n"
            + "\n".join(
                f"- {name}: {status}"
                for name, status in meta.behavioural_feature_availability.items()
            )
            + "\n\nIdentity evidence is unavailable for individual missing-identity rows. "
            "Calendar dates, weekdays and actual GPS travel are unsupported.\n"
        ).encode(),
    )
    return summary
