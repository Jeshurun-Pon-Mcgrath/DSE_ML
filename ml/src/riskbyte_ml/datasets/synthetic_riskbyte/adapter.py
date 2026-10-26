"""Synthetic source verification and reuse of the audited two-table join boundary."""

import json

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import LoadedDataset
from riskbyte_ml.data.streaming import load_streamed_csv
from riskbyte_ml.datasets.join_models import MemoryAudit
from riskbyte_ml.datasets.synthetic_riskbyte import GENERATOR_VERSION
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig, SyntheticConfig, digest
from riskbyte_ml.datasets.two_table import audited_left_join


def load_dataset(config: SyntheticConfig):
    if not all(
        p.is_file() for p in (config.transaction_path, config.identity_path, config.metadata_path)
    ):
        raise DatasetError(
            "synthetic_files_missing",
            "Synthetic CSVs or generation metadata are absent; run the generator explicitly",
        )
    try:
        if config.metadata_path.stat().st_size > 1_000_000:
            raise ValueError
        manifest = json.loads(config.metadata_path.read_bytes())
        GeneratorConfig.model_validate(manifest["config"])
        if (
            manifest["dataset_type"] != "SYNTHETIC"
            or manifest["generator_version"] != GENERATOR_VERSION
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise DatasetError(
            "synthetic_manifest_invalid", "Synthetic generation metadata is invalid or incompatible"
        ) from None
    tx = load_streamed_csv(
        config.table_config("transaction"),
        chunk_rows=config.chunk_rows,
        max_memory_bytes=config.max_retained_memory_bytes,
        optimize_all_strings=True,
    )
    identity = load_streamed_csv(
        config.table_config("identity"),
        chunk_rows=config.chunk_rows,
        max_memory_bytes=config.max_retained_memory_bytes - tx.memory_after_bytes,
        optimize_all_strings=True,
    )
    sources = {"transaction": tx.dataset.fingerprint, "identity": identity.dataset.fingerprint}
    fingerprint = digest(
        {"sources": sources, "config": manifest["config"], "generator_version": GENERATOR_VERSION}
    )
    if fingerprint != manifest["fingerprint"] or sources != manifest["source_fingerprints"]:
        raise DatasetError(
            "synthetic_fingerprint_mismatch",
            "Synthetic contents do not match the generation manifest",
        )
    frame, audit = audited_left_join(
        tx.dataset.frame,
        identity.dataset.frame,
        config.analysis_config(tx.dataset.frame.columns),
        key="TransactionID",
        target="isFraud",
        required=("TransactionAmt", "TransactionDT", "behavioural_entity_id"),
        prefix="synthetic",
    )
    memory = int(frame.memory_usage(index=True, deep=True).sum())
    if memory > config.max_retained_memory_bytes:
        raise DatasetError("dataset_memory_limit", "Joined synthetic frame exceeds memory budget")
    memory_audit = MemoryAudit(
        transaction_before_bytes=tx.memory_before_bytes,
        transaction_after_bytes=tx.memory_after_bytes,
        identity_before_bytes=identity.memory_before_bytes,
        identity_after_bytes=identity.memory_after_bytes,
        joined_before_bytes=memory,
        joined_after_bytes=memory,
        optimized_columns=sorted(set((*tx.optimized_columns, *identity.optimized_columns))),
    )
    return (
        LoadedDataset(frame, config.analysis_config(frame.columns), fingerprint),
        audit,
        memory_audit,
        manifest,
    )
