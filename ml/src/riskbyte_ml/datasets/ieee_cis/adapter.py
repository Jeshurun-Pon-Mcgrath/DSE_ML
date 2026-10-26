"""Audited transaction-preserving join and content/configuration provenance."""

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import LoadedDataset
from riskbyte_ml.data.streaming import load_streamed_csv, optimize_categories
from riskbyte_ml.datasets.ieee_cis.config import (
    ADAPTER_VERSION,
    AMOUNT,
    ELAPSED,
    JOIN_KEY,
    TARGET,
    IEEEConfig,
    verify_sources,
)
from riskbyte_ml.datasets.ieee_cis.models import JoinAudit, MemoryAudit
from riskbyte_ml.datasets.two_table import JoinRejected as JoinRejected
from riskbyte_ml.datasets.two_table import audited_left_join


@dataclass(frozen=True)
class JoinedDataset:
    dataset: LoadedDataset
    join_audit: JoinAudit
    memory: MemoryAudit
    source_fingerprints: dict[str, str]
    configuration_fingerprint: str


def configuration_fingerprint(config: IEEEConfig) -> str:
    policy = config.model_dump(mode="json", exclude={"transaction_path", "identity_path"})
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def combined_fingerprint(sources: dict[str, str], config: IEEEConfig) -> str:
    manifest = {
        "adapter_version": ADAPTER_VERSION,
        "join": "left:one_to_one:preserve_order",
        "source_contents": sources,
        "configuration": configuration_fingerprint(config),
    }
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def join_tables(transaction: pd.DataFrame, identity: pd.DataFrame, config: IEEEConfig):
    return audited_left_join(
        transaction,
        identity,
        config.analysis_config(transaction.columns),
        key=JOIN_KEY,
        target=TARGET,
        required=(AMOUNT, ELAPSED),
        prefix="ieee_cis",
    )


def load_joined(config: IEEEConfig) -> JoinedDataset:
    verify_sources(config)
    transaction_policy = config.table_config("transaction")
    identity_policy = config.table_config("identity")
    if not config.optimize_categories:
        transaction_policy = transaction_policy.model_copy(update={"categorical_columns": ()})
        identity_policy = identity_policy.model_copy(update={"categorical_columns": ()})
    transaction = load_streamed_csv(
        transaction_policy,
        chunk_rows=config.chunk_rows,
        max_memory_bytes=config.max_retained_memory_bytes,
        optimize_all_strings=config.optimize_categories,
    )
    remaining = config.max_retained_memory_bytes - transaction.memory_after_bytes
    if remaining <= 0:
        raise DatasetError("dataset_memory_limit", "No retained-memory budget remains for identity")
    identity = load_streamed_csv(
        identity_policy,
        chunk_rows=config.chunk_rows,
        max_memory_bytes=remaining,
        optimize_all_strings=config.optimize_categories,
    )
    frame, audit = join_tables(transaction.dataset.frame, identity.dataset.frame, config)
    policy = config.analysis_config(frame.columns)
    categories = policy.categorical_columns if config.optimize_categories else ()
    before, after, optimized = optimize_categories(frame, categories)
    if after > config.max_retained_memory_bytes:
        raise DatasetError("dataset_memory_limit", "Joined data exceeds retained-memory budget")
    sources = {
        "transaction": transaction.dataset.fingerprint,
        "identity": identity.dataset.fingerprint,
    }
    memory = MemoryAudit(
        transaction_before_bytes=transaction.memory_before_bytes,
        transaction_after_bytes=transaction.memory_after_bytes,
        identity_before_bytes=identity.memory_before_bytes,
        identity_after_bytes=identity.memory_after_bytes,
        joined_before_bytes=before,
        joined_after_bytes=after,
        optimized_columns=sorted(
            set((*transaction.optimized_columns, *identity.optimized_columns, *optimized))
        ),
    )
    return JoinedDataset(
        LoadedDataset(frame, policy, combined_fingerprint(sources, config)),
        audit,
        memory,
        sources,
        configuration_fingerprint(config),
    )
