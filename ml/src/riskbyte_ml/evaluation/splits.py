"""Immutable fingerprint-bound row-position splits, with chronological tie boundaries."""

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.models import ReportModel
from riskbyte_ml.datasets.synthetic_riskbyte.config import digest
from riskbyte_ml.features.behaviour import FEATURE_PIPELINE_VERSION, finite_numeric

SPLIT_VERSION = "split-v1:no-time-tie-crossing"


class SplitConfig(ReportModel):
    strategy: Literal["temporal", "random_stratified"] = "temporal"
    train_ratio: float = Field(default=0.7, gt=0, lt=1, allow_inf_nan=False)
    validation_ratio: float = Field(default=0.15, gt=0, lt=1, allow_inf_nan=False)
    seed: int = 42

    @model_validator(mode="after")
    def ratios(self):
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("A positive test fraction is required")
        return self


class PartitionSummary(ReportModel):
    row_count: int
    fraud_count: int
    fraud_prevalence: float
    entity_count: int
    elapsed_min: float
    elapsed_max: float
    scenario_distribution: dict[str, int]
    hard_negative_count: int


class SplitArtifact(ReportModel):
    dataset_fingerprint: str
    row_order_fingerprint: str
    split_strategy: Literal["temporal", "random_stratified"]
    split_version: str = SPLIT_VERSION
    feature_pipeline_version: str = FEATURE_PIPELINE_VERSION
    split_id: str
    config: SplitConfig
    seed: int | None
    indexes: dict[str, list[int]]
    summaries: dict[str, PartitionSummary]
    entity_overlap: dict[str, int]


def validate_partition(artifact, frame):
    if (
        artifact.feature_pipeline_version != FEATURE_PIPELINE_VERSION
        or artifact.split_version != SPLIT_VERSION
    ):
        raise ValueError("Incompatible split or feature version")
    indexes = [i for values in artifact.indexes.values() for i in values]
    if sorted(indexes) != list(range(len(frame))) or set(artifact.indexes) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Split must partition all rows exactly once")
    if not all(artifact.indexes.values()):
        raise ValueError("Every partition requires observations")
    if digest(frame.TransactionID.astype(str).tolist()) != artifact.row_order_fingerprint:
        raise ValueError("Dataset row order does not match split")


def make_split(frame, fingerprint, config=None, evaluation=None):
    config = config or SplitConfig()
    target = frame.isFraud.astype("string")
    if target.isna().any() or not target.isin(["0", "1"]).all():
        raise ValueError("Unsupported or missing target values")
    labels = target.astype(int).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("Valid binary target with both classes required for splitting")
    times = finite_numeric(frame.TransactionDT)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Finite elapsed time required")
    if config.strategy == "temporal":
        order = np.argsort(times, kind="stable")
        a, b = (
            int(len(frame) * config.train_ratio),
            int(len(frame) * (config.train_ratio + config.validation_ratio)),
        )
        while 0 < a < len(order) and times[order[a]] == times[order[a - 1]]:
            a += 1
        while 0 < b < len(order) and times[order[b]] == times[order[b - 1]]:
            b += 1
        parts = (order[:a], order[a:b], order[b:])
    else:
        from sklearn.model_selection import train_test_split

        train, remaining = train_test_split(
            np.arange(len(frame)),
            train_size=config.train_ratio,
            stratify=labels,
            random_state=config.seed,
        )
        val, test = train_test_split(
            remaining,
            train_size=config.validation_ratio / (1 - config.train_ratio),
            stratify=labels[remaining],
            random_state=config.seed,
        )
        parts = (np.sort(train), np.sort(val), np.sort(test))
    names = ("train", "validation", "test")
    if any(len(p) == 0 for p in parts):
        raise ValueError("Temporal ties or ratios produced an empty partition")
    summaries, entities = {}, {}
    for name, rows in zip(names, parts, strict=True):
        subset = frame.iloc[rows]
        entities[name] = set(subset.behavioural_entity_id.astype(str))
        ev = evaluation.iloc[rows] if evaluation is not None else None
        summaries[name] = PartitionSummary(
            row_count=len(rows),
            fraud_count=int(labels[rows].sum()),
            fraud_prevalence=float(labels[rows].mean()),
            entity_count=len(entities[name]),
            elapsed_min=float(times[rows].min()),
            elapsed_max=float(times[rows].max()),
            scenario_distribution={}
            if ev is None
            else {
                str(k): int(v)
                for k, v in ev.loc[ev.fraud_scenario != "", "fraud_scenario"].value_counts().items()
            },
            hard_negative_count=0 if ev is None else int(ev.hard_negative.sum()),
        )
    config_data = config.model_dump(mode="json")
    indexes = {name: rows.tolist() for name, rows in zip(names, parts, strict=True)}
    artifact = SplitArtifact(
        dataset_fingerprint=fingerprint,
        row_order_fingerprint=digest(frame.TransactionID.astype(str).tolist()),
        split_strategy=config.strategy,
        config=config,
        seed=config.seed if config.strategy != "temporal" else None,
        split_id=digest(
            {
                "fingerprint": fingerprint,
                "policy": config_data,
                "version": SPLIT_VERSION,
                "features": FEATURE_PIPELINE_VERSION,
                "indexes": indexes,
            }
        ),
        indexes=indexes,
        summaries=summaries,
        entity_overlap={
            f"{a}_{b}": len(entities[a] & entities[b])
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
        },
    )
    validate_partition(artifact, frame)
    return artifact


def save_split(artifact: SplitArtifact, directory: Path):
    path = directory / (artifact.split_id.removeprefix("sha256:") + ".json")
    if path.exists():
        existing = SplitArtifact.model_validate_json(path.read_bytes())
        if existing != artifact:
            raise ValueError("Existing split artifact is incompatible")
    else:
        write_artifact(path, artifact.model_dump_json(indent=2).encode())
    return path
