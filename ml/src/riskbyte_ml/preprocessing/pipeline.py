"""Train-only numeric and categorical transforms with sparse family outputs."""

import io
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.datasets.synthetic_riskbyte.config import EXCLUDED, digest
from riskbyte_ml.evaluation.splits import SplitArtifact
from riskbyte_ml.features.behaviour import FEATURE_PIPELINE_VERSION, finite_numeric
from riskbyte_ml.features.manifest import FeatureSpec


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    family: Literal["linear_distance", "tree_boosting"] = "linear_distance"
    scaling: Literal["standard", "robust", "none"] = "standard"
    rare_min_count: int = Field(default=5, ge=1)
    max_onehot_categories: int = Field(default=256, ge=2)
    frequency_columns: tuple[str, ...] = ("device_id", "merchant_id")
    numeric_safety_bound: float = Field(default=1e12, gt=0, allow_inf_nan=False)


class Preprocessor:
    def __init__(self, config=None):
        self.config = config or PreprocessingConfig()
        self.fitted = False

    def fit_training(
        self, features: pd.DataFrame, split: SplitArtifact, manifest: list[FeatureSpec]
    ):
        if self.fitted:
            raise ValueError("Fitted preprocessing is immutable; construct a new instance to refit")
        if list(features.index) != list(range(len(features))):
            raise ValueError("Feature rows must retain canonical dataset positions")
        all_rows = [i for rows in split.indexes.values() for i in rows]
        if sorted(all_rows) != list(range(len(features))) or any(
            not rows for rows in split.indexes.values()
        ):
            raise ValueError("Recorded split must partition feature rows")
        if split.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
            raise ValueError("Incompatible feature version")
        if features.attrs.get("row_order_fingerprint") != split.row_order_fingerprint:
            raise ValueError("Feature row provenance does not match the recorded split")
        specs = {s.name: s for s in manifest}
        if len(specs) != len(manifest) or set(features) != set(specs) or set(features) & EXCLUDED:
            raise ValueError("Exact approved feature manifest required; outcome metadata forbidden")
        if any(
            s.leakage_status != "APPROVED_PRE_OUTCOME"
            or self.config.family not in s.model_eligibility
            for s in manifest
        ):
            raise ValueError("Feature not approved for this preprocessing family")
        self.columns = list(features)
        self.manifest = manifest
        self.categorical = [n for n in self.columns if specs[n].feature_family == "categorical"]
        self.numeric = [n for n in self.columns if n not in self.categorical]
        train = features.iloc[split.indexes["train"]]
        values = self._numeric(train)
        self.medians = np.array(
            [np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0 for v in values.T]
        )
        filled = np.where(np.isnan(values), self.medians, values)
        self.scaler = None
        if self.config.family == "linear_distance" and self.config.scaling != "none":
            self.scaler = StandardScaler() if self.config.scaling == "standard" else RobustScaler()
            self.scaler.fit(filled)
        self.counts, self.rare_values = {}, {}
        for name in self.categorical:
            counts = self._category(train[name]).value_counts().to_dict()
            self.counts[name] = counts
            self.rare_values[name] = {
                k for k, v in counts.items() if v >= self.config.rare_min_count
            } | {"m:"}
        self.frequency = [
            n
            for n in self.categorical
            if n in self.config.frequency_columns
            or len(self.rare_values[n]) > self.config.max_onehot_categories
        ]
        self.onehot = [n for n in self.categorical if n not in self.frequency]
        self.encoder = None
        if self.onehot:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, dtype=np.float64
            )
            self.encoder.fit(self._group_categories(train))
        self.train_count = len(train)
        self.feature_names = [
            *self.numeric,
            *(n + "__missing" for n in self.numeric),
            *(n + "__train_frequency" for n in self.frequency),
            *(self.encoder.get_feature_names_out(self.onehot).tolist() if self.encoder else []),
        ]
        self.dataset_fingerprint = split.dataset_fingerprint
        self.split_id = split.split_id
        self.train_indexes_fingerprint = digest(split.indexes["train"])
        self.pipeline_version = FEATURE_PIPELINE_VERSION
        self.preprocessing_id = digest(
            {
                "split": self.split_id,
                "config": self.config.model_dump(mode="json"),
                "version": self.pipeline_version,
                "feature_order": self.columns,
                "horizon_seconds": features.attrs.get("relative_time_horizon_seconds"),
                "learned_medians": self.medians.tolist(),
                "learned_categories": self.counts,
                "scaler_state": {
                    n: getattr(self.scaler, n).tolist()
                    for n in ("center_", "mean_", "scale_")
                    if self.scaler is not None and hasattr(self.scaler, n)
                },
            }
        )
        self.fitted = True
        return self

    def _numeric(self, frame):
        values = np.column_stack([finite_numeric(frame[n]) for n in self.numeric])
        return np.clip(values, -self.config.numeric_safety_bound, self.config.numeric_safety_bound)

    @staticmethod
    def _category(series):
        return series.map(
            lambda v: "m:" if pd.isna(v) or not str(v).strip() else "v:" + str(v)
        ).astype(str)

    def _group_categories(self, frame):
        return pd.DataFrame(
            {
                n: self._category(frame[n]).map(
                    lambda v, n=n: v if v in self.rare_values[n] else "r:"
                )
                for n in self.onehot
            },
            index=frame.index,
        )

    def transform(self, features):
        if not self.fitted or list(features.columns) != self.columns:
            raise ValueError("Fitted state and exact ordered feature columns required")
        values = self._numeric(features)
        missing = np.isnan(values).astype(float)
        filled = np.where(np.isnan(values), self.medians, values)
        if self.scaler is not None:
            filled = self.scaler.transform(filled)
        filled = np.nan_to_num(
            filled,
            nan=0,
            posinf=self.config.numeric_safety_bound,
            neginf=-self.config.numeric_safety_bound,
        )
        blocks = [sparse.csr_matrix(filled), sparse.csr_matrix(missing)]
        if self.frequency:
            frequency = np.column_stack(
                [
                    self._category(features[n]).map(self.counts[n]).fillna(0).to_numpy(dtype=float)
                    / self.train_count
                    for n in self.frequency
                ]
            )
            blocks.append(sparse.csr_matrix(frequency))
        if self.encoder is not None:
            blocks.append(self.encoder.transform(self._group_categories(features)))
        result = sparse.hstack(blocks, format="csr")
        if not np.isfinite(result.data).all() or result.shape[1] != len(self.feature_names):
            raise ValueError("Invalid model matrix")
        return result

    def save(self, directory: Path):
        if not self.fitted:
            raise ValueError("Cannot serialize unfitted preprocessing")
        path = directory / (self.preprocessing_id.removeprefix("sha256:") + ".joblib")
        buffer = io.BytesIO()
        joblib.dump(self, buffer, compress=3)
        payload = buffer.getvalue()
        if path.exists():
            loaded = load_preprocessor(path, self.dataset_fingerprint, self.split_id)
            if loaded.preprocessing_id != self.preprocessing_id:
                raise ValueError("Existing preprocessing artifact mismatch")
        else:
            write_artifact(path, payload)
        return path

    def encoded_manifest(self):
        by_name = {s.name: s for s in self.manifest}
        entries = []
        for name in self.feature_names:
            source = (
                name
                if name in by_name
                else next(
                    n
                    for n in sorted(self.columns, key=len, reverse=True)
                    if name.startswith(n + "_") or name.startswith(n + "__")
                )
            )
            spec = by_name[source]
            transform = (
                "missing indicator"
                if name.endswith("__missing")
                else "train-only frequency"
                if name.endswith("__train_frequency")
                else "train-only rare grouping + one-hot"
                if source in self.onehot
                else "train-only median; " + (self.config.scaling if self.scaler else "unscaled")
            )
            entries.append(
                spec.model_copy(
                    update={"name": name, "kind": "encoded", "transformation": transform}
                )
            )
        return entries


def load_preprocessor(path: Path, fingerprint: str, split_id: str):
    """Load trusted local artifacts only; joblib is executable serialization."""
    value = joblib.load(path)
    if (
        not isinstance(value, Preprocessor)
        or not value.fitted
        or value.dataset_fingerprint != fingerprint
        or value.split_id != split_id
        or value.pipeline_version != FEATURE_PIPELINE_VERSION
    ):
        raise ValueError("Incompatible preprocessing artifact")
    return value


def matrix_summary(matrix):
    return {
        "rows": matrix.shape[0],
        "encoded_feature_count": matrix.shape[1],
        "representation": "CSR sparse",
        "nonzero_count": matrix.nnz,
        "memory_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
        "equivalent_dense_bytes": matrix.shape[0] * matrix.shape[1] * matrix.dtype.itemsize,
    }
