"""Reviewable raw/derived and encoded feature lineage."""

from typing import Literal

from riskbyte_ml.data.models import ReportModel
from riskbyte_ml.datasets.synthetic_riskbyte.config import EXCLUDED
from riskbyte_ml.features.behaviour import CATEGORICAL, NOVELTY


class FeatureSpec(ReportModel):
    name: str
    source_columns: list[str]
    kind: Literal["raw", "derived", "encoded"]
    feature_family: str
    transformation: str
    interpretability_class: str
    behavioural_dependency: bool
    temporal_dependency: bool
    missingness_strategy: str
    leakage_status: Literal["APPROVED_PRE_OUTCOME", "EXCLUDED"]
    availability_requirements: list[str]
    model_eligibility: list[str]


def feature_manifest(features, raw_columns):
    specs = []
    for name in features:
        if name in EXCLUDED:
            raise ValueError("Outcome, metadata and identifiers cannot be model features")
        raw = name in raw_columns
        temporal = name in {"elapsed_day_index", "relative_time_position"}
        behavioural = not raw and not temporal
        sources = (
            [name]
            if raw
            else ["TransactionDT"]
            if temporal
            else ["behavioural_entity_id", "TransactionDT"]
        )
        if behavioural and ("amount" in name or name == "amount_history_available"):
            sources.append("TransactionAmt")
        for novelty, column in NOVELTY.items():
            if name.startswith(novelty + "_"):
                sources.append(column)
        classification = (
            "ANONYMIZED"
            if raw and name.startswith("V")
            else "TEMPORAL"
            if temporal
            else "INTERPRETABLE"
        )
        if raw and name in {"device_id", "merchant_id"}:
            classification = "PARTIALLY_INTERPRETABLE"
        specs.append(
            FeatureSpec(
                name=name,
                source_columns=sources,
                kind="raw" if raw else "derived",
                feature_family="categorical"
                if name in CATEGORICAL
                else "temporal"
                if temporal
                else "behavioural"
                if behavioural
                else "numeric",
                transformation="preserved pre-outcome context"
                if raw
                else "strictly prior history"
                if behavioural
                else "configured elapsed index; no calendar anchor",
                interpretability_class=classification,
                behavioural_dependency=behavioural,
                temporal_dependency=not raw,
                missingness_strategy="train median + indicator / categorical missing token",
                leakage_status="APPROVED_PRE_OUTCOME",
                availability_requirements=sources,
                model_eligibility=["linear_distance", "tree_boosting"],
            )
        )
    return specs


def column_catalog(columns):
    catalog = {}
    for name in columns:
        tags = ["ANONYMIZED" if name.startswith("V") and name[1:].isdigit() else "INTERPRETABLE"]
        if name == "TransactionID":
            tags = ["IDENTIFIER", "EXCLUDED"]
        elif name == "TransactionDT":
            tags = ["TEMPORAL", "EXCLUDED"]
        elif name == "behavioural_entity_id":
            tags = ["BEHAVIOURAL_KEY", "EXCLUDED"]
        elif name == "isFraud":
            tags = ["TARGET", "EXCLUDED"]
        elif name in {"fraud_scenario", "hard_negative", "hard_negative_type"}:
            tags = ["TARGET_METADATA", "EXCLUDED"]
        elif name in {"merchant_id", "device_id"}:
            tags = ["IDENTIFIER", "PARTIALLY_INTERPRETABLE"]
        catalog[name] = tags
    return catalog
