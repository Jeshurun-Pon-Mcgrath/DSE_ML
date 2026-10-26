"""Entirely synthetic, small offline fixtures; no competition data or model training."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.adapter import load_dataset
from riskbyte_ml.datasets.synthetic_riskbyte.config import (
    EXCLUDED,
    SCENARIOS,
    GeneratorConfig,
    SyntheticConfig,
)
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate, save_generated
from riskbyte_ml.datasets.synthetic_riskbyte.pipeline import integrate, run_phase2
from riskbyte_ml.evaluation.splits import (
    SplitArtifact,
    SplitConfig,
    make_split,
    save_split,
    validate_partition,
)
from riskbyte_ml.features.behaviour import (
    FEATURE_PIPELINE_VERSION,
    behavioural_features,
    build_features,
)
from riskbyte_ml.features.manifest import column_catalog, feature_manifest
from riskbyte_ml.preprocessing.pipeline import PreprocessingConfig, Preprocessor, load_preprocessor


@pytest.fixture(scope="module")
def generated():
    return generate(
        GeneratorConfig(
            transaction_count=1200, entity_count=20, merchant_count=24, fraud_prevalence=0.08
        )
    )


@pytest.fixture
def source(tmp_path, generated):
    manifest = save_generated(generated, tmp_path)
    config = SyntheticConfig(
        transaction_path=tmp_path / "train_transaction.csv",
        identity_path=tmp_path / "train_identity.csv",
        metadata_path=tmp_path / "generation_metadata.json",
    )
    return config, manifest


def test_generator_deterministic(generated, tmp_path):
    second = generate(generated.config)
    pd.testing.assert_frame_equal(generated.transaction, second.transaction)
    pd.testing.assert_frame_equal(generated.identity, second.identity)
    a = save_generated(generated, tmp_path / "a")
    b = save_generated(second, tmp_path / "b")
    assert a["fingerprint"] == b["fingerprint"]
    assert (tmp_path / "a/train_transaction.csv").read_bytes() == (
        tmp_path / "b/train_transaction.csv"
    ).read_bytes()


def test_different_seed(generated):
    changed = generate(generated.config.model_copy(update={"seed": 51}))
    assert not generated.transaction.equals(changed.transaction)


@pytest.mark.parametrize(
    "condition",
    [
        "entities",
        "merchants",
        "devices",
        "classes",
        "imbalance",
        "hard_negatives",
        "scenarios",
        "heterogeneity",
        "coverage",
        "missingness",
        "chronology",
        "schema",
    ],
)
def test_nontrivial_dataset(generated, condition):
    tx, identity, ev = generated.transaction, generated.identity, generated.evaluation
    checks = {
        "entities": tx.behavioural_entity_id.nunique() > 1,
        "merchants": tx.merchant_id.nunique() > 1,
        "devices": identity.device_id.nunique() > 1,
        "classes": set(tx.isFraud) == {0, 1},
        "imbalance": abs(tx.isFraud.mean() - generated.config.fraud_prevalence) <= 1 / len(tx),
        "hard_negatives": ev.hard_negative.sum() > 0
        and (tx.loc[ev.hard_negative, "isFraud"] == 0).all(),
        "scenarios": set(ev.fraud_scenario) - {""} == set(SCENARIOS),
        "heterogeneity": generated.profiles.typical_amount.max()
        > generated.profiles.typical_amount.min() * 10,
        "coverage": 0 < len(identity) < len(tx),
        "missingness": tx.isna().sum().sum() + identity.isna().sum().sum() > 0,
        "chronology": tx.TransactionDT.is_monotonic_increasing,
        "schema": {
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "isFraud",
            "behavioural_entity_id",
            "merchant_id",
        }
        <= set(tx),
    }
    assert checks[condition]


def test_join_profile_and_manifest(source):
    config, manifest = source
    dataset, audit, memory, _ = load_dataset(config)
    assert audit.join_kind == "left" and audit.cardinality == "one_to_one"
    assert len(dataset.frame) == audit.joined_rows == manifest["transaction_count"]
    assert audit.target_integrity_verified
    assert audit.matched_identity_rows == manifest["identity_count"]
    assert audit.unmatched_transaction_rows > 0
    assert memory.transaction_after_bytes <= memory.transaction_before_bytes
    profile, metadata = integrate(config)
    assert profile.quality.validation_status == "PASSED"
    assert profile.eda.fraud_by_hour.status == "UNAVAILABLE_WITH_REASON"
    assert metadata.dataset_type == "synthetic"
    assert set(metadata.behavioural_feature_availability.values()) == {"SUPPORTED"}
    assert "fraud_scenario" not in dataset.frame


def test_source_protection_and_tampering(source, generated):
    config, _ = source
    before = config.transaction_path.read_bytes()
    with pytest.raises(DatasetError, match="overwrite"):
        save_generated(generated, config.transaction_path.parent)
    assert config.transaction_path.read_bytes() == before
    config.transaction_path.write_bytes(before.replace(b"merchant-", b"merchanz-", 1))
    with pytest.raises(DatasetError) as error:
        load_dataset(config)
    assert error.value.code == "synthetic_fingerprint_mismatch"


def test_missing_sources(tmp_path):
    with pytest.raises(DatasetError) as error:
        load_dataset(SyntheticConfig(transaction_path=tmp_path / "absent.csv"))
    assert error.value.code == "synthetic_files_missing"


def history():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [0, 100, 100, 500, 1000],
            "TransactionAmt": [10, 20, 100, 30, 40],
            "behavioural_entity_id": ["a"] * 5,
            "merchant_id": ["m1", "m2", "m3", "m1", "m2"],
            "device_id": ["d1", "d2", "d3", None, "d1"],
            "billing_region": ["r1", "r2", "r3", "r1", "r2"],
            "product_code": ["p1", "p2", "p3", "p1", "p2"],
            "merchant_category": ["c1", "c2", "c3", "c1", "c2"],
            "isFraud": [0, 1, 0, 0, 1],
        }
    )


@pytest.mark.parametrize(
    "feature,expected",
    [
        ("prior_mean_amount", 10),
        ("prior_median_amount", 10),
        ("prior_amount_std", 0),
        ("prior_transaction_count", 1),
        ("transactions_prior_5m", 1),
        ("time_since_previous_transaction", 100),
        ("amount_minus_prior_mean", 10),
        ("amount_ratio_to_prior_mean", 2),
        ("merchant_seen_before", False),
        ("device_seen_before", False),
        ("billing_region_seen_before", False),
        ("product_seen_before", False),
    ],
)
def test_strictly_prior_values(feature, expected):
    result = behavioural_features(history())
    assert result.loc[1, feature] == expected
    assert result.loc[2, "prior_transaction_count"] == 1
    assert result.loc[3, "prior_median_amount"] == 20
    assert result.loc[3, "prior_mean_amount"] == pytest.approx(130 / 3)
    assert result.loc[3, "prior_amount_std"] == pytest.approx(np.std([10, 20, 100]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("TransactionAmt", 99999),
        ("merchant_id", "futuremerchant"),
        ("device_id", "futuredevice"),
        ("billing_region", "futureregion"),
        ("product_code", "futureproduct"),
        ("TransactionDT", 999999),
    ],
)
def test_future_mutation_leaves_earlier_features_unchanged(column, value):
    frame = history()
    before = behavioural_features(frame)
    frame.loc[4, column] = value
    after = behavioural_features(frame)
    pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])


def test_current_and_same_time_are_not_history():
    frame = history()
    before = behavioural_features(frame)
    frame.loc[1, "TransactionAmt"] = 4000
    after = behavioural_features(frame)
    assert after.loc[1, "prior_mean_amount"] == before.loc[1, "prior_mean_amount"]
    assert after.loc[2, "prior_mean_amount"] == before.loc[2, "prior_mean_amount"]
    assert before.loc[3, "transactions_prior_5m"] == 0
    assert before.loc[3, "transactions_prior_30m"] == 3


def test_new_entity_and_missing_evidence():
    result = behavioural_features(history())
    assert result.loc[0, "behavioural_history_count"] == 0
    assert result.loc[0, "is_new_behavioural_entity"]
    assert not result.loc[0, "behaviour_feature_available"]
    assert pd.isna(result.loc[0, "prior_mean_amount"])
    assert not result.loc[3, "device_novelty_available"]
    assert pd.isna(result.loc[3, "device_seen_before"])


@pytest.mark.parametrize(
    "amounts", [[0, 0, 10, 20, 30], [None, None, 10, 0, 30], [np.inf, 0, 10, 20, 30]]
)
def test_amount_ratio_edges(amounts):
    frame = history()
    frame.TransactionAmt = amounts
    result = behavioural_features(frame)
    assert not np.isinf(result.select_dtypes(include="number").to_numpy()).any()
    assert pd.isna(result.loc[1, "amount_ratio_to_prior_mean"])
    assert pd.isna(result.loc[1, "amount_zscore_prior"])


def test_no_calendar_and_order_invariance():
    frame = history()
    baseline = behavioural_features(frame)
    shuffled = frame.sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(baseline, behavioural_features(shuffled).sort_index())
    assert not any("weekday" in n or "calendar" in n or "month" in n for n in baseline)


@pytest.mark.parametrize(
    "metadata", ["isFraud", "fraud_scenario", "hard_negative", "hard_negative_type"]
)
def test_outcome_metadata_never_used(metadata):
    frame = history()
    baseline = build_features(frame)
    frame[metadata] = ["outcome_changed"] * len(frame)
    pd.testing.assert_frame_equal(baseline, build_features(frame))
    assert not set(baseline) & EXCLUDED


@pytest.mark.parametrize("strategy", ["temporal", "random_stratified"])
def test_split_determinism_and_prevalence(generated, strategy, tmp_path):
    config = SplitConfig(strategy=strategy)
    a = make_split(generated.transaction, "fixture-fingerprint", config, generated.evaluation)
    b = make_split(generated.transaction, "fixture-fingerprint", config, generated.evaluation)
    assert a == b
    assert sum(s.row_count for s in a.summaries.values()) == len(generated.transaction)
    if strategy == "temporal":
        assert a.summaries["train"].elapsed_max < a.summaries["validation"].elapsed_min
        assert a.summaries["validation"].elapsed_max < a.summaries["test"].elapsed_min
    else:
        assert all(abs(s.fraud_prevalence - 0.08) < 0.01 for s in a.summaries.values())
    path = save_split(a, tmp_path)
    assert SplitArtifact.model_validate_json(path.read_bytes()) == a
    assert save_split(a, tmp_path) == path
    shuffled = generated.transaction.iloc[::-1]
    with pytest.raises(ValueError):
        validate_partition(a, shuffled)


def test_temporal_boundary_ties():
    frame = history()
    config = SplitConfig(train_ratio=0.3, validation_ratio=0.4)
    split = make_split(frame, "fixture", config)
    assert split.indexes["train"] == [0]
    assert split.indexes["validation"] == [1, 2]


def prepare(generated):
    tx = generated.transaction.merge(
        generated.identity, on="TransactionID", how="left", validate="one_to_one"
    )
    features = build_features(tx)
    split = make_split(tx, "synthetic-fixture", evaluation=generated.evaluation)
    return features, split, feature_manifest(features, tx.columns)


@pytest.mark.parametrize("mutation", ["numeric", "category", "target_metadata"])
def test_training_state_independent_of_held_out_values(generated, mutation):
    features, split, specs = prepare(generated)
    first = Preprocessor().fit_training(features, split, specs)
    changed = features.copy()
    heldout = split.indexes["validation"] + split.indexes["test"]
    if mutation == "numeric":
        changed.loc[heldout, "TransactionAmt"] = 99999999
    elif mutation == "category":
        changed.loc[heldout, "card_network"] = "future-only-network"
    else:
        changed["fraud_scenario"] = "leakage"
        with pytest.raises(ValueError):
            Preprocessor().fit_training(changed, split, specs)
        return
    second = Preprocessor().fit_training(changed, split, specs)
    np.testing.assert_array_equal(first.medians, second.medians)
    np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)
    assert first.counts == second.counts
    assert first.rare_values == second.rare_values
    assert first.feature_names == second.feature_names
    assert (
        first.transform(features.iloc[split.indexes["train"]])
        != second.transform(features.iloc[split.indexes["train"]])
    ).nnz == 0


@pytest.mark.parametrize(
    "family,scaling",
    [
        ("linear_distance", "standard"),
        ("linear_distance", "robust"),
        ("linear_distance", "none"),
        ("tree_boosting", "none"),
    ],
)
def test_numeric_category_alignment_sparse_and_serialization(generated, family, scaling, tmp_path):
    features, split, specs = prepare(generated)
    # Force a training all-missing numeric column to test documented fallback + flag.
    features["identity_age_days"] = np.nan
    features.loc[0, "TransactionAmt"] = np.inf
    processor = Preprocessor(PreprocessingConfig(family=family, scaling=scaling)).fit_training(
        features, split, specs
    )
    validation = features.iloc[split.indexes["validation"]].copy()
    validation["card_network"] = "unseen-network"
    matrix = processor.transform(validation)
    assert sparse.isspmatrix_csr(matrix)
    assert np.isfinite(matrix.data).all()
    assert matrix.shape[1] == len(processor.feature_names)
    assert matrix.shape[1] == processor.transform(features.iloc[split.indexes["train"]]).shape[1]
    assert not set(processor.feature_names) & EXCLUDED
    assert len(processor.encoded_manifest()) == len(processor.feature_names)
    if family == "tree_boosting":
        assert processor.scaler is None
    path = processor.save(tmp_path)
    restored = load_preprocessor(path, "synthetic-fixture", split.split_id)
    assert (restored.transform(validation) != matrix).nnz == 0
    with pytest.raises(ValueError):
        load_preprocessor(path, "wrong-fingerprint", split.split_id)
    with pytest.raises(ValueError):
        processor.fit_training(features, split, specs)


def test_feature_version_and_manifest(generated):
    features, split, specs = prepare(generated)
    assert split.feature_pipeline_version == FEATURE_PIPELINE_VERSION
    assert all(s.source_columns and s.model_eligibility for s in specs)
    assert column_catalog(["behavioural_entity_id", "fraud_scenario", "V1"])["V1"] == ["ANONYMIZED"]
    with pytest.raises(ValueError):
        Preprocessor().fit_training(
            features, split.model_copy(update={"feature_pipeline_version": "other"}), specs
        )


def test_anonymized_optional_features():
    data = generate(
        GeneratorConfig(
            transaction_count=100, entity_count=10, merchant_count=12, anonymized_feature_count=2
        )
    )
    assert {"V1", "V2"} <= set(data.transaction)
    specs = feature_manifest(build_features(data.transaction), data.transaction.columns)
    assert [s.interpretability_class for s in specs if s.name in {"V1", "V2"}] == ["ANONYMIZED"] * 2


def test_complete_small_pipeline(source, tmp_path):
    config, manifest = source
    summary = run_phase2(config, tmp_path / "artifacts")
    assert summary["dataset"]["dataset_fingerprint"] == manifest["fingerprint"]
    assert all(v["serialization_reload_verified"] for v in summary["families"].values())
    output = json.loads((tmp_path / "artifacts/features/feature_manifest.json").read_text())
    assert output["derived_feature_count"] > 0
    assert set(output["families"]) == {"linear_distance", "tree_boosting"}


def test_unseen_merchant_scenarios_use_actual_prior_history(generated):
    tx = generated.transaction
    ev = generated.evaluation
    features = behavioural_features(tx)
    selected = ev.fraud_scenario.eq("NEW_MERCHANT_AMOUNT_DEVIATION")
    assert selected.any()
    assert not features.loc[selected, "merchant_seen_before"].fillna(False).any()


def test_fraud_mechanisms_and_legitimate_bursts(generated):
    tx, ev = generated.transaction, generated.evaluation
    features = behavioural_features(tx)
    profiles = generated.profiles.set_index("behavioural_entity_id")
    baseline = tx.behavioural_entity_id.map(profiles.typical_amount)
    spike = ev.fraud_scenario.eq("AMOUNT_SPIKE")
    assert (tx.loc[spike, "TransactionAmt"] > baseline[spike] * 3).all()
    failures = ev.fraud_scenario.eq("REPEATED_FAILED_ATTEMPTS")
    assert (tx.loc[failures, "failed_attempts_before"] >= 3).all()
    slow = ev.fraud_scenario.eq("LOW_AND_SLOW")
    assert (tx.loc[slow, "TransactionAmt"] < baseline[slow] * 0.5).all()
    festival = ev.hard_negative_type.eq("SHOPPING_FESTIVAL")
    assert (features.loc[festival, "transactions_prior_5m"] >= 2).any()
    assert tx.loc[festival, "isFraud"].eq(0).all()


@pytest.mark.parametrize(
    "kind,category", [("LAPTOP_PURCHASE", "electronics"), ("MEDICAL_EXPENSE", "healthcare")]
)
def test_hard_negative_context_is_semantically_consistent(generated, kind, category):
    selected = generated.evaluation.hard_negative_type.eq(kind)
    assert selected.any()
    assert generated.transaction.loc[selected, "merchant_category"].eq(category).all()
    assert generated.transaction.loc[selected, "isFraud"].eq(0).all()


def test_added_future_record_does_not_change_earlier_features():
    frame = history()
    earlier = build_features(frame)
    future = frame.iloc[[-1]].copy()
    future.TransactionID = 6
    future.TransactionDT = 9999999
    future.TransactionAmt = 9999999
    future.merchant_id = "unseen-future"
    changed = build_features(pd.concat([frame, future], ignore_index=True)).iloc[: len(frame)]
    pd.testing.assert_frame_equal(earlier, changed)


def test_misaligned_feature_provenance_is_rejected(generated):
    features, split, specs = prepare(generated)
    features.attrs["row_order_fingerprint"] = "wrong-order"
    with pytest.raises(ValueError, match="provenance"):
        Preprocessor().fit_training(features, split, specs)


def test_unknown_frequency_and_rare_categories_are_train_only(generated):
    features, split, specs = prepare(generated)
    processor = Preprocessor(PreprocessingConfig(rare_min_count=10)).fit_training(
        features, split, specs
    )
    heldout = features.iloc[split.indexes["test"]].copy()
    heldout.device_id = "never-observed-device"
    heldout.card_network = "never-observed-network"
    before = {name: values.copy() for name, values in processor.counts.items()}
    matrix = processor.transform(heldout)
    frequency_index = processor.feature_names.index("device_id__train_frequency")
    assert matrix[:, frequency_index].nnz == 0
    assert processor.counts == before
    assert "v:never-observed-network" not in processor.rare_values["card_network"]


@pytest.mark.parametrize("value", [0.5, "fraud", None])
def test_unsupported_split_target_fails(value):
    frame = history()
    frame.isFraud = frame.isFraud.astype(object)
    frame.loc[0, "isFraud"] = value
    with pytest.raises(ValueError, match="target"):
        make_split(frame, "fixture")


def test_synthetic_duplicate_join_keys_rejected(source):
    config, _ = source
    from riskbyte_ml.datasets.two_table import JoinRejected, audited_left_join

    tx = pd.DataFrame({"TransactionID": ["a", "a"], "isFraud": ["0", "1"]})
    identity = pd.DataFrame({"TransactionID": ["a"]})
    with pytest.raises(JoinRejected) as error:
        audited_left_join(
            tx,
            identity,
            config.analysis_config(tx.columns),
            key="TransactionID",
            target="isFraud",
            prefix="synthetic",
        )
    assert error.value.code == "synthetic_duplicate_join_keys"
