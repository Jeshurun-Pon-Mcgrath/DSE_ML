"""All IEEE-shaped records are synthetic fixtures, never actual Kaggle data."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.streaming import load_streamed_csv
from riskbyte_ml.datasets.ieee_cis.__main__ import main
from riskbyte_ml.datasets.ieee_cis.adapter import (
    JoinRejected,
    combined_fingerprint,
    join_tables,
    load_joined,
)
from riskbyte_ml.datasets.ieee_cis.config import IEEEConfig, load_ieee_config
from riskbyte_ml.datasets.ieee_cis.pipeline import integrate, save_integration


@pytest.fixture
def ieee_config(tmp_path):
    fixtures = Path(__file__).parents[1] / "fixtures"
    tx, identity = tmp_path / "train_transaction.csv", tmp_path / "train_identity.csv"
    tx.write_bytes((fixtures / "synthetic_ieee_transaction.csv").read_bytes())
    identity.write_bytes((fixtures / "synthetic_ieee_identity.csv").read_bytes())
    return IEEEConfig(
        transaction_path=tx,
        identity_path=identity,
        chunk_rows=3,
        entity_candidates=(
            ("card1", "card2"),
            ("card1", "addr1"),
            ("DeviceInfo",),
            ("card6", "DeviceInfo"),
        ),
        min_entity_count_for_recommendation=2,
    )


def test_left_join_preserves_order_target_and_coverage(ieee_config):
    result = load_joined(ieee_config)
    assert result.dataset.frame["TransactionID"].tolist() == [str(i) for i in range(1001, 1009)]
    assert result.dataset.frame["isFraud"].tolist() == ["0", "0", "1", "0", "0", "1", "0", "0"]
    audit = result.join_audit
    assert (audit.transaction_rows_before, audit.identity_rows_before, audit.joined_rows) == (
        8,
        6,
        8,
    )
    assert (
        audit.matched_identity_rows,
        audit.unmatched_transaction_rows,
        audit.unused_identity_rows,
    ) == (5, 3, 1)
    assert audit.identity_coverage_percentage == 62.5
    assert audit.target_integrity_verified
    assert set(audit.columns_added) == {"DeviceType", "DeviceInfo", "id_12"}
    assert pd.isna(result.dataset.frame.loc[2, "DeviceInfo"])
    # An identity record with blank attributes still counts as a joined record.
    assert result.dataset.frame.loc[3, "DeviceInfo"] == ""


@pytest.mark.parametrize("side", ["transaction", "identity"])
def test_duplicate_keys_rejected_with_audit(ieee_config, side):
    path = getattr(ieee_config, side + "_path")
    rows = path.read_text().splitlines()
    path.write_text("\n".join([*rows, rows[1]]) + "\n")
    with pytest.raises(JoinRejected) as error:
        load_joined(ieee_config)
    assert error.value.code == "ieee_cis_duplicate_join_keys"
    assert getattr(error.value.audit, "duplicate_" + side + "_keys") == 1
    assert error.value.audit.joined_rows is None


def test_overlapping_identity_columns_cannot_replace_target(ieee_config):
    tx = pd.read_csv(ieee_config.transaction_path, dtype="string", keep_default_na=False)
    identity = pd.read_csv(ieee_config.identity_path, dtype="string", keep_default_na=False)
    identity["isFraud"] = "1"
    with pytest.raises(JoinRejected, match="conflict"):
        join_tables(tx, identity, ieee_config)


def test_missing_keys_rejected(ieee_config):
    text = ieee_config.identity_path.read_text().replace("1001,", ",", 1)
    ieee_config.identity_path.write_text(text)
    with pytest.raises(JoinRejected) as error:
        load_joined(ieee_config)
    assert error.value.audit.missing_identity_keys == 1


@pytest.mark.parametrize("side", ["transaction", "identity", "both"])
def test_missing_files_report_configuration_error_no_fake_replacements(
    ieee_config, side, tmp_path, capsys
):
    if side in {"transaction", "both"}:
        ieee_config.transaction_path.unlink()
    if side in {"identity", "both"}:
        ieee_config.identity_path.unlink()
    config = tmp_path / "ieee.json"
    config.write_text(ieee_config.model_dump_json())
    output = tmp_path / "reports"
    assert main(["--config", str(config), "--output-dir", str(output)]) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "dataset_configuration_error"
    assert body["error"]["code"] == "ieee_cis_files_missing"
    assert not output.exists()
    assert str(tmp_path) not in json.dumps(body)


def test_feature_catalog_and_contract_are_reviewable(ieee_config):
    profile, report = integrate(ieee_config)
    catalog = {item.column: item for item in report.feature_catalog}
    assert catalog["V127"].classifications == ["ANONYMIZED"]
    assert "PARTIALLY_INTERPRETABLE" in catalog["card1"].classifications
    assert "POTENTIAL_BEHAVIOURAL_KEY" in catalog["DeviceInfo"].classifications
    assert "EXCLUDED_CANDIDATE" in catalog["fraud_status"].classifications
    assert "IDENTIFIER" in catalog["TransactionID"].classifications
    assert catalog["isFraud"].classifications == ["TARGET"]
    contract = report.feature_contract
    assert contract.proposed_only
    assert "TransactionAmt" in contract.raw_predictive_features
    assert "V127" in contract.anonymized_engineered_features
    assert "card1" in contract.categorical_features
    assert "TransactionID" in contract.identifier_excluded_fields
    assert "addr1" in contract.missingness_indicators
    assert profile.quality.target.fraud_count == 2
    assert profile.quality.target.positive_class_prevalence == 0.25
    assert "fraud_status" in profile.unique_counts  # Report candidates, never drop columns.


def test_elapsed_semantics_prevent_fake_calendar_analyses(ieee_config):
    profile, report = integrate(ieee_config)
    assert profile.eda.fraud_by_hour.status == "UNAVAILABLE_WITH_REASON"
    assert (
        profile.eda.fraud_by_day_of_week.reason == "elapsed_time_has_no_confirmed_calendar_anchor"
    )
    assert report.elapsed_seconds.maximum == 90000
    assert "undisclosed" in report.temporal_semantics
    support = {item.feature: item.support for item in report.behavioural_feasibility}
    assert support["relative_elapsed_hour_day_index"] == "SUPPORTED"
    assert support["calendar_hour_weekday_month"] == "UNSUPPORTED"
    assert support["merchant_novelty"] == "UNSUPPORTED"
    assert support["verified_customer_history"] == "UNSUPPORTED"


def test_entity_statistics_exclude_incomplete_keys(ieee_config):
    _, report = integrate(ieee_config)
    cards, addresses, devices, _ = report.entity_key_study
    assert cards.unique_entities == 3
    assert cards.median_transactions_per_entity == 2
    assert cards.mean_transactions_per_entity == pytest.approx(8 / 3)
    assert cards.maximum_transactions == 4
    assert cards.singleton_entity_percentage == 0
    assert cards.statistically_useful
    assert addresses.missing_rows == 1 and addresses.eligible_rows == 7
    assert addresses.singleton_entity_percentage == pytest.approx(100 / 3)
    assert devices.eligible_rows == 4
    assert devices.singleton_entity_percentage == 100
    assert devices.statistically_useful is False
    assert cards.proxy_variation_entity_percentage["DeviceInfo"] == pytest.approx(100 / 3)
    assert report.recommended_behavioural_groups[0] == cards.name


def test_no_entity_recommendation_when_not_statistically_useful(ieee_config):
    _, report = integrate(
        ieee_config.model_copy(update={"min_entity_count_for_recommendation": 20})
    )
    assert report.recommended_behavioural_groups == []
    assert (
        next(
            item for item in report.behavioural_feasibility if item.feature == "amount_deviation"
        ).support
        == "UNSUPPORTED"
    )


def test_absent_entity_fields_are_truthfully_unavailable(ieee_config):
    frame = pd.read_csv(ieee_config.transaction_path, dtype="string", keep_default_na=False)
    frame.drop(columns=["addr2"]).to_csv(ieee_config.transaction_path, index=False)
    _, report = integrate(
        ieee_config.model_copy(update={"entity_candidates": (("card1", "addr2"),)})
    )
    assert report.entity_key_study[0].status == "UNAVAILABLE_WITH_REASON"
    assert report.entity_key_study[0].reason == "required_fields_absent"
    assert report.entity_key_study[0].unique_entities is None


def test_fingerprint_tracks_both_contents_and_policy_not_paths(ieee_config, tmp_path):
    first = load_joined(ieee_config)
    assert first.dataset.fingerprint == load_joined(ieee_config).dataset.fingerprint
    moved = tmp_path / "moved.csv"
    moved.write_bytes(ieee_config.transaction_path.read_bytes())
    assert (
        load_joined(ieee_config.model_copy(update={"transaction_path": moved})).dataset.fingerprint
        == first.dataset.fingerprint
    )
    identity = ieee_config.identity_path
    identity.write_text(identity.read_text().replace("synthetic-device-A", "changed-device"))
    assert load_joined(ieee_config).dataset.fingerprint != first.dataset.fingerprint
    changed = ieee_config.model_copy(update={"chunk_rows": 4})
    assert combined_fingerprint(first.source_fingerprints, changed) != first.dataset.fingerprint


def test_storage_optimization_is_lossless_and_measured(ieee_config):
    result = load_joined(ieee_config)
    assert result.memory.transaction_after_bytes <= result.memory.transaction_before_bytes
    assert result.memory.identity_after_bytes <= result.memory.identity_before_bytes
    assert result.memory.peak_process_memory_bytes is None
    # Synthetic large integers/fractional lexemes remain exact, without float downcasts.
    text = ieee_config.transaction_path.read_text().replace(
        "1001,0,0,10,", "9007199254740993,0,0,10.00000000000000000001,"
    )
    ieee_config.transaction_path.write_text(text)
    frame = load_joined(ieee_config).dataset.frame
    assert frame.loc[0, "TransactionID"] == "9007199254740993"
    assert frame.loc[0, "TransactionAmt"] == "10.00000000000000000001"


def test_stream_batches_deterministic_and_category_codes_not_measures(ieee_config):
    first = load_joined(ieee_config)
    second = load_joined(ieee_config.model_copy(update={"chunk_rows": 1}))
    pd.testing.assert_frame_equal(first.dataset.frame, second.dataset.frame)
    profile, _ = integrate(ieee_config)
    assert "card1" in profile.schema_metadata.categorical_columns
    assert "card1" not in profile.descriptive_statistics


@pytest.mark.parametrize(
    "policy", ["max_bytes_per_file", "max_rows_per_file", "max_retained_memory_bytes"]
)
def test_resource_limits_fail_explicitly(ieee_config, policy):
    with pytest.raises(DatasetError) as error:
        load_joined(ieee_config.model_copy(update={policy: 1}))
    assert error.value.code in {"dataset_limit_exceeded", "dataset_memory_limit"}


def test_stream_rejects_malformed_and_duplicate_headers(ieee_config):
    for text in ("x,x\n1,2\n", "x,y\n1,2,3\n", 'x\n"unfinished'):
        ieee_config.transaction_path.write_text(text)
        with pytest.raises(DatasetError):
            load_streamed_csv(
                ieee_config.table_config("transaction"), chunk_rows=1, max_memory_bytes=100000
            )


def test_reports_preserve_sources_and_do_not_leak_paths(ieee_config, tmp_path):
    before = [
        path.read_bytes() for path in (ieee_config.transaction_path, ieee_config.identity_path)
    ]
    profile, report = integrate(ieee_config)
    output = tmp_path / "artifacts"
    save_integration(profile, report, output, ieee_config)
    assert (output / "data_profile.json").exists()
    assert (output / "ENTITY_KEY_STUDY.md").exists()
    assert (output / "feature_contract.json").exists()
    assert str(tmp_path) not in report.model_dump_json()
    assert [
        path.read_bytes() for path in (ieee_config.transaction_path, ieee_config.identity_path)
    ] == before
    assert report.ordering_review[0].file_order_monotonic
    assert report.id_elapsed_rank_correlation == pytest.approx(1)
    assert any(item.code == "leakage_target_duplicate" for item in report.leakage_candidates)


def test_configuration_paths_relative_and_invalid_entity_target_rejected(ieee_config, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"transaction_path": "train_transaction.csv", "identity_path": "train_identity.csv"}
        )
    )
    loaded = load_ieee_config(config)
    assert loaded.transaction_path == ieee_config.transaction_path
    with pytest.raises(ValidationError):
        IEEEConfig(entity_candidates=(("isFraud",),))


def test_simultaneous_events_do_not_establish_prior_history(ieee_config):
    frame = pd.read_csv(ieee_config.transaction_path, dtype="string", keep_default_na=False)
    frame["TransactionDT"] = "100"
    frame.to_csv(ieee_config.transaction_path, index=False)
    _, report = integrate(ieee_config)
    assert report.entity_key_study[0].rows_with_strictly_prior_elapsed_time_percentage == 0
    support = {item.feature: item.support for item in report.behavioural_feasibility}
    assert support["time_since_previous_transaction"] == "UNSUPPORTED"


def test_negative_elapsed_values_fail_quality_without_fabricating_time(ieee_config):
    ieee_config.transaction_path.write_text(
        ieee_config.transaction_path.read_text().replace("1001,0,0,", "1001,0,-60,")
    )
    profile, report = integrate(ieee_config)
    assert profile.quality.validation_status == "FAILED"
    assert any(w.code == "ieee_negative_elapsed_time" for w in profile.warnings)
    assert (
        next(
            item
            for item in report.behavioural_feasibility
            if item.feature == "relative_elapsed_hour_day_index"
        ).support
        == "UNSUPPORTED"
    )


def test_category_optimization_reduces_repeated_string_memory(ieee_config):
    frame = pd.read_csv(ieee_config.transaction_path, dtype="string", keep_default_na=False)
    expanded = pd.concat([frame] * 50, ignore_index=True)
    expanded["TransactionID"] = [str(i) for i in range(len(expanded))]
    expanded.to_csv(ieee_config.transaction_path, index=False)
    loaded = load_joined(ieee_config)
    assert loaded.memory.transaction_after_bytes < loaded.memory.transaction_before_bytes
    assert loaded.memory.optimized_columns


def test_device_conditioned_groups_cannot_claim_device_novelty(ieee_config):
    identity = ieee_config.identity_path
    identity.write_text(identity.read_text().replace("synthetic-device-B", "synthetic-device-A"))
    _, report = integrate(ieee_config.model_copy(update={"entity_candidates": (("DeviceInfo",),)}))
    assert report.entity_key_study[0].statistically_useful
    assert (
        next(
            item for item in report.behavioural_feasibility if item.feature == "device_novelty"
        ).support
        == "UNSUPPORTED"
    )


def test_invalid_target_preserved_and_reported(ieee_config):
    tx = ieee_config.transaction_path
    tx.write_text(tx.read_text().replace("1001,0,0,", "1001,unknown,0,"))
    profile, report = integrate(ieee_config)
    assert report.join_audit.target_integrity_verified
    assert profile.quality.validation_status == "FAILED"
    assert profile.quality.target.unsupported_count == 1
    assert report.ordering_review[0].status == "UNAVAILABLE_WITH_REASON"


def test_rejected_cli_join_saves_only_rejected_audit(ieee_config, tmp_path, capsys):
    rows = ieee_config.identity_path.read_text().splitlines()
    ieee_config.identity_path.write_text("\n".join([*rows, rows[1]]) + "\n")
    config = tmp_path / "config.json"
    config.write_text(ieee_config.model_dump_json())
    output = tmp_path / "audit"
    assert main(["--config", str(config), "--output-dir", str(output)]) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["error"]["code"] == "ieee_cis_duplicate_join_keys"
    assert json.loads((output / "join_audit.json").read_text())["status"] == "REJECTED"
    assert not (output / "data_profile.json").exists()


def test_invalid_encoding_is_sanitized(ieee_config):
    with pytest.raises(DatasetError) as error:
        load_joined(ieee_config.model_copy(update={"encoding": "base64_codec"}))
    assert error.value.code == "dataset_encoding_unsupported"


def test_optimized_and_unoptimized_values_and_statistics_match(ieee_config):
    optimized = load_joined(ieee_config)
    plain = load_joined(ieee_config.model_copy(update={"optimize_categories": False}))
    for name in plain.dataset.frame:
        pd.testing.assert_series_equal(
            optimized.dataset.frame[name].astype("string"),
            plain.dataset.frame[name].astype("string"),
        )
    optimized_profile, _ = integrate(ieee_config)
    plain_profile, _ = integrate(ieee_config.model_copy(update={"optimize_categories": False}))
    assert optimized_profile.descriptive_statistics == plain_profile.descriptive_statistics
    assert optimized_profile.quality.target == plain_profile.quality.target
