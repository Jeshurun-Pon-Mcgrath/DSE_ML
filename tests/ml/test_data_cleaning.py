import json

import pandas as pd
import pytest

from riskbyte_ml.data.cleaning import apply_cleaning, save_processed
from riskbyte_ml.data.cli import main
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import load_csv
from riskbyte_ml.data.models import CleaningOperation, CleaningPlan
from riskbyte_ml.data.pipeline import profile_dataset


def plan(dataset, operations):
    return CleaningPlan(source_fingerprint=dataset.fingerprint, operations=operations)


def test_cleaning_copy_audit_and_separate_outputs(synthetic_config, tmp_path):
    source = synthetic_config.source_path.read_bytes()
    dataset = load_csv(synthetic_config)
    proposed = profile_dataset(synthetic_config).cleaning_plan
    assert proposed.automatically_applied is False
    cleaned, audit = apply_cleaning(dataset, proposed)
    assert len(cleaned) == 11 and len(dataset.frame) == 12
    assert pd.isna(cleaned.loc[7, "channel"])
    assert cleaned["value"].dtype.kind == "f"
    assert any(item.operation == "parse_datetime" for item in audit.operations)
    assert audit.operations[0].rows_affected == 1
    assert audit.operations[0].before.row_count == 12
    assert audit.operations[0].after.row_count == 11
    assert all(item.reason for item in audit.operations)
    output = tmp_path / "processed.csv"
    audit_path = save_processed(dataset, cleaned, audit, output)
    assert output.exists() and audit_path.exists()
    assert json.loads(audit_path.read_text())["source_fingerprint"] == dataset.fingerprint
    assert synthetic_config.source_path.read_bytes() == source
    with pytest.raises(DatasetError) as error:
        save_processed(dataset, cleaned, audit, output)
    assert error.value.code == "dataset_output_exists"


def test_cleaning_cannot_overwrite_source(synthetic_config):
    dataset = load_csv(synthetic_config)
    cleaned, audit = apply_cleaning(dataset, plan(dataset, []))
    with pytest.raises(DatasetError) as error:
        save_processed(dataset, cleaned, audit, synthetic_config.source_path)
    assert error.value.code == "dataset_source_overwrite_forbidden"


def test_stale_plan_and_missing_cleaning_column(synthetic_config):
    dataset = load_csv(synthetic_config)
    with pytest.raises(DatasetError, match="fingerprint"):
        apply_cleaning(dataset, CleaningPlan(source_fingerprint="wrong", operations=[]))
    operation = CleaningOperation(
        operation="coerce_numeric", column="absent", reason="Explicit test"
    )
    with pytest.raises(DatasetError) as error:
        apply_cleaning(dataset, plan(dataset, [operation]))
    assert error.value.code == "dataset_cleaning_column_missing"


def test_invalid_numeric_requires_explicit_loss_policy(csv_factory):
    dataset = load_csv(
        csv_factory("amount\n12\nbad\ninf\n", mode="explicit", numeric_columns=("amount",))
    )
    operation = CleaningOperation(
        operation="coerce_numeric", column="amount", reason="Explicit test"
    )
    with pytest.raises(DatasetError):
        apply_cleaning(dataset, plan(dataset, [operation]))
    coerced, audit = apply_cleaning(
        dataset, plan(dataset, [operation.model_copy(update={"errors": "coerce"})])
    )
    assert int(coerced.amount.isna().sum()) == 2
    assert audit.operations[0].errors_policy == "coerce"
    assert audit.operations[0].after.missing_count == 2
    assert dataset.frame.amount.tolist() == ["12", "bad", "inf"]


def test_malformed_datetime_requires_explicit_loss_policy(csv_factory):
    dataset = load_csv(
        csv_factory("date\n2024-01-01\nbad\n", mode="explicit", timestamp_column="date")
    )
    operation = CleaningOperation(operation="parse_datetime", column="date", reason="Explicit test")
    with pytest.raises(DatasetError):
        apply_cleaning(dataset, plan(dataset, [operation]))
    cleaned, audit = apply_cleaning(
        dataset, plan(dataset, [operation.model_copy(update={"errors": "coerce"})])
    )
    assert int(cleaned.date.isna().sum()) == 1
    assert audit.operations[0].after.missing_count == 1


def test_failed_later_operation_does_not_partially_mutate_dataset(csv_factory):
    dataset = load_csv(csv_factory("n\nbad\nbad\n"))
    original = dataset.frame.copy(deep=True)
    operations = [
        CleaningOperation(operation="remove_duplicates", reason="Explicit test"),
        CleaningOperation(operation="coerce_numeric", column="n", reason="Explicit test"),
    ]
    with pytest.raises(DatasetError):
        apply_cleaning(dataset, plan(dataset, operations))
    pd.testing.assert_frame_equal(dataset.frame, original)


def test_cli_unconfigured_creates_no_fake_artifacts(tmp_path, capsys):
    assert main(["profile", "--output-dir", str(tmp_path / "unused")]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dataset_not_configured"}
    assert not (tmp_path / "unused").exists()


def test_cli_end_to_end_profile_and_clean(synthetic_config, tmp_path, capsys):
    configuration = tmp_path / "config.json"
    configuration.write_text(synthetic_config.model_dump_json(), encoding="utf-8")
    artifacts = tmp_path / "reports"
    assert main(["profile", "--config", str(configuration), "--output-dir", str(artifacts)]) == 0
    profile = json.loads((artifacts / "data_profile.json").read_text())
    cleaning_plan = tmp_path / "plan.json"
    cleaning_plan.write_text(json.dumps(profile["cleaning_plan"]), encoding="utf-8")
    assert (
        main(
            [
                "clean",
                "--config",
                str(configuration),
                "--plan",
                str(cleaning_plan),
                "--output",
                str(tmp_path / "processed.csv"),
            ]
        )
        == 0
    )
    assert "cleaning_completed" in capsys.readouterr().out


def test_cli_dataset_errors_are_safe_json(tmp_path, capsys):
    assert main(["profile", "--config", str(tmp_path / "private-path.json")]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "dataset_config_invalid"
    assert str(tmp_path) not in json.dumps(payload)
