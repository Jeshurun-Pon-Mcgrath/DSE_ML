import json

import pandas as pd
import pytest

from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.ingestion import load_csv
from riskbyte_ml.data.inspection import inspect_schema
from riskbyte_ml.data.pipeline import profile_dataset, save_reports
from riskbyte_ml.data.statistics import descriptive_statistics, outlier_overview
from riskbyte_ml.data.validation import validate_dataset
from riskbyte_ml.data.values import parse_timestamps


def quality(config):
    dataset = load_csv(config)
    return validate_dataset(dataset, inspect_schema(dataset))


def test_duplicates_missing_and_constants(synthetic_config):
    report = quality(synthetic_config)
    assert report.duplicates.extra_rows == 1
    assert report.duplicates.rows_in_duplicate_groups == 2
    assert report.missing_values["channel"].count == 1
    assert report.missing_values["channel"].percentage == pytest.approx(100 / 12)
    assert any(w.code == "constant_column" and w.column == "unchanged" for w in report.warnings)
    assert report.duplicates.identifier_extra_values["party_key"] > 0


def test_unique_identifiers_are_distinct_from_repeated_entities(synthetic_config):
    report = quality(
        synthetic_config.model_copy(update={"unique_identifier_columns": ("event_key",)})
    )
    assert report.validation_status == "FAILED"
    warnings = {
        w.column: w.severity for w in report.warnings if w.code == "duplicate_identifier_values"
    }
    assert warnings == {"event_key": "ERROR", "party_key": "INFO"}


def test_explicit_numeric_invalid_infinite_and_missing(csv_factory):
    config = csv_factory(
        "value\n10\ninf\n-inf\nwrong\n \nNA\n",
        mode="explicit",
        numeric_columns=("value",),
        na_values=("NA",),
    )
    report = quality(config)
    assert report.invalid_numeric_values["value"] == 1
    assert report.infinite_numeric_values["value"] == 2
    assert report.missing_values["value"].count == 2
    assert report.validation_status == "FAILED"
    stats = profile_dataset(config).descriptive_statistics["value"]
    assert (stats.count, stats.invalid_count, stats.infinite_count, stats.missing_count) == (
        1,
        1,
        2,
        2,
    )


def test_unsupported_target_labels_are_validation_failures(csv_factory):
    config = csv_factory(
        "decision\ngood\nbad\nunknown\n \n",
        mode="explicit",
        target_column="decision",
        positive_class="bad",
        negative_class="good",
    )
    target = quality(config).target
    assert target.status == "invalid"
    assert target.unsupported_count == 1 and target.missing_count == 1
    assert target.valid_count == 2
    assert profile_dataset(config).eda.class_distribution.status == "UNAVAILABLE_WITH_REASON"


def test_explicit_positive_class_and_imbalance(csv_factory):
    # Synthetic inverted numeric convention: 0 is fraud; no code assumes 1 means fraud.
    config = csv_factory(
        "result\n" + "1\n" * 99 + "0\n",
        mode="explicit",
        target_column="result",
        positive_class="0",
        negative_class="1",
    )
    report = quality(config)
    assert report.target.fraud_count == 1
    assert report.target.legitimate_count == 99
    assert report.target.imbalance_ratio == 99
    assert report.target.positive_class_prevalence == pytest.approx(0.01)
    assert report.target.fraud_percentage == 1.0
    assert report.target.minority_class == "0" and report.target.majority_class == "1"
    assert any(w.code == "class_imbalance" for w in report.warnings)


def test_single_class_has_no_infinity_ratio(csv_factory):
    config = csv_factory(
        "y\ngood\ngood\n",
        mode="explicit",
        target_column="y",
        positive_class="bad",
        negative_class="good",
    )
    report = quality(config)
    assert report.target.status == "valid"
    assert report.target.imbalance_ratio is None
    assert any(w.code == "single_class_target" for w in report.warnings)


def test_leakage_duplicate_names_and_relationships(csv_factory):
    rows = "\n".join(f"{i % 2},{i % 2},{10 * (i % 2)},done,category{i}" for i in range(40))
    config = csv_factory(
        "y,copy,perfect,final_decision,unique_text\n" + rows + "\n",
        mode="explicit",
        target_column="y",
        positive_class="1",
        negative_class="0",
    )
    report = quality(config)
    findings = {w.column: w.code for w in report.warnings if w.code.startswith("leakage_")}
    assert findings["copy"] == "leakage_target_duplicate"
    assert findings["perfect"] == "leakage_target_relationship"
    assert findings["final_decision"] == "leakage_outcome_name"
    assert "unique_text" not in findings
    assert all(w.inferred for w in report.warnings if w.code.startswith("leakage_"))
    assert load_csv(config).frame.shape[1] == 5


def test_missing_optional_fields_are_truthfully_unavailable(csv_factory):
    config = csv_factory(
        "x\n1\n2\n",
        mode="explicit",
        timestamp_column="absent_time",
        amount_column="absent_amount",
        identifier_columns=("absent_id",),
    )
    report = profile_dataset(config)
    assert report.schema_metadata.timestamp.reason == "configured_field_absent"
    assert report.eda.amount_distribution.status == "UNAVAILABLE_WITH_REASON"
    assert report.eda.fraud_by_hour.status == "UNAVAILABLE_WITH_REASON"
    assert any(w.code == "configured_optional_field_absent" for w in report.warnings)


def test_malformed_timestamps_are_reported(csv_factory):
    config = csv_factory(
        "date,y\n2024-01-01,good\nbroken,bad\n",
        mode="explicit",
        timestamp_column="date",
        target_column="y",
        positive_class="bad",
        negative_class="good",
    )
    report = profile_dataset(config)
    assert report.quality.validation_status == "FAILED"
    assert report.quality.malformed_timestamps["date"] == 1
    assert report.eda.fraud_by_hour.reason == "configured_timestamps_contain_invalid_values"


def test_statistics_exact_values_and_missing_handling():
    result = descriptive_statistics(
        pd.Series(["1", "2", "3", "4", "", "inf", "bad"]), DatasetConfig()
    )
    assert result.count == 4
    assert (result.missing_count, result.invalid_count, result.infinite_count) == (1, 1, 1)
    assert result.mean == result.median == 2.5
    assert (result.minimum, result.maximum, result.q1, result.q3, result.iqr) == (
        1.0,
        4.0,
        1.75,
        3.25,
        1.5,
    )
    assert result.variance == pytest.approx(5 / 3)
    assert result.standard_deviation == pytest.approx((5 / 3) ** 0.5)
    assert result.skewness == pytest.approx(0)
    assert result.percentiles["50"] == 2.5


@pytest.mark.parametrize(
    "values", [[None, "", "bad"], ["4"], ["4", "4", "4"], ["1e308", "-1e308", "1e308"]]
)
def test_degenerate_and_extreme_statistics_always_emit_strict_json(values):
    stats = descriptive_statistics(pd.Series(values), DatasetConfig())
    json.loads(stats.model_dump_json(), parse_constant=lambda value: pytest.fail(value))
    if stats.count < 2:
        assert stats.variance is None
    if len(values) == 3 and values == ["4", "4", "4"]:
        assert stats.standard_deviation == 0 and stats.skewness is None


def test_iqr_and_configurable_zscore_outliers():
    values = pd.Series(["0"] * 20 + ["100"])
    result = outlier_overview(values, DatasetConfig(zscore_threshold=3))
    assert result.iqr_outlier_count == 1
    assert result.zscore_outlier_count == 1
    assert result.lower_bound == result.upper_bound == 0
    assert outlier_overview(values, DatasetConfig(zscore_threshold=5)).zscore_outlier_count == 0
    assert outlier_overview(pd.Series([1, 1, 1]), DatasetConfig()).zscore_reason == "zero_variance"


def test_all_eda_analyses_and_class_hour_totals(synthetic_config):
    report = profile_dataset(synthetic_config)
    assert report.eda.class_distribution.data.fraud_count == 2
    assert report.eda.fraud_amount_comparison.data["fraud"].mean == 700
    assert sum(item.fraud_count for item in report.eda.fraud_by_hour.data) == 2
    assert sum(item.total_count for item in report.eda.fraud_by_day_of_week.data) == 12
    assert report.eda.missing_data.status == "AVAILABLE"
    assert report.eda.numeric_distributions.status == "AVAILABLE"
    assert report.eda.outliers.data["value"].iqr_outlier_count == 2
    assert report.eda.categorical_frequencies.data["channel"].non_missing_count == 11
    assert report.eda.correlation.status == "UNAVAILABLE_WITH_REASON"


def test_correlation_pair_counts_and_finite_output(csv_factory):
    config = csv_factory(
        "x,z,id\n1,2,A\n2,4,B\n3,6,C\n4,,D\n", mode="explicit", identifier_columns=("id",)
    )
    correlation = profile_dataset(config).eda.correlation.data
    assert correlation.values["x"]["z"] == pytest.approx(1)
    assert correlation.pair_counts["x"]["z"] == 3
    assert "id" in correlation.excluded_columns


def test_elapsed_timestamp_has_no_fabricated_weekday(csv_factory):
    config = csv_factory(
        "clock,y\n0,good\n3600,bad\n",
        mode="explicit",
        timestamp_column="clock",
        timestamp_kind="elapsed",
        timestamp_unit="s",
        target_column="y",
        positive_class="bad",
        negative_class="good",
    )
    assert (
        profile_dataset(config).eda.fraud_by_day_of_week.reason
        == "elapsed_time_has_no_confirmed_calendar_anchor"
    )


def test_profile_and_artifacts_are_reproducible_and_do_not_change_source(
    synthetic_config, tmp_path
):
    source_bytes = synthetic_config.source_path.read_bytes()
    first = profile_dataset(synthetic_config)
    second = profile_dataset(synthetic_config)
    assert first.model_dump(exclude={"generated_at"}) == second.model_dump(exclude={"generated_at"})
    save_reports(first, tmp_path)
    artifact = json.loads((tmp_path / "data_profile.json").read_text(encoding="utf-8"))
    assert artifact["source_path"] == str(synthetic_config.source_path.resolve())
    assert artifact["fingerprint"] == first.fingerprint
    assert "Cleaning philosophy" in (tmp_path / "data_quality_report.md").read_text()
    assert synthetic_config.source_path.read_bytes() == source_bytes


def test_local_timezone_preserves_offsets_and_localizes_naive_times():
    # Synthetic times intentionally mix naive, Z and short/long offset forms.
    config = DatasetConfig(timezone="Asia/Kolkata")
    parsed = parse_timestamps(
        pd.Series(["2024-01-01T10:00:00", "2024-01-01T10:00:00Z", "2024-01-01T10:00:00+02"]),
        config,
    )
    assert list(parsed.dt.hour) == [10, 15, 13]
    assert list(parsed.dt.minute) == [0, 30, 30]


def test_invalid_dst_times_are_reported_instead_of_guessed():
    config = DatasetConfig(timezone="America/New_York")
    parsed = parse_timestamps(pd.Series(["2024-03-10T02:30:00", "2024-11-03T01:30:00"]), config)
    assert parsed.isna().all()


def test_datetime_candidates_tolerate_small_malformed_fraction(csv_factory):
    config = csv_factory("when\n" + "2024-01-01\n" * 19 + "broken\n")
    schema = inspect_schema(load_csv(config))
    assert schema.datetime_like_columns == ["when"]
    assert schema.potential_timestamps[0].inferred is True
    assert schema.timestamp.available is False
