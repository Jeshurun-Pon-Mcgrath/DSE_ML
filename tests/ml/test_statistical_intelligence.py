"""Small synthetic fixtures only; never read the MEDIUM dataset or use the internet."""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.adapter import load_dataset
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig, SyntheticConfig
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate as generate_dataset
from riskbyte_ml.datasets.synthetic_riskbyte.generator import save_generated
from riskbyte_ml.evaluation.splits import SplitConfig, make_split, save_split
from riskbyte_ml.features.behaviour import FEATURE_PIPELINE_VERSION, build_features
from riskbyte_ml.statistics import verification
from riskbyte_ml.statistics.analysis import (
    correlations,
    describe,
    groups,
    hard_negative_analysis,
    hypotheses,
    novelty_test,
    scenario_analysis,
)
from riskbyte_ml.statistics.methods import (
    benjamini_hochberg,
    categorical_test,
    cliffs_delta,
    mann_whitney,
    prevalence_interval,
    summarize,
)
from riskbyte_ml.statistics.models import StatisticalBundle, StatisticsResponse
from riskbyte_ml.statistics.pipeline import generate


def test_descriptives_match_hand_calculated_sample():
    s = summarize([1, 2, 3, 4, 5])
    assert (s.count, s.mean, s.median, s.minimum, s.maximum) == (5, 3, 3, 1, 5)
    assert s.variance == pytest.approx(2.5)
    assert s.standard_deviation == pytest.approx(np.sqrt(2.5))
    assert (s.q1, s.q3, s.iqr) == pytest.approx((2, 4, 2))
    assert s.skewness == pytest.approx(0, abs=1e-14)
    assert s.kurtosis == pytest.approx(-1.2)
    assert s.percentiles["95"] == pytest.approx(4.8)


def test_skewness_and_missing_accounting():
    s = summarize([1, 1, 1, 2, 20, None, "bad", np.inf, -np.inf])
    assert s.count == 5
    assert (s.missing_count, s.invalid_count, s.infinite_count) == (1, 1, 2)
    assert s.skewness > 1
    assert "NaN" not in s.model_dump_json() and "Infinity" not in s.model_dump_json()


@pytest.mark.parametrize("values", [[], [None, np.nan], [np.inf, "bad"]])
def test_empty_summary_safe(values):
    result = summarize(values)
    assert result.count == 0
    assert result.mean is None and result.variance is None and result.kurtosis is None


@pytest.mark.parametrize("values", [[7], [7, 7, 7, 7]])
def test_zero_variance_moments(values):
    result = summarize(values)
    assert result.skewness is None and result.kurtosis is None
    assert result.variance == (None if len(values) == 1 else 0)


def test_extreme_numeric_summary_does_not_emit_invalid_numbers():
    s = summarize([-1e308, 1e308, 1e308, -1e308])
    assert s.mean == pytest.approx(0)
    assert s.variance is None  # True variance exceeds finite float representation.
    json.loads(s.model_dump_json())


def test_mann_whitney_exact_known_example():
    result = mann_whitney([19, 22, 16, 29, 24], [20, 11, 17, 12])
    assert result.statistic == 17
    assert result.p_value == pytest.approx(0.1111111111111111)
    assert "exact" in result.method
    assert result.effect_size.value == pytest.approx(0.7)


def test_mann_whitney_small_tied_permutation_is_deterministic():
    first = mann_whitney([0, 0, 1], [0, 1, 1])
    second = mann_whitney([0, 0, 1], [0, 1, 1])
    assert first.p_value == second.p_value
    assert "permutation" in first.method


def test_mann_whitney_missing_and_constant():
    result = mann_whitney([2, 2, None], [2, 2, np.inf])
    assert result.p_value == 1 and result.effect_size.value == 0
    assert result.excluded_counts == {"first": 1, "second": 1}
    assert mann_whitney([None], [1, 2]).status == "UNAVAILABLE_WITH_REASON"


def test_extreme_ranks_preserved_and_p_underflow_labelled():
    result = mann_whitney([1e-320, 2e-320, 1e308], [3e-320, 4e-320, 1.5e308])
    assert result.statistic == 2
    assert result.effect_size.value == pytest.approx(-5 / 9)
    separated = mann_whitney(np.arange(2000), np.arange(2000) + 10000)
    assert separated.p_value == 0 and separated.p_value_underflow


@pytest.mark.parametrize(
    "x,y,delta",
    [([2, 3], [0, 1], 1), ([0, 1], [2, 3], -1), ([0, 1], [0, 1], 0), ([1, 2], [1, 1], 0.5)],
)
def test_cliffs_delta_direction_and_ties(x, y, delta):
    assert cliffs_delta(x, y) == delta
    assert cliffs_delta(y, x) == -delta


def test_delta_without_finite_data_is_unavailable():
    assert cliffs_delta([np.inf], [1, 2]) is None


def test_chi_square_and_cramers_v_known_table():
    result = categorical_test([[30, 20], [20, 30]])
    assert result.statistic == 4
    assert result.p_value == pytest.approx(0.04550026389635857)
    assert result.effect_size.value == pytest.approx(0.2)
    assert result.effect_size.odds_ratio == pytest.approx(2.25)


def test_fisher_selected_for_sparse_two_by_two():
    result = categorical_test([[1, 9], [11, 3]])
    # One expected cell equals 5, so explicitly lower the total for sparse selection.
    result = categorical_test([[1, 3], [4, 0]])
    assert "Fisher" in result.method
    assert result.p_value == pytest.approx(1 / 7)
    assert result.effect_size.odds_ratio == 0


def test_zero_marginal_and_unbounded_odds_safe():
    assert categorical_test([[0, 0], [2, 3]]).status == "UNAVAILABLE_WITH_REASON"
    result = categorical_test([[2, 0], [0, 2]])
    assert result.effect_size.odds_ratio is None
    assert result.effect_size.odds_ratio_reason
    assert "Infinity" not in result.model_dump_json()


@pytest.mark.parametrize(
    "table", [[[1, -1], [2, 3]], [[1.2, 3], [4, 5]], [[np.nan, 1], [2, 3]], [[1, 2]]]
)
def test_invalid_contingencies_rejected(table):
    with pytest.raises(ValueError):
        categorical_test(table)


def test_sparse_larger_table_returns_unavailable():
    assert categorical_test([[1, 1], [1, 1], [1, 1]]).status == "UNAVAILABLE_WITH_REASON"


def test_bh_correction_known_order_and_unavailable_family_members():
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.002]) == pytest.approx([0.02, 0.04, 0.04, 0.008])
    result = benjamini_hochberg([0.01, None, 0.01])
    assert result == [0.015, None, 0.015]
    assert benjamini_hochberg([]) == []


@pytest.mark.parametrize("values", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_bh_invalid_p_rejected(values):
    with pytest.raises(ValueError):
        benjamini_hochberg(values)


def test_wilson_interval_reference_and_empty():
    p = prevalence_interval(20, 1000)
    assert p.prevalence == 0.02
    assert p.lower == pytest.approx(0.012983682916415, rel=1e-8)
    assert p.upper == pytest.approx(0.030690005229, rel=1e-8)
    assert prevalence_interval(0, 0).prevalence is None
    assert prevalence_interval(0, 10).lower == pytest.approx(0, abs=1e-15)
    assert prevalence_interval(10, 10).upper == pytest.approx(1)


@pytest.mark.parametrize("fraud,rows", [(-1, 10), (11, 10), (0, -1)])
def test_invalid_prevalence_counts(fraud, rows):
    with pytest.raises(ValueError):
        prevalence_interval(fraud, rows)


def test_strict_schema_rejects_nonfinite_and_bad_envelope():
    with pytest.raises(ValidationError):
        summarize([1, 2]).model_validate({**summarize([1, 2]).model_dump(), "mean": float("inf")})
    with pytest.raises(ValidationError):
        StatisticsResponse(status="statistics_ready")
    assert (
        StatisticsResponse(status="dataset_not_configured").model_dump_json()
        == '{"status":"dataset_not_configured"}'
    )


@pytest.fixture(scope="module")
def small_statistical_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic_statistics_fixture")
    source = directory / "data"
    generated = generate_dataset(
        GeneratorConfig(
            transaction_count=800,
            entity_count=20,
            merchant_count=24,
            fraud_prevalence=0.08,
            duration_days=3,
        )
    )
    manifest = save_generated(generated, source)
    config = SyntheticConfig(
        transaction_path=source / "train_transaction.csv",
        identity_path=source / "train_identity.csv",
        metadata_path=source / "generation_metadata.json",
    )
    dataset, _, _, _ = load_dataset(config)
    features = build_features(dataset.frame, horizon_seconds=3 * 86400)
    split = make_split(dataset.frame, dataset.fingerprint, SplitConfig(), generated.evaluation)
    root = directory / "artifacts"
    save_split(split, root / "splits")
    (root / "features").mkdir()
    (root / "features" / "feature_manifest.json").write_text(
        json.dumps(
            {
                "dataset_fingerprint": dataset.fingerprint,
                "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
                "split_id": split.split_id,
            }
        ),
        encoding="utf-8",
    )
    return config, root, manifest, dataset.frame, generated.evaluation, features, split


@pytest.fixture
def frozen_fixture(small_statistical_fixture, monkeypatch):
    # Only test policy constants are patched; production CLI has no fingerprint override flag.
    config, root, manifest, frame, evaluation, features, split = small_statistical_fixture
    monkeypatch.setattr(verification, "FROZEN_FINGERPRINT", manifest["fingerprint"])
    monkeypatch.setattr(verification, "FROZEN_SPLIT_ID", split.split_id)
    return config, root, manifest, frame, evaluation, features, split


def test_hard_negative_and_scenario_counts(small_statistical_fixture):
    _, _, _, frame, evaluation, features, _ = small_statistical_fixture
    hard = hard_negative_analysis(features, frame.isFraud, evaluation)
    scenario = scenario_analysis(features, frame.isFraud, evaluation)
    assert sum(g.count for g in hard.groups.values()) == len(frame)
    assert hard.groups["hard_negative_legitimate"].count == int(evaluation.hard_negative.sum())
    assert sum(g.count for g in scenario.scenarios.values()) == int(frame.isFraud.astype(int).sum())
    assert sum(g.proportion for g in scenario.scenarios.values()) == pytest.approx(1)
    assert "fraud_scenario" not in features and "hard_negative" not in features


def test_primary_hypotheses_and_group_effects(small_statistical_fixture):
    _, _, _, frame, evaluation, features, split = small_statistical_fixture
    rows = split.indexes["train"]
    report = hypotheses(features.iloc[rows], frame.isFraud.iloc[rows], evaluation.iloc[rows])
    assert report.family_size == 6
    assert {h.id for h in report.hypotheses} == {f"RQ{i}" for i in range(1, 7)}
    assert all(h.h0 and h.h1 and h.outcome.assumption_checks for h in report.hypotheses)
    assert all(
        h.adjusted_p_value >= h.outcome.p_value
        for h in report.hypotheses
        if h.adjusted_p_value is not None
    )
    descriptive = describe(features, groups(frame.isFraud, evaluation))
    assert descriptive.group_counts["all"] == 800


def test_missing_novelty_never_becomes_seen_or_new():
    result = novelty_test(
        pd.Series([False, True, None, False, True, None]),
        np.array([True, True, True, False, False, False]),
    )
    assert result.contingency_table == [[1, 1], [1, 1]]
    assert result.excluded_counts == {"fraud": 1, "legitimate": 1}


def test_correlations_omit_identifiers_and_handle_missing_constants(small_statistical_fixture):
    features = small_statistical_fixture[5].copy()
    features["account_age_days"] = 0
    features["identity_age_days"] = np.nan
    report = correlations(features)
    assert len(report.pairs) == 66
    assert "TransactionID" not in report.columns and "isFraud" not in report.columns
    assert all(p.reason for p in report.pairs if "account_age_days" in (p.first, p.second))
    assert all(
        p.pair_count == 0 for p in report.pairs if "identity_age_days" in (p.first, p.second)
    )


def test_frozen_fingerprint_enforcement(small_statistical_fixture):
    config, root, *_ = small_statistical_fixture
    with pytest.raises(DatasetError, match="frozen Phase 3 fingerprint"):
        generate(config, root, root / "statistics")
    assert not (root / "statistics" / "analysis_bundle.json").exists()


def test_full_small_pipeline_serialization_and_no_calendar(frozen_fixture):
    config, root, *_ = frozen_fixture
    bundle = generate(config, root, root / "statistics")
    restored = StatisticalBundle.model_validate_json(
        (root / "statistics" / "analysis_bundle.json").read_bytes()
    )
    assert restored == bundle
    assert len(bundle.temporal.buckets) <= 4
    assert all(b.elapsed_day_index >= 0 for b in bundle.temporal.buckets)
    assert bundle.prevalence.partitions["overall"].rows == 800
    assert "calendar" not in bundle.temporal.model_dump()
    assert len(list((root / "statistics").glob("*.json"))) == 8


@pytest.mark.parametrize("change", ["source", "evaluation", "version", "split", "missing_split"])
def test_changed_provenance_aborts_without_output(frozen_fixture, tmp_path, change):
    import shutil

    config, root, *_ = frozen_fixture
    destination = tmp_path / "isolated"
    shutil.copytree(root.parent, destination)
    config = config.model_copy(
        update={
            name: destination / "data" / getattr(config, name).name
            for name in ("transaction_path", "identity_path", "metadata_path")
        }
    )
    root = destination / "artifacts"
    if change == "source":
        with config.transaction_path.open("ab") as stream:
            stream.write(b"\n")
    elif change == "evaluation":
        (destination / "data" / "evaluation_metadata.csv").write_text("tampered", encoding="utf-8")
    elif change == "version":
        path = root / "features" / "feature_manifest.json"
        value = json.loads(path.read_bytes())
        value["feature_pipeline_version"] = "changed"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif change == "split":
        path = next((root / "splits").glob("*.json"))
        value = json.loads(path.read_bytes())
        value["indexes"]["train"][0] = value["indexes"]["test"][0]
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        next((root / "splits").glob("*.json")).unlink()
    output = root / "new_statistics"
    with pytest.raises(DatasetError):
        generate(config, root, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "summary",
        "hypotheses",
        "correlations",
        "prevalence",
        "scenarios",
        "hard-negatives",
        "temporal",
    ],
)
def test_statistics_api_unconfigured(endpoint):
    from fastapi.testclient import TestClient

    from riskbyte_api.core.config import Settings
    from riskbyte_api.main import create_app

    with TestClient(create_app(Settings(_env_file=None, dataset_config_path=None))) as client:
        result = client.get("/api/v1/statistics/" + endpoint)
        assert result.status_code == 200
        assert result.json() == {"status": "dataset_not_configured"}


def test_statistics_api_ready_openapi_privacy_and_missing_artifacts(frozen_fixture, monkeypatch):
    from fastapi.testclient import TestClient

    from riskbyte_api.core.config import Settings
    from riskbyte_api.main import create_app
    from riskbyte_api.services import statistics as service

    config, root, manifest, _, _, _, split = frozen_fixture
    generate(config, root, root / "statistics")
    monkeypatch.setattr(service, "FROZEN_FINGERPRINT", manifest["fingerprint"])
    monkeypatch.setattr(service, "FROZEN_SPLIT_ID", split.split_id)
    config_path = root.parent / "adapter.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        dataset_config_path=config_path,
        dataset_adapter="synthetic_riskbyte",
        statistics_artifacts_dir=root / "statistics",
    )
    with TestClient(create_app(settings)) as client:
        for endpoint in (
            "summary",
            "hypotheses",
            "correlations",
            "prevalence",
            "scenarios",
            "hard-negatives",
            "temporal",
        ):
            result = client.get("/api/v1/statistics/" + endpoint)
            assert result.status_code == 200, result.text
            body = result.json()
            assert body["status"] == "statistics_ready"
            assert body["provenance"]["dataset_fingerprint"] == manifest["fingerprint"]
            assert str(root) not in result.text and "NaN" not in result.text
        schema = client.get("/openapi.json").json()
        assert "HypothesisReport" in schema["components"]["schemas"]
        bundle_path = root / "statistics" / "analysis_bundle.json"
        payload = bundle_path.read_bytes()
        bundle_path.unlink()
        assert client.get("/api/v1/statistics/summary").json() == {
            "status": "statistics_not_generated"
        }
        bundle_path.write_bytes(payload)
        corrupted = json.loads(payload)
        corrupted["provenance"]["artifact_version"] = "old"
        bundle_path.write_text(json.dumps(corrupted), encoding="utf-8")
        error = client.get("/api/v1/statistics/summary")
        assert (
            error.status_code == 422
            and error.json()["error"]["code"] == "statistics_bundle_invalid"
        )
        bundle_path.write_bytes(payload)
        # Content changed despite preserved attributes cannot serve a stale report.
        import os

        original = config.transaction_path.read_bytes()
        stat = config.transaction_path.stat()
        config.transaction_path.write_bytes(
            original.replace(b"TransactionAmt", b"TransactionAmX", 1)
        )
        os.utime(config.transaction_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        try:
            result = client.get("/api/v1/statistics/summary")
            assert result.status_code == 422
            assert result.json()["error"]["code"] == "statistics_fingerprint_mismatch"
            assert client.get("/health").status_code == 200
        finally:
            config.transaction_path.write_bytes(original)


def test_other_dataset_adapter_statistics_unavailable():
    from fastapi.testclient import TestClient

    from riskbyte_api.core.config import Settings
    from riskbyte_api.main import create_app

    with TestClient(
        create_app(Settings(_env_file=None, dataset_config_path="unused.json"))
    ) as client:
        result = client.get("/api/v1/statistics/summary")
        assert result.status_code == 200
        assert result.json()["status"] == "statistics_unavailable"
        assert result.json()["reason"]
