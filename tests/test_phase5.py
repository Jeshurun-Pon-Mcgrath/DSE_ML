"""Small synthetic fixtures only; no real project partitions or network dependencies."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from riskbyte_api.core.config import Settings
from riskbyte_api.main import create_app
from riskbyte_ml.anomaly.analysis import complementarity, distribution, evaluate, select_threshold
from riskbyte_ml.anomaly.development import DevelopmentInputs, identity_projection, prefix_csv
from riskbyte_ml.anomaly.engine import (
    INTERPRETABLE,
    AnomalyConfig,
    EmpiricalNormalizer,
    IsolationEngine,
    NumericPreparation,
    feature_contract,
)
from riskbyte_ml.anomaly.schemas import Phase5Response
from riskbyte_ml.behavioural.deviation import (
    SIGNALS,
    BehaviourConfig,
    DeviationEngine,
)
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig, SyntheticConfig
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate, save_generated
from riskbyte_ml.evaluation.splits import make_split
from riskbyte_ml.features.behaviour import build_features
from riskbyte_ml.features.manifest import feature_manifest
from riskbyte_ml.statistics.verification import verify_fingerprint


@pytest.fixture
def development_fixture(tmp_path, monkeypatch):
    """A separate 1,200-row synthetic test source; original project data is never opened."""
    generated = generate(
        GeneratorConfig(transaction_count=1200, entity_count=20, merchant_count=24)
    )
    directory = tmp_path / "data"
    manifest = save_generated(generated, directory)
    full = generated.transaction.merge(
        generated.identity, on="TransactionID", how="left", sort=False
    )
    split = make_split(full, manifest["fingerprint"], evaluation=generated.evaluation)
    features = build_features(full)
    specs = feature_manifest(features, set(full))
    config = SyntheticConfig(
        transaction_path=directory / "train_transaction.csv",
        identity_path=directory / "train_identity.csv",
        metadata_path=directory / "generation_metadata.json",
    )
    monkeypatch.setattr("riskbyte_ml.anomaly.development.verify_sources", lambda _: manifest)
    monkeypatch.setattr(
        "riskbyte_ml.anomaly.development.verify_frozen_files",
        lambda _: {"features": [s.model_dump() for s in specs]},
    )
    monkeypatch.setattr("riskbyte_ml.anomaly.development.frozen_artifacts", lambda _: split)
    return config, tmp_path / "artifacts", full, split


def test_real_projection_matches_prefix_features_without_test(development_fixture):
    config, root, full, split = development_fixture
    inputs = DevelopmentInputs(config, root)
    size = len(split.indexes["train"]) + len(split.indexes["validation"])
    assert len(inputs.frame) == size < len(full)
    assert set(inputs.frame.TransactionID).isdisjoint(
        set(full.TransactionID.iloc[size:].astype(str))
    )
    assert inputs.audit["test_rows_decoded"] == 0
    reference = build_features(full.iloc[:size])
    np.testing.assert_allclose(
        inputs.features.prior_mean_amount.astype(float),
        reference.prior_mean_amount.astype(float),
        equal_nan=True,
    )


def test_tampered_frozen_split_rejected(development_fixture, monkeypatch):
    config, root, _, split = development_fixture
    changed = split.model_copy(
        update={"indexes": {**split.indexes, "validation": split.indexes["validation"][::-1]}}
    )
    monkeypatch.setattr("riskbyte_ml.anomaly.development.frozen_artifacts", lambda _: changed)
    with pytest.raises(DatasetError):
        DevelopmentInputs(config, root)


def test_end_to_end_small_pipeline_never_releases_test(development_fixture, monkeypatch):
    from sklearn.tree import DecisionTreeClassifier

    from riskbyte_ml.anomaly import pipeline
    from riskbyte_ml.anomaly.schemas import Phase5Bundle
    from riskbyte_ml.models.supervised.inputs import TestGate

    config, root, *_ = development_fixture

    def forbidden(*args):
        pytest.fail("Phase 5 cannot release a test gate")

    monkeypatch.setattr(TestGate, "release", forbidden)

    def baseline(_, inputs):
        # Test-only frozen baseline fits ONLY fixture TRAIN numeric features.
        names = feature_contract(inputs.features, "interpretable")
        prep = NumericPreparation.fit(inputs.features.iloc[: inputs.train_count], names)
        matrix = prep.transform(inputs.features)
        tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(
            matrix[: inputs.train_count], inputs.labels[: inputs.train_count]
        )
        return tree.predict_proba(matrix)[:, 1], {
            "model_id": "synthetic-test-baseline",
            "registry_hash": "synthetic-test-hash",
        }

    monkeypatch.setattr(pipeline, "frozen_baseline", baseline)
    monkeypatch.setattr(pipeline, "file_hash", lambda _: "synthetic-test-hash")
    result = pipeline.run(config, root, AnomalyConfig(n_estimators=10, max_samples=32, n_jobs=1))
    Phase5Bundle.model_validate_json(result.model_dump_json())
    saved = pd.read_csv(root / "phase5" / "development_scores.csv")
    assert set(saved.partition) == {"train", "validation"}
    assert len(saved) == result.leakage_audit["decoded_transaction_rows"]
    assert result.leakage_audit["test_rows_scored"] == 0
    assert len(result.study) == 4
    assert result.selected["population"] == "legitimate_train"


@pytest.fixture
def events():
    """Synthetic prior-history fixture: two entities, known novelty, equal-time peers."""
    return pd.DataFrame(
        {
            "TransactionID": [str(i) for i in range(12)],
            "TransactionDT": [0, 0, 100, 100, 200, 200, 300, 300, 400, 400, 400, 500],
            "TransactionAmt": [
                10.0,
                100.0,
                20.0,
                110.0,
                15.0,
                120.0,
                15.0,
                130.0,
                90.0,
                140.0,
                1000.0,
                150.0,
            ],
            "behavioural_entity_id": ["a", "b"] * 6,
            "merchant_id": ["m1"] * 8 + ["new", "m1", "future", "m1"],
            "device_id": ["d1"] * 8 + ["new", "d1", "future", "d1"],
            "billing_region": ["r1"] * 8 + ["new", "r1", "future", "r1"],
            "product_code": ["p1"] * 12,
            "failed_attempts_before": [0] * 8 + [5, 0, 0, 0],
            "account_age_days": [100] * 12,
            "identity_age_days": [50] * 12,
            "is_international": [0] * 12,
        }
    )


@pytest.fixture
def numeric_fixture():
    rng = np.random.default_rng(7)
    values = pd.DataFrame({name: rng.normal(size=240) for name in INTERPRETABLE})
    values["behavioural_history_count"] = np.arange(240)
    return values


def fitted(features):
    return IsolationEngine.fit(
        features,
        list(features),
        AnomalyConfig(n_estimators=12, max_samples=64, n_jobs=1),
        population="mixed_train",
        feature_family="interpretable",
    )


def test_isolation_reproducible(numeric_fixture):
    np.testing.assert_array_equal(
        fitted(numeric_fixture).score(numeric_fixture)[1],
        fitted(numeric_fixture).score(numeric_fixture)[1],
    )


@pytest.mark.parametrize(
    "column",
    [
        "isFraud",
        "fraud_scenario",
        "hard_negative",
        "hard_negative_type",
        "behavioural_entity_id",
        "TransactionID",
    ],
)
def test_metadata_cannot_be_fit_feature(numeric_fixture, column):
    values = numeric_fixture.assign(**{column: 1})
    with pytest.raises(ValueError):
        NumericPreparation.fit(values, list(values))
    with pytest.raises(ValueError):
        feature_contract(values, "broad_numeric")


def test_label_changes_cannot_change_scores(numeric_fixture):
    names = feature_contract(numeric_fixture, "interpretable")
    config = AnomalyConfig(n_estimators=10, max_samples=32, n_jobs=1)
    first = numeric_fixture.assign(isFraud=0, fraud_scenario="A", hard_negative=False)
    second = numeric_fixture.assign(isFraud=1, fraud_scenario="B", hard_negative=True)
    engines = [
        IsolationEngine.fit(
            x, names, config, population="mixed_train", feature_family="interpretable"
        )
        for x in [first, second]
    ]
    np.testing.assert_array_equal(engines[0].score(first)[1], engines[1].score(second)[1])


def test_anomaly_bounds_monotonic_and_raw_orientation(numeric_fixture):
    raw, scores = fitted(numeric_fixture).score(numeric_fixture)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert (np.diff(scores[np.argsort(-raw)]) >= 0).all()


def test_training_only_normalization(numeric_fixture):
    engine = fitted(numeric_fixture)
    reference = engine.normalization.reference.copy()
    medians = engine.preparation.medians.copy()
    engine.score(numeric_fixture * 1000)
    np.testing.assert_array_equal(engine.normalization.reference, reference)
    np.testing.assert_array_equal(engine.preparation.medians, medians)


def test_empirical_zero_mass_and_unavailable():
    normalizer = EmpiricalNormalizer.fit([0, 0, 0, 1, 2])
    assert normalizer.transform([0])[0] == 0
    assert normalizer.transform([100])[0] == 1
    assert np.isnan(normalizer.transform([np.nan])[0])
    with pytest.raises(ValueError):
        EmpiricalNormalizer.fit([np.nan])


@pytest.mark.parametrize("value", ["bad", 0.0, 0.6, float("nan")])
def test_invalid_contamination(value):
    with pytest.raises(ValidationError):
        AnomalyConfig(contamination_candidates=[value])


@pytest.mark.parametrize("value", ["auto", 0.005, 0.01, 0.02, 0.03, 0.05])
def test_contamination_candidates(value):
    assert AnomalyConfig(contamination_candidates=[value]).contamination_candidates == [value]


def test_feature_contract_ignores_categories_and_clock(events):
    features = build_features(events)
    names = feature_contract(features, "broad_numeric")
    assert "merchant_id" not in names
    assert "elapsed_day_index" not in names
    assert "relative_time_position" not in names
    with pytest.raises(ValueError):
        feature_contract(features, "unknown")


def test_serialization_reload(numeric_fixture):
    engine = fitted(numeric_fixture)
    buffer = io.BytesIO()
    joblib.dump(engine, buffer)
    buffer.seek(0)
    np.testing.assert_array_equal(
        joblib.load(buffer).score(numeric_fixture)[1], engine.score(numeric_fixture)[1]
    )


def test_numeric_nonfinite_preparation(numeric_fixture):
    frame = numeric_fixture.copy()
    frame.iloc[0, 0] = np.inf
    frame.iloc[:, 1] = np.nan
    prep = NumericPreparation.fit(frame, list(frame))
    matrix = prep.transform(frame)
    assert np.isfinite(matrix).all()
    assert matrix.shape[1] == 2 * frame.shape[1]


@pytest.mark.parametrize("position", [0, 1])
def test_cold_start_unavailable(events, position):
    features = build_features(events)
    result = DeviationEngine.fit(features).explain(features, position)
    assert result.behaviour_score is None
    assert result.behaviour_confidence == 0
    assert result.history_count == 0
    assert not result.history_available


def test_confidence_short_and_long_history(events):
    features = build_features(events)
    engine = DeviationEngine.fit(features)
    test = features.iloc[[4]].copy()
    test["behavioural_history_count"] = 1
    short = engine.explain(test, 0)
    test["behavioural_history_count"] = 100
    long = engine.explain(test, 0)
    assert 0 < short.behaviour_confidence < long.behaviour_confidence <= 1


@pytest.mark.parametrize("signal", list(SIGNALS))
def test_component_bounds_and_contribution(events, signal):
    features = build_features(events)
    engine = DeviationEngine.fit(features.iloc[:8])
    result = engine.explain(features, 8)
    component = next(x for x in result.components if x.signal == signal)
    assert component.score is None or 0 <= component.score <= 1
    assert 0 <= result.behaviour_score <= 1
    assert sum(x.contribution or 0 for x in result.components) == pytest.approx(
        result.behaviour_score
    )


@pytest.mark.parametrize("signal", ["merchant_novelty", "device_novelty", "region_novelty"])
def test_novelty_semantics(events, signal):
    features = build_features(events)
    engine = DeviationEngine.fit(features.iloc[:8])
    novel = next(x for x in engine.explain(features, 8).components if x.signal == signal)
    seen = next(x for x in engine.explain(features, 4).components if x.signal == signal)
    assert novel.score == 1
    assert seen.score == 0


def test_failed_attempt_component(events):
    features = build_features(events)
    engine = DeviationEngine.fit(features.iloc[:8])
    assert (
        next(
            x for x in engine.explain(features, 8).components if x.signal == "failed_attempts"
        ).score
        == 1
    )


def test_behaviour_train_only_deterministic_reload(events):
    features = build_features(events)
    engine = DeviationEngine.fit(features.iloc[:8])
    first = engine.explain(features, 8)
    engine.score_arrays(features.iloc[8:])
    assert engine.explain(features, 8) == first
    buffer = io.BytesIO()
    joblib.dump(engine, buffer)
    buffer.seek(0)
    assert joblib.load(buffer).explain(features, 8) == first


@pytest.mark.parametrize("column", ["TransactionAmt", "merchant_id", "device_id", "billing_region"])
def test_future_change_never_changes_earlier_behaviour(events, column):
    first = build_features(events)
    changed = events.copy()
    changed.loc[11, column] = 999999 if column == "TransactionAmt" else "changed_future"
    second = build_features(changed)
    pd.testing.assert_frame_equal(first.iloc[:11], second.iloc[:11])
    engine = DeviationEngine.fit(first.iloc[:8])
    np.testing.assert_array_equal(
        engine.score_arrays(first.iloc[:11])[0], engine.score_arrays(second.iloc[:11])[0]
    )


def test_current_and_equal_time_events_excluded(events):
    features = build_features(events)
    assert features.prior_transaction_count.iloc[0] == 0
    assert features.transactions_prior_5m.iloc[0] == 0
    assert features.prior_transaction_count.iloc[8] == features.prior_transaction_count.iloc[10]
    assert features.prior_mean_amount.iloc[8] == features.prior_mean_amount.iloc[10]


def test_behaviour_ignores_outcomes(events):
    first = build_features(events.assign(isFraud=0, fraud_scenario="X", hard_negative=True))
    second = build_features(events.assign(isFraud=1, fraud_scenario="Y", hard_negative=False))
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(
        DeviationEngine.fit(first).score_arrays(first)[0],
        DeviationEngine.fit(second).score_arrays(second)[0],
    )


def test_bad_behaviour_weights():
    with pytest.raises(ValidationError):
        BehaviourConfig(weights={"device_novelty": 1.0})
    with pytest.raises(ValidationError):
        BehaviourConfig(weights=dict.fromkeys(SIGNALS, 0.0))


def test_complementarity_four_cells():
    result = complementarity([1, 1, 1, 1, 0], [1, 0, 1, 0, 0], [0, 1, 1, 0, 1])
    assert [result[k] for k in ["supervised_only", "anomaly_only", "both", "neither"]] == [
        1,
        1,
        1,
        1,
    ]
    assert result["additional_false_positives"] == 1


def test_validation_threshold_budgets():
    y = np.r_[np.zeros(100, int), np.ones(3, int)]
    scores = np.r_[np.linspace(0, 0.8, 100), [0.7, 0.9, 1.0]]
    hard = np.r_[np.ones(10, bool), np.zeros(93, bool)]
    result = select_threshold(y, scores, hard, AnomalyConfig())
    assert scores[result["threshold"] <= scores].size > 0
    assert ((scores[:100] >= result["threshold"]).sum() / 100) <= 0.02
    assert result["source"] == "validation only"


def test_unavailable_scores_are_not_zero_risk():
    meta = pd.DataFrame(
        {
            "hard_negative": [False, False],
            "fraud_scenario": ["", "AMOUNT_SPIKE"],
            "hard_negative_type": ["", ""],
        }
    )
    report = evaluate([0, 1], [np.nan, np.nan], 0.5, meta)
    assert report["score_available"] == 0
    assert report["unavailable_fraud"] == 1
    assert report["pr_auc"] is None


def test_prefix_never_parses_test_record(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_bytes(b"TransactionID,amount\n1,10\n2,20\n3,\xffmalformed-test\n")
    assert len(prefix_csv(path, 2, "ascii")) == 2


def test_identity_excluded_fields_never_decoded(tmp_path):
    path = tmp_path / "synthetic_identity.csv"
    path.write_bytes(b"TransactionID,device\n1,known\n999,\xff-test-context\n2,known\n")
    projected = identity_projection(path, {"1", "2"}, "ascii")
    assert list(projected.TransactionID) == ["1", "2"]


def test_fingerprint_failure_precedes_decoding(monkeypatch):
    def reject(_):
        verify_fingerprint("sha256:changed")

    def forbidden(*args):
        pytest.fail("Decoding must not start after fingerprint failure")

    monkeypatch.setattr("riskbyte_ml.anomaly.development.verify_sources", reject)
    monkeypatch.setattr("riskbyte_ml.anomaly.development.prefix_csv", forbidden)
    with pytest.raises(DatasetError):
        DevelopmentInputs(None, None)


@pytest.mark.parametrize(
    "route",
    [
        "anomaly/status",
        "anomaly/metrics",
        "anomaly/scenarios",
        "anomaly/hard-negatives",
        "behaviour/summary",
        "behaviour/components",
        "behaviour/validation-analysis",
    ],
)
def test_api_unconfigured(route):
    app = create_app(Settings(_env_file=None, environment="test"))
    with TestClient(app) as client:
        response = client.get("/api/v1/" + route)
    assert response.status_code == 200
    assert response.json()["status"] == "dataset_not_configured"
    Phase5Response.model_validate(response.json())


def test_zero_variance_missing_distribution():
    assert distribution([1, 1, np.nan])["median"] == 1
    assert distribution([np.nan])["mean"] is None


def test_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Phase5Response(status="phase5_ready", unexpected=True)
