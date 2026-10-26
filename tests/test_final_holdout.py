"""Essential synthetic fixtures; never open the real final holdout for tests."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig
from riskbyte_ml.datasets.synthetic_riskbyte.generator import csv_bytes, generate
from riskbyte_ml.final_holdout.pipeline import continuation, verify_locked
from riskbyte_ml.inference.engine import future_features


def fixture():
    cfg = GeneratorConfig(transaction_count=100, entity_count=10, merchant_count=30, seed=99)
    d = generate(cfg)
    frame = d.transaction.merge(d.identity, on="TransactionID", how="left")
    return SimpleNamespace(
        frame=frame, generation={"config": cfg.model_dump(mode="json"), "transaction_count": 100}
    )


def test_original_generator_unchanged():
    cfg = GeneratorConfig(transaction_count=100, entity_count=10, merchant_count=30, seed=99)
    data = generate(cfg)
    assert (
        hashlib.sha256(csv_bytes(data.transaction) + csv_bytes(data.identity)).hexdigest()
        == "393aaee6539305eeebe0720def5ddd657536597e6adb0121e097344e491fbb9b"
    )


def test_future_continuation_determinism_and_context():
    previous = fixture()
    a, b = continuation(previous, 100), continuation(previous, 100)
    assert csv_bytes(a.transaction) == csv_bytes(b.transaction)
    assert a.config.seed == 2026
    assert a.transaction.TransactionDT.min() > 90 * 86400
    assert a.transaction.TransactionID.min() > 100
    assert a.transaction.behavioural_entity_id.str.startswith("entity-future-").any()
    assert (~a.transaction.behavioural_entity_id.str.startswith("entity-future-")).any()
    assert not a.transaction.TransactionID.duplicated().any()


def test_strict_prior_continuing_and_new_entity():
    history = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [1],
            "TransactionAmt": [10],
            "behavioural_entity_id": ["known"],
        }
    )
    incoming = pd.DataFrame(
        {
            "TransactionID": [2, 3, 4],
            "TransactionDT": [2, 2, 3],
            "TransactionAmt": [20, 30, 1000],
            "behavioural_entity_id": ["known", "new", "known"],
        }
    )
    features = future_features(history, incoming)
    assert features.behavioural_history_count.tolist() == [1, 0, 2]
    assert features.prior_mean_amount.iloc[0] == 10
    changed = incoming.copy()
    changed.loc[2, "TransactionAmt"] = 500000
    pd.testing.assert_series_equal(features.iloc[0], future_features(history, changed).iloc[0])
    assert np.isnan(features.prior_mean_amount.iloc[1])


def test_unlocked_holdout_is_not_final(tmp_path):
    from riskbyte_ml.final_holdout.pipeline import save

    save(tmp_path / "holdout_manifest.json", {"status": "READY_UNTOUCHED"})
    with pytest.raises(DatasetError):
        verify_locked(tmp_path)


def test_evaluation_cannot_restart(monkeypatch, tmp_path):
    import riskbyte_ml.final_holdout.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "prepare", lambda *a: ({"status": "EVALUATION_IN_PROGRESS"}, None)
    )
    with pytest.raises(DatasetError, match="already started"):
        pipeline.evaluate_once(None, tmp_path)


def test_locked_integrity_and_tamper(tmp_path):
    from riskbyte_ml.final_holdout.pipeline import save
    from riskbyte_ml.statistics.verification import file_hash

    directory = tmp_path / "artifacts" / "final_holdout"
    source = tmp_path / "data" / "final_holdout"
    source.mkdir(parents=True)
    (source / "fixture.csv").write_text("synthetic fixture only")
    save(
        directory / "holdout_manifest.json",
        {"status": "FINAL_EVALUATED_LOCKED", "holdout_fingerprint": "synthetic-fixture"},
    )
    save(directory / "final_metrics.json", {"synthetic_fixture": True})
    save(
        directory / "integrity.json",
        {
            "evaluation_count": 1,
            "holdout_fingerprint": "synthetic-fixture",
            "artifacts": {
                name: file_hash(directory / name)
                for name in ["holdout_manifest.json", "final_metrics.json"]
            },
            "sources": {"fixture.csv": file_hash(source / "fixture.csv")},
        },
    )
    assert verify_locked(directory)["status"] == "FINAL_EVALUATED_LOCKED"
    (directory / "final_metrics.json").write_text("{}")
    with pytest.raises(DatasetError):
        verify_locked(directory)
