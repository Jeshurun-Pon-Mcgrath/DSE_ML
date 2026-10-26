"""A predeclared continuation and irreversible final-evaluation journal."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskbyte_ml.abrf.contracts import A_THRESHOLD, P_THRESHOLD
from riskbyte_ml.anomaly.analysis import distribution, evaluate
from riskbyte_ml.anomaly.development import DevelopmentInputs
from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import GeneratorConfig, digest
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate, save_generated
from riskbyte_ml.inference.engine import FrozenSystem, future_features
from riskbyte_ml.statistics.verification import file_hash, read_json

SEED = 2026
CONTINUATION = "stateful-continuation-v1"


def save(path, value, overwrite=False):
    write_artifact(path, json.dumps(value, indent=2, allow_nan=False).encode(), overwrite=overwrite)


def checkpoint(development, continuing=1600):
    cfg = GeneratorConfig.model_validate(development.generation["config"])
    rng = np.random.default_rng(cfg.seed)
    weights = rng.lognormal(0, 0.65, cfg.entity_count)
    rng.multinomial(cfg.rows - 2 * cfg.entity_count, weights / weights.sum())
    baseline = rng.lognormal(np.log(2000), 1.15, cfg.entity_count).clip(100, 40000)
    spread = rng.uniform(0.25, 0.7, cfg.entity_count)
    home = rng.integers(0, 8, cfg.entity_count)
    preferred = rng.integers(0, int(cfg.merchant_count * 0.7), (cfg.entity_count, 5))
    age = rng.integers(5, 3000, cfg.entity_count)
    common = rng.integers(0, 3, cfg.entity_count)
    seen = {
        str(e): {int(str(v).split("-")[-1]) for v in values.dropna()}
        for e, values in development.frame.groupby("behavioural_entity_id").merchant_id
    }
    histories = []
    for e in range(continuing):
        histories.append(seen.get(f"entity-{e:05d}", set()))
    return {
        "continuing_entities": continuing,
        "baseline": baseline,
        "spread": spread,
        "home": home,
        "preferred": preferred,
        "account_age": age + 90,
        "common_type": common,
        "merchant_history": histories,
        "weights": weights,
    }


def continuation(development, rows=30000):
    original = GeneratorConfig.model_validate(development.generation["config"])
    continuing = int(original.entity_count * 0.8)
    cfg = original.model_copy(
        update={
            "transaction_count": rows,
            "seed": SEED,
            "duration_days": 27,
            "merchant_count": original.merchant_count + max(12, int(original.merchant_count * 0.2)),
        }
    )
    data = generate(cfg, checkpoint=checkpoint(development, continuing))
    offset = 90 * 86400
    data.transaction["TransactionDT"] += offset
    data.transaction["TransactionID"] += development.generation["transaction_count"]
    for frame in [data.identity, data.evaluation]:
        frame["TransactionID"] += development.generation["transaction_count"]
    new_entity = (
        data.transaction.behavioural_entity_id.str.extract(r"(\d+)$")[0].astype(int) >= continuing
    )
    data.transaction.loc[new_entity, "behavioural_entity_id"] = data.transaction.loc[
        new_entity, "behavioural_entity_id"
    ].str.replace("entity-", "entity-future-", regex=False)
    data.identity["device_id"] = data.identity.device_id.map(
        lambda v: str(v).replace("dev-new-", "dev-future-new-")
        if str(v).startswith("dev-new-")
        else str(v).replace("dev-", "dev-future-")
        if int(str(v).split("-")[1]) >= continuing
        else str(v)
    )
    return data


def verify_locked(directory):
    manifest = read_json(directory / "holdout_manifest.json")
    if manifest["status"] != "FINAL_EVALUATED_LOCKED":
        raise DatasetError("final_evaluation_unavailable", "Final holdout is not locked", 503)
    integrity = read_json(directory / "integrity.json")
    if (
        integrity["evaluation_count"] != 1
        or integrity["holdout_fingerprint"] != manifest["holdout_fingerprint"]
    ):
        raise DatasetError("final_holdout_changed", "Final evaluation journal is inconsistent")
    for name, expected in integrity["artifacts"].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise DatasetError("final_holdout_changed", "Locked final artifacts changed")
    data_dir = directory.parent.parent / "data" / "final_holdout"
    for name, expected in integrity["sources"].items():
        if file_hash(data_dir / name) != expected:
            raise DatasetError("final_holdout_changed", "Locked holdout source changed")
    return manifest


def prepare(config, root):
    system = FrozenSystem(config, root)
    directory = root / "final_holdout"
    if (directory / "holdout_manifest.json").exists():
        manifest = read_json(directory / "holdout_manifest.json")
        return manifest, system
    development = DevelopmentInputs(config, root)
    data = continuation(development)
    data_dir = root.parent / "data" / "final_holdout"
    save_generated(data, data_dir, overwrite=False)
    sources = {
        name: file_hash(data_dir / name)
        for name in [
            "train_transaction.csv",
            "train_identity.csv",
            "evaluation_metadata.csv",
            "generation_metadata.json",
        ]
    }
    fingerprint = digest(
        {
            "sources": {k: v for k, v in sources.items() if k.endswith(".csv")},
            "configuration": data.config.model_dump(mode="json"),
            "continuation": CONTINUATION,
            "history_checkpoint": "frozen TRAIN/VALIDATION 85000",
            "seed": SEED,
        }
    )
    manifest = {
        "generator_version": "synthetic-riskbyte-v1",
        "continuation_version": CONTINUATION,
        "seed": SEED,
        "transaction_count": len(data.transaction),
        "identity_count": len(data.identity),
        "joined_count": len(data.transaction),
        "TransactionDT_start": int(data.transaction.TransactionDT.min()),
        "TransactionDT_end": int(data.transaction.TransactionDT.max()),
        "holdout_fingerprint": fingerprint,
        "created_timestamp": datetime.now(UTC).isoformat(),
        "status": "READY_UNTOUCHED",
        "sources": sources,
        "frozen_system": system.provenance,
        "continuing_entities": 1600,
        "new_entities": 400,
        "merchant_pool": 360,
        "history_checkpoint_rows": 85000,
        "original_test_scored": False,
        "history_limitation": "Original test is not decoded; history during the gap is unobserved.",
    }
    save(directory / "holdout_manifest.json", manifest)
    save(directory / "ready_untouched.json", manifest)
    return manifest, system


def evaluate_once(config, root):
    directory = root / "final_holdout"
    manifest, system = prepare(config, root)
    if manifest["status"] == "FINAL_EVALUATED_LOCKED":
        return verify_locked(directory)
    if manifest["status"] != "READY_UNTOUCHED":
        raise DatasetError(
            "final_evaluation_consumed", "Evaluation already started; no automatic retry", 409
        )
    for name, expected in manifest["sources"].items():
        if file_hash(root.parent / "data" / "final_holdout" / name) != expected:
            raise DatasetError("final_holdout_changed", "Prepared source differs before scoring")
    # Exclusive journal is the irreversible scoring gate; no reset or tune interface exists.
    save(
        directory / "evaluation_started.json",
        {
            "holdout_fingerprint": manifest["holdout_fingerprint"],
            "started": datetime.now(UTC).isoformat(),
        },
    )
    manifest["status"] = "EVALUATION_IN_PROGRESS"
    save(directory / "holdout_manifest.json", manifest, overwrite=True)
    development = DevelopmentInputs(config, root)
    data_dir = root.parent / "data" / "final_holdout"
    tx = pd.read_csv(
        data_dir / "train_transaction.csv", usecols=lambda c: c != "isFraud", keep_default_na=True
    )
    identity = pd.read_csv(data_dir / "train_identity.csv")
    if tx.TransactionID.duplicated().any() or identity.TransactionID.duplicated().any():
        raise ValueError("Duplicate holdout keys")
    frame = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one", sort=False)
    if len(frame) != manifest["joined_count"] or frame.TransactionDT.min() <= 90 * 86400:
        raise ValueError("Invalid future join or temporal range")
    features = future_features(development.frame.drop(columns="isFraud"), frame)
    s = system.score(features)
    flags = s["risk"] >= system.fusion_config.decision_threshold
    predictions = pd.DataFrame(
        {
            "TransactionID": frame.TransactionID,
            "P": s["P"],
            "A": s["A"],
            "B": s["B"],
            "confidence": s["confidence"],
            "risk": s["risk"],
            "investigation_flag": flags,
        }
    )
    predictions["risk_level"] = np.array(["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"])[
        np.searchsorted(system.fusion_config.risk_boundaries, s["risk"], side="right")
    ]
    write_artifact(
        directory / "predictions.csv", predictions.to_csv(index=False).encode(), overwrite=False
    )
    # Outcomes are opened only after component scores/decisions are durably saved.
    y = pd.read_csv(data_dir / "train_transaction.csv", usecols=["isFraud"]).isFraud.to_numpy(int)
    metadata = pd.read_csv(data_dir / "evaluation_metadata.csv", keep_default_na=False)
    if (
        metadata.TransactionID.tolist() != frame.TransactionID.tolist()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Holdout target/metadata integrity failed")
    supervised = evaluate(y, s["P"], P_THRESHOLD, metadata)
    abrf = evaluate(y, s["risk"], system.fusion_config.decision_threshold, metadata)
    for row in [supervised, abrf]:
        row["specificity"] = 1 - row["fpr"]
        row["fnr"] = 1 - row["recall"]
        row["accuracy"] = (row["confusion"]["tp"] + row["confusion"]["tn"]) / len(y)
        row["score_semantics"] = (
            "Calibrated synthetic supervised probability"
            if row is supervised
            else "Fused risk, not fraud probability"
        )
    pflag = s["P"] >= P_THRESHOLD
    metrics = {
        "finding_label": "Synthetic academic finding",
        "holdout_fingerprint": manifest["holdout_fingerprint"],
        "transaction_count": len(y),
        "fraud_count": int(y.sum()),
        "fraud_prevalence": float(y.mean()),
        "hard_negative_count": int(metadata.hard_negative.sum()),
        "supervised": supervised,
        "abrf": abrf,
        "improvement": {
            "additional_fraud": int(((y == 1) & flags & ~pflag).sum()),
            "lost_fraud": int(((y == 1) & ~flags & pflag).sum()),
            "additional_false_positives": int(((y == 0) & flags).sum() - ((y == 0) & pflag).sum()),
            **{key: abrf[key] - supervised[key] for key in ["precision", "recall", "f1", "fpr"]},
        },
    }
    scenarios = {
        name: {
            "count": row["count"],
            "supervised": supervised["scenario_recall"][name],
            "abrf": row,
            "anomaly": None,
        }
        for name, row in abrf["scenario_recall"].items()
    }
    anomaly = evaluate(y, s["A"], A_THRESHOLD, metadata)
    anomaly["abrf_weight"] = 0.0
    for name in scenarios:
        scenarios[name]["anomaly"] = anomaly["scenario_recall"][name]
    hard = {
        name: {
            "count": row["count"],
            "supervised": supervised["hard_negative_types"][name],
            "abrf": row,
        }
        for name, row in abrf["hard_negative_types"].items()
    }
    hm = metadata.hard_negative.to_numpy(bool)
    behaviour = {
        "normal_legitimate": distribution(s["B"][(y == 0) & ~hm]),
        "hard_negative": distribution(s["B"][hm]),
        "fraud": distribution(s["B"][y == 1]),
        "confidence": distribution(s["confidence"]),
        "cold_start_percentage": float((features.behavioural_history_count.eq(0)).mean() * 100),
    }
    masks = {
        "true_positive": (y == 1) & flags,
        "true_negative": (y == 0) & ~flags,
        "false_positive": (y == 0) & flags,
        "false_negative": (y == 1) & ~flags,
        "hard_negative_rejected": hm & ~flags,
        "hard_negative_flagged": hm & flags,
        **{
            name: metadata.fraud_scenario.eq(name).to_numpy()
            for name in ["LOW_AND_SLOW", "VELOCITY_BURST", "INTERNATIONAL_ANOMALY"]
        },
    }
    cases = []
    for name, mask in masks.items():
        ids = np.flatnonzero(mask)
        if not len(ids):
            cases.append(
                {
                    "case": name,
                    "status": "UNAVAILABLE_WITH_REASON",
                    "reason": "No eligible holdout record",
                }
            )
            continue
        i = int(ids[0])
        explanation = system.explain(frame.iloc[i], features.iloc[[i]])
        if not np.isclose(explanation.fusion.final_risk_score, s["risk"][i], atol=1e-12, rtol=0):
            raise ValueError("Case explanation differs from locked predictions")
        cases.append(
            {
                "case": name,
                "status": "AVAILABLE",
                "evaluation_only": {
                    "isFraud": int(y[i]),
                    "fraud_scenario": metadata.fraud_scenario.iloc[i],
                    "hard_negative": bool(hm[i]),
                },
                "explanation": explanation.model_dump(mode="json"),
            }
        )
    reference = development.features.iloc[development.train_count :]
    from riskbyte_ml.abrf.inputs import load_inputs

    old, _, _ = load_inputs(config, root)
    old = old[old.partition.eq("validation")]
    drift = {
        name: {
            "development_validation": distribution(reference[name]),
            "future_holdout": distribution(features[name]),
        }
        for name in [
            "TransactionAmt",
            "transactions_prior_30m",
            "amount_ratio_to_prior_mean",
            "failed_attempts_before",
        ]
    }
    drift.update(
        {
            "behaviour_score": {
                "development_validation": distribution(old.B),
                "future_holdout": distribution(s["B"]),
            },
            "abrf_score": {
                "development_validation": distribution(
                    system.bundle.validation
                    and pd.read_csv(
                        root
                        / "abrf"
                        / "c8b265bcaa4c91db3a41a4ee8f708281ee4fe224bcf8ad369a7121922bb057b0"
                        / "development_fusion_scores.csv.gz"
                    )
                    .query("partition=='validation'")
                    .final_risk_score
                ),
                "future_holdout": distribution(s["risk"]),
            },
            "fraud_prevalence": {
                "development_validation": 266 / 15000,
                "future_holdout": float(y.mean()),
            },
            "scope": "Synthetic descriptive shift, no new tests or tuning",
        }
    )

    # Persist outcome-free demo contexts/features; frontend does not receive generator labels.
    def safe(value):
        if pd.isna(value):
            return None
        return value.item() if isinstance(value, np.generic) else value

    demo = []
    for i in range(len(frame)):
        demo.append(
            json.dumps(
                {
                    "transaction": {k: safe(v) for k, v in frame.iloc[i].items()},
                    "features": {k: safe(v) for k, v in features.iloc[i].items()},
                    "prediction": {k: safe(v) for k, v in predictions.iloc[i].items()},
                },
                allow_nan=False,
            )
        )
    write_artifact(
        directory / "demo_records.jsonl.gz",
        gzip.compress(("\n".join(demo) + "\n").encode(), mtime=0),
        overwrite=False,
    )
    reports = {
        "final_metrics.json": metrics,
        "scenario_metrics.json": scenarios,
        "hard_negative_metrics.json": hard,
        "behavioural_metrics.json": behaviour,
        "anomaly_metrics.json": anomaly,
        "case_studies.json": {"cases": cases},
        "drift_summary.json": drift,
    }
    for name, value in reports.items():
        save(directory / name, value)
    FrozenSystem(
        config, root
    )  # Verify unchanged components before final publication; no re-evaluation.
    manifest["status"] = "FINAL_EVALUATED_LOCKED"
    save(directory / "holdout_manifest.json", manifest, overwrite=True)
    hashes = {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}
    save(
        directory / "integrity.json",
        {
            "artifacts": hashes,
            "sources": manifest["sources"],
            "holdout_fingerprint": manifest["holdout_fingerprint"],
            "evaluation_count": 1,
        },
    )
    return verify_locked(directory)
