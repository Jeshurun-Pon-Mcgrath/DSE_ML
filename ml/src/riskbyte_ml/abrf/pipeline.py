"""Offline fusion research on frozen development scores; component training is absent."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskbyte_ml.abrf.contracts import (
    A_THRESHOLD,
    ABRF_VERSION,
    B_THRESHOLD,
    P_THRESHOLD,
    ABRFBundle,
    FusionConfig,
    FusionInput,
    SearchConfig,
    Weights,
)
from riskbyte_ml.abrf.engine import fuse, fusion_arrays
from riskbyte_ml.abrf.inputs import load_inputs, safe_child, verify_components
from riskbyte_ml.abrf.search import candidates, eligible_constraint, rank_candidate, threshold_curve
from riskbyte_ml.anomaly.analysis import distribution, evaluate, rates
from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import digest
from riskbyte_ml.statistics.verification import file_hash


def save_json(path, value):
    write_artifact(path, json.dumps(value, allow_nan=False, indent=2).encode(), overwrite=False)


def score_frame(frame, candidate):
    return fusion_arrays(
        frame.P,
        frame.A,
        frame.B,
        frame.behaviour_confidence,
        Weights.model_validate(candidate["weights"]),
        candidate["curve"],
    )


def verify_completed(directory, bundle, provenance):
    if bundle.provenance != provenance:
        raise DatasetError("abrf_provenance_changed", "Frozen fusion input provenance changed")
    for name, expected in bundle.artifact_hashes.items():
        if file_hash(safe_child(directory, name)) != expected:
            raise DatasetError("abrf_artifact_changed", "Published fusion artifact changed")
    records = json.loads((directory / "candidates.json").read_bytes())
    selected = [r for r in records if r["selected"]]
    if len(selected) != 1 or selected[0]["candidate_id"] != bundle.selected_candidate_id:
        raise DatasetError("abrf_selection_invalid", "Exactly one frozen configuration required")
    frozen = FusionConfig.model_validate_json((directory / "selected_config.json").read_bytes())
    if frozen != bundle.selected:
        raise DatasetError("abrf_selection_invalid", "Frozen selected configuration differs")


def run(config, root: Path, search=None):
    search = search or SearchConfig()
    _, _, provenance = verify_components(config, root)
    run_id = digest(
        {"version": ABRF_VERSION, "sources": provenance, "search": search.model_dump(mode="json")}
    )[7:]
    directory = root / "abrf" / run_id
    bundle_path = directory / "analysis_bundle.json"
    if bundle_path.exists():
        bundle = ABRFBundle.model_validate_json(bundle_path.read_bytes())
        verify_completed(directory, bundle, provenance)
        return bundle, directory
    if directory.exists() and any(directory.iterdir()):
        raise DatasetError(
            "abrf_incomplete_run", "Incomplete fusion output requires explicit review"
        )
    # The active selection is frozen; a changed search requires a separate experiment.
    if (root / "abrf" / "active.json").exists():
        raise DatasetError(
            "abrf_selection_frozen", "Active ABRF selection is frozen; no silent replacement"
        )
    frame, phase5, provenance = load_inputs(config, root)
    train = frame.loc[frame.partition.eq("train")].reset_index(drop=True)
    validation = frame.loc[frame.partition.eq("validation")].reset_index(drop=True)
    y = validation.ground_truth.to_numpy(int)
    directory.mkdir(parents=True)
    records, hashes, threshold_count = [], {}, 0
    candidate_list = list(candidates(search))
    for candidate in candidate_list:
        scores, *_ = score_frame(validation, candidate)
        curve, choices = threshold_curve(y, scores, validation, search)
        threshold_count += len(curve["threshold"])
        filename = "thresholds/" + candidate["candidate_id"] + ".csv.gz"
        content = pd.DataFrame(curve).to_csv(index=False, float_format="%.17g").encode()
        write_artifact(directory / filename, gzip.compress(content, mtime=0), overwrite=False)
        hashes[filename] = file_hash(directory / filename)
        for policy, point in choices.items():
            if point is None:
                records.append(
                    {
                        **candidate,
                        "policy": policy,
                        "status": "UNAVAILABLE_WITH_REASON",
                        "reason": "No threshold meets budgets, minimum recall and scenario floors",
                        "selected": False,
                        "rank": None,
                    }
                )
                continue
            metric = evaluate(y, scores, point["threshold"], validation)
            metric["fnr"] = 1 - metric["recall"]
            metric["score_semantics"] = "RiskByte fused risk score; NOT fraud probability"
            records.append(
                {
                    **candidate,
                    "policy": policy,
                    "status": "AVAILABLE",
                    "threshold": point["threshold"],
                    "metrics": metric,
                    "selected": False,
                    "rank": None,
                }
            )
    policies = {}
    for name, policy in search.policies.items():
        ranked = sorted(
            [r for r in records if r["policy"] == name and r["status"] == "AVAILABLE"],
            key=lambda r: rank_candidate(r, policy),
            reverse=True,
        )
        for position, record in enumerate(ranked, 1):
            record["rank"] = position
        policies[name] = ranked[0] if ranked else {"status": "UNAVAILABLE_WITH_REASON"}
    selected = policies[search.default_policy]
    if selected.get("status") != "AVAILABLE":
        raise DatasetError("abrf_no_feasible_policy", "No candidate satisfies the default policy")
    selected["selected"] = True
    weights = Weights.model_validate(selected["weights"])
    train_scores, *_ = score_frame(train, selected)
    boundaries = [float(np.quantile(train_scores, q)) for q in search.risk_band_quantiles]
    fusion_config = FusionConfig(
        weights=weights,
        confidence_curve=selected["curve"],
        policy=search.default_policy,
        decision_threshold=selected["threshold"],
        risk_boundaries=boundaries,
        reason_evidence_cutoff=search.reason_evidence_cutoff,
        limited_confidence_cutoff=search.limited_confidence_cutoff,
    )
    # Config reload is validated before publication. This serializes no executable model.
    assert FusionConfig.model_validate_json(fusion_config.model_dump_json()) == fusion_config
    scores, effective, contributions = score_frame(validation, selected)
    flags = scores >= fusion_config.decision_threshold
    pflags = validation.P.ge(P_THRESHOLD).to_numpy(bool)
    aflags = validation.A.ge(A_THRESHOLD).to_numpy(bool)
    bflags = validation.B.ge(B_THRESHOLD).to_numpy(bool)
    baseline = evaluate(y, validation.P.to_numpy(float), P_THRESHOLD, validation)
    or_metrics = evaluate(y, (pflags | aflags).astype(float), 0.5, validation)
    # Binary OR has no meaningful continuous risk-ranking AUC.
    or_metrics["pr_auc"] = None
    or_metrics["roc_auc"] = None
    or_metrics["score_semantics"] = "frozen binary OR comparator; not a continuous risk score"
    ablation = []
    default_records = [
        r for r in records if r["status"] == "AVAILABLE" and r["policy"] == search.default_policy
    ]
    for variant in ["Supervised", "Supervised + Anomaly", "Supervised + Behaviour", "Full ABRF"]:
        pool = [r for r in default_records if r["variant"] == variant]
        winner = (
            max(pool, key=lambda r: rank_candidate(r, search.policies[search.default_policy]))
            if pool
            else None
        )
        ablation.append(
            {
                "variant": variant,
                "candidate": winner,
                "metrics": winner["metrics"] if winner else None,
            }
        )
    ablation.extend(
        [
            {"variant": "Naive OR baseline", "metrics": or_metrics},
            {"variant": "Frozen supervised comparator", "metrics": baseline},
        ]
    )
    constraint_study = {}
    full = [r for r in default_records if r["variant"] == "Full ABRF"]
    for constraint in ["unconstrained", "supervised_dominant", "supervised_floor_0.4"]:
        pool = [r for r in full if eligible_constraint(r, constraint)]
        constraint_study[constraint] = (
            max(pool, key=lambda r: rank_candidate(r, search.policies[search.default_policy]))
            if pool
            else {"status": "UNAVAILABLE_WITH_REASON"}
        )
    confidence_study = {}
    for curve_name in search.confidence_curves:
        pool = [r for r in full if r["curve"] == curve_name]
        confidence_study[curve_name] = (
            max(pool, key=lambda r: rank_candidate(r, search.policies[search.default_policy]))
            if pool
            else {"status": "UNAVAILABLE_WITH_REASON"}
        )
    scenarios = {}
    for name, subgroup in selected["metrics"]["scenario_recall"].items():
        mask = validation.fraud_scenario.eq(name).to_numpy(bool) & (y == 1)
        scenarios[name] = {
            "count": subgroup["count"],
            "supervised": rates(mask, pflags),
            "anomaly": rates(mask, aflags),
            "abrf": rates(mask, flags),
            "P": distribution(validation.P.to_numpy(float)[mask]),
            "A": distribution(validation.A.to_numpy(float)[mask]),
            "B": distribution(validation.B.to_numpy(float)[mask]),
            "confidence": distribution(validation.behaviour_confidence.to_numpy(float)[mask]),
        }
    # Counterfactual calculations describe score allocation, not causal explanations.
    hard_analysis = {}
    hard = validation.hard_negative.to_numpy(bool)
    for name in selected["metrics"]["hard_negative_types"]:
        mask = hard & validation.hard_negative_type.eq(name).to_numpy(bool)
        flagged = mask & flags
        hard_analysis[name] = {
            "supervised": rates(mask, pflags),
            "anomaly": rates(mask, aflags),
            "abrf": rates(mask, flags),
            "crossing_allocations": {},
        }
        for index, component in enumerate(["supervised", "anomaly", "behaviour"]):
            # Removal keeps the other effective weights fixed; no renormalization is implied.
            necessary = flagged & (
                (scores - contributions[:, index]) < fusion_config.decision_threshold
            )
            hard_analysis[name]["crossing_allocations"][component] = int(necessary.sum())
    support = (~pflags) & (~aflags) & bflags
    support_analysis = {
        "definition": "frozen P negative, frozen A negative, frozen B high",
        "fraud": rates(y == 1, support),
        "legitimate": rates(y == 0, support),
        "hard_negative": rates(hard, support),
        "fraud_scenarios": {
            name: int(
                (support & validation.fraud_scenario.eq(name).to_numpy(bool) & (y == 1)).sum()
            )
            for name in scenarios
        },
    }
    improvement = {
        "additional_fraud_recovered": int(((y == 1) & ~pflags & flags).sum()),
        "previously_detected_fraud_lost": int(((y == 1) & pflags & ~flags).sum()),
        "net_fraud_detection_change": int(((y == 1) & flags).sum() - ((y == 1) & pflags).sum()),
        "additional_false_positives": int(((y == 0) & ~pflags & flags).sum()),
        "previous_false_positives_removed": int(((y == 0) & pflags & ~flags).sum()),
        "net_false_positive_change": int(((y == 0) & flags).sum() - ((y == 0) & pflags).sum()),
        "net_hard_negative_false_positive_change": int(
            (hard & flags).sum() - (hard & pflags).sum()
        ),
        "scenarios_improved": [
            name
            for name, row in scenarios.items()
            if row["abrf"]["flagged"] > row["supervised"]["flagged"]
        ],
        "scenarios_degraded": [
            name
            for name, row in scenarios.items()
            if row["abrf"]["flagged"] < row["supervised"]["flagged"]
        ],
    }
    rationale = (
        "Highest validation policy objective across all component ablations; ties use "
        "average precision, precision, recall, hard-negative/legitimate FPR and simplicity. "
        "Same budgets and frozen strong-scenario detection floors apply to every variant. "
        "No requirement forces nonzero anomaly/behaviour weight when evidence does not support it."
    )
    all_scores, all_weights, all_contributions = score_frame(frame, selected)
    traced = frame.copy()
    traced["final_risk_score"] = all_scores
    traced["investigation_flag"] = all_scores >= fusion_config.decision_threshold
    traced["supervised_prediction"] = frame.P.ge(P_THRESHOLD)
    traced["anomaly_prediction"] = frame.A.ge(A_THRESHOLD)
    traced["behaviour_available"] = np.isfinite(frame.B.to_numpy(float))
    traced["decision_threshold"] = fusion_config.decision_threshold
    traced["risk_level"] = np.array(["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"])[
        np.searchsorted(fusion_config.risk_boundaries, all_scores, side="right")
    ]
    for name, value in fusion_config.weights.model_dump().items():
        traced["base_weight_" + name] = value
    reason_masks = {
        "SUPERVISED_THRESHOLD_EXCEEDED": traced.supervised_prediction.to_numpy(bool),
        "HIGH_ANOMALY_SCORE": traced.anomaly_prediction.to_numpy(bool),
        "HIGH_BEHAVIOURAL_DEVIATION": frame.B.ge(B_THRESHOLD).to_numpy(bool),
        "LIMITED_BEHAVIOURAL_HISTORY": (~traced.behaviour_available.to_numpy(bool))
        | frame.behaviour_confidence.lt(fusion_config.limited_confidence_cutoff).to_numpy(bool),
    }
    traced["reason_codes"] = [
        ";".join(code for code, mask in reason_masks.items() if mask[row])
        for row in range(len(frame))
    ]
    for i, name in enumerate(["supervised", "anomaly", "behaviour"]):
        traced["effective_weight_" + name] = all_weights[:, i]
        traced["contribution_" + name] = all_contributions[:, i]
    content = gzip.compress(traced.to_csv(index=False).encode(), mtime=0)
    write_artifact(directory / "development_fusion_scores.csv.gz", content, overwrite=False)
    hashes["development_fusion_scores.csv.gz"] = file_hash(
        directory / "development_fusion_scores.csv.gz"
    )
    example = validation.iloc[0]
    record = FusionInput(
        transaction_id=str(example.TransactionID),
        supervised_probability=float(example.P),
        supervised_prediction=bool(example.P >= P_THRESHOLD),
        anomaly_score=float(example.A),
        anomaly_prediction=bool(example.A >= A_THRESHOLD),
        behaviour_score=float(example.B) if np.isfinite(example.B) else None,
        behaviour_confidence=float(example.behaviour_confidence),
        behaviour_available=bool(np.isfinite(example.B)),
    )
    save_json(directory / "trace_example.json", fuse(record, fusion_config).model_dump(mode="json"))
    save_json(directory / "candidates.json", records)
    save_json(directory / "selected_config.json", fusion_config.model_dump(mode="json"))
    save_json(
        directory / "selection_metadata.json",
        {
            "provenance": provenance,
            "validation_metrics": selected["metrics"],
            "rationale": rationale,
            "candidate_id": selected["candidate_id"],
            "risk_band_source": "TRAIN score quantiles; labels unused",
        },
    )
    for name in [
        "trace_example.json",
        "candidates.json",
        "selected_config.json",
        "selection_metadata.json",
    ]:
        hashes[name] = file_hash(directory / name)
    # Recheck input snapshots before committing a completed selection.
    _, _, after = verify_components(config, root)
    if after != provenance:
        raise DatasetError("abrf_sources_changed", "Frozen inputs changed during fusion research")
    bundle = ABRFBundle(
        generated_at=datetime.now(UTC).isoformat(),
        provenance=provenance,
        search_configuration=search,
        selected=fusion_config,
        selected_candidate_id=selected["candidate_id"],
        selection_rationale=rationale,
        validation=selected["metrics"],
        ablation=ablation,
        policies=policies,
        constraint_study=constraint_study,
        confidence_study=confidence_study,
        scenario_analysis=scenarios,
        hard_negative_analysis=hard_analysis,
        support_analysis=support_analysis,
        improvement=improvement,
        search_summary={
            "candidate_configurations": len(candidate_list),
            "unique_weight_combinations": len(
                {tuple(c["weights"].values()) for c in candidate_list}
            ),
            "thresholds_evaluated": threshold_count,
            "policy_configurations": len(records),
            "threshold_reporting": "all exact threshold metrics in compressed per-candidate CSVs",
        },
        leakage_audit={
            "original_temporal_test_accessed": False,
            "test_rows_scored": 0,
            "future_holdout_generated": False,
            "component_models_retrained": False,
            "component_artifacts_unchanged": True,
            "fusion_inputs_label_free": True,
            "selection_source": "VALIDATION only",
            "risk_bands_source": "TRAIN only",
            "source_io": "opaque source checksums; only frozen development score CSV decoded",
        },
        artifact_hashes=hashes,
    )
    save_json(bundle_path, bundle.model_dump(mode="json"))
    save_json(
        root / "abrf" / "active.json", {"run_id": run_id, "bundle_hash": file_hash(bundle_path)}
    )
    return bundle, directory
