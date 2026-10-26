"""Scenario-driven simulation with persistent heterogeneous synthetic entities."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte import DATASET_NAME, GENERATOR_VERSION, NOTICE
from riskbyte_ml.datasets.synthetic_riskbyte.config import (
    HARD_NEGATIVES,
    GeneratorConfig,
    digest,
)


@dataclass
class GeneratedDataset:
    transaction: pd.DataFrame
    identity: pd.DataFrame
    evaluation: pd.DataFrame
    profiles: pd.DataFrame
    config: GeneratorConfig


def generate(config: GeneratorConfig, *, checkpoint=None) -> GeneratedDataset:
    rng = np.random.default_rng(config.seed)
    n, entities, merchants = config.rows, config.entity_count, config.merchant_count
    # Frequency, amount and dispersion vary independently between synthetic entities.
    weights = rng.lognormal(0, 0.65, entities)
    if checkpoint is not None:
        weights[: checkpoint["continuing_entities"]] = checkpoint["weights"][
            : checkpoint["continuing_entities"]
        ]
    counts = np.full(entities, 2) + rng.multinomial(n - 2 * entities, weights / weights.sum())
    entity = np.repeat(np.arange(entities), counts)
    offsets = np.r_[0, counts.cumsum()]
    baseline = rng.lognormal(np.log(2000), 1.15, entities).clip(100, 40_000)
    spread = rng.uniform(0.25, 0.7, entities)
    home = rng.integers(0, 8, entities)
    preferred = rng.integers(0, int(merchants * 0.7), (entities, 5))
    account_age = rng.integers(5, 3000, entities)
    common_type = rng.integers(0, 3, entities)
    # Optional continuation context replaces persistent profiles, not attack logic or RNG draws.
    # The original checkpoint=None path remains byte-for-byte reproducible.
    if checkpoint is not None:
        for name, values in [
            ("baseline", baseline),
            ("spread", spread),
            ("home", home),
            ("preferred", preferred),
            ("account_age", account_age),
            ("common_type", common_type),
        ]:
            count = checkpoint["continuing_entities"]
            values[:count] = checkpoint[name][:count]
    # Select attack episodes, then enact their mechanisms. Labels are not Bernoulli draws.
    scenario = np.full(n, "", dtype=object)
    budget = round(n * config.fraud_prevalence)
    occupied = np.zeros(n, dtype=bool)
    episode = 0
    while budget:
        e = int(rng.integers(entities))
        start = int(rng.integers(offsets[e] + 1, offsets[e + 1]))
        name = config.scenarios[episode % len(config.scenarios)]
        length = min(
            budget,
            int(rng.integers(3, 8))
            if name in {"VELOCITY_BURST", "ACCOUNT_TAKEOVER", "LOW_AND_SLOW", "MIXED_MODE"}
            else 1,
            int(offsets[e + 1] - start),
        )
        slots = np.arange(start, start + length)
        if occupied[slots].any():
            continue
        scenario[slots], occupied[slots] = name, True
        budget -= length
        episode += 1
    hard = np.full(n, "", dtype=object)
    hard_budget, hard_episode = round(n * config.hard_negative_rate), 0
    while hard_budget:
        e = int(rng.integers(entities))
        start = int(rng.integers(offsets[e], offsets[e + 1]))
        name = HARD_NEGATIVES[hard_episode % len(HARD_NEGATIVES)]
        length = min(
            hard_budget,
            int(offsets[e + 1] - start),
            int(rng.integers(3, 8)) if name == "SHOPPING_FESTIVAL" else 1,
        )
        slots = np.arange(start, start + length)
        if occupied[slots].any() or (hard[slots] != "").any():
            continue
        hard[slots] = name
        hard_budget -= length
        hard_episode += 1
    # Heterogeneous normal event sequences; bursts alter inter-arrival gaps before outcomes.
    elapsed = np.zeros(n, dtype=np.int64)
    for e in range(entities):
        a, b = int(offsets[e]), int(offsets[e + 1])
        gaps = rng.exponential(1, b - a)
        gaps *= config.duration_days * 86400 * rng.uniform(0.75, 0.98) / gaps.sum()
        gaps = np.maximum(1, gaps.astype(np.int64))
        local = scenario[a:b]
        fast = np.isin(local, ["VELOCITY_BURST", "ACCOUNT_TAKEOVER", "MIXED_MODE"])
        slow = local == "LOW_AND_SLOW"
        festival = hard[a:b] == "SHOPPING_FESTIVAL"
        gaps[fast | festival] = rng.integers(2, 35, (fast | festival).sum())
        gaps[slow] = rng.integers(600, 3600, slow.sum())
        elapsed[a:b] = np.cumsum(gaps) + int(rng.integers(0, 86400))
    amount = rng.lognormal(np.log(baseline[entity]), spread[entity]).clip(1, 1_000_000)
    merchant = preferred[entity, rng.integers(0, 5, n)]
    unusual = rng.random(n) < 0.08
    merchant[unusual] = rng.integers(0, merchants, unusual.sum())
    billing = home[entity].copy()
    shipping = billing.copy()
    device = np.array(
        [f"dev-{e:05d}-{int(v)}" for e, v in zip(entity, rng.integers(0, 2, n), strict=True)],
        dtype=object,
    )
    international = (rng.random(n) < 0.015).astype(int)
    failures = rng.poisson(0.2, n)
    tx_types = np.array(["purchase", "transfer", "bill_payment"], dtype=object)[common_type[entity]]
    tx_types[rng.random(n) < 0.1] = "purchase"
    merchant_history = [set() for _ in range(entities)]
    if checkpoint is not None:
        for e, history in enumerate(checkpoint["merchant_history"]):
            merchant_history[e] = set(history)
    for i in range(n):
        s, h, e = scenario[i], hard[i], entity[i]
        mixed = (
            set(
                rng.choice(
                    ["amount", "device", "region", "merchant", "failures"],
                    size=int(rng.integers(2, 6)),
                    replace=False,
                )
            )
            if s == "MIXED_MODE"
            else set()
        )
        # Overlapping signals and stochastic strengths prevent a single threshold shortcut.
        spike = s in {
            "AMOUNT_SPIKE",
            "NEW_DEVICE_HIGH_AMOUNT",
            "NEW_MERCHANT_AMOUNT_DEVIATION",
            "ACCOUNT_TAKEOVER",
        }
        if spike or "amount" in mixed or h in {"LAPTOP_PURCHASE", "MEDICAL_EXPENSE"}:
            amount[i] = baseline[e] * rng.uniform(3.5, 13)
        if s == "LOW_AND_SLOW":
            amount[i] = baseline[e] * rng.uniform(0.12, 0.45)
        if (
            s in {"NEW_DEVICE_HIGH_AMOUNT", "ACCOUNT_TAKEOVER"}
            or "device" in mixed
            or h == "NEW_LEGITIMATE_DEVICE"
        ):
            device[i] = f"dev-new-{i:07d}"
        if (
            s in {"REGION_CHANGE", "ACCOUNT_TAKEOVER"}
            or "region" in mixed
            or h == "TRAVEL_PURCHASE"
        ):
            billing[i] = (home[e] + int(rng.integers(1, 8))) % 8
            shipping[i] = billing[i]
        if (
            s == "NEW_MERCHANT_AMOUNT_DEVIATION"
            or "merchant" in mixed
            or h
            in {
                "NEW_MERCHANT_PURCHASE",
                "LAPTOP_PURCHASE",
                "MEDICAL_EXPENSE",
            }
        ):
            unseen = [m for m in range(merchants) if m not in merchant_history[e]]
            if h in {"LAPTOP_PURCHASE", "MEDICAL_EXPENSE"}:
                category_index = 1 if h == "LAPTOP_PURCHASE" else 2
                matching = [m for m in unseen if m % 6 == category_index]
                # Large legitimate purchases may revisit a known specialist merchant.
                unseen = matching or [m for m in range(merchants) if m % 6 == category_index]
            if not unseen:
                raise ValueError(
                    "Merchant pool exhausted; increase merchant_count for novelty scenarios"
                )
            merchant[i] = int(rng.choice(unseen))
        merchant_history[e].add(int(merchant[i]))
        if s in {"REPEATED_FAILED_ATTEMPTS", "ACCOUNT_TAKEOVER"} or "failures" in mixed:
            failures[i] = int(rng.integers(3, 13))
        elif h == "NEW_LEGITIMATE_DEVICE":
            failures[i] = int(rng.integers(0, 7))
        if s == "INTERNATIONAL_ANOMALY" or h == "TRAVEL_PURCHASE":
            international[i] = 1
        if s == "INTERNATIONAL_ANOMALY":
            amount[i] *= rng.uniform(1.3, 3)
    # Merchant/device descriptive context is deterministic from the synthetic ID, not target.
    categories = np.array(
        ["groceries", "electronics", "healthcare", "travel", "fashion", "utilities"]
    )
    transaction = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": elapsed,
            "TransactionAmt": amount.round(2),
            "isFraud": occupied.astype(int),
            "behavioural_entity_id": [f"entity-{e:05d}" for e in entity],
            "merchant_id": [f"merchant-{m:04d}" for m in merchant],
            "merchant_category": categories[merchant % len(categories)],
            "transaction_type": tx_types,
            "product_code": categories[merchant % len(categories)],
            "card_type": np.where(entity % 3 == 0, "credit", "debit"),
            "card_network": np.array(["visa", "mastercard", "rupay"])[entity % 3],
            "billing_region": [f"region-{r}" for r in billing],
            "shipping_region": [f"region-{r}" for r in shipping],
            "P_emaildomain": np.array(["mail.example", "inbox.example", "post.example"])[
                entity % 3
            ],
            "R_emaildomain": np.array(["shop.example", "pay.example", "store.example"])[
                merchant % 3
            ],
            "failed_attempts_before": failures,
            "account_age_days": account_age[entity] + elapsed // 86400,
            "is_international": international,
            "is_card_present": (rng.random(n) < 0.25).astype(int),
        }
    )
    for v in range(config.anonymized_feature_count):
        # Non-outcome numeric context mixtures, deliberately anonymized, never label copies.
        transaction[f"V{v + 1}"] = (
            np.log1p(amount) * rng.normal(0, 0.2) + rng.normal(0, 1, n)
        ).round(6)
    # Independent fixed coverage sample; missingness does not condition on the target.
    eligible = np.sort(rng.choice(n, round(n * config.identity_coverage), replace=False))
    identity = pd.DataFrame(
        {
            "TransactionID": transaction.TransactionID.iloc[eligible].to_numpy(),
            "DeviceType": np.where(entity[eligible] % 2 == 0, "mobile", "desktop"),
            "DeviceInfo": np.where(
                entity[eligible] % 2 == 0, "synthetic-phone", "synthetic-computer"
            ),
            "device_id": device[eligible],
            "os_family": np.array(["android", "windows", "ios"])[entity[eligible] % 3],
            "browser_family": np.array(["chromium", "firefox", "webkit"])[entity[eligible] % 3],
            "identity_region": [f"region-{r}" for r in billing[eligible]],
            "identity_age_days": np.maximum(
                0, account_age[entity[eligible]] - rng.integers(0, 5, len(eligible))
            ),
        }
    )
    for column, rate in config.missingness.items():
        frame = transaction if column in transaction else identity
        frame.loc[rng.random(len(frame)) < rate, column] = None
    evaluation = pd.DataFrame(
        {
            "TransactionID": transaction.TransactionID,
            "fraud_scenario": scenario,
            "hard_negative": hard != "",
            "hard_negative_type": hard,
        }
    )
    # IDs are assigned before sorting; ID magnitude is not a clock or predictive feature.
    transaction = transaction.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)
    evaluation = transaction[["TransactionID"]].merge(
        evaluation, on="TransactionID", validate="one_to_one"
    )
    profiles = pd.DataFrame(
        {
            "behavioural_entity_id": [f"entity-{e:05d}" for e in range(entities)],
            "typical_amount": baseline,
            "log_amount_std": spread,
            "transaction_count": counts,
            "usual_region": home,
            "common_transaction_type": common_type,
        }
    )
    return GeneratedDataset(transaction, identity, evaluation, profiles, config)


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode("utf-8")


def save_generated(dataset: GeneratedDataset, directory: Path, *, overwrite=False):
    names = (
        "train_transaction.csv",
        "train_identity.csv",
        "evaluation_metadata.csv",
        "entity_profiles.csv",
        "generation_metadata.json",
    )
    if not overwrite and any((directory / name).exists() for name in names):
        raise DatasetError(
            "synthetic_output_exists", "Dataset exists; explicit --overwrite required"
        )
    tx, identity = csv_bytes(dataset.transaction), csv_bytes(dataset.identity)
    source_hashes = {
        "transaction": "sha256:" + hashlib.sha256(tx).hexdigest(),
        "identity": "sha256:" + hashlib.sha256(identity).hexdigest(),
    }
    config = dataset.config.model_dump(mode="json")
    fingerprint = digest(
        {"sources": source_hashes, "config": config, "generator_version": GENERATOR_VERSION}
    )
    fraud = int(dataset.transaction.isFraud.sum())
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_type": "SYNTHETIC",
        "description": "IEEE-CIS-inspired synthetic financial transaction dataset",
        "notice": NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": dataset.config.seed,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "transaction_count": len(dataset.transaction),
        "identity_count": len(dataset.identity),
        "entity_count": int(dataset.transaction.behavioural_entity_id.nunique()),
        "merchant_count": int(dataset.transaction.merchant_id.nunique()),
        "device_count": int(dataset.identity.device_id.nunique()),
        "fraud_count": fraud,
        "fraud_prevalence": fraud / len(dataset.transaction),
        "identity_coverage": len(dataset.identity) / len(dataset.transaction),
        "scenario_distribution": dataset.evaluation.loc[
            dataset.evaluation.fraud_scenario != "", "fraud_scenario"
        ]
        .value_counts()
        .to_dict(),
        "hard_negative_count": int(dataset.evaluation.hard_negative.sum()),
        "hard_negative_distribution": dataset.evaluation.loc[
            dataset.evaluation.hard_negative, "hard_negative_type"
        ]
        .value_counts()
        .to_dict(),
        "missingness_summary": {
            "transaction": dataset.transaction.isna().sum().to_dict(),
            "identity": dataset.identity.isna().sum().to_dict(),
        },
        "source_fingerprints": source_hashes,
        "fingerprint": fingerprint,
        "evaluation_fingerprint": "sha256:"
        + hashlib.sha256(csv_bytes(dataset.evaluation)).hexdigest(),
    }
    for name, payload in zip(
        names,
        (
            tx,
            identity,
            csv_bytes(dataset.evaluation),
            csv_bytes(dataset.profiles),
            __import__("json").dumps(manifest, indent=2, allow_nan=False).encode(),
        ),
        strict=True,
    ):
        write_artifact(directory / name, payload)
    return manifest
