# RiskByte synthetic dataset

RiskByte uses an **IEEE-CIS-inspired synthetic financial transaction dataset** for this
academic development environment because the real competition files are not locally available.
**The dataset structure and problem framing are inspired by fraud-detection datasets such
as IEEE-CIS, but all generated records are synthetic.** Nothing reconstructs competition
records, claims IEEE-CIS distribution matching, or downloads external data.

## Reproducible generator

`riskbyte_ml.datasets.synthetic_riskbyte` owns the versioned simulation. Presets are SMALL
10,000, MEDIUM 100,000 and LARGE 500,000 transactions. JSON can override row count, entity/
merchant counts, seed, fraud proportion, hard-negative rate, duration, identity coverage,
context missingness and enabled scenarios. Tests use hundreds or 1,200 rows; LARGE is never
generated automatically. Canonical development configuration is MEDIUM, seed 42, 90 elapsed
days, 2% fraud and 70% identity coverage. These are simulation choices, not banking statistics.
The 2% default is intentionally higher than some real banking datasets to make experiments
manageable for teaching; prevalence remains configurable.

Each entity has a persistent lognormal amount baseline, variable amount dispersion,
preferred merchants, usual region, common transaction type and heterogeneous activity rate.
Entities have two initial allocated transactions plus a weighted multinomial activity
allocation. Baselines vary substantially, so a large amount can be ordinary for one entity
and unusual for another. Merchants have stable synthetic categories/products. Entities have
two common synthetic device IDs. All domains use `.example`; no actual personal records,
payment credentials, sensitive demographic characteristics or real account identities exist.

Attack episodes are placed into entity sequences until the configured fraud-row budget
is exhausted. A row is labelled because its attack mechanism is enacted, not by a final
Bernoulli fraud draw. Episode lengths make scenario row shares unequal. Scenario scheduling
cycles through configured archetypes; small budgets may not include every archetype. Small
or truncated episodes need not have enough observations for strong collective evidence.

| Fraud scenario                | Mechanism                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| AMOUNT_SPIKE                  | 3.5–13 times entity baseline, overlapping legitimate large purchases                 |
| VELOCITY_BURST                | Consecutive events with 2–34 second gaps                                             |
| NEW_DEVICE_HIGH_AMOUNT        | New device ID and increased amount; evidence depends on identity coverage            |
| REGION_CHANGE                 | Billing/identity region differs from usual region; no GPS claim                      |
| NEW_MERCHANT_AMOUNT_DEVIATION | Merchant absent from entity's prior generated history plus increased amount          |
| REPEATED_FAILED_ATTEMPTS      | 3–12 simulated attempts before the transaction outcome                               |
| ACCOUNT_TAKEOVER              | Burst with new device, region change, higher amount and failed attempts              |
| LOW_AND_SLOW                  | Several small amounts, 0.12–0.45 baseline, with 10–59 minute gaps                    |
| INTERNATIONAL_ANOMALY         | International flag against predominantly domestic histories; amount perturbation     |
| MIXED_MODE                    | Burst plus a random combination of 2–5 amount/device/region/merchant/failure signals |

The merchant pool must contain unseen candidates for novelty scenarios; exhaustion fails
explicitly rather than mislabelling seen merchants as new. Amounts are nonnegative simulated
INR-denominated units, rounded to cents; they are not financial accounting evidence. LOW_AND_SLOW
is a scenario benchmark, not a claim that current features perfectly distinguish every episode.

Legitimate hard negatives include laptop purchases, medical expenses, travel/international
purchases, new legitimate devices, new merchants, and multi-event shopping-festival bursts.
Signals overlap attack mechanisms. Hard negatives retain `isFraud=0`; new-device hard negatives
can also have multiple failed attempts. The current simulation still has learnability biases.

## Schema and availability

Transaction CSV fields:

| Field(s)                          | Synthetic meaning / role                                                    |
| --------------------------------- | --------------------------------------------------------------------------- |
| TransactionID                     | Unique exact join/audit ID, assigned before chronological sorting; excluded |
| TransactionDT                     | Elapsed seconds since synthetic start; chronology and prior-history source  |
| TransactionAmt                    | Nonnegative simulated transaction amount                                    |
| isFraud                           | Known simulation outcome: 1 attack, 0 legitimate; target only               |
| behavioural_entity_id             | Persistent simulated account profile; history key, excluded from prediction |
| merchant_id, merchant_category    | Synthetic merchant and purchase context                                     |
| transaction_type, product_code    | Purchase/transfer/bill payment and product context                          |
| card_type, card_network           | Synthetic card attributes                                                   |
| billing_region, shipping_region   | Discrete synthetic regions, not coordinates                                 |
| P_emaildomain, R_emaildomain      | Non-personal `.example` context domains                                     |
| failed_attempts_before            | Simulated pre-outcome authentication/payment attempt signal                 |
| account_age_days                  | Synthetic account age at elapsed event time                                 |
| is_international, is_card_present | Simulated channel/context flags                                             |
| Optional V1–V20                   | Non-outcome noisy amount/context mixtures, ANONYMIZED; default disabled     |

Identity CSV fields: TransactionID, DeviceType, DeviceInfo, device_id, os_family,
browser_family, identity_region and identity_age_days. Identity age describes the synthetic
registered identity context, not physical device age. Device IDs denote known simulated
devices; the metadata description is coarse. A target-independent sample of transaction
IDs receives identity rows. Missingness is independently injected into approved context
fields, never conditioned on the label. Identity absence and injected nulls are measured
separately. Missingness does not imply low risk.

Evaluation-only `evaluation_metadata.csv` contains TransactionID, fraud_scenario,
hard_negative and hard_negative_type. `entity_profiles.csv` records simulation baselines.
Neither sidecar is joined into predictive inputs. All outcome metadata is TARGET_METADATA /
EXCLUDED. The sidecar has an independent content hash and exact row-order verification.
Normal dataset API responses expose aggregate provenance, not these row annotations.

## Generation and artifacts

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.datasets.synthetic_riskbyte --preset medium --seed 42 --run-pipeline
# Custom generation without preprocessing:
.\.venv\Scripts\python.exe -m riskbyte_ml.datasets.synthetic_riskbyte --config ml/config/synthetic_generator.example.json --output-dir ml/data/synthetic_riskbyte
# Reproduce analysis/preprocessing without changing the CSVs:
.\.venv\Scripts\python.exe -m riskbyte_ml.preprocessing --config ml/config/synthetic_riskbyte.example.json
```

Generation refuses existing dataset outputs unless `--overwrite` is supplied. Use one writer
per output directory; individual files publish atomically, the dataset bundle does not.
Generation timestamps are provenance and excluded from the deterministic fingerprint.
Fingerprint hashes both exact CSV byte hashes, normalized generator configuration and version.
Reproducibility assumes the recorded dependency versions; CSV format/newlines are fixed.
Changing a seed/configuration/version changes experiment identity. Data/artifacts remain ignored.

Default files live in `ml/data/synthetic_riskbyte/`. The generation manifest is copied to
`ml/artifacts/synthetic_dataset_manifest.json`. Profiling/quality/EDA/schema/join/feasibility
reports live in `ml/artifacts/synthetic_riskbyte/`; splits in `ml/artifacts/splits/`, feature
metadata in `ml/artifacts/features/feature_manifest.json`, trusted fitted preprocessing in
`ml/artifacts/preprocessing/`, and the run summary in `ml/artifacts/phase2_summary.json`.
No classifier or anomaly model is fitted, and no raw dataset is embedded in preprocessing.
No application startup generates data; absent source files stay structured dataset errors.

## Limitations and ethics

**The data is synthetic. It cannot prove production banking performance.** Fraud-generating
rules influence learnability, and performance may be optimistic when scenarios are too
deterministic. Later ABRF results must be interpreted as performance on this academic
simulation. Synthetic behavioural identities are not real bank customers. Real-world concept
drift is only approximated by differing activity lifetimes and scenario timing, not a calibrated
banking process. Label delay, investigations, reversals, balances, multi-currency accounting,
verified people and GPS travel are not simulated. Device evidence is intentionally incomplete.

Attack labels are known simulation truth, not conclusions about actual people. Explain future
results as synthetic benchmark findings; never present anonymized V-feature contributions as
human-readable business causes. Do not use this simulation alone to approve a production fraud
system, customer restriction, fairness claim or calibrated banking probability.
