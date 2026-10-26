# Dataset recommendation for Phase 2

Recommendation: **IEEE-CIS Fraud Detection**, starting with its labelled transaction CSV,
if the project requires genuine transaction data. It combines a realistic fraud task with
some interpretable transaction fields, while retaining substantial anonymization. This is
an engineering recommendation, not a result from training RiskByte. Nothing was downloaded.

| Criterion                 | IEEE-CIS                                                                   | PaySim                                                            | European credit-card fraud (ULB)                                    |
| ------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| Classification usefulness | Real card-not-present labels; rich mixed features                          | Mobile-money fraud mechanisms; useful simulated task              | Established real binary classification baseline                     |
| Imbalance                 | Source training fraud rate about 3.5%, as reported by FDB                  | Rare fraud; measure exact prevalence after ingestion              | 492 frauds among 284,807 events in source description               |
| Interpretability          | Amount/product/card/email/device context; many anonymized fields remain    | Amount, transaction type, origin/destination entities             | V1–V28 are anonymized PCA features; amount/time remain readable     |
| Behavioural history       | Possible proxy grouping; no guaranteed customer identifier                 | Origin/destination IDs support exploratory histories              | No usable customer/merchant history identifiers                     |
| Time                      | `TransactionDT` is elapsed from an undisclosed reference                   | `step` is elapsed simulation hours                                | Relative time across two days; no confirmed calendar origin         |
| Final-year suitability    | Strong realism and research scope, more engineering work                   | Good explanatory prototype if clearly labelled synthetic          | Simple reproducible baseline, weaker behavioural explanations       |
| Access/license            | Kaggle account/rules acceptance; subject to competition rules              | Kaggle public dataset, CC BY-SA 4.0                               | Check original dataset terms/attribution; FDB links DbCL            |
| Preprocessing             | High: missingness, categoricals, anonymized fields, optional identity join | Moderate: temporal/entity semantics and leakage review; large CSV | Lower: numeric inputs, but transformed feature meanings unavailable |

IEEE-CIS has transaction and identity files joined by `TransactionID`; identity information
is missing for some events. The competition data listing reports a roughly 683 MB training
transaction CSV, above Phase 1's default 256 MB limit. Review RAM and explicitly adjust
bounds before loading; in-memory string storage can be much larger. Multi-file joins are
not implemented in Phase 1. Start with transaction-only inspection; design and validate an
identity join in an authorized later phase. `TransactionDT` is a timedelta, **not an actual
timestamp**: do not invent calendar dates for weekday/hour charts. Card-based proxy groups
must never be represented as verified customer identity.
[Official dataset description](https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv).

FDB's authors describe Vesta provenance, the 590,540 labelled transaction/identity records,
source imbalance and the anonymized features. Their benchmark creates a composite entity
fingerprint; this does not establish a true customer ID. These details support selecting
IEEE-CIS for a real-data project, with explicit behavioural limitations.
[Amazon Science Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/README.md).

**PaySim is the practical alternative when readable behavioural fields matter most**, but
it is synthetic, generated from a simulator informed by aggregated real financial logs.
It must not be described as real customer transactions. Its hour steps cannot establish
calendar weekdays. The publisher explicitly warns against using balance fields for fraud
detection because detected fraudulent transactions are cancelled; retain and flag those
fields during Phase 1, then review their exclusion before any later model. The dataset
license is CC BY-SA 4.0 (distinct from the simulator software license).
[Creator's dataset documentation](https://www.kaggle.com/ealaxi/paysim1/metadata).

The common European dataset is useful as a baseline, but **PCA-transformed V1–V28 make it
weaker for human-readable behavioural explanations**. Its lack of customer histories also
limits a behavioural engine. FDB documents its numeric PCA features, retained time/amount,
two-day coverage and extreme imbalance. Prefer IEEE-CIS for genuine data with richer
context; prefer PaySim only with an explicit synthetic-data qualification.
[FDB source documentation](https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/README.md).

## Exact next step

Manually review/accept IEEE-CIS access and competition terms, acquire `train_transaction.csv`
and put it under ignored `ml/data/raw/`. Record source/version/license in a local manifest.
Create `ml/config/dataset.local.json` in inspection mode, select the real path and deliberately
set resource limits after a RAM review. Run Phase 1 profiling and review quality/leakage.
Only then select explicit `isFraud` labels (`1` fraud / `0` legitimate, documented by the
host), `TransactionAmt`, unique `TransactionID`, and elapsed `TransactionDT` with verified
units. The host also describes label maturation; preserve that context in the dictionary.
[Vesta host's label description](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/100304).

Do not commence training, history features or Phase 2 automatically. Source figures above
are literature descriptions, not RiskByte-generated metrics.
