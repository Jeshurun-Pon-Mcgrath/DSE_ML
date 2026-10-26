# Phase 2 preprocessing report

These are measured observations from the MEDIUM **IEEE-CIS-inspired synthetic financial
transaction dataset**, not IEEE-CIS records or real banking statistics. Checked 2026-10-12
local time; timestamps in generated manifests use UTC.

## Provenance and generator methodology

Generator: `synthetic-riskbyte-v1`; seed: **42**; preset: MEDIUM; duration: 90 elapsed days.
Generation timestamp: `2026-10-11T19:17:54.517513+00:00`.
Dataset fingerprint: `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`.
Feature pipeline version: `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`.

Full MEDIUM regeneration in memory reproduced both exact CSV byte hashes. The timestamp
is deliberately excluded from that fingerprint. Entity-specific amount/frequency/context
profiles drive normal activity. Attack episodes enact ten mechanisms; hard negatives overlap
suspicious signals. Labels follow enacted scenario logic, not a final independent random draw.
Identity selection and injected context missingness do not condition on the target. Scenario
and hard-negative annotations remain evaluation-only in an independently hashed sidecar.
Laptop and medical hard negatives use matching electronics/healthcare merchants.
See [methodology and schema](SYNTHETIC_DATASET.md).

## Actual dataset and join measurements

| Measurement                 | Actual value |
| --------------------------- | ------------ |
| transaction_rows            | 100,000      |
| identity_rows               | 70,000       |
| joined_rows                 | 100,000      |
| entity_count                | 2,000        |
| merchant_count              | 300          |
| device_count                | 4,892        |
| fraud_count                 | 2,000        |
| fraud_prevalence            | 2.000%       |
| identity_coverage           | 70.000%      |
| hard_negative_count         | 5,000        |
| unmatched_transactions      | 30,000       |
| duplicate/missing join keys | 0 / 0        |
| majority/minority ratio     | 49:1         |

The one-to-one LEFT join preserved every transaction, ID order and target value. Device
count means observed available identity devices, not every latent device in the simulation.
Entities are explicit fictional account profiles; no proxy-person identity inference is needed.

## Scenario and legitimate hard-negative distribution

| Fraud scenario                | Rows |
| ----------------------------- | ---- |
| ACCOUNT_TAKEOVER              | 401  |
| VELOCITY_BURST                | 382  |
| MIXED_MODE                    | 371  |
| LOW_AND_SLOW                  | 367  |
| REPEATED_FAILED_ATTEMPTS      | 80   |
| REGION_CHANGE                 | 80   |
| AMOUNT_SPIKE                  | 80   |
| NEW_DEVICE_HIGH_AMOUNT        | 80   |
| NEW_MERCHANT_AMOUNT_DEVIATION | 80   |
| INTERNATIONAL_ANOMALY         | 79   |

| Legitimate hard negative | Rows |
| ------------------------ | ---- |
| SHOPPING_FESTIVAL        | 2405 |
| MEDICAL_EXPENSE          | 519  |
| TRAVEL_PURCHASE          | 519  |
| NEW_MERCHANT_PURCHASE    | 519  |
| NEW_LEGITIMATE_DEVICE    | 519  |
| LAPTOP_PURCHASE          | 519  |

Shares differ because collective/burst scenarios contain multiple rows per episode. Every
temporal partition contains every configured fraud archetype, but some validation/test
archetypes have only 9–19 rows; future per-scenario recall estimates will be uncertain.

## Quality, missingness and EDA

Existing Phase 1 validation: **PASSED**; **24 retained
findings**, zero ERROR findings. No warnings are suppressed and no source rows/outliers deleted.

| Joined column     | Missing rows | Missing share |
| ----------------- | ------------ | ------------- |
| shipping_region   | 1917         | 1.917%        |
| P_emaildomain     | 3013         | 3.013%        |
| R_emaildomain     | 2986         | 2.986%        |
| DeviceType        | 30000        | 30.000%       |
| DeviceInfo        | 34239        | 34.239%       |
| device_id         | 30000        | 30.000%       |
| os_family         | 30000        | 30.000%       |
| browser_family    | 30000        | 30.000%       |
| identity_region   | 32729        | 32.729%       |
| identity_age_days | 30000        | 30.000%       |

Findings include structural identity absence, injected blanks, high cardinality of entity/
merchant/device fields, uniqueness of event/time fields and intentional class imbalance.
No duplicates, unsupported target labels or invalid/infinite numeric source values were found.
No statistical leakage candidate was raised by generic heuristics; that does not prove absence.
Targets, evaluation annotations, event IDs and raw entity/time identifiers are explicitly
excluded. Device/merchant popularity, synthetic correlations and time patterns require later
ablations. Binary-field IQR flags are not evidence of continuous-amount abnormality.

Available EDA: class and amount distribution, class amount comparison, numeric correlation,
IQR/z-score outliers, missingness, numeric summaries, categorical frequencies, relative
elapsed-day fraud frequency, fraud scenario distribution and legitimate hard-negative shares.
Calendar hour/weekday EDA is unavailable; no fictitious calendar conversion is performed.
Synthetic amount mean: **4795.03**; median: **2021.53**; range:
**15.20–482802.32** simulated INR units. Amount IQR outliers:
**9,777**; absolute z-score >3: **1,288**.
Neither outlier method supplies fraud labels. Measured identity coverage by target: fraud
**69.100%**, legitimate
**70.018%**. This is independent-sampling
variation, not a label-conditioned identity availability rule.

## Split strategy and actual partitions

Temporal is primary: earlier training and later validation/test, without equal-time boundary
crossing. Random stratification is a comparison with overlapping periods and globally prior
unlabelled history. Integer/fraction rounding can change a partition by one row.
See [split and historical-availability policy](SPLIT_STRATEGY.md).

| Strategy          | Partition  | Rows   | Fraud rows | Prevalence | Entities | Hard negatives |
| ----------------- | ---------- | ------ | ---------- | ---------- | -------- | -------------- |
| temporal          | train      | 70,000 | 1517       | 2.167143%  | 2000     | 3664           |
| temporal          | validation | 15,000 | 266        | 1.773333%  | 1870     | 686            |
| temporal          | test       | 15,000 | 217        | 1.446667%  | 1724     | 650            |
| random_stratified | train      | 70,000 | 1400       | 2.000000%  | 2000     | 3460           |
| random_stratified | validation | 14,999 | 300        | 2.000133%  | 1972     | 778            |
| random_stratified | test       | 15,001 | 300        | 1.999867%  | 1969     | 762            |

| Strategy          | Train/validation overlap | Train/test overlap | Validation/test overlap |
| ----------------- | ------------------------ | ------------------ | ----------------------- |
| temporal          | 1870                     | 1724               | 1683                    |
| random_stratified | 1972                     | 1969               | 1942                    |

Primary split ID: `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`.
Artifacts preserve row indexes, fingerprint, row-order hash, strategy/version, ratios, seed
where applicable, prevalence and complete per-scenario distributions. Existing identical
splits are verified/reused. No resampling, model fitting, tuning or test-set selection occurs.

## Feature lineage and strictly prior history

Joined source columns: **26**; raw predictive features:
**22**; derived features:
**31**; unencoded features:
**53**. Default V-feature count is zero; optional
V1–V20 are explicitly ANONYMIZED and carry no human business explanation.

Enabled history includes prior counts; amount mean/median/population std/min/max; amount
differences/ratios/z-scores; previous-event gap; 5m/30m/1h/24h velocity; merchant/category/
device/region/product novelty; cold-start and availability flags. Failed attempts are a
raw pre-outcome signal. Relative day/position uses the configured 7,776,000-second horizon,
never a future maximum. Equal-time entity peers read the same earlier state before updates.
Velocity windows are [event_time-window, event_time). No current/future event or target is
used in history. Missing/zero denominators produce unavailable values, never Infinity.

| Availability flag                   | True rows | Share of transactions |
| ----------------------------------- | --------- | --------------------- |
| is_new_behavioural_entity           | 2000      | 2.000%                |
| behaviour_feature_available         | 98000     | 98.000%               |
| amount_history_available            | 98000     | 98.000%               |
| merchant_novelty_available          | 98000     | 98.000%               |
| merchant_category_novelty_available | 98000     | 98.000%               |
| device_novelty_available            | 68000     | 68.000%               |
| billing_region_novelty_available    | 98000     | 98.000%               |
| product_novelty_available           | 98000     | 98.000%               |

All nine requested behavioural feature families are SUPPORTED at dataset level. Device
evidence remains incomplete for individual rows. Calendar dates, weekdays and GPS travel
remain unsupported. Synthetic entities are not real bank customers.

## Train-only preprocessing, encoding and scaling

Medians, scaler statistics, rare-category dictionaries, frequencies and one-hot categories
fit only recorded training positions. Validation/test transformations never refit. All-missing
training numeric columns use zero plus a missing indicator; every numeric feature has a
stable indicator column. Invalid/Infinity analytic values become missing, and a fixed ±1e12
safety bound prevents extreme arithmetic overflow in model-preparation views without altering
sources. Categorical missing tokens are collision-safe; unknown categories map to rare/unknown.

Merchant/device IDs use train-only frequency proportions. Fourteen manageable context fields
use training rare grouping (default minimum count 5) and sparse one-hot encoding. Entity IDs,
targets and outcome metadata are excluded; no target encoding exists. Recorded linear/distance
preprocessing uses RobustScaler; StandardScaler/no-scaling alternatives are tested. Tree/
boosting preprocessing is unscaled. Both families produce **144 aligned encoded columns**.
Feature manifests include source lineage, family, raw/derived/encoded status, transformations,
interpretability, history/time dependencies, missingness, leakage approval and model eligibility.

Preprocessing IDs hash the split, configuration, feature version/order and fitted state.
Joblib artifacts contain trusted preprocessing/metadata, not full raw datasets. Every MEDIUM
partition was transformed and compared exactly following serialization/reload. Dataset/feature/
split versions and source row-order provenance guard incompatible reuse. Future semantic
changes to generator schema, behavioural key/calculation or encoding must bump their versions.

## Memory and dimensionality

| Retained view / estimate        | Bytes       | MiB    |
| ------------------------------- | ----------- | ------ |
| Transaction before optimization | 106,391,664 | 101.46 |
| Transaction after optimization  | 19,248,075  | 18.36  |
| Identity before optimization    | 31,779,685  | 30.31  |
| Identity after optimization     | 5,050,653   | 4.82   |
| Joined frame                    | 20,796,420  | 19.83  |
| Unencoded feature frame         | 130,094,754 | 124.07 |

| Family          | Partition  | Encoded features | CSR bytes  | Equivalent dense bytes |
| --------------- | ---------- | ---------------- | ---------- | ---------------------- |
| linear_distance | train      | 144              | 30,266,528 | 80,640,000             |
| linear_distance | validation | 144              | 6,359,704  | 17,280,000             |
| linear_distance | test       | 144              | 6,362,164  | 17,280,000             |
| tree_boosting   | train      | 144              | 39,082,964 | 80,640,000             |
| tree_boosting   | validation | 144              | 8,391,400  | 17,280,000             |
| tree_boosting   | test       | 144              | 8,391,280  | 17,280,000             |

These are measured retained dataframe/CSR estimates, **not peak process RSS**. Generation,
joins, history and numerical blocks allocate additional working memory. One-hot remains sparse;
numeric-only preparation blocks are temporarily dense. LARGE has not been generated. Exact
prior medians retain sorted per-entity amounts; insertion cost grows with long histories.
No persistent production streaming history store or distributed out-of-core processing exists.

KNN is not trained or runtime-benchmarked. Exact brute-force MEDIUM test evaluation would
consider 70,000 × 15,000 = **1.05 billion** train/query pairs at 144 features. Sparse storage
does not remove search cost; full MEDIUM interactive KNN is not recommended without sampling
or a measured approximate-search design. A nominal LARGE 350,000 × 75,000 evaluation would
entail 26.25 billion pairs. These are sizing calculations, not measured KNN timings.

## Verification and leakage tests

**220 offline tests passed**: 148 retained cases and 72 Phase 2 cases. All fixtures are small
and synthetic; no test downloads data or generates MEDIUM/LARGE. Future amount/time/merchant/
device/region/product mutations and appended future records cannot change earlier features.
Current/simultaneous events cannot enter their history. Validation/test numeric/category
changes cannot alter training medians/scalers/rare dictionaries/frequencies. Target changes
cannot affect features; targets, fraud_scenario and hard_negative metadata cannot enter matrices.

Further coverage includes join protection, deterministic source hashes, scenario semantics,
legitimate hard-negative context, unseen categories, cold start, zero-denominator/all-missing
handling, sparse alignment, serialization/reload, split versions/row-order and API privacy/
unconfigured states. Python lint/format, compilation, imports of 67 API/ML modules, pip check,
frontend lint/typecheck/build/format verification and actual MEDIUM API endpoint smoke passed.
Synthetic smoke, MEDIUM generation, both split strategies, all profiles/features/reports,
preprocessing reload and full-size deterministic source regeneration were executed.

```powershell
npm run lint
npm run typecheck
npm run build
npm run format:check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q apps/api/src ml/src tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m riskbyte_ml.datasets.synthetic_riskbyte --preset medium --seed 42 --run-pipeline
.\.venv\Scripts\python.exe -m riskbyte_ml.preprocessing --config ml/config/synthetic_riskbyte.example.json
```

The development dataset was explicitly regenerated with `--overwrite` after correcting laptop/
medical merchant semantics. No user-supplied dataset was overwritten. Regeneration is now
frozen by the above fingerprint; subsequent analysis reads unchanged source CSVs. Generated
data/sidecars/splits/preprocessors/manifests and credential-free local `.env` are Git-ignored,
verified using an isolated temporary Git check-ignore invocation; nothing was staged/committed.
The small smoke uses only `.tools/phase2-smoke` with explicit overwrite for that synthetic output.

The ignored local `.env` now selects the generated dataset. All seven existing dataset endpoints
return real synthetic reports; integration metadata includes provenance/notice/split but no
local paths or evaluation row labels. No generation, preprocessing fit or model training runs
in API requests. Frontend gains only an exported notice constant; no layout redesign occurs.

## Mandatory limitations and exact next step

**The data is synthetic. It cannot prove production banking performance.** Fraud-generating
rules influence learnability. Performance may be optimistic if scenarios are too deterministic.
Later ABRF results must be interpreted as performance on this academic simulation. Synthetic
behavioural identities are not real bank customers. Real-world concept drift is only approximated.
Small per-scenario test counts limit later recall precision. Merchant/device popularity and
coarse correlated category profiles can reflect generator bias. Device evidence is incomplete.

The random comparator uses overlapping periods and earlier unlabelled held-out context, not
independent future-period evaluation. Sequential holdouts allow earlier observations to become
history without label feedback. Past attack events can contaminate behavioural baselines;
no target-aware history filtering is allowed. Neither aggregate EDA nor future model performance
on these rules is causal evidence or a validated banking probability. No real IEEE-CIS rows,
calendar anchor, GPS coordinates, matured production labels or banking access controls exist.

This dataset fingerprint, primary temporal split ID and feature manifest are frozen and
reused by [Phase 3 statistical intelligence](STATISTICAL_ANALYSIS.md). The next recommended
phase is separately authorized Phase 4 baseline classification with scenario-level recall
and hard-negative false-positive evaluation. No Logistic Regression,
Decision Tree, KNN, Random Forest, XGBoost, Isolation Forest, SHAP, ABRF, hyperparameter tuning
or final model selection has begun.
