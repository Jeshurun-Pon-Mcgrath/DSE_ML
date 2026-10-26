# Supervised model evaluation — Synthetic-data findings

All measurements describe the RiskByte Synthetic Fraud Dataset: an IEEE-CIS-inspired academic simulation. They do not establish banking performance, causal explanations or production-ready calibrated probabilities.

Dataset fingerprint: `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`.
Feature version: `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`.
Primary temporal split: `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`.
Frozen feature manifest content hash: `sha256:2668cf738689f229b9da566f20c3ad69918889dd946e72452f757f07146acd67`.
Model version: `supervised-classification-v1`; registry version: `model-registry-v1`.
Experiment ID: `sha256:09a0b32d0fe7536432dc58797347cbd71dfbef8e71d5bebe56fffb528db4c75a`.
Generated UTC: `2026-10-13T17:40:10.653748+00:00`.

## Method and isolation contract

Existing verified train-only RobustScaler/median/rare-grouping/frequency/OHE preprocessing serves Logistic Regression/KNN. Unscaled tree preprocessing serves Decision Tree, Random Forest and optional XGBoost. Both frozen CSR families retain 144 encoded columns. No dataset, split, feature manifest or temporal preprocessor is modified/refitted.

Models fit only the 70,000 training rows. Earlier chronological validation rows select restrained hyperparameters and fit calibrators; later validation rows select calibration method, operating threshold and final algorithm. Internal validation blocks do not change the frozen split. No random cross-validation or preprocessing across temporal folds. ParameterSampler generates a small reproducible fixed-validation search, including default and weighted behaviour. XGBoost uses only the earlier validation block for early stopping when installed. Nonconverged candidates are ineligible. No SMOTE or target encoding.

Validation block counts: `{'search_calibration': 7500, 'selection': 7500}`. Dependency versions: `{'numpy': '2.5.3', 'pandas': '3.0.6', 'scikit-learn': '1.9.0', 'scipy': '1.18.1', 'joblib': '1.6.0', 'xgboost': 'unavailable'}`.

Sigmoid calibration fits clipped-logit scores on earlier validation; isotonic requires at least 1,000 fitting rows and 100 fraud observations. On later validation, choose the lowest Brier alternative only if it improves raw Brier by >1% without AP loss >0.01. Otherwise retain raw probabilities. Reliability bins and Brier scores expose empirical probability quality; they cannot validate real-world banking probabilities.

Default BALANCED policy maximizes validation F1 subject to FPR <=1% and hard-negative FPR <=10%; ties prefer recall then precision. HIGH_RECALL maximizes recall with FPR <=2% and hard-negative FPR <=15%. LOW_FALSE_POSITIVE minimizes FPR with recall >=60% and hard-negative FPR <=5%. Policies are configurable simulation criteria, not bank-approved cost thresholds. All exact distinct score candidates are considered; charts retain bounded points. Decision is P(fraud) >= threshold; >1 is an explicit all-negative endpoint/fallback.

PR-AUC is non-interpolated average precision, computed from full predictions, not a trapezoid through sampled display points. Low prevalence makes accuracy and ROC-AUC alone poor selection criteria. A false positive flags a legitimate transaction; a false negative misses a simulated attack. Hard-negative and scenario counts are evaluation metadata only.

## Primary later-validation comparison

These rows support selection. Calibration method and threshold use this block, so even these validation results contain selection optimism. Accuracy is secondary.

| Model               | Precision   | Recall      | F1          | PR-AUC (AP) | ROC-AUC     | FPR         | FNR         |
| ------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Logistic Regression | 0.723684    | 0.416667    | 0.528846    | 0.576306    | 0.955868    | 0.002850    | 0.583333    |
| Decision Tree       | 0.928571    | 0.590909    | 0.722222    | 0.712918    | 0.921760    | 0.000814    | 0.409091    |
| KNN                 | 0.609195    | 0.401515    | 0.484018    | 0.449341    | 0.831087    | 0.004615    | 0.598485    |
| Random Forest       | 0.765432    | 0.469697    | 0.582160    | 0.693541    | 0.975380    | 0.002579    | 0.530303    |
| XGBoost             | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| Rule Baseline       | 0.976744    | 0.318182    | 0.480000    | 0.401771    | 0.813325    | 0.000136    | 0.681818    |

## Entire-validation comparison

Includes earlier calibration-fitting rows; not an independent generalization estimate.

| Model               | Precision   | Recall      | F1          | PR-AUC (AP) | ROC-AUC     | FPR         | FNR         |
| ------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Logistic Regression | 0.714286    | 0.413534    | 0.523810    | 0.581268    | 0.961191    | 0.002986    | 0.586466    |
| Decision Tree       | 0.903030    | 0.560150    | 0.691415    | 0.713196    | 0.931052    | 0.001086    | 0.439850    |
| KNN                 | 0.538012    | 0.345865    | 0.421053    | 0.372508    | 0.812309    | 0.005362    | 0.654135    |
| Random Forest       | 0.769231    | 0.451128    | 0.568720    | 0.674403    | 0.980182    | 0.002443    | 0.548872    |
| XGBoost             | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| Rule Baseline       | 0.964706    | 0.308271    | 0.467236    | 0.381471    | 0.798906    | 0.000204    | 0.691729    |

## Operating points, calibration and imbalance

| Model               | Threshold   | Calibration | Selection Brier | Imbalance strategy                                                             |
| ------------------- | ----------- | ----------- | --------------- | ------------------------------------------------------------------------------ |
| Logistic Regression | 0.411401    | none        | 0.010655        | class_weight=None; default and weighted candidates compared                    |
| Decision Tree       | 0.366541    | sigmoid     | 0.008349        | class_weight=balanced; default and weighted candidates compared                |
| KNN                 | 0.264563    | none        | 0.012846        | Representative stratified training sample; no class weights/oversampling       |
| Random Forest       | 0.338782    | none        | 0.008868        | class_weight=None; default and weighted candidates compared                    |
| XGBoost             | unavailable | unavailable | unavailable     | Planned comparison: scale_pos_weight=1 versus training negative/positive ratio |
| Rule Baseline       | 0.666667    | rule_score  | unavailable     | None; fixed training-knowledge rules                                           |

Every trained model also records its 0.5-threshold metrics, selected train diagnostics, candidate parameters/results, full threshold-policy curves and tree global importances.

## Hard negatives — entire validation

| Model               | Hard negatives | False positives | False-positive rate |
| ------------------- | -------------- | --------------- | ------------------- |
| Logistic Regression | 686            | 38              | 0.055394            |
| Decision Tree       | 686            | 9               | 0.013120            |
| KNN                 | 686            | 76              | 0.110787            |
| Random Forest       | 686            | 36              | 0.052478            |
| XGBoost             | unavailable    | unavailable     | unavailable         |
| Rule Baseline       | 686            | 3               | 0.004373            |

Each subtype is reported independently in the JSON registry, with explicit unavailable rates for empty subgroups. Unusual legitimate purchases must not be interpreted as fraud solely because an amount/device/velocity is unusual.

## Scenario recall — entire validation

| Scenario                      | Logistic Regression | Decision Tree | KNN      | Random Forest | XGBoost     | Rule Baseline |
| ----------------------------- | ------------------- | ------------- | -------- | ------------- | ----------- | ------------- |
| AMOUNT_SPIKE                  | 0.111111            | 0.444444      | 0.222222 | 0.111111      | unavailable | 0.000000      |
| VELOCITY_BURST                | 0.071429            | 0.023810      | 0.214286 | 0.047619      | unavailable | 0.000000      |
| NEW_DEVICE_HIGH_AMOUNT        | 0.181818            | 0.454545      | 0.545455 | 0.545455      | unavailable | 0.454545      |
| REGION_CHANGE                 | 0.000000            | 0.785714      | 0.000000 | 0.000000      | unavailable | 0.000000      |
| NEW_MERCHANT_AMOUNT_DEVIATION | 0.052632            | 0.473684      | 0.210526 | 0.052632      | unavailable | 0.052632      |
| REPEATED_FAILED_ATTEMPTS      | 0.700000            | 0.700000      | 0.500000 | 0.700000      | unavailable | 0.000000      |
| ACCOUNT_TAKEOVER              | 1.000000            | 1.000000      | 0.829268 | 1.000000      | unavailable | 0.878049      |
| LOW_AND_SLOW                  | 0.163636            | 0.418182      | 0.000000 | 0.218182      | unavailable | 0.000000      |
| INTERNATIONAL_ANOMALY         | 0.000000            | 0.076923      | 0.000000 | 0.000000      | unavailable | 0.000000      |
| MIXED_MODE                    | 0.884615            | 0.903846      | 0.615385 | 0.961538      | unavailable | 0.769231      |

Counts/true positives accompany every rate in the registry. Small subgroups have uncertain recall; evaluation-only fraud_scenario never enters matrices or rule inputs.

## Selection and final temporal test

Decision Tree selected by later-validation average precision 0.712918; lexicographic tie-breaks: recall, precision, F1, hard-negative FPR, minimum scenario recall, Brier, batch prediction time. Accuracy/random comparator/test results are never selection criteria.

Selection, calibrated artifact hash and operating threshold were exclusively published before the one-use test gate opened. Only the selected supervised classifier was scored on the primary temporal test. No refitting/tuning followed. Other models' test metrics remain unavailable. Phase 3 had already inspected aggregate test statistics; this is Phase 4 model-selection isolation, not a claim that prior exploration never saw test data.

| Partition  | Rows  | Fraud prevalence |
| ---------- | ----- | ---------------- |
| train      | 70000 | 0.021671         |
| validation | 15000 | 0.017733         |
| test       | 15000 | 0.014467         |

| Scope            | Precision | Recall   | F1       | PR-AUC   | ROC-AUC  | FPR      | FNR      | Brier    |
| ---------------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| selection        | 0.928571  | 0.590909 | 0.722222 | 0.712918 | 0.921760 | 0.000814 | 0.409091 | 0.008349 |
| whole validation | 0.903030  | 0.560150 | 0.691415 | 0.713196 | 0.931052 | 0.001086 | 0.439850 | 0.008689 |
| final test       | 0.906977  | 0.539171 | 0.676301 | 0.679004 | 0.932577 | 0.000812 | 0.460829 | 0.007338 |

Final test confusion: `{'tp': 117, 'fp': 12, 'tn': 14771, 'fn': 100}`.
Final test hard negatives: `{'count': 650, 'flagged': 6, 'rate': 0.009230769230769232}`.

| Scenario                      | Test fraud count | Detected | Recall   |
| ----------------------------- | ---------------- | -------- | -------- |
| AMOUNT_SPIKE                  | 13               | 6        | 0.461538 |
| VELOCITY_BURST                | 38               | 0        | 0.000000 |
| NEW_DEVICE_HIGH_AMOUNT        | 16               | 5        | 0.312500 |
| REGION_CHANGE                 | 13               | 11       | 0.846154 |
| NEW_MERCHANT_AMOUNT_DEVIATION | 11               | 4        | 0.363636 |
| REPEATED_FAILED_ATTEMPTS      | 9                | 6        | 0.666667 |
| ACCOUNT_TAKEOVER              | 51               | 51       | 1.000000 |
| LOW_AND_SLOW                  | 34               | 13       | 0.382353 |
| INTERNATIONAL_ANOMALY         | 10               | 0        | 0.000000 |
| MIXED_MODE                    | 22               | 21       | 0.954545 |

## Error analysis — selected final test

Definitions: High amount exceeds training 95th percentile; velocity burst means strictly prior 30m count >=3; novelty unavailable is not new.

False positives: `{'count': 12, 'hard_negative_count': 6, 'missing_identity_count': 4, 'median_amount': 2239.9750000000004, 'median_amount_ratio': 1.311572256978915, 'median_velocity_30m': 1.0, 'new_device_count': 0, 'new_merchant_count': 1, 'high_amount_count': 1, 'velocity_burst_count': 3, 'scenarios': {}, 'hard_negative_types': {'SHOPPING_FESTIVAL': 5, 'NEW_LEGITIMATE_DEVICE': 1}}`.

False negatives: `{'count': 100, 'hard_negative_count': 0, 'missing_identity_count': 32, 'median_amount': 1865.41, 'median_amount_ratio': 0.8937128565030616, 'median_velocity_30m': 0.0, 'new_device_count': 7, 'new_merchant_count': 19, 'high_amount_count': 11, 'velocity_burst_count': 19, 'scenarios': {'VELOCITY_BURST': 38, 'LOW_AND_SLOW': 21, 'NEW_DEVICE_HIGH_AMOUNT': 11, 'INTERNATIONAL_ANOMALY': 10, 'NEW_MERCHANT_AMOUNT_DEVIATION': 7, 'AMOUNT_SPIKE': 7, 'REPEATED_FAILED_ATTEMPTS': 3, 'REGION_CHANGE': 2, 'MIXED_MODE': 1}, 'hard_negative_types': {}}`.

These are aggregate diagnostics, not exposed individual transactions. Investigate hard-negative/new-device/high-amount/velocity false positives and weak-signal or missing-identity false negatives without adjusting this already frozen model.

## Timing and KNN practicality

| Model               | Search fit seconds | Chosen fit seconds | Validation predict seconds |
| ------------------- | ------------------ | ------------------ | -------------------------- |
| Logistic Regression | 7.307963           | 1.496964           | 0.001480                   |
| Decision Tree       | 5.719636           | 1.434910           | 0.004342                   |
| KNN                 | 0.009747           | 0.002610           | 1.893381                   |
| Random Forest       | 93.327530          | 18.727507          | 0.103950                   |
| XGBoost             | not fitted         | not fitted         | unavailable                |
| Rule Baseline       | not fitted         | not fitted         | 0.006672                   |

Selected test batch prediction: 0.004425 seconds; average 0.294980 microseconds/transaction. This excludes preprocessing, history, HTTP and queueing, and is not a real-time SLA.

Random Forest uses serial tree aggregation for exact serialization/probability tie reproducibility. Other estimators obey the bounded thread/working-memory budget.

KNN benchmark: `{'sample_size': 5000, 'fraud_count': 108, 'query_count': 256, 'fit_seconds': 0.0021912000001975684, 'prediction_seconds': 0.04411739999977726, 'matrix_memory_bytes': 2161668, 'full_pair_count': 525000000, 'projected_full_prediction_seconds': 18.095027343658643, 'constrained': True, 'reason': 'Representative stratified training subset; full fit/prediction exceeds configured pair/time budget', 'metrics': {'rows': 256, 'fraud_count': 5, 'prevalence': 0.01953125, 'threshold': 0.5, 'precision': 1.0, 'recall': 0.2, 'f1': 0.33333333333333337, 'pr_auc': 0.522953869047619, 'pr_auc_definition': 'average precision; non-interpolated', 'roc_auc': 0.8908366533864542, 'brier_score': 0.011901495778112402, 'accuracy': 0.984375, 'specificity': 1.0, 'fpr': 0.0, 'fnr': 0.8, 'confusion': {'tp': 1, 'fp': 0, 'tn': 251, 'fn': 4}, 'probability_note': 'P(fraud); empirical calibration requires review'}}`.

Projected full timing is an explicit extrapolation from the measured sample, not a completed MEDIUM KNN run. Retained CSR bytes and a 64 MiB distance working budget describe memory behaviour, not isolated peak RSS. Constrained train diagnostics use the fit subset; KNN stores that subset's encoded vectors by design.

KNN: Representative stratified training subset; full fit/prediction exceeds configured pair/time budget

XGBoost: XGBoost package absent; network installation blocked in this environment. No XGBoost measurements are claimed.

## Secondary saved random-split benchmark

Run only after the primary selection/test freeze. Strong available primary models reuse chosen hyperparameters, with new train-only random-split preprocessing saved separately. Only random validation is scored at raw threshold 0.5; random test is unscored. AP is more comparable than differently calibrated operating-point rates. Random training reuses some primary test rows; these results cannot inform any primary model decision. Prior unlabelled history is transductive, so this is not independent future-period evaluation.

| Model         | Temporal selection AP | Random validation AP | Random fit seconds |
| ------------- | --------------------- | -------------------- | ------------------ |
| Decision Tree | 0.712918              | 0.751413             | 1.428500           |
| Random Forest | 0.693541              | 0.804193             | 19.166561          |

## Phase 3 connection and feature importance

Phase 3 found strong velocity separation and weak amount-ratio separation between fraud and hard negatives. The measured model allocations below permit reviewing consistency; correlations/importance do not demonstrate causal reasons or guarantee false-positive improvement. SHAP is not implemented. Anonymized features would retain ANONYMIZED tags; the frozen generator enables no V features.

| Selected encoded feature        | Global importance | Interpretability        |
| ------------------------------- | ----------------- | ----------------------- |
| time_since_previous_transaction | 0.605367          | INTERPRETABLE           |
| amount_ratio_to_prior_mean      | 0.149620          | INTERPRETABLE           |
| failed_attempts_before          | 0.069370          | INTERPRETABLE           |
| billing_region_seen_before      | 0.056011          | INTERPRETABLE           |
| is_international                | 0.050789          | INTERPRETABLE           |
| behavioural_history_count       | 0.013553          | INTERPRETABLE           |
| device_id__train_frequency      | 0.008186          | PARTIALLY_INTERPRETABLE |
| amount_minus_prior_mean         | 0.004138          | INTERPRETABLE           |
| prior_median_amount             | 0.003588          | INTERPRETABLE           |
| TransactionAmt                  | 0.003555          | INTERPRETABLE           |
| account_age_days                | 0.003457          | INTERPRETABLE           |
| relative_time_position          | 0.003300          | TEMPORAL                |
| prior_amount_std                | 0.002937          | INTERPRETABLE           |
| amount_zscore_prior             | 0.002832          | INTERPRETABLE           |
| elapsed_day_index               | 0.002721          | TEMPORAL                |

## Curves and reusable artifacts

Local reports: `ml/artifacts/evaluation/model_comparison.json`, five report-ready CSV tables, per-model validation PR/ROC/reliability JSON, final_test_metrics.json and standalone SVGs. PR/ROC charts use recorded finite curve points; AUC uses all scored records. Whole-validation curves include calibration-fitting rows. Models/metadata are under `ml/artifacts/models/`; only trusted, checksummed local joblib artifacts may be loaded. API reads JSON only.

![Whole-validation PR](../ml/artifacts/evaluation/validation_pr.svg)

![Whole-validation ROC](../ml/artifacts/evaluation/validation_roc.svg)

![Selected final-test PR](../ml/artifacts/evaluation/selected_test_pr.svg)

## Leakage audit

- preprocessing: PASS: frozen bytes, training-index hash, ordered raw/encoded manifest, no temporal refit
- target_and_metadata: PASS: source-lineage allowlist excludes target/scenario/hard-negative annotations
- behaviour: PASS: existing strictly prior builder; current/equal/future event and target invariance tests retained
- split: PASS: exact frozen temporal and secondary random split bytes preserved
- threshold_and_calibration: PASS: validation blocks only; no test parameter selection
- final_test: PASS: durable model/threshold/calibrator freeze before one gate access (1)
- random_comparator: PASS: post-freeze secondary experiment; never feeds primary selection

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.models.supervised --config ml/config/synthetic_riskbyte.example.json --experiment-config ml/config/supervised.example.json
```

A completed identical experiment returns the saved registry without refitting or reopening the test gate. Changed settings/dependencies are rejected in that output directory. A frozen/interrupted experiment cannot silently retrain or repeat the test. Use one offline writer; fresh output directories do not make an already inspected test scientifically unseen.

Read-only routes: GET /api/v1/models, /comparison, /selected, /{model_id}/metrics, /{model_id}/scenario-recall, /{model_id}/threshold-analysis. No public prediction API, startup training or frontend redesign.

Optional XGBoost setup in an unrestricted environment: `python -m pip install -e "./ml[boosting]"`. Adding a missing algorithm after final evaluation requires a separately declared experiment; do not silently reuse the primary test as untouched or change this frozen selected classifier.

## Limitations

- All findings use synthetic academic data and cannot establish production banking performance.
- Scenario rules influence learnability; synthetic identities are not real customers and drift is approximate.
- Validation is subdivided chronologically: earlier search/calibration and later model/calibration/threshold selection. Entire-validation metrics include calibration-fitting rows and are not independent estimates.
- Later validation is reused for calibration method, threshold and algorithm selection, so selection metrics are optimistic; no nested temporal replication is claimed.
- Phase 3 already inspected aggregate test statistics; test remains excluded from Phase 4 fitting/selection, but was not unseen for all prior exploration.
- Repeated entities/episodes and small scenario counts limit uncertainty and independent generalization claims.
- KNN can retain only a stratified training subset; its comparison has a different fit budget. Memory figures describe retained CSR, not isolated peak RSS.
- Batch prediction timings are measured offline and include no HTTP/history/preprocessing service latency; no real-time readiness claim.
- Tree importances are exploratory correlated-feature allocations, not causal explanations; SHAP remains absent.
- Random comparator training reuses rows from different primary partitions after primary selection/test freeze; it cannot inform any primary decision.
- Missing identity evidence and contaminated prior attack history can impair detection. No target-aware history filtering or synthetic record deletion.
- No SMOTE, target encoding, Isolation Forest, SHAP, ABRF or public transaction prediction is implemented.

## References

[Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html), [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

## Reviewed measured findings

**Synthetic-data finding:** the Decision Tree was selected among the available trained
classifiers by later-validation AP **0.712918**, followed by Random Forest **0.693541**,
Logistic Regression **0.576306** and sampled KNN **0.449341**. XGBoost was not installed;
pip lookup failed and direct dependency retrieval was blocked with Windows socket permission
error 10013. No XGBoost training or measurements occurred. Its implementation and explicit
unavailable registry entry remain ready for an environment with the optional dependency.

The selected tree uses `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=30`, and
`class_weight=balanced`. Sigmoid calibration reduced selection Brier from **0.025904** to
**0.008349** without changing AP. Isotonic achieved lower Brier **0.007834**, but reduced
AP to **0.683089**, exceeding the prespecified allowed loss; it was not selected.
The BALANCED threshold **0.3665406264377367** maximizes selection F1 under the declared
FPR/subgroup caps. It is not a globally optimal banking threshold. The validation-only
HIGH_RECALL alternative had recall **65.91%** at threshold **0.225526**, but the frozen
default was not changed after test evaluation.

| Decision Tree scope        | PR-AUC   | Precision | Recall   | F1       |
| -------------------------- | -------- | --------- | -------- | -------- |
| Training diagnostic        | 0.830740 | 0.865766  | 0.633487 | 0.731633 |
| Later validation selection | 0.712918 | 0.928571  | 0.590909 | 0.722222 |
| Entire validation          | 0.713196 | 0.903030  | 0.560150 | 0.691415 |
| Frozen temporal test       | 0.679004 | 0.906977  | 0.539171 | 0.676301 |

The selected-model AP drop from selection to test is **0.033914**; recall drops **5.17
percentage points**. The largest observed scenario recall drop is new-device/high-amount:
**3/3** detected in the selection block versus **5/16** on test (**68.75 percentage points**).
That comparison is highly uncertain because selection contains only three examples.
Entire-validation recall for the same scenario was **5/11**, so the whole-validation/test
gap is smaller. These are simulated temporal-generalization characteristics, not automatically
implementation bugs, and no tuning follows them.

The most difficult selected-model test scenarios tied at zero recall: **velocity burst
(0/38)** and **international anomaly (0/10)**. Low-and-slow recall was **13/34**. Account
takeover was **51/51**, and mixed mode **21/22**. These results expose substantial blind spots
despite high overall accuracy. Of 100 false negatives, **32** lacked identity information;
the aggregate does not prove that missing identity caused those errors.

Test hard-negative FPR was **6/650 = 0.9231%**. The largest named false-positive subtype was
shopping festival (**5 of 12 total false positives**); one was a legitimate new-device
hard negative. Six false positives were ordinary legitimate records. Laptop, medical,
travel and new-merchant hard-negative test groups had zero observed false positives;
small groups do not establish zero population risk.

Whole-validation hard-negative FPR ranking: Rule Baseline **0.4373%** (3/686), Decision Tree
**1.3120%** (9/686), Random Forest **5.2478%** (36/686), Logistic Regression **5.5394%**
(38/686), KNN **11.0787%** (76/686). Later-selection supervised ranking: Decision Tree
**0.6116%** (2/327), Logistic Regression and Random Forest **5.8104%** (19/327 each), KNN
**9.7859%** (32/327). Constraint satisfaction is specific to the selection block; it is
not guaranteed in other periods. Rule Baseline remains unselected: it has fewer false
positives but substantially lower recall and AP, so ML does not dominate every tradeoff.

The selected tree's largest impurity allocations were previous-event gap (**0.60537**),
amount ratio (**0.14962**), failed attempts (**0.06937**) and region novelty (**0.05601**).
This supports a qualitative connection to prior-time and deviation signals studied in
Phase 3. It does not establish causal importance or successful detection of all velocity
attacks. The tree had fewer unusual-purchase false positives than the linear/KNN comparators,
but that improvement cannot be causally attributed to one feature without later ablation.

Random validation AP exceeded primary later-temporal validation by **0.038495** for Decision
Tree and **0.110651** for Random Forest. Different prevalence, training composition, evaluation
sizes and raw-versus-calibrated operating points limit direct comparison. The random result
cannot justify changing the already frozen primary selection.

## Completed-run integrity

Cached runs and model-report API responses verify actual classifier byte hashes, the saved
selection freeze, the one-use test-start marker and final-test metrics against the registry.
They read JSON/checksums without loading executable model objects. Corrupted completed
artifacts fail closed rather than silently retraining or repeating test scoring.

Initial target reads validate integrity and partition counts only. No test labels, predictions
or error summaries are supplied to fitting, hyperparameter search, calibration, threshold
selection or primary model ranking. The final test gate opens after durable selection.

## Verification results

**319 offline tests passed**, including 44 supervised-phase cases. Frontend lint,
typecheck, production build and format verification passed. Python Ruff lint/format,
compilation and `pip check` passed. All **89** API/ML modules imported without automatic
training or scoring. Tests use small synthetic fixtures, never MEDIUM or network access.

The complete frozen MEDIUM experiment was executed once, then an identical CLI invocation
verified/reused saved results without refitting or reopening test scoring. Health and all
six model-report endpoints passed actual MEDIUM smoke requests. XGBoost metrics correctly
returned `models_unavailable`; unknown IDs returned structured 404 responses. API payloads
contained no local source paths, raw transaction rows, NaN or Infinity. The final test-start
journal retained exactly one access and unchanged content through cached CLI/API checks.

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
.\.venv\Scripts\python.exe -m riskbyte_ml.models.supervised --config ml/config/synthetic_riskbyte.example.json --experiment-config ml/config/supervised.example.json
```

## Next recommended phase

Review the selected classifier, hard-negative/scenario tradeoffs and unavailable dependency. Then separately authorize Phase 5 anomaly detection with training-only reference populations and score normalization. Do not fuse scores, implement SHAP/ABRF or deploy based on this synthetic benchmark. No Phase 5 work is performed here.
