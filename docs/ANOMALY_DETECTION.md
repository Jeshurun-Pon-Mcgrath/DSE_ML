# Phase 5 anomaly detection

**Synthetic-data finding.** Dataset `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`; feature pipeline `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`; temporal split `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`. Development contains 70,000 TRAIN and 15,000 VALIDATION rows. The original temporal test partition was not decoded into records, scored, or used for Phase 5 decisions. Full files are read as opaque bytes for checksums; identity projection checks excluded numeric join keys without decoding excluded context fields.

## Method and selection

Isolation Forest isolates unusual observations using randomly partitioned trees. Shorter
average isolation paths indicate global unusualness, not proof of fraud. Version:
`anomaly-detection-v1`. Four candidates compare interpretable versus broader numeric
features and legitimate-only versus mixed training. No categorical one-hot matrix is fed
to the detector. The broader contract excludes identifiers, categories and absolute elapsed
position to avoid identifying future time as automatically anomalous.

The reference contract deliberately prefers known legitimate TRAIN observations; mixed
training is a methodological comparator. TRAIN labels select the normal reference cohort,
but fitting receives X only. Validation labels evaluate fitted candidates and thresholds.
No outcome/scenario/hard-negative field enters X. The selected candidate has the highest
validation average precision among eligible legitimate-reference candidates.

| Features      | Reference        | Raw/prepared columns | AP       | ROC-AUC  | Recall   | Legitimate FPR | Hard-negative FPR | Fit seconds |
| ------------- | ---------------- | -------------------- | -------- | -------- | -------- | -------------- | ----------------- | ----------- |
| interpretable | legitimate_train | 21/42                | 0.438004 | 0.918292 | 36.0902% | 0.4751%        | 9.9125%           | 1.016       |
| interpretable | mixed_train      | 21/42                | 0.412656 | 0.915338 | 34.5865% | 0.4683%        | 9.7668%           | 0.893       |
| broad_numeric | legitimate_train | 30/60                | 0.123592 | 0.834182 | 21.0526% | 1.9954%        | 8.7464%           | 0.944       |
| broad_numeric | mixed_train      | 30/60                | 0.087706 | 0.812896 | 13.9098% | 1.9818%        | 8.0175%           | 1.000       |

Selected: **interpretable**, **legitimate_train**, **68,483**
reference rows; **21** numeric signals plus one missing indicator
per signal (**42** prepared columns). Seed 42, 150 trees,
maximum sample 1,024 per tree. These are restrained prespecified settings, not a giant search.

Selected fields: `TransactionAmt`, `amount_ratio_to_prior_mean`, `amount_zscore_prior`, `amount_minus_prior_mean`, `transactions_prior_5m`, `transactions_prior_30m`, `transactions_prior_1h`, `transactions_prior_24h`, `time_since_previous_transaction`, `failed_attempts_before`, `merchant_seen_before`, `device_seen_before`, `billing_region_seen_before`, `product_seen_before`, `account_age_days`, `identity_age_days`, `is_international`, `behavioural_history_count`, `is_new_behavioural_entity`, `behaviour_feature_available`, `device_novelty_available`.

## Missingness and score semantics

Each candidate fits medians on its TRAIN reference only. All-missing columns use zero plus
an explicit missing indicator. Nonfinite values become missing; numerical preparation uses
the existing documented fixed ±1e12 safety bound. No source rows/columns are deleted.
No scaling or one-hot expansion is necessary for this numeric isolation study.

The original `score_samples` output is preserved, with lower values more anomalous.
Raw anomaly severity is its negative. A strict-left empirical CDF of TRAIN reference severity
maps severity to [0,1]: `searchsorted(sorted_training_scores, x, side='left') / n`.
This deterministic monotonic mapping is frozen before validation. Values beyond the training
reference clip naturally to 0/1; ties share a percentile. **A is not fraud probability.**

Contamination candidates: auto, .005, .01, .02, .03, .05. For fixed trees, contamination
changes the native decision offset, not tree fitting/ranking. Thus offsets are evaluated
without redundantly fitting six identical forests. Numeric offsets use TRAIN-reference
quantiles; auto uses raw severity 0.5. Native decisions use strict raw comparison.
Chosen native contamination: **0.03**. This is not an estimate of
fraud prevalence. The operational analysis threshold is chosen separately on normalized A.

## Validation operating point

Threshold **0.9626184600557802** maximizes fraud recall subject to overall legitimate
FPR ≤2% and overall hard-negative rate ≤10%; ties favor precision, fewer FPs, higher threshold.
Continuous scores remain primary. Caps constrain overall hard negatives, not every subtype.
Raw threshold points and every contamination comparison remain in `analysis_bundle.json`.

AP **0.438004**, ROC-AUC **0.918292**, precision
**57.8313%**, recall **36.0902%**, F1 **0.444444**,
legitimate FPR **0.4751%**, hard-negative rate
**9.9125%**
(68/686).
Confusion: {'tp': 96, 'fp': 70, 'tn': 14664, 'fn': 170}. AP means non-interpolated average precision.

## Artifacts, API and execution

`ml/artifacts/phase5/isolation_forest.joblib` contains the trusted fitted model, feature list,
imputer, sorted normalization reference and provenance. `anomaly_metadata.json` records
population, contamination, feature study and validation threshold; the coherent
`analysis_bundle.json` binds versions, seed/configuration, split, timestamp, baseline hash
and executable artifact hashes. No full raw dataset is serialized.
GET `/api/v1/anomaly/{status,metrics,scenarios,hard-negatives}` reads JSON/checksums only.
No prediction endpoint, model fitting, or unpickling occurs in API requests.

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.anomaly `
  --config ml/config/synthetic_riskbyte.example.json `
  --experiment-config ml/config/anomaly.example.json `
  --behaviour-config ml/config/behaviour.example.json
```

Completed identical runs reuse verified artifacts; changed configuration or incompatible
frozen sources abort. Executable joblib artifacts are trusted local files only.

## Limitations

Synthetic academic data cannot establish production banking performance. Generator rules
influence separability. Normal training contains unusual legitimate hard negatives; anomaly
is not equivalent to fraud. Validation is reused for feature-family and threshold selection,
so its measurements are development results. Episode/entity dependence and small scenario
counts limit uncertainty claims. Empirical percentiles may shift under drift; they are not
calibrated probabilities. Numerical preparation retains a modest dense matrix and no OHE.
No original-test candidate evaluation, ABRF, SHAP or final system logic is implemented.
