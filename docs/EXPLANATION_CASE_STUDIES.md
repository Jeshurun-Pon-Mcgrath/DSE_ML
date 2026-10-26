# Validation explanation case studies

All entries are **Synthetic-data findings**. IDs refer to fictional transactions and do not
identify people. Cases come from VALIDATION only. Selection uses the earliest eligible
event for five prespecified outcome/scenario categories; the normal legitimate case is the
minimum-risk eligible event, with chronology breaking ties. Failures are included alongside
successes. Ground truth stays in the offline evaluation wrapper, outside the live contract.

## Actual frozen measurements

| Case                             | Transaction ID | Calibrated P | A diagnostic | B        | Confidence | ABRF risk | Decision    |
| -------------------------------- | -------------- | ------------ | ------------ | -------- | ---------- | --------- | ----------- |
| Account takeover true positive   | 2845           | 0.940109     | 0.999401     | 0.780966 | 0.821429   | 0.926799  | Investigate |
| Recovered low-and-slow           | 69117          | 0.295488     | 0.753574     | 0.421656 | 0.759036   | 0.305301  | Investigate |
| Recovered international anomaly  | 58423          | 0.313237     | 0.896909     | 0.209823 | 0.438462   | 0.313237  | Investigate |
| Shopping-festival false positive | 14302          | 0.367705     | 0.950455     | 0.418622 | 0.591837   | 0.370847  | Investigate |
| Minimum-risk normal legitimate   | 86722          | 0.001191     | 0.231225     | 0.207060 | 0.428571   | 0.001191  | No flag     |
| Remaining fraud false negative   | 10604          | 0.291835     | 0.884920     | 0.338519 | 0.460976   | 0.291835  | No flag     |

The underlying tree probabilities are respectively 1.0, 0.963148, 0.970902, 0.985447,
0.0 and 0.961284. These differ substantially from calibrated P; a tree SHAP decomposition
must not be presented as a decomposition of calibrated P. The original v1 report lacked SHAP. The active v1.1 snapshot now contains genuine tree
SHAP factors; calibration remains outside the attribution decomposition.

## What these traces demonstrate

- The takeover case is supported strongly by the classifier and behaviour. Anomaly is
  high but contributes exactly zero to ABRF.
- The selected low-and-slow case lies below the ABRF threshold on P alone; the confidence-
  gated behavioural contribution of 0.032796, after P renormalization, takes risk above it.
- The international case has confidence below 0.5. B contributes zero. Recovery comes
  from the lower frozen ABRF investigation threshold, not anomaly or behavioural fusion.
- The shopping-festival legitimate event is a false positive. Its supervised contribution
  is 0.345017 and behavioural contribution 0.025830. Anomaly contributes zero. Its classifier
  term alone already exceeds the investigation threshold at these effective weights.
- The minimum-risk case has actual frozen band **GUARDED**, because equality to the first
  boundary enters the higher band. It is never falsely displayed as LOW. Its risk is
  far below the investigation threshold; missing/limited history is not proof of safety.
- The missed fraud case has P below the investigation threshold and confidence below the
  behaviour gate. Neither high raw tree output nor diagnostic anomaly implies a positive
  final investigation decision.

## Hard-negative false positives

All **11** hard-negative false positives are retained: **8 shopping-festival** events and
**3 new-legitimate-device** events. Five have unavailable current identity context; two
have confidence below the behaviour gate. Mean P=0.379280, B=0.467614, A=0.934990.

At fixed effective weights, subtracting the supervised term removes all 11 flags;
subtracting the behavioural term removes none. This is an arithmetic diagnostic, not
causation or a comparison with renormalized P-only fusion. Behaviour is modest; the anomaly
term is exactly zero. Actual classifier SHAP factors now appear below; they are model
contributions, not causal explanations of financial behaviour.

## Remaining false negatives

All **98** missed validation fraud events are explained, including:

| Scenario                        | Missed |
| ------------------------------- | ------ |
| Velocity burst                  | 41     |
| Low-and-slow                    | 22     |
| New merchant / amount deviation | 10     |
| New device / high amount        | 6      |
| International anomaly           | 6      |
| Mixed mode                      | 5      |
| Region change                   | 3      |
| Amount spike                    | 3      |
| Repeated failed attempts        | 2      |

Mean P=0.137305, mean A=0.848899, mean B=0.405591; median confidence=0.603922.
**42** lack current identity context; **21** fall below the confidence gate. None is flagged
by the final frozen decision. High diagnostic anomaly evidence cannot recover them because
the active anomaly weight is zero. These patterns describe remaining limitations, not a
reason to retune frozen models, weights or thresholds during Phase 7.

## Artifacts and safe demo access

`case_studies.json`, `hard_negative_explanations.json` and `false_negative_explanations.json`
contain actual component evidence and exact traces, plus separately labelled evaluation
ground truth. Normal transaction explanation responses exclude scenario/hard-negative/target
metadata. Transaction-level HTTP lookup serves only these precomputed validation records;
other IDs return a structured 404 and never trigger on-demand scoring.

Original temporal test accessed for these explanations: **NO**. No future holdout was
generated or evaluated. SHAP and all explanation evidence are non-causal; no output proves
guilt, human intent or production financial performance.

## Phase 7.1 genuine SHAP case table

All entries are synthetic academic findings. IDs and non-SHAP evidence are unchanged.

| Case / ID                                | Evaluation label | RiskByte decision | Final score | Strongest increasing tree SHAP                            | Strongest behavioural component | Anomaly diagnostic            | Outcome |
| ---------------------------------------- | ---------------- | ----------------- | ----------- | --------------------------------------------------------- | ------------------------------- | ----------------------------- | ------- |
| account_takeover_true_positive / 2845    | 1                | Investigate       | 0.926799    | time_since_previous_transaction (+0.184710; value 20)     | device_novelty (1.000000)       | A=0.999401, zero contribution | TP      |
| recovered_low_and_slow / 69117           | 1                | Investigate       | 0.305301    | time_since_previous_transaction (+0.358150; value 1217)   | amount_deviation (0.952706)     | A=0.753574, zero contribution | TP      |
| recovered_international_anomaly / 58423  | 1                | Investigate       | 0.313237    | is_international (+0.554502; value 1)                     | amount_zscore (0.884774)        | A=0.896909, zero contribution | TP      |
| shopping_festival_false_positive / 14302 | 0                | Investigate       | 0.370847    | time_since_previous_transaction (+0.506646; value 12)     | short_time_gap (0.986912)       | A=0.950455, zero contribution | FP      |
| normal_legitimate_low_risk / 86722       | 0                | No flag           | 0.001191    | device_id__train_frequency (+0.003611; value 5.71429e-05) | amount_zscore (0.777715)        | A=0.231225, zero contribution | TN      |
| remaining_fraud_false_negative / 10604   | 1                | No flag           | 0.291835    | amount_ratio_to_prior_mean (+0.453770; value 5.68565)     | amount_zscore (0.985743)        | A=0.884920, zero contribution | FN      |

The low-and-slow example has a small amount ratio (0.258868), yet that deviation increases
the tree output. Its wording therefore does not claim a high transaction amount.
International status raises the underlying tree output, but recovery in this example comes
from the lower ABRF threshold: behavioural confidence is below its gate and anomaly weight
is zero. The normal case's strongest decreasing factor is prior-event gap (-0.303637);
its actual band remains GUARDED. The missed case has a strong positive amount-ratio
attribution (+0.453770), offset by decreasing features and separate calibration.

### Actual top factors, including protective evidence

#### account_takeover_true_positive (2845)

| Direction  | Feature                         | Encoded value | SHAP          | Interpretability        |
| ---------- | ------------------------------- | ------------- | ------------- | ----------------------- |
| Increasing | time_since_previous_transaction | 20            | +0.18470982   | INTERPRETABLE           |
| Increasing | amount_ratio_to_prior_mean      | 4.0106215     | +0.12643073   | INTERPRETABLE           |
| Increasing | failed_attempts_before          | 10            | +0.11329087   | INTERPRETABLE           |
| Increasing | billing_region_seen_before      | 0             | +0.087364401  | INTERPRETABLE           |
| Increasing | device_id__train_frequency      | 0             | +0.023615442  | PARTIALLY_INTERPRETABLE |
| Decreasing | prior_valid_amount_count        | 92            | -0.039750654  | INTERPRETABLE           |
| Decreasing | TransactionAmt                  | 49770.54      | -0.0068905509 | INTERPRETABLE           |
| Decreasing | prior_min_amount                | 2525.05       | -0.0034985183 | INTERPRETABLE           |
| Decreasing | product_code_v:electronics      | 1             | -0.0028037086 | INTERPRETABLE           |
| Decreasing | prior_transaction_count         | 92            | -0.0020241329 | INTERPRETABLE           |

#### recovered_low_and_slow (69117)

| Direction  | Feature                         | Encoded value | SHAP          | Interpretability |
| ---------- | ------------------------------- | ------------- | ------------- | ---------------- |
| Increasing | time_since_previous_transaction | 1217          | +0.35814997   | INTERPRETABLE    |
| Increasing | amount_ratio_to_prior_mean      | 0.25886817    | +0.20843018   | INTERPRETABLE    |
| Increasing | prior_valid_amount_count        | 63            | +0.023817495  | INTERPRETABLE    |
| Increasing | TransactionAmt                  | 122.25        | +0.0043534122 | INTERPRETABLE    |
| Increasing | merchant_seen_before            | 1             | +0.0022914362 | INTERPRETABLE    |
| Decreasing | failed_attempts_before          | 0             | -0.061248179  | INTERPRETABLE    |
| Decreasing | prior_transaction_count         | 63            | -0.019841378  | INTERPRETABLE    |
| Decreasing | transactions_prior_24h          | 1             | -0.014104795  | INTERPRETABLE    |
| Decreasing | behavioural_history_count       | 63            | -0.013277259  | INTERPRETABLE    |
| Decreasing | billing_region_seen_before      | 1             | -0.0089875168 | INTERPRETABLE    |

#### recovered_international_anomaly (58423)

| Direction  | Feature                         | Encoded value | SHAP          | Interpretability |
| ---------- | ------------------------------- | ------------- | ------------- | ---------------- |
| Increasing | is_international                | 1             | +0.55450153   | INTERPRETABLE    |
| Increasing | amount_ratio_to_prior_mean      | 2.156909      | +0.11788368   | INTERPRETABLE    |
| Increasing | merchant_seen_before            | 1             | +0.0065717751 | INTERPRETABLE    |
| Increasing | prior_amount_std                | 1685.6703     | +0.0042238412 | INTERPRETABLE    |
| Increasing | prior_median_amount             | 2285.83       | +0.0036547713 | INTERPRETABLE    |
| Decreasing | time_since_previous_transaction | 189050        | -0.13357657   | INTERPRETABLE    |
| Decreasing | failed_attempts_before          | 0             | -0.0318674    | INTERPRETABLE    |
| Decreasing | TransactionAmt                  | 5627.43       | -0.012017068  | INTERPRETABLE    |
| Decreasing | prior_min_amount                | 705.46        | -0.0099672117 | INTERPRETABLE    |
| Decreasing | amount_minus_prior_mean         | 3018.4047     | -0.0092604028 | INTERPRETABLE    |

#### shopping_festival_false_positive (14302)

| Direction  | Feature                         | Encoded value | SHAP          | Interpretability |
| ---------- | ------------------------------- | ------------- | ------------- | ---------------- |
| Increasing | time_since_previous_transaction | 12            | +0.50664611   | INTERPRETABLE    |
| Increasing | prior_amount_std                | 8436.7178     | +0.1061517    | INTERPRETABLE    |
| Increasing | relative_time_position          | 0.59787616    | +0.0081153925 | INTERPRETABLE    |
| Increasing | amount_zscore_prior             | 0.62036726    | +0.0029895588 | INTERPRETABLE    |
| Increasing | prior_median_amount             | 2925.13       | +0.0016928827 | INTERPRETABLE    |
| Decreasing | failed_attempts_before          | 0             | -0.045867412  | INTERPRETABLE    |
| Decreasing | prior_min_amount                | 1479.05       | -0.039223691  | INTERPRETABLE    |
| Decreasing | TransactionAmt                  | 10359.15      | -0.012440227  | INTERPRETABLE    |
| Decreasing | amount_minus_prior_mean         | 5233.8634     | -0.0092929583 | INTERPRETABLE    |
| Decreasing | billing_region_seen_before      | 1             | -0.0089604608 | INTERPRETABLE    |

#### normal_legitimate_low_risk (86722)

| Direction  | Feature                         | Encoded value | SHAP           | Interpretability        |
| ---------- | ------------------------------- | ------------- | -------------- | ----------------------- |
| Increasing | device_id__train_frequency      | 5.7142857e-05 | +0.0036111973  | PARTIALLY_INTERPRETABLE |
| Increasing | relative_time_position          | 0.59268596    | +0.0022635184  | INTERPRETABLE           |
| Increasing | merchant_seen_before            | 1             | +0.0021428955  | INTERPRETABLE           |
| Increasing | merchant_category_v:healthcare  | 0             | +0.0015198218  | INTERPRETABLE           |
| Increasing | product_code_v:electronics      | 0             | +0.00077750858 | INTERPRETABLE           |
| Decreasing | time_since_previous_transaction | 82061         | -0.3036369     | INTERPRETABLE           |
| Decreasing | amount_ratio_to_prior_mean      | 0.73637573    | -0.083982421   | INTERPRETABLE           |
| Decreasing | failed_attempts_before          | 0             | -0.044469071   | INTERPRETABLE           |
| Decreasing | billing_region_seen_before      | 1             | -0.026721763   | INTERPRETABLE           |
| Decreasing | TransactionAmt                  | 9939.5        | -0.010002801   | INTERPRETABLE           |

#### remaining_fraud_false_negative (10604)

| Direction  | Feature                         | Encoded value | SHAP          | Interpretability |
| ---------- | ------------------------------- | ------------- | ------------- | ---------------- |
| Increasing | amount_ratio_to_prior_mean      | 5.6856488     | +0.4537697    | INTERPRETABLE    |
| Increasing | amount_minus_prior_mean         | 15079.005     | +0.10113902   | INTERPRETABLE    |
| Increasing | prior_median_amount             | 3586.4        | +0.070040907  | INTERPRETABLE    |
| Increasing | product_code_v:electronics      | 0             | +0.01023835   | INTERPRETABLE    |
| Increasing | merchant_seen_before            | 1             | +0.0075552862 | INTERPRETABLE    |
| Decreasing | time_since_previous_transaction | 36502         | -0.12006822   | INTERPRETABLE    |
| Decreasing | failed_attempts_before          | 0             | -0.02234922   | INTERPRETABLE    |
| Decreasing | TransactionAmt                  | 18297.13      | -0.014477056  | INTERPRETABLE    |
| Decreasing | billing_region_seen_before      | 1             | -0.0090778303 | INTERPRETABLE    |
| Decreasing | account_age_days                | 2835          | -0.0065728485 | INTERPRETABLE    |

All 11 hard-negative and 98 false-negative records retain both directions and their
original evidence. Machine-readable error summaries provide top-K factor recurrence by
scenario/type. Failed attempts decreases tree output in 95/98 missed records, including
41/41 velocity bursts and 22/22 low-and-slow misses. Time gap increases it in all these
velocity/low-and-slow misses; supporting evidence alone does not guarantee a positive
calibrated/fused decision. SHAP does not explain the deterministic behavioural engine or
prove intent, causation or guilt. Original temporal test scoring: **NO**.
