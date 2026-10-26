# ABRF validation tables

**Synthetic-data findings only.**
Dataset `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`;
feature pipeline `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`; split `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`.
Version `abrf-v1`. Only the existing 70,000 TRAIN / 15,000 VALIDATION score artifacts
are decoded. The original temporal test is not used or scored. All component bytes,
normalization snapshots, versions and thresholds are verified before and after research.
No Decision Tree, Isolation Forest or behavioural component is fitted or modified.

## Component ablations

| Variant                               | Precision | Recall   | F1       | AP          | FPR     | Hard-negative FPR | Threshold           |
| ------------------------------------- | --------- | -------- | -------- | ----------- | ------- | ----------------- | ------------------- |
| Supervised (policy-optimized control) | 82.5871%  | 62.4060% | 0.710921 | 0.713196    | 0.2375% | 1.6035%           | 0.2954875597718248  |
| Supervised + Anomaly                  | 82.9897%  | 60.5263% | 0.700000 | 0.744180    | 0.2240% | 1.7493%           | 0.34940769388415355 |
| Supervised + Behaviour                | 82.3529%  | 63.1579% | 0.714894 | 0.745857    | 0.2443% | 1.6035%           | 0.29755749437222373 |
| Full ABRF                             | 82.5000%  | 62.0301% | 0.708155 | 0.737926    | 0.2375% | 2.1866%           | 0.3627480762421854  |
| Naive OR baseline                     | 65.9919%  | 61.2782% | 0.635478 | N/A: binary | 0.5701% | 10.9329%          | 0.5                 |
| Frozen supervised comparator          | 90.3030%  | 56.0150% | 0.691415 | 0.713196    | 0.1086% | 1.3120%           | 0.3665406264377367  |

P-only optimization is a fair control; the historical Phase 4 threshold remains unchanged.
Naive OR ranking AP is unavailable because it emits only a binary union, not an R score.

## Scenario recall

| Scenario                      | N   | Frozen supervised recall | Frozen anomaly recall | Selected ABRF recall |
| ----------------------------- | --- | ------------------------ | --------------------- | -------------------- |
| AMOUNT_SPIKE                  | 9   | 44.4444%                 | 0.0000%               | 66.6667%             |
| VELOCITY_BURST                | 42  | 2.3810%                  | 23.8095%              | 2.3810%              |
| NEW_DEVICE_HIGH_AMOUNT        | 11  | 45.4545%                 | 27.2727%              | 45.4545%             |
| REGION_CHANGE                 | 14  | 78.5714%                 | 0.0000%               | 78.5714%             |
| NEW_MERCHANT_AMOUNT_DEVIATION | 19  | 47.3684%                 | 0.0000%               | 47.3684%             |
| REPEATED_FAILED_ATTEMPTS      | 10  | 70.0000%                 | 0.0000%               | 80.0000%             |
| ACCOUNT_TAKEOVER              | 41  | 100.0000%                | 100.0000%             | 100.0000%            |
| LOW_AND_SLOW                  | 55  | 41.8182%                 | 0.0000%               | 60.0000%             |
| INTERNATIONAL_ANOMALY         | 13  | 7.6923%                  | 0.0000%               | 53.8462%             |
| MIXED_MODE                    | 52  | 90.3846%                 | 80.7692%              | 90.3846%             |

## Hard-negative results and contribution crossings

| Hard-negative type    | N   | Frozen P flags | ABRF flags | ABRF FPR | P/A/B crossing allocations                      |
| --------------------- | --- | -------------- | ---------- | -------- | ----------------------------------------------- |
| LAPTOP_PURCHASE       | 92  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| TRAVEL_PURCHASE       | 89  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| MEDICAL_EXPENSE       | 75  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| NEW_LEGITIMATE_DEVICE | 90  | 1              | 3          | 3.3333%  | {'supervised': 3, 'anomaly': 0, 'behaviour': 0} |
| NEW_MERCHANT_PURCHASE | 78  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| SHOPPING_FESTIVAL     | 262 | 8              | 8          | 3.0534%  | {'supervised': 8, 'anomaly': 0, 'behaviour': 0} |

## Improvement accounting

```json
{
  "additional_fraud_recovered": 19,
  "previously_detected_fraud_lost": 0,
  "net_fraud_detection_change": 19,
  "additional_false_positives": 20,
  "previous_false_positives_removed": 0,
  "net_false_positive_change": 20,
  "net_hard_negative_false_positive_change": 2,
  "scenarios_improved": [
    "AMOUNT_SPIKE",
    "REPEATED_FAILED_ATTEMPTS",
    "LOW_AND_SLOW",
    "INTERNATIONAL_ANOMALY"
  ],
  "scenarios_degraded": []
}
```

## Interpretation against the fair control

**Synthetic-data finding:** selected weights are **0.9 / 0 / 0.1**. The zero anomaly
weight is a measured selection result, not a claim that anomaly detection is universally
unhelpful. The default does not preserve Phase 5's velocity-burst recovery: selected ABRF
detects 1/42, matching the frozen supervised comparator, versus anomaly 10/42.

The frozen supervised operating point detects 149 fraud rows with 16 false positives.
Threshold-optimized P-only detects 166 with 35 false positives. Selected ABRF detects
168 with 36 false positives: the marginal behavioural gain over the fair control is
**two additional fraud cases at one additional false positive**, not 19 cases attributable
solely to behavioural fusion. Precision declines from 90.30% at the frozen P threshold to
82.35% at the selected ABRF operating point; improved recall/F1 is an explicit tradeoff.
An independent future holdout is needed before interpreting this modest difference as robust.

Supervised-dominant and wp>=0.4 filters select the same best full three-signal candidate
as the unconstrained study. They showed no measured validation improvement in this run.

## Completed verification

**436 offline tests passed**, including **50 Phase 6 cases**. Frontend lint, typecheck,
production build and Prettier format checks passed. Python Ruff lint/format, compileall,
pip check and imports of **108 API/ML modules** passed. The small synthetic fusion smoke,
full bounded search, configuration reload, cached replay, six actual API report endpoints,
health/OpenAPI, path privacy and finite JSON checks passed.

All **85,000** exported development traces were checked for contribution sums and
unavailable-behaviour fallback. Nullable cold-start conversion and exact floating-point
round trips have regression coverage. The preliminary search and interrupted trace-export
attempt remain archived; no historical results were silently overwritten. All eight Phase 5
files and the Phase 4 test-access journal remained byte-identical. No component was retrained.
Original temporal test accessed for ABRF fitting/tuning/scoring: **NO**.
Future holdout: **PREPARATION_ONLY_NOT_GENERATED**.

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
```

## Selection and reproducibility

Configuration: `04328ed2bb150369`, version `abrf-v1`, policy `BALANCED`.
Base weights `{'wp': 0.9, 'wa': 0.0, 'wb': 0.1}`, confidence `gated_0.5`, threshold
`0.29755749437222373`. Risk boundaries `[0.001191118506145397, 0.027236456910269238, 0.44883536556906983, 0.9401087692599148]`.
Highest validation policy objective across all component ablations; ties use average precision, precision, recall, hard-negative/legitimate FPR and simplicity. Same budgets and frozen strong-scenario detection floors apply to every variant. No requirement forces nonzero anomaly/behaviour weight when evidence does not support it.

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.abrf `
  --config ml/config/synthetic_riskbyte.example.json `
  --search-config ml/config/abrf.example.json
```

Original temporal test accessed for ABRF fitting/tuning/scoring: **NO**.
Opaque full-source hashing verifies provenance without decoding original transaction records.
Fresh final holdout: **not generated or evaluated**. Models and component thresholds unchanged.
All figures are synthetic validation development observations; validation reuse, generator
bias, dependence and incomplete context prevent production banking claims.
