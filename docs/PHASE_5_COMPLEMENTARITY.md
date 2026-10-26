# Phase 5 supervised complementarity

**Synthetic-data finding.** Dataset `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`; feature pipeline `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`; temporal split `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`. Development contains 70,000 TRAIN and 15,000 VALIDATION rows. The original temporal test partition was not decoded into records, scored, or used for Phase 5 decisions. Full files are read as opaque bytes for checksums; identity projection checks excluded numeric join keys without decoding excluded context fields.

## Frozen supervised reference

Decision Tree `decision-tree-459fbcf71e8c2360` remains byte-identical, with unchanged
sigmoid calibration and threshold `0.3665406264377367`. Phase 4 registry,
selection history and test results are never rewritten. Existing test blind spots motivate
research questions only; no old test records inform new thresholds or normalization.
P is the supervised fraud probability; A global anomaly and B entity deviation are separate.

## What anomaly detection recovers

Validation fraud contingency: supervised only **67**, anomaly
only **14**, both **82**, neither
**103**. The classifier detects **149**
of 266 fraud rows. Analysis-only OR covers **163**,
recovering **14** additional fraud rows while adding
**68** false positives. This tradeoff does NOT justify
deploying an OR rule. No operational ensemble/fusion logic exists.

| Fraud scenario                | N   | Supervised recall | Anomaly recall | Median B            |
| ----------------------------- | --- | ----------------- | -------------- | ------------------- |
| AMOUNT_SPIKE                  | 9   | 44.4444%          | 0.0000%        | 0.32431143868586665 |
| VELOCITY_BURST                | 42  | 2.3810%           | 23.8095%       | 0.4213823852057754  |
| NEW_DEVICE_HIGH_AMOUNT        | 11  | 45.4545%          | 27.2727%       | 0.3316739120199223  |
| REGION_CHANGE                 | 14  | 78.5714%          | 0.0000%        | 0.22913918094406543 |
| NEW_MERCHANT_AMOUNT_DEVIATION | 19  | 47.3684%          | 0.0000%        | 0.4020853930466294  |
| REPEATED_FAILED_ATTEMPTS      | 10  | 70.0000%          | 0.0000%        | 0.2412028983702692  |
| ACCOUNT_TAKEOVER              | 41  | 100.0000%         | 100.0000%      | 0.7342754529788764  |
| LOW_AND_SLOW                  | 55  | 41.8182%          | 0.0000%        | 0.4089103594496086  |
| INTERNATIONAL_ANOMALY         | 13  | 7.6923%           | 0.0000%        | 0.23413090652242471 |
| MIXED_MODE                    | 52  | 90.3846%          | 80.7692%       | 0.7050648240498252  |

## Dedicated validation investigations

**VELOCITY_BURST**: anomaly flags 10/42; supervised flags 1/42. Anomaly-score Cliff's delta versus normal legitimate is 0.948030; B delta is 0.904877. Median B is 0.421382 (analysis threshold 0.482368).

**LOW_AND_SLOW**: anomaly flags 0/55; supervised flags 23/55. Anomaly-score Cliff's delta versus normal legitimate is 0.566498; B delta is 0.920175. Median B is 0.408910 (analysis threshold 0.482368).

**INTERNATIONAL_ANOMALY**: anomaly flags 0/13; supervised flags 1/13. Anomaly-score Cliff's delta versus normal legitimate is 0.956287; B delta is 0.213532. Median B is 0.234131 (analysis threshold 0.482368).

Ranking separation does not imply adequate threshold recall under hard-negative constraints.
Longer-window and relative amount evidence remains available for later frozen-component
ablations, with no special scenario-aware fraud rule. Global unusualness can overlap hard
negatives. Travel hard negatives are separately retained; international activity is not
fraud and is not a documented entity-specific international novelty signal.

All focused raw-signal and A/B quantiles/effects are saved under `investigations` in the bundle.
These are descriptive synthetic effects without independent population inference.

## Hard-negative protection and overlap

| Legitimate hard-negative type | N   | Anomaly flags | Anomaly rate | Behaviour flags | Behaviour rate       |
| ----------------------------- | --- | ------------- | ------------ | --------------- | -------------------- |
| LAPTOP_PURCHASE               | 92  | 5             | 5.4348%      | 11              | 0.11956521739130435  |
| TRAVEL_PURCHASE               | 89  | 0             | 0.0000%      | 1               | 0.011235955056179775 |
| MEDICAL_EXPENSE               | 75  | 9             | 12.0000%     | 9               | 0.12                 |
| NEW_LEGITIMATE_DEVICE         | 90  | 1             | 1.1111%      | 8               | 0.08888888888888889  |
| NEW_MERCHANT_PURCHASE         | 78  | 0             | 0.0000%      | 1               | 0.01282051282051282  |
| SHOPPING_FESTIVAL             | 262 | 53            | 20.2290%     | 38              | 0.1450381679389313   |

Hard-negative A median **0.917447** overlaps
fraud A median **0.934371**; normal legitimate A median
is **0.477002**. B cohort distributions and
confidence/quantiles appear in the behavioural report. Large/shopping-burst legitimate
purchases can resemble attacks. Overall 10% hard-negative caps do not ensure subtype parity.

## Artifacts and audit

`ml/artifacts/phase5/` contains two executable engine objects, anomaly/behaviour metadata,
coherent analysis bundle, complementarity and leakage-audit JSON, and an 85,000-row
evaluation-only development score sidecar. Separate TRAIN diagnostics use validation-frozen
thresholds and are not unbiased generalization estimates. All source/artifact versions and
hashes remain bound to the original frozen experiment. Test rows decoded/scored: **0**.
Opaque full-source checksum I/O and excluded identity join-key membership checks are necessary
integrity/projection operations, not test feature/label analysis. The isolation tests enforce
bounded record decoding, metadata exclusion, training-only mappings and cold-start semantics.

## Completed verification

**386 offline tests passed**, including **67 Phase 5 cases**. Frontend lint, typecheck,
production build and Prettier format verification passed. Python Ruff lint/format,
compilation, dependency verification and imports of **99 API/ML modules** passed.
The small synthetic integration smoke, full TRAIN/VALIDATION experiment, exact engine
serialization/reload, cached-run reuse, seven actual report endpoints, health/OpenAPI,
JSON finiteness and path privacy checks passed. The Phase 4 test-access journal checksum
remained unchanged. Original temporal test records decoded/scored for this phase: **zero**.

**Synthetic-data finding:** the 14 additional validation detections comprised **9 velocity
burst**, **4 mixed-mode**, and **1 new-device/high-amount** transaction. These counts were
verified from the checksum-bound development score sidecar, using round-trip float parsing.
Later ablations must verify the sidecar hash and use `float_precision="round_trip"` when
reading its CSV probabilities/scores, to preserve decisions exactly at saved thresholds.

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
.\.venv\Scripts\python.exe -m riskbyte_ml.anomaly `
  --config ml/config/synthetic_riskbyte.example.json `
  --experiment-config ml/config/anomaly.example.json `
  --behaviour-config ml/config/behaviour.example.json
```

## Limitations and next phase

These are synthetic development findings. Validation reuse, episode/entity dependence,
few scenario observations, engineered learnability and component correlations limit claims.
No production banking performance or calibrated A/B fraud probabilities can be inferred.
Phase 6 should compare classifier-only against confidence-aware P/A/B ablations, explicit
missing-signal policies and constrained calibrated fusion using TRAIN/VALIDATION only.
Start from hard-negative budgets; anomaly recovers only modest incremental coverage at a
substantial added false-positive cost. Freeze all components/weights/thresholds before
creating/evaluating a fresh future holdout under FINAL_HOLDOUT_PLAN.md. Phase 6 is not implemented.
