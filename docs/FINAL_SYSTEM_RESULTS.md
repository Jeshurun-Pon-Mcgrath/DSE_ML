# Final system results — canonical performance source

All evaluation records are **synthetic academic data**. This file is the only canonical
source for final frontend performance pages, project report, PPT, viva and demo metrics.
Historical Phase 3–7 development metrics must not be presented as final performance.
Exact machine-readable values are the checksum-locked final_holdout artifacts; displayed
values below are rounded. Live/dashboard counts are separate operational demo counts.

## Frozen system and one-use protocol

Generator synthetic-riskbyte-v1 with stateful-continuation-v1; seed 2026. All classifier,
sigmoid calibration, preprocessing, Isolation Forest normalization, behaviour components,
ABRF settings and risk bands remain unchanged. Development fingerprint:
sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66.
Feature version: synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1.
Explainability: explainability-v1.1-shap / SHAP 0.53.0.
ABRF weights P=0.9, A=0, B=0.1; linear behaviour contribution only when confidence>=0.5;
investigation threshold 0.29755749437222373. Supervised historical threshold remains
0.3665406264377367. No threshold was selected using this holdout.

READY_UNTOUCHED was durably recorded before an exclusive evaluation-start journal.
Component scores/decisions were saved before evaluation labels were opened. Evaluation
ran once and transitioned to FINAL_EVALUATED_LOCKED; there is no reset/retry/tune path.
Subsequent commands verify cached artifacts without repeating scoring/evaluation.
Only nine selected cases received SHAP. This is now an observed final evaluation,
not an untouched reserve for any further model changes.

Holdout fingerprint: `sha256:20e72b996b4c38d3f28f70e4b0b2826a053dbb7c3deebaa2e5e3897c3df4fe03`.

Transactions **30,000**, identity rows **21,000**, joined rows **30,000**.
Fraud **600 (2.00%)**; hard negatives **1,500**.
Elapsed TransactionDT range **7776513–10142412 seconds**, beyond the original 90-day horizon.

Continuity preserves 1,600 existing profiles and introduces 400 new observed entities,
known/new devices and a merchant pool of 360 (60 added). History is the frozen 85,000-row
TRAIN/VALIDATION checkpoint plus strictly prior future events. Original temporal test
events are not decoded or rescored. The intervening gap is unobserved; missing history is
not claimed to be complete bank history. Attack/hard-negative/identity/missingness logic
is unchanged. The original generator path passes a byte-level regression fixture.

## Final comparison

PR-AUC means non-interpolated average precision. Accuracy is secondary.

| System            | Precision | Recall   | F1       | PR-AUC   | ROC-AUC  | FPR       | Hard-negative FPR |
| ----------------- | --------- | -------- | -------- | -------- | -------- | --------- | ----------------- |
| Frozen Supervised | 0.904762  | 0.538333 | 0.675026 | 0.689682 | 0.914624 | 0.115646% | 0.800000%         |
| Final ABRF        | 0.820755  | 0.580000 | 0.679688 | 0.717291 | 0.948376 | 0.258503% | 1.466667%         |

| System            | TP  | FP  | TN    | FN  | FNR      | Specificity | Secondary accuracy |
| ----------------- | --- | --- | ----- | --- | -------- | ----------- | ------------------ |
| Frozen Supervised | 323 | 34  | 29366 | 277 | 0.461667 | 0.998844    | 0.989633           |
| Final ABRF        | 348 | 76  | 29324 | 252 | 0.420000 | 0.997415    | 0.989067           |

ABRF recovered **25 additional fraud cases**, lost **zero**, and added **42 false positives**.
Hard-negative false positives increased from **12 to 22**. Recall increased by 4.1667
percentage points, precision decreased by 8.4007 points, F1 increased by 0.004661 and
FPR increased by 0.142857 points. This is a detection/false-positive tradeoff, not unqualified
superiority. The comparison includes different frozen operating thresholds; it does not
establish that behavioural contribution alone caused the additional detections.

## Scenario recall

| Scenario                      | Count | Supervised detected | ABRF detected | Supervised recall | ABRF recall | Diagnostic anomaly recall |
| ----------------------------- | ----- | ------------------- | ------------- | ----------------- | ----------- | ------------------------- |
| AMOUNT_SPIKE                  | 29    | 6                   | 6             | 20.6897%          | 20.6897%    | 6.8966%                   |
| VELOCITY_BURST                | 105   | 4                   | 4             | 3.8095%           | 3.8095%     | 20.9524%                  |
| NEW_DEVICE_HIGH_AMOUNT        | 28    | 11                  | 13            | 39.2857%          | 46.4286%    | 17.8571%                  |
| REGION_CHANGE                 | 28    | 21                  | 21            | 75.0000%          | 75.0000%    | 0.0000%                   |
| NEW_MERCHANT_AMOUNT_DEVIATION | 28    | 18                  | 18            | 64.2857%          | 64.2857%    | 3.5714%                   |
| REPEATED_FAILED_ATTEMPTS      | 28    | 16                  | 17            | 57.1429%          | 60.7143%    | 0.0000%                   |
| ACCOUNT_TAKEOVER              | 115   | 115                 | 115           | 100.0000%         | 100.0000%   | 100.0000%                 |
| LOW_AND_SLOW                  | 112   | 40                  | 50            | 35.7143%          | 44.6429%    | 0.0000%                   |
| INTERNATIONAL_ANOMALY         | 28    | 1                   | 13            | 3.5714%           | 46.4286%    | 7.1429%                   |
| MIXED_MODE                    | 99    | 91                  | 91            | 91.9192%          | 91.9192%    | 87.8788%                  |

## Hard negatives

| Type                  | Count | Supervised false positives | ABRF false positives | ABRF FPR |
| --------------------- | ----- | -------------------------- | -------------------- | -------- |
| LAPTOP_PURCHASE       | 170   | 0                          | 0                    | 0.0000%  |
| TRAVEL_PURCHASE       | 169   | 0                          | 1                    | 0.5917%  |
| MEDICAL_EXPENSE       | 169   | 0                          | 0                    | 0.0000%  |
| NEW_LEGITIMATE_DEVICE | 169   | 1                          | 4                    | 2.3669%  |
| NEW_MERCHANT_PURCHASE | 169   | 0                          | 0                    | 0.0000%  |
| SHOPPING_FESTIVAL     | 654   | 11                         | 17                   | 2.5994%  |

## Behaviour and anomaly diagnostics

Median B: normal legitimate **0.196493**, hard negative **0.371988**, fraud **0.452362**.
Mean confidence **0.619922**, median **0.680488**; cold-start percentage **1.3333%**.
Undefined behaviour is excluded from score summaries with availability counts recorded;
new entities retain null B and low confidence, never an assumed zero-risk history.

Isolation Forest diagnostic PR-AUC **0.325220**, fraud recall **39.00%**, legitimate FPR **1.897959%**, hard-negative FPR **11.80%**.
Anomaly ABRF weight/contribution is **zero**. Isolation Forest does not influence active final risk.

## Selected final cases

| Case                   | Transaction | Evaluation ground truth | Final risk | Decision    |
| ---------------------- | ----------- | ----------------------- | ---------- | ----------- |
| true_positive          | 118827      | 1                       | 0.496920   | Investigate |
| true_negative          | 108768      | 0                       | 0.026024   | No flag     |
| false_positive         | 115879      | 0                       | 0.369846   | Investigate |
| false_negative         | 128561      | 1                       | 0.131041   | No flag     |
| hard_negative_rejected | 108768      | 0                       | 0.026024   | No flag     |
| hard_negative_flagged  | 115879      | 0                       | 0.369846   | Investigate |
| LOW_AND_SLOW           | 103007      | 1                       | 0.249851   | No flag     |
| VELOCITY_BURST         | 104865      | 1                       | 0.224724   | No flag     |
| INTERNATIONAL_ANOMALY  | 107337      | 1                       | 0.019922   | No flag     |

These fixed post-evaluation examples include TP/TN/FP/FN, rejected/flagged hard negatives,
low-and-slow, velocity burst and international anomaly. Actual classifier SHAP, behaviour,
anomaly diagnostics and exact fusion traces remain separate. Ground truth is evaluation-only
and never enters live inputs. SHAP explains the underlying tree, not sigmoid calibration
or the entire system, and does not establish causation, guilt or human motivation.

## Compact synthetic shift summary

| Signal                     | Development validation median | Future holdout median |
| -------------------------- | ----------------------------- | --------------------- |
| TransactionAmt             | 2000.205000                   | 1990.840000           |
| transactions_prior_30m     | 0.000000                      | 0.000000              |
| amount_ratio_to_prior_mean | 0.838529                      | 0.835121              |
| failed_attempts_before     | 0.000000                      | 0.000000              |
| behaviour_score            | 0.203477                      | 0.206355              |
| abrf_score                 | 0.014981                      | 0.013712              |

Fraud prevalence changes from 1.7733% in development validation to the configured 2% in
this holdout. The continuation introduces later relative time, older continuing accounts,
new profiles/merchants and cold starts; frozen normalization does not adapt. These are
synthetic descriptive shifts, not new hypothesis tests or evidence of real-world drift.

## API, integrity and limitations

Final evaluation endpoints read locked artifacts only. The dashboard/transaction store
contains actual holdout demo predictions plus later submitted transactions; its live counts
must not replace final evaluation denominators. Analyst state never changes scores or labels.
Components are startup-loaded once; preprocessing remains fitted on original TRAIN only.
No final models, weights, thresholds, calibration or feature semantics were retuned.

Synthetic data cannot prove production banking performance. Generator rules influence
learnability, entities/episodes are dependent, fraud mechanisms remain simplified and new
entity fraud cold starts are limited by generator scheduling. History omits the original
test-period gap. Velocity-burst detection remains weak. Real-world input authenticity,
customer ownership, calibrated population risk and production throughput are not established.
SQLite is tested for development; PostgreSQL needs its driver and reviewed Alembic migration.
Alembic installation was blocked in this environment, so migration execution is not claimed.

Final holdout state: **FINAL_EVALUATED_LOCKED**; evaluation count **1**.

Verification: **539 tests passed, zero skipped**. Ruff lint/format, compileall,
pip check, import verification (129 modules), frontend lint/typecheck/build/format
and live FastAPI smoke tests passed. The dependency lock still matches all 42 pinned
runtime packages. Locked holdout source/report checksums and frozen component integrity
passed; the original temporal test was not scored in this phase. Smoke-test submissions
use an isolated local database and do not alter final evaluation artifacts.

Recommendation: **Proceed directly to premium React frontend implementation.**
No frontend implementation is performed by this phase.
