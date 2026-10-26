# ABRF research

**Synthetic-data findings only.**
Dataset `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`;
feature pipeline `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`; split `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`.
Version `abrf-v1`. Only the existing 70,000 TRAIN / 15,000 VALIDATION score artifacts
are decoded. The original temporal test is not used or scored. All component bytes,
normalization snapshots, versions and thresholds are verified before and after research.
No Decision Tree, Isolation Forest or behavioural component is fitted or modified.

## Motivation and exact formula

Adaptive Behavioural Risk Fusion is a RiskByte-specific, transparent fusion methodology,
not a claim of universally optimal or novel mathematics. P is the frozen calibrated
supervised probability; A is global anomaly and B entity-relative deviation. A/B/R are NOT
fraud probabilities. Phase 4's frozen tree is precise but misses some velocity/weak-signal
patterns. Phase 5's analysis recovered 14 additional validation fraud rows at 68 added false
positives. High deviation alone cannot justify a fraud decision.

Let `g(C)` be the selected confidence transform and `I_B` behaviour availability.
`d = wp + wa + wb * I_B * g(C_b)`.
`R = (wp*P + wa*A + wb*I_B*g(C_b)*B) / d`.
For unavailable B the behavioural term is omitted, not evaluated with an assumed safe B=0.
Effective weights are `(wp/d, wa/d, wb*I_B*g(C_b)/d)`; contributions multiply those weights
by the respective available signals. Their sum is R. Weights are finite/nonnegative,
sum to 1 and retain positive P/A fallback mass. P/A are required; no undocumented missing-P/A
fallback exists. If C_b=0, B has no influence even when a score exists. C_b=1 gives full base
behavioural weight. Dynamic weighting responds to reliability, not transaction outcomes.

## Confidence and bounded search

Compare linear `g(C)=C`, and gates `g(C)=C if C>=0.5 else 0` / `C>=0.75`.
There is no neural/meta-classifier or behavioural-component retuning. Confidence is the
unchanged Phase 5 count/availability measure, not calibrated confidence in a fraud claim.
All grid candidates and confidence rules are specified before inspecting Phase 6 results.
Grid divisions: 10; each P weight is positive.
Zero A/B weights are eligible so ablations can reject a component whose value is unsupported.
Evaluated **55** weight combinations,
**145** weight/curve candidates,
**1,977,017** exact threshold locations and
**435** candidate-policy configurations.
All threshold metrics, including AP/FPR/FNR/hard-negative FPR/TP/FP/FN, are stored in
compressed per-candidate CSV files rather than huge API payloads. AP is constant across
thresholds for a given ranking and means non-interpolated average precision.

## Operating policies, fairness and selection

BALANCED maximizes F1 under FPR≤1%, hard-negative FPR≤5%.
HIGH_RECALL maximizes recall under FPR≤2%, hard-negative FPR≤10%.
STRICT minimizes false positives with recall≥60%, FPR≤0.5%, hard-negative FPR≤3%.
Phase 6 deliberately tightens BALANCED hard-negative control versus earlier 10% conventions.
Every variant uses the same validation rows, budgets, confidence choices where applicable
and exact threshold search. Protected scenario floors require at least the frozen P detections
for account takeover/mixed mode, identically across variants. Other regressions are reported.
The default policy **BALANCED** is declared before search, to favor precision/recall balance
and stronger hard-negative control. Candidate ranking uses its policy objective, then AP,
precision, recall, subgroup/overall FPR and simpler confidence logic; accuracy is not primary.
Scenario metadata controls offline evaluation/protection only and never enters live fusion.

The P-only policy threshold is a Phase 6 control, NOT a change to Phase 4's frozen threshold.
Both are reported. Frozen binary OR uses the unchanged P/A thresholds and has no meaningful
continuous ranking AP; its AP is deliberately unavailable rather than relabelled.

| Variant                               | Precision | Recall   | F1       | AP          | FPR     | Hard-negative FPR | Threshold           |
| ------------------------------------- | --------- | -------- | -------- | ----------- | ------- | ----------------- | ------------------- |
| Supervised (policy-optimized control) | 82.5871%  | 62.4060% | 0.710921 | 0.713196    | 0.2375% | 1.6035%           | 0.2954875597718248  |
| Supervised + Anomaly                  | 82.9897%  | 60.5263% | 0.700000 | 0.744180    | 0.2240% | 1.7493%           | 0.34940769388415355 |
| Supervised + Behaviour                | 82.3529%  | 63.1579% | 0.714894 | 0.745857    | 0.2443% | 1.6035%           | 0.29755749437222373 |
| Full ABRF                             | 82.5000%  | 62.0301% | 0.708155 | 0.737926    | 0.2375% | 2.1866%           | 0.3627480762421854  |
| Naive OR baseline                     | 65.9919%  | 61.2782% | 0.635478 | N/A: binary | 0.5701% | 10.9329%          | 0.5                 |
| Frozen supervised comparator          | 90.3030%  | 56.0150% | 0.691415 | 0.713196    | 0.1086% | 1.3120%           | 0.3665406264377367  |

## Supervised-priority and confidence comparisons

The full three-signal candidates are compared unrestricted, with `wp>=wa,wp>=wb`, and
with `wp>=0.4`, using the same policy. These filters do not receive separate special tuning.
No constraint is claimed to improve robustness unless observed measurements support it.

| Study                | Weights                           | Confidence | F1       | AP       | Hard-negative FPR |
| -------------------- | --------------------------------- | ---------- | -------- | -------- | ----------------- |
| unconstrained        | {'wp': 0.7, 'wa': 0.1, 'wb': 0.2} | gated_0.75 | 0.708155 | 0.737926 | 2.1866%           |
| supervised_dominant  | {'wp': 0.7, 'wa': 0.1, 'wb': 0.2} | gated_0.75 | 0.708155 | 0.737926 | 2.1866%           |
| supervised_floor_0.4 | {'wp': 0.7, 'wa': 0.1, 'wb': 0.2} | gated_0.75 | 0.708155 | 0.737926 | 2.1866%           |

| Study      | Weights                           | Confidence | F1       | AP       | Hard-negative FPR |
| ---------- | --------------------------------- | ---------- | -------- | -------- | ----------------- |
| linear     | {'wp': 0.8, 'wa': 0.1, 'wb': 0.1} | linear     | 0.701903 | 0.753121 | 2.6239%           |
| gated_0.5  | {'wp': 0.7, 'wa': 0.1, 'wb': 0.2} | gated_0.5  | 0.702586 | 0.751989 | 2.9155%           |
| gated_0.75 | {'wp': 0.7, 'wa': 0.1, 'wb': 0.2} | gated_0.75 | 0.708155 | 0.737926 | 2.1866%           |

Other policy operating points:

| Study       | Weights                           | Confidence | F1       | AP       | Hard-negative FPR |
| ----------- | --------------------------------- | ---------- | -------- | -------- | ----------------- |
| BALANCED    | {'wp': 0.9, 'wa': 0.0, 'wb': 0.1} | gated_0.5  | 0.714894 | 0.745857 | 1.6035%           |
| HIGH_RECALL | {'wp': 0.7, 'wa': 0.0, 'wb': 0.3} | linear     | 0.673913 | 0.754979 | 9.1837%           |
| STRICT      | {'wp': 0.9, 'wa': 0.0, 'wb': 0.1} | gated_0.5  | 0.706402 | 0.745857 | 1.3120%           |

## Frozen final development configuration

```json
{
  "version": "abrf-v1",
  "weights": {
    "wp": 0.9,
    "wa": 0.0,
    "wb": 0.1
  },
  "confidence_curve": "gated_0.5",
  "policy": "BALANCED",
  "decision_threshold": 0.29755749437222373,
  "risk_boundaries": [
    0.001191118506145397, 0.027236456910269238, 0.44883536556906983, 0.9401087692599148
  ],
  "reason_evidence_cutoff": 0.95,
  "limited_confidence_cutoff": 0.5
}
```

Highest validation policy objective across all component ablations; ties use average precision, precision, recall, hard-negative/legitimate FPR and simplicity. Same budgets and frozen strong-scenario detection floors apply to every variant. No requirement forces nonzero anomaly/behaviour weight when evidence does not support it.

Selected precision 82.3529%, recall 63.1579%, F1 0.714894,
AP 0.745857, ROC-AUC 0.957140, FPR 0.2443%,
hard-negative FPR 1.6035%. Confusion: {'tp': 168, 'fp': 36, 'tn': 14698, 'fn': 98}.
Compared with frozen P: 19 newly recovered fraud rows,
0 previously detected fraud lost,
net fraud change 19; 20 added
false positives, 0 previous false positives removed,
net FP change 20, net hard-negative FP change 2.

## Risk bands, trace and calibration

Risk boundaries are TRAIN fused-score quantiles [0.5, 0.9, 0.99, 0.999],
without label use: **[0.001191118506145397, 0.027236456910269238, 0.44883536556906983, 0.9401087692599148]**. LOW/GUARDED/ELEVATED/HIGH/CRITICAL bins are
left-inclusive at internal boundaries; R=1 is CRITICAL. The investigation threshold is a
separate validation operating choice; a risk category is not a statement of guilt.
Quantile risk bands are development categories, not validated banking risk bands.

`FusionInput` excludes all target/scenario/hard-negative fields. `fuse` exposes validated
inputs, base/effective weights, contributions, R, category, threshold, flag and deterministic
evidence codes. Optional novelty/velocity/amount reason codes require supplied available
structured component evidence. The existing score sidecar lacks these fine-grained signals,
so research traces do not invent them. Contributions and subtractive crossing allocations
are calculation explanations, not SHAP or causal attribution. Multiple components can be
necessary for a crossing; counts need not be mutually exclusive.

No calibration is applied to R. A strictly monotonic calibration preserves ranking and
available threshold decisions, so it cannot improve AP/recall-FPR tradeoffs by itself.
Isotonic ties can change ranking and require a separately justified calibration protocol.
No empirical Brier improvement is claimed. R remains a normalized decision-support score;
future probability claims would require independent calibration evidence.

## Behaviour-only support and scenario/hard-negative findings

Frozen P-negative/A-negative/B-high support: {'definition': 'frozen P negative, frozen A negative, frozen B high', 'fraud': {'count': 266, 'flagged': 20, 'rate': 0.07518796992481203}, 'legitimate': {'count': 14734, 'flagged': 120, 'rate': 0.00814442785394326}, 'hard_negative': {'count': 686, 'flagged': 47, 'rate': 0.06851311953352769}, 'fraud_scenarios': {'AMOUNT_SPIKE': 0, 'VELOCITY_BURST': 6, 'NEW_DEVICE_HIGH_AMOUNT': 0, 'REGION_CHANGE': 0, 'NEW_MERCHANT_AMOUNT_DEVIATION': 3, 'REPEATED_FAILED_ATTEMPTS': 0, 'ACCOUNT_TAKEOVER': 0, 'LOW_AND_SLOW': 11, 'INTERNATIONAL_ANOMALY': 0, 'MIXED_MODE': 0}}.
Signals overlap unusual legitimate transactions, so high B does not automatically imply fraud.

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

| Hard-negative type    | N   | Frozen P flags | ABRF flags | ABRF FPR | P/A/B crossing allocations                      |
| --------------------- | --- | -------------- | ---------- | -------- | ----------------------------------------------- |
| LAPTOP_PURCHASE       | 92  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| TRAVEL_PURCHASE       | 89  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| MEDICAL_EXPENSE       | 75  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| NEW_LEGITIMATE_DEVICE | 90  | 1              | 3          | 3.3333%  | {'supervised': 3, 'anomaly': 0, 'behaviour': 0} |
| NEW_MERCHANT_PURCHASE | 78  | 0              | 0          | 0.0000%  | {'supervised': 0, 'anomaly': 0, 'behaviour': 0} |
| SHOPPING_FESTIVAL     | 262 | 8              | 8          | 3.0534%  | {'supervised': 8, 'anomaly': 0, 'behaviour': 0} |

Improved scenarios: ['AMOUNT_SPIKE', 'REPEATED_FAILED_ATTEMPTS', 'LOW_AND_SLOW', 'INTERNATIONAL_ANOMALY']. Degraded scenarios: [].
International separability is described using saved validation P/A/B/confidence distributions;
no international-fraud rule is added. Low-and-slow and velocity results must be read alongside
global/hard-negative costs, not selected as isolated success measures.

## Reproducibility, API, final holdout and limitations

Ignored `ml/artifacts/abrf/<run-id>/` stores selected configuration, provenance/selection
metadata, all ranked candidates, threshold CSVs, development scores, a computed trace and
coherent report bundle. Exclusive writes preserve history; changed searches cannot silently
replace the active selected configuration. Cached runs verify all checksums without refitting.
GET `/api/v1/abrf/{status,config,validation,ablation,scenarios,hard-negatives}` serves
read-only typed JSON reports, never a public prediction API or raw transaction records.

Fresh holdout status: **PREPARATION_ONLY_NOT_GENERATED**. Generation was explicitly deferred;
current generator lacks reviewed stateful future continuation. FINAL_HOLDOUT_PLAN.md remains
the governing protocol. No fresh dataset is generated or scored in this phase.

Synthetic data cannot prove production banking performance. Generator rules affect learnability;
confidence is provisional; P/A/B share correlated signals; validation was already reused in
earlier phases and is reused for this search, so results are development findings, not unbiased
generalization. Scenario counts are small and episodes/entities dependent. Overall subgroup
caps do not guarantee every subtype's rate. Missing-history fallback and shift in confidence
or score percentiles need fresh future stress testing. ABRF is project-specific, not universally
optimal. No SHAP, final inference endpoint, UI redesign or Phase 7 implementation occurs.
