# Explainable intelligence: Phase 7.1

All findings below concern **synthetic academic records**, not real banking customers.
The active snapshot is **explainability-v1.1-shap**, SHAP **AVAILABLE**.
Historical explainability-v1 remains preserved separately; its unavailable SHAP state is
not rewritten as a successful run.

## Dependencies and frozen inputs

Python 3.12.6, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.0 and SHAP 0.53.0 were
verified. The ML explainability extra pins shap==0.53.0. The existing requirements lock
retains all previous versions and adds only shap==0.53.0, slicer==0.0.8 and tqdm==4.70.1.
It matches `pip freeze --exclude-editable` exactly: 42 pinned packages. No pip upgrade
or unrelated dependency upgrade was performed. Editable ML metadata was refreshed with
`pip install --no-deps --no-build-isolation -e ./ml`.

Dataset: `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`.
Feature version: `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`.
Temporal split: `split-v1:no-time-tie-crossing`, ID
`sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`.
Classifier: `decision-tree-459fbcf71e8c2360`, `supervised-classification-v1`.
The frozen sigmoid calibrator, preprocessor, Phase 5 engines, ABRF pointer, weights,
thresholds and risk bands are unchanged. Source and artifact hashes are checked before
and after generation. Validation P/A/B/confidence and all fusion traces reproduce the
previous results. No model fitting, calibration, feature selection or threshold tuning
occurs in this phase.

## Exact model and SHAP semantics

TreeExplainer explains the **underlying Decision Tree positive-class raw probability**.
Configuration: `feature_perturbation="tree_path_dependent"`, `model_output="raw"`, exact
SHAP with additivity checks. The reference uses frozen TRAIN path counts recorded by the
fitted, class-weighted tree. No explicit background sample is needed; sample size zero,
seed not applicable. Global evaluation covers all 15,000 validation rows deterministically
in batches of 256. No original test or future holdout is used.

The installed SHAP/sklearn combination returns `(samples, 144 transformed features, 2
classes)`, with two expected values. Class order is explicitly checked as [0,1], and class
1 is selected. Compatibility tests also cover legacy lists and 2D arrays, scalar/vector
expected values, invalid dimensions and mismatched ordering. Every model feature index
must match both frozen preprocessing and the manifest. These choices follow the official
[TreeExplainer output contract](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

The sigmoid calibration mapping is **not** included in TreeExplainer attribution.
`raw_tree_probability` / `raw_probability` and `calibrated_supervised_probability` /
`calibrated_probability` are explicit equivalent field pairs. The output states
`explained_model="underlying_decision_tree"` and
`calibrated_probability_shap_explained=false`. SHAP is not an explanation of the complete
RiskByte system.

Expected positive-class value: **0.500000000000048**. Maximum absolute residual across all validation records: **5.80092e-13** against tolerance 1e-7. Expected value + sum(SHAP) reproduces raw tree probability,
not calibrated P. The weighted reference expectation is not population fraud prevalence.

## Global transformed-feature ranking

| Rank | Feature                         | Family                     | Mean absolute SHAP | Interpretability        |
| ---- | ------------------------------- | -------------------------- | ------------------ | ----------------------- |
| 1    | time_since_previous_transaction | relative_time              | 0.288030717        | INTERPRETABLE           |
| 2    | amount_ratio_to_prior_mean      | behavioural_amount_history | 0.080736414        | INTERPRETABLE           |
| 3    | failed_attempts_before          | failed_attempts            | 0.052658089        | INTERPRETABLE           |
| 4    | billing_region_seen_before      | region                     | 0.026170705        | INTERPRETABLE           |
| 5    | is_international                | region                     | 0.018821357        | INTERPRETABLE           |
| 6    | prior_valid_amount_count        | behavioural_amount_history | 0.015586765        | INTERPRETABLE           |
| 7    | amount_minus_prior_mean         | behavioural_amount_history | 0.012949106        | INTERPRETABLE           |
| 8    | behavioural_history_count       | account_history            | 0.007295913        | INTERPRETABLE           |
| 9    | prior_transaction_count         | account_history            | 0.005511017        | INTERPRETABLE           |
| 10   | TransactionAmt                  | transaction_amount         | 0.004814087        | INTERPRETABLE           |
| 11   | prior_median_amount             | behavioural_amount_history | 0.004334699        | INTERPRETABLE           |
| 12   | merchant_seen_before            | merchant                   | 0.004317652        | INTERPRETABLE           |
| 13   | device_id__train_frequency      | device                     | 0.003668750        | PARTIALLY_INTERPRETABLE |
| 14   | prior_mean_amount               | behavioural_amount_history | 0.003401924        | INTERPRETABLE           |
| 15   | transactions_prior_24h          | velocity                   | 0.003335011        | INTERPRETABLE           |

The complete 144-feature ranking preserves exact values, source/transformed feature,
source columns, raw/derived status, original manifest class and transformation. Local
feature values are actual encoded/imputed inputs, not silently reconstructed raw values.
Anonymized fields never receive business semantics. This active feature set contains no
anonymized V-feature family; no fictitious anonymized importance row is added.

## Family aggregation

Family importance is the sum of mean absolute SHAP of its member transformed features.
This is not signed grouped attribution; absolute aggregation can favour larger families.
All underlying feature-level results remain available. Relative time denotes elapsed time,
not dates or clock hour. International status is grouped with region context rather than
account history; this presentation mapping refinement changes no predictive manifest.

| Rank | Family                     | Aggregated mean absolute SHAP |
| ---- | -------------------------- | ----------------------------- |
| 1    | relative_time              | 0.293357791                   |
| 2    | behavioural_amount_history | 0.126358076                   |
| 3    | failed_attempts            | 0.052658089                   |
| 4    | region                     | 0.047838299                   |
| 5    | account_history            | 0.015308915                   |
| 6    | merchant                   | 0.007892840                   |
| 7    | transaction_amount         | 0.004814087                   |
| 8    | device                     | 0.003668750                   |
| 9    | velocity                   | 0.003335011                   |
| 10   | product                    | 0.002808332                   |
| 11   | payment_context            | 0.001504867                   |
| 12   | email_context              | 0.000230183                   |
| 13   | identity                   | 0.000000000                   |

## Phase 3 consistency

This is a qualitative post-hoc comparison, not a new statistical test. Descriptive ranking
cutoffs are recorded in metadata. Marginal associations and conditional model attributions
need not agree: features are correlated, the tree was class-weighted, interactions matter,
and Phase 3 hypotheses use TRAIN while SHAP aggregation uses VALIDATION.

| Phase 3 finding                            | SHAP relationship | Actual interpretation                                                                                                                            |
| ------------------------------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Strong prior 30-minute velocity separation | PARTIALLY_ALIGNED | Prior-event gap is rank 1, but direct 30-minute count is rank 141; time gap is not the same feature as rolling velocity.                         |
| Failed-attempt association                 | ALIGNED           | failed_attempts_before is rank 3.                                                                                                                |
| Device novelty association                 | DIFFERENT         | device_seen_before is rank 76 with negligible contribution; device frequency is rank 13 and must not be called novelty.                          |
| Weak merchant novelty                      | ALIGNED           | merchant_seen_before is rank 12 with comparatively small importance.                                                                             |
| Hard-negative amount overlap               | PARTIALLY_ALIGNED | Amount ratio is positive in recorded top-K for six hard-negative false positives; selected errors cannot establish full-population separability. |

## Four separate evidence layers

1. Supervised: actual tree SHAP, five increasing/decreasing factors by default, stable
   sign/rank handling, source metadata and deterministic wording. Zero values appear in
   neither direction. Full local attribution vectors are saved offline.
2. Behaviour: unchanged deterministic components, ranked by contribution, with raw signal
   values and availability. Confidence is `n/(n+20)*available_component_weight_fraction`.
   Count, coverage and device/merchant comparison availability explain confidence; duration
   is not used. Current identity availability is context, not an extra multiplier.
3. Anomaly: unchanged original Isolation Forest score, normalized A, threshold and flag.
   Observable signals are diagnostic context, not Isolation Forest feature attribution.
4. Fusion: the exact frozen ABRF implementation and its calculation trace.

```text
R = (wp*P + wa*A + wb*g(Cb)*B)/(wp + wa + wb*g(Cb))
wp=0.9, wa=0, wb=0.1
g(Cb)=Cb if Cb>=0.5, otherwise 0
Unavailable B is excluded; remaining P/A weights renormalize.
Investigation threshold = 0.29755749437222373
```

Anomaly remains `diagnostic_only=true`, `effective_abrf_weight=0`,
`influenced_final_risk=false`, even for A near one. Reason codes can report diagnostic
evidence without implying mathematical influence. Bands are unchanged:
0.001191118506145397, 0.027236456910269238, 0.44883536556906983,
0.9401087692599148; equality enters the higher band. R is a decision-support risk score,
not fraud probability or guilt. Phase 6 behaviour added only two detections beyond the fair
P-only optimized control; most improvement over the old operating point was threshold
optimization. Tree SHAP does not reattribute that improvement to behavioural fusion.

## Case and error findings

The same six IDs (2845, 69117, 58423, 14302, 86722, 10604) remain fixed. All 11
hard-negative false positives and all 98 missed validation fraud records now contain real
SHAP factors. The generation compares every non-SHAP field with the historical snapshot;
behaviour, anomaly, risk, flags, reasons, context and evaluation metadata are preserved.
See [report-ready cases](EXPLANATION_CASE_STUDIES.md).

Shopping-festival false positives have time gap among top increasing factors in 8/8,
and amount ratio in 6/8. New-legitimate-device false positives have failed attempts among
top increasing factors in 3/3. These are model explanations, not conclusions about people.
Behaviour is separate and anomaly contributes zero.

Among 98 false negatives, failed attempts is among top decreasing factors in 95;
region history in 72; amount ratio in 43. Velocity-burst false negatives have time gap
among top increasing factors in 41/41 but failed attempts among decreasing factors in
41/41. Low-and-slow misses have time gap increasing in 22/22, while failed attempts decrease
in 22/22, prior transaction count in 17/22 and valid amount-history count in 16/22.
These frequencies are explicitly **top-K recurrence**, not complete-group mean SHAP or
causality. Both supporting and protective factors, calibration and frozen fusion decisions
must be considered; a positive SHAP signal alone does not guarantee an investigation flag.

## Genuine SHAP performance and memory

The following are five-repeat medians using the cached explainer. These measure SHAP,
bounded densification, raw-tree prediction and independent additivity checks only;
they exclude behavioural/anomaly/fusion explanation work. Constructor timing excludes
package import. No cold-process import latency or production capacity is claimed.

| Measurement            | Actual time |
| ---------------------- | ----------- |
| Explainer construction | 1.3534 ms   |
| One record             | 1.8120 ms   |
| 10 records             | 3.4203 ms   |
| 100 records            | 13.2970 ms  |

Validation CSR input is 8,391,400 bytes (17,280,000 dense). Complete positive-class SHAP
values occupy 17,280,000 bytes as a 15,000x144 float64 array; compressed NPZ is 2,635,441
bytes. A 256-row dense input uses 294,912 bytes; two-class SHAP output uses up to 589,824
bytes, excluding tree/library overhead. Aggregation/serialization also allocate complete
arrays and compression buffers. Process peak RSS was not measured; no production capacity
claim follows from these development measurements.

## Immutable snapshot, API and reproduction

Nine outputs under `ml/artifacts/explainability/`: global_feature_importance.json,
feature_family_importance.json, case_studies.json, hard_negative_explanations.json,
false_negative_explanations.json, explainer_metadata.json, validation_shap_values.npz,
analysis_bundle.json and snapshot_freeze.json. The freeze binds bundle/artifact hashes,
historical v1 hash, provenance, unchanged cases and holdout isolation. No silent overwrite.
Original v1 is preserved at `ml/artifacts/explainability_snapshots/explainability-v1/`;
the initial presentation-mapping draft is retained as an unpublished historical draft.

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.explainability --config ml/config/synthetic_riskbyte.example.json --snapshot-version explainability-v1.1-shap --baseline-dir ml/artifacts/explainability_snapshots/explainability-v1
```

A completed snapshot is hash-verified and reused. For a separately declared experiment,
select a new `--output-dir`; it cannot silently replace a frozen snapshot.
Existing routes report explainability_ready and SHAP AVAILABLE with version, explainer
and explained-model fields. Global/family routes serve genuine ranks; precomputed validation
lookup includes genuine factors and excludes evaluation labels. Requests never initialize
SHAP, unpickle models, fit or score records, or expose local paths. NPZ stays offline.

## Isolation and limitations

Original temporal test accessed for Phase 7.1 explanation scoring: **NO**. Only the safe
85,000-row development prefix is decoded; only 15,000 validation records are scored or
explained. TRAIN fitted-tree counts provide the reference. Full sources are checksummed as
opaque bytes; excluded identity fields are not decoded. Future holdout remains
**PREPARATION_ONLY_NOT_GENERATED** and was neither generated nor evaluated.

SHAP explains model behaviour. It does **not** establish causation, criminal intent, fraud
guilt or human motivation. All records are synthetic; generator rules influence learning,
correlated signals complicate attribution, and repeated validation reuse is not independent
production evaluation. Anonymized semantics are unavailable. Basic stability tests are not
formal robustness certification. Human review is recommended.

## Verification

**522 tests passed, zero skipped.** The original genuine SHAP test now imports SHAP and
passes. Added coverage verifies installed 3D output semantics, list/2D compatibility,
expected values and additivity, fixed case IDs, historical evidence preservation,
scenario/type factor summaries, complete attribution serialization, API-ready SHAP fields
and snapshot freeze/tamper handling. Tests use small synthetic fixtures, never protected
dataset partitions. Python Ruff checks, formatting, compileall, pip check and 119 module
imports passed. The dependency lock still matches all 42 installed pins.
Frontend lint, typecheck, production build and format verification also passed.

The published artifact audit independently recomputed all 144 global means from the
15,000x144 NPZ and all family sums. It verified local factors and additivity for all 115
case/error records, unchanged historical hash, component hashes and zero anomaly
contributions. All existing explanation routes and all six fixed transaction lookups
returned HTTP 200 / explainability_ready; an uncomputed ID returned the structured 404.
No request performed model scoring. Reproduction reused the hash-verified frozen snapshot.

Verification commands:

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

Phase 7.1 completes genuine SHAP support without starting another phase. Frozen supervised,
calibration, anomaly, behaviour and ABRF artifacts remain unchanged. Original temporal test
explanation scoring: **NO**. Future holdout: **PREPARATION_ONLY_NOT_GENERATED**.
Final Phase 7.1 status: **COMPLETE**.
