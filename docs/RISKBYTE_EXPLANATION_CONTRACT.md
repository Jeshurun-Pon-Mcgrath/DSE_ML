# RiskByte explanation contract: explainability-v1.1-shap

The contract describes synthetic academic decision-support evidence. It contains four
separate layers and never includes target/scenario/hard-negative fields as live inputs.
The typed definitions live in `riskbyte_ml.explainability.schemas`; extra fields, invalid
numbers, contradictory trace values and invented unavailable SHAP values are rejected.

## Layer fields and scope

| Layer        | Fields                                                                                                                                                   | Meaning                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Risk summary | final_risk_score, risk_level, investigation_flag                                                                                                         | Frozen ABRF result                                                    |
| Supervised   | model identity, raw probability, calibrated P, frozen threshold/prediction, calibration provenance, SHAP availability, top increasing/decreasing factors | Tree output attribution only when genuine SHAP is available           |
| Behaviour    | score, confidence, history count/availability, ranked components, confidence measurements                                                                | Exact deterministic deviation and history reliability                 |
| Anomaly      | raw and normalized score, threshold/flag, observable signals, diagnostic-only flag, effective weight, influenced-final-risk                              | Isolation Forest diagnostics, without unsupported feature attribution |
| Fusion       | ABRF version, P/A/B/C, base/effective weights, contributions, denominator, confidence transform, risk band, threshold/decision, reason codes             | Exact frozen calculation                                              |
| Limitations  | scope, synthetic notice, non-causal notice                                                                                                               | No probability/guilt/causation claims for risk or deviation scores    |

SHAP factors carry encoded feature value, actual signed SHAP value, display name, source
lineage, family, raw/derived status and both manifest/presentation interpretability classes.
Zero contributions appear in neither increasing nor decreasing lists. Top-K defaults to
five per direction and is bounded at 20. Unavailable SHAP uses empty factor lists and null
base/residual with an explicit reason; it is not a zero-attribution explanation.

## Calculation consistency

`R=(wp*P+wa*A+wb*g(Cb)*B)/(wp+wa+wb*g(Cb))`, with `wp=0.9, wa=0, wb=0.1` and
`g(Cb)=Cb` when confidence is at least 0.5, otherwise zero. Unavailable B is excluded and
remaining weights renormalize. Confidence is not a probability of fraud.

All scores, contributions, risk levels, flags and aggregate reason codes must reproduce
the frozen Phase 6 result. The investigation decision is `R >= 0.29755749437222373`; risk
band equality enters the higher band. Anomaly always has zero influence in the selected
configuration, even when its diagnostic flag is true. `HIGH_ANOMALY_SCORE` is an evidence
code, never an assertion that A raised R. Component evidence is separate from aggregate
frozen reason codes; Phase 7 does not rewrite those historical codes.

Raw tree probability and sigmoid-calibrated P are separately exposed. Tree SHAP does not
explain the calibration mapping or the complete RiskByte system. Any additivity mismatch,
feature misalignment, frozen artifact mismatch or disagreement between layers fails
generation instead of producing contradictory explanation text.

## Read-only API

- `GET /api/v1/explainability/status`
- `GET /api/v1/explainability/global`
- `GET /api/v1/explainability/families`
- `GET /api/v1/explainability/case-studies`
- `GET /api/v1/explainability/validation-errors`
- `GET /api/v1/explainability/validation/{transaction_id}`

Routes return typed `ExplanationResponse` envelopes. States are `dataset_not_configured`,
`explainability_not_generated`, `explainability_unavailable`, `explainability_partial` and
`explainability_ready`. The historical v1 snapshot was partial. The current v1.1 snapshot reports
explainability_ready with genuine SHAP AVAILABLE. Research case/error endpoints intentionally include
evaluation-only labels in a separate wrapper. Transaction lookup returns only the strict
outcome-free unified explanation. Uncomputed IDs receive a structured 404.

Requests verify frozen provenance and published artifact hashes, then read bounded JSON.
They never read raw dataset rows, unpickle models, fit anything, initialize SHAP or score
transactions. No local source path is returned. This is not the final public inference API.

## Artifact lifecycle and limits

The offline command publishes six JSON reports and `analysis_bundle.json` last, with
checksums and isolation audit. Successful genuine SHAP execution additionally saves
`validation_shap_values.npz` for offline complete attributions; HTTP never exposes the
full matrix. Completed snapshots are immutable; experiments use separate output directories.
Joblib is executed only offline after pinned hashes succeed, from trusted local artifacts.

TRAIN/VALIDATION source decoding uses the established safe development projection;
integrity checks may read complete sources as opaque bytes. Original temporal test rows
are neither materialized nor scored. The future holdout stays
`PREPARATION_ONLY_NOT_GENERATED`.

SHAP describes model-output contributions, not causation, criminal intent, fraud guilt or
human motives. Synthetic data cannot establish production performance. Deterministic
templates use only documented feature semantics and contribution signs; anonymized features
never receive fabricated business explanations. Human review is recommended for flags.

## Phase 7.1 frozen extension

The schema accepts historical explainability-v1 and active explainability-v1.1-shap.
The latter cannot have unavailable SHAP. Probability aliases raw_tree_probability and
calibrated_supervised_probability match their existing raw_probability/calibrated_probability
fields exactly. TreeExplainer still explains only the underlying Decision Tree.
Status adds SHAP version, explainer type and explained model. No endpoint was added.

List/2D/3D output normalization and scalar/vector expected values are checked against
actual raw class-1 output, ordered features and finite values. Additivity tolerance is 1e-7.
Full validation attributions reside in the offline NPZ; normal responses remain top-K.
Both metadata and freeze record identify the preserved historical snapshot and fixed IDs.
Non-SHAP evidence is compared field-for-field before publication. Artifact hashes and
snapshot_freeze.json are verified for v1.1 reads; frozen snapshots cannot be silently rewritten.

Risk scores, risk bands, calibration, weights and decisions are unchanged. Anomaly weight
and contribution remain zero. Ground truth stays in research/evaluation wrappers, never
in the normal transaction explanation. SHAP is non-causal and all records are synthetic.
The original test is not scored; the future holdout remains PREPARATION_ONLY_NOT_GENERATED.
