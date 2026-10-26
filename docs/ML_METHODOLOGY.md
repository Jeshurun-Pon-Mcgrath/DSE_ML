# ML methodology

## Phase 6 fusion research

Confidence-aware weighted P/A/B fusion consumes the checksum-bound Phase 5 development
sidecar, never original temporal test records. Fitting/scoring frozen components is unnecessary.
Validation compares a bounded simplex grid and three transparent confidence rules under
identical policy budgets and protected account-takeover/mixed-mode detection floors.
All exact thresholds are retained in compressed artifacts. Risk bands use training score
quantiles without labels. The frozen supervised threshold is preserved; a separately tuned
P-only control makes the ablation fair. Zero component weights are eligible when validation
does not support their inclusion. Selection is frozen before any fresh holdout generation.
See [ABRF research](ABRF_RESEARCH.md) for actual improvements, regressions and validation-reuse
limitations. No real-world probability or universally optimal algorithm claim is made.

## Phase 5 development contract

Isolation Forest compares two dedicated numeric feature sets and legitimate-only/mixed TRAIN
reference populations. Targets only select the known-normal training cohort; fitting receives
features without labels. Empirical score mappings and medians use TRAIN only. VALIDATION
selects feature family/native contamination offsets and an anomaly threshold with legitimate
and hard-negative budgets. Behaviour components use training-only mappings, configurable
provisional weights, explicit missing evidence and count/coverage confidence. P/A/B remain
separate; analytical OR coverage is never a deployed decision rule.

The previously observed temporal test records are inaccessible to Phase 5 development.
Full-source opaque hashes verify provenance; excluded identity keys serve only projection
membership, without decoding excluded context. No old test fitting/scoring/threshold tuning
occurs. Validation reuse is acknowledged, and a fresh future holdout is required after ABRF
and every component/threshold is frozen. See [Phase 5 measurements](PHASE_5_COMPLEMENTARITY.md)
and [final holdout plan](FINAL_HOLDOUT_PLAN.md). Historical phase sections below remain unchanged.

## Phase 4 supervised experiments

The dataset, temporal split, feature manifest and fitted temporal preprocessors remain
frozen. Classifiers fit training rows only. Earlier chronological validation rows support
a restrained seeded parameter search, XGBoost early stopping where available, and sigmoid/
isotonic calibration fitting. Later validation rows select calibration method, threshold
policy and algorithm. Default BALANCED maximizes F1 under 1% overall FPR and 10% hard-negative
FPR caps; alternative policies are explicit. Primary model ranking starts with average
precision, then recall/precision/F1, subgroup rates, calibration and inference practicality.
Accuracy is secondary; no SMOTE is applied to these already synthetic records.

One classifier/calibrator/threshold artifact is frozen durably before a one-use test gate
opens. No fitting or selection uses test predictions; final test metrics are saved separately.
Only afterward does a limited saved-random-split validation comparator run, with separate
train-only preprocessing and no feedback into primary decisions. Completed identical runs
reuse the registry without repeating test evaluation; changed/frozen incomplete runs fail
closed. Aggregated test exploration from Phase 3 remains explicitly acknowledged.

See [SUPERVISED_MODEL_EVALUATION.md](SUPERVISED_MODEL_EVALUATION.md) for measured results,
unavailable dependencies, calibration tradeoffs, scenario recall, hard-negative false positives,
timing, error analysis and limitations. Synthetic results cannot establish banking performance.

## Phase 3 statistical intelligence

The frozen synthetic MEDIUM fingerprint and temporal split are enforced before statistical
execution. Six prespecified training-partition research questions use rank tests or categorical
association tests, nominal BH FDR correction and effect sizes. Full-source scenario/hard-negative
and relative-time summaries are exploratory, not held-out performance. No model is fitted.
See [STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md) for actual Synthetic-data findings,
assumptions, confidence intervals and reproducible commands. Repeated entity/episode dependence
prevents treating nominal p-values or Wilson intervals as calibrated banking-population inference.

## Current Phase 2 decision

The approved academic source is now a **fully synthetic** benchmark, not IEEE-CIS.
See [SYNTHETIC_DATASET.md](SYNTHETIC_DATASET.md), [SPLIT_STRATEGY.md](SPLIT_STRATEGY.md)
and [PHASE_2_PREPROCESSING_REPORT.md](PHASE_2_PREPROCESSING_REPORT.md). Phase 2 generates
scenario-based transactions and hard negatives, validates the audited left join, computes
strictly prior context and fits only training preprocessing. No predictive/anomaly model,
SHAP or ABRF is implemented. Earlier phase status sections below are historical scope.

Every predictive input has allowlisted source lineage. Targets, TransactionID, raw
TransactionDT, behavioural_entity_id and evaluation-only sidecars are excluded. Fixed
duration-relative time features do not use a future maximum or invented calendar anchor.
Numeric preparation uses finite views, train-only medians, a zero fallback for all-missing
training columns and explicit missing indicators for every numeric feature. A documented
fixed ±1e12 model-preparation bound prevents extreme arithmetic overflow without modifying
source data. Zero-variance scalers stay finite. Cold-start history is unavailable, not low risk.

Categorical missing values use a collision-safe explicit token. Training-only rare grouping
precedes sparse one-hot encoding; unseen categories map to rare/unknown without learning them.
High-cardinality merchant/device IDs use train-only frequency proportions, not raw ID
one-hot vectors. This can still encode simulated collection/popularity bias and needs later
ablations. Frequency encoding is not target encoding. Entity IDs remain history keys only.
Robust-scaled linear/distance and unscaled tree/boosting families have aligned CSR outputs;
StandardScaler and no-scaling alternatives are configurable. Scaler/encoder objects are
preprocessors, not trained fraud classifiers. Fitted artifacts omit raw training records.

Synthetic EDA is descriptive, not held-out performance evidence. Later evaluation must
report recall by scenario and false-positive rates on legitimate hard negatives, acknowledge
generator-induced learnability and avoid production-banking claims. No model metric exists yet.

## Status and purpose

Phase 1 implements descriptive data intelligence; Phase 0 inference interfaces remain unimplemented. No dataset has been approved, no feature has been
computed, no model has been trained and no performance claim is made. The working domain
is transaction risk/fraud screening; confirm dataset, target and business costs before
implementation. Do not interpret anomaly or behavioural scores as fraud probabilities.

## Implemented Phase 1 methodology

Inspect source-preserving CSV strings before configuring actual roles and both target
classes. Missing fields are documented rather than invented. Heuristics are candidates;
retain leakage candidates pending review. Finite-only descriptive statistics explicitly
count missing, invalid and infinite values. Missing/unsupported target labels fail quality
validation. Imbalance is descriptive and triggers no resampling or model fitting.

IQR and configurable z-score checks identify outliers without deletion: fraud may genuinely
be unusual. Each EDA result declares availability; elapsed timestamps cannot justify calendar
charts. Cleaning is reviewed, source-hash-bound and applied to a copy with separate outputs
and audits. Source/configuration hashes, dependency versions and pipeline versions support
reproducibility. See [DATA_PIPELINE.md](DATA_PIPELINE.md) for formulas and limits.
EDA is exploratory, not held-out evidence of performance; associations are not causation.
Synthetic test metrics verify behavior only and never represent a real fraud dataset.

## Phase 1.5 safeguards

The IEEE adapter verifies exact target and row preservation through a one-to-one left
join, fingerprints both sources/configuration, treats TransactionDT as elapsed seconds
without calendar inference, and studies complete proxy groups without identity claims.
All real-data metrics remain unavailable because the local training files are absent.
Category storage is lossless representation, not fitted preprocessing. No feature is
materialized, no missing values imputed and no model trained. Entity repetition is only
a feasibility screen; split stability, collisions and strict prior-time support require
review. See [entity study](ENTITY_KEY_STUDY.md), [leakage review](IEEE_CIS_LEAKAGE_REVIEW.md)
and [proposed feature contract](IEEE_CIS_FEATURE_CONTRACT.md).

## Future experimental sequence

1. Approve data provenance, licensing, privacy handling, label definition and availability
   timestamps. Establish a versioned data dictionary and immutable snapshot manifest.
2. Audit missingness, duplicates, class imbalance, delayed labels, entity overlap and drift.
   Exclude identifiers and post-outcome fields from model inputs unless justified explicitly.
3. Split chronologically into training, validation/calibration and held-out test periods.
   Use grouped evaluation for entities where appropriate. Record boundaries, seeds and
   label maturation windows; fit transforms using training data only.
4. Start with an interpretable baseline, then compare suitable tree-based classifiers.
   Algorithm selection is deferred until feature types and scale are known. Rebalancing
   belongs inside training folds; never contaminate validation/test distributions.
5. Train a separate anomaly baseline on a documented reference population. Define raw
   score orientation and learn normalization from training/calibration data, not requests.
6. Define behavioural features using only events preceding the assessment timestamp.
   Evaluate cold-start behavior, sparse histories and adversarial manipulation.
7. Compare classifier-only, anomaly-only, behavioural-only and fused variants. Calibrate
   classifier probabilities and ABRF weights/thresholds on separate validation data.
8. Evaluate once on the untouched test period after choosing the system. Report uncertainty
   and subgroup results; failed acceptance requires a new experiment, not test-set tuning.

## Evaluation plan

Prioritize precision-recall AUC, recall at an agreed false-positive budget, precision at an
investigation capacity, confusion matrices and business-weighted expected cost. Report
ROC AUC as supplementary context. Use reliability diagrams, Brier score and calibration
error when making probability claims. Bootstrap by time/entity where independent sample
assumptions are invalid. Acceptance thresholds and cost assumptions are not set in Phase 0.

Measure end-to-end latency, explanation cost, coverage, missing-data behavior and artifact
load time. Evaluate drift and fairness on approved groupings without collecting additional
sensitive attributes by default. Separate monitoring drift from confirmed performance
degradation; production labels may arrive late.

## Reproducibility and explanations

Every experiment records code revision, dataset/split hashes, feature schema, dependencies,
random seeds, hyperparameters, training hardware and evaluation results. Artifacts are
immutable and include fitted preprocessing. Notebooks are exploratory; production logic
belongs in the package. No model or dataset is loaded merely by importing a module.

Future SHAP explanations use a documented training-only background sample, output space
and feature ordering. Test additivity and stability. Explicitly distinguish explanation
of classifier output from explanation of ABRF. Explanation values describe model behavior,
not causation or proof of fraudulent intent.

## Phase 7 implementation

Explainability separates underlying-tree SHAP from calibrated P, deterministic behavioural
evidence, anomaly diagnostics and exact frozen ABRF calculation. The tree-path-dependent
explainer uses frozen TRAIN path counts and requires no explicit background sample.
No validation/test reference distribution is fitted. Phase 7.1 completed genuine SHAP
0.53.0 execution and froze explainability-v1.1-shap while preserving original v1 history.

Validation P/A/B/confidence and all fusion scores/contributions/bands/flags are reproduced
before publication. Six balanced validation cases and all hard-negative false positives /
fraud false negatives are retained as synthetic academic evidence. Labels remain solely in
evaluation wrappers. Original temporal test explanation scoring is prohibited and was not
performed; no fresh holdout is generated. Neither these explanations nor existing repeated
validation analysis prove production generalization, causality or guilt.

See [explainability methodology](EXPLAINABLE_AI.md) for genuine SHAP scope, actual measurements,
calibration distinction, immutable artifacts and non-causal interpretation.
