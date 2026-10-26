# Fresh final system holdout plan

## Phase 8 authorized protocol

The user now authorizes one final future evaluation: 30,000 transactions, seed 2026,
27 synthetic elapsed days after the original 90-day horizon. This supersedes the older
preparation-only authorization below. No model, calibrator, normalizer, fusion weight,
threshold or risk band is changed. Primary metrics and scenario/hard-negative analyses
are the user's fixed Phase 8 requirements; no acceptance-based tuning or retry exists.

The optional stateful-continuation-v1 checkpoint leaves the generator's original path
byte-identical (small deterministic regression fixture). It reuses persistent normal
profiles, preference arrays, frequency weights and observed merchant histories for
1,600 existing entities; 400 entities are new to observed history. Merchant pool extends
from 300 to 360, with unchanged category and attack semantics. New device/transaction ID
namespaces prevent collisions. Existing account ages advance by 90 elapsed days.
Fraud/hard-negative/identity/missingness mechanisms stay the frozen generator mechanisms.

History checkpoint deliberately contains only the 85,000 TRAIN/VALIDATION events;
original test events are neither decoded nor rescored. The gap after validation is
unobserved, so this is a conditional future academic simulation, not complete bank history.
No history during that gap is invented. Relative-time normalization retains the frozen
90-day denominator; future positions can exceed one. This intentional extrapolation and
context extension must be reported as holdout shift, without retraining.

READY_UNTOUCHED is persisted before an exclusive evaluation-start journal. It transitions
irreversibly to EVALUATION_IN_PROGRESS and then FINAL_EVALUATED_LOCKED. An interrupted
evaluation cannot automatically retry. Component predictions are saved before evaluation
labels are opened. Only selected final demo cases receive SHAP. Failed performance must
be reported honestly, never repaired with tuning or another seed.

## Why a new holdout is necessary

Phase 6 preparation status: **PREPARATION_ONLY_NOT_GENERATED**. ABRF research uses existing
TRAIN/VALIDATION artifacts and publishes an immutable selected configuration. No future
extension, target-distribution inspection or holdout scoring is authorized in this phase.
The reviewed stateful-continuation adapter and explicit generation authorization remain
prerequisites; a distinct seed/time offset alone is not sufficient to create valid future history.
SHAP/explanation changes must not become a route to retuning frozen component/fusion decisions.

The original temporal test period was inspected in earlier phases. Phase 5 does not use
its records for fitting, selection, normalization, behavioural weights, thresholds or
fusion decisions. Reusing it after the full system evolves would not yield independent
final-system evidence. Phase 5 creates no new holdout.

## Preconditions and governance

Freeze and hash the supervised classifier/calibrator/threshold, anomaly model/feature
contract/reference mapping/threshold, behavioural components/configuration/confidence,
ABRF algorithm/missing-signal policy/weights/calibration/thresholds and feature versions.
Predeclare primary metrics, alert budgets, scenario and hard-negative analyses and acceptance
criteria. Record generator version, random-seed policy and dependency versions. A designated
final-evaluation command must fail closed if any component changes and publish an exclusive
one-use evaluation journal before exposing labels/results. Failed acceptance requires a new
experiment and independently reserved holdout, not iterative tuning of this period.

## Proposed future synthetic extension

After explicit authorization, generate a chronological extension strictly after the frozen
synthetic elapsed-time horizon, using the same generator semantics and a distinct reserved
deterministic seed. Preserve persistent synthetic entity profiles, merchant/device context,
prior event-history state and account/identity age semantics. Verify unique event IDs and
strictly greater times; define equal-time batching, identity coverage and missingness rules.
Include both continuing entities and explicitly identified new entities under a prespecified
proportion. Predeclare scenario/hard-negative scheduling independently of measured results.

The current generator does NOT implement stateful future continuation. Adding a seed/time
offset to a new independently generated dataset does not automatically produce credible
continuing behaviour. A dedicated reviewed continuation adapter/state checkpoint and tests
will be required, while preserving the frozen generator's scenario semantics. If semantics
must change, version them and label the holdout a shifted-generator external stress test
rather than an interchangeable sample from the old benchmark.

Do not extend training, refit medians/encoders/CDFs or tune any thresholds with future data.
Scoring may use earlier unlabelled future transactions as strict-prior event history under
the frozen sequential policy, never future outcomes or equal-time peers. Evaluation metadata
is stored separately and opened only after predictions/components are durably recorded.

## Final evaluation

Evaluate once, after all frozen-system checks pass. Report PR-AUC, precision/recall, legitimate
and hard-negative false-positive rates, scenario recall, confidence/availability, calibration
where probability is claimed, time/entity subgroup stability, ablations and measured timing.
Use suitable episode/entity/time-respecting uncertainty estimates where support is adequate.
Report abstentions and cold-start coverage explicitly. Compare to frozen classifier-only
results on this same fresh period without changing either comparator.

Every conclusion remains a **Synthetic-data finding**. A fresh academic simulation cannot
prove production banking performance; generator-induced learnability, simplified identities,
missing label-delay/process dynamics and approximate concept drift remain limitations.
