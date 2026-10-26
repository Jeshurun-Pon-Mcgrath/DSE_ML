# ABRF design

## Implemented Phase 6 boundary

Adaptive Behavioural Risk Fusion is a project-specific, transparent RiskByte methodology,
not a claim of universally optimal or novel mathematics. Version `abrf-v1` consumes three
separate signals: calibrated supervised estimate P, normalized anomaly score A and
behavioural deviation B with behavioural confidence C_b. A/B/R are not fraud probabilities.
The selected development configuration comes from actual validation comparison, not arbitrary
one-third coefficients. See [research](ABRF_RESEARCH.md) and [validation](ABRF_VALIDATION.md).

**Weights are initially configurable and must ultimately be validated and calibrated
experimentally.** This phase supplies synthetic validation evidence; it does not establish
universal coefficients or calibrated production banking risk.

## Exact confidence-aware formulation

Base weights are finite/nonnegative and sum to one. At least one P/A weight must remain
positive so cold-start fallback is defined. P/A are required finite inputs in [0,1]. B may
be unavailable; unavailable B requires confidence zero and remains null in the trace.

Let I_B indicate availability, and g be the selected confidence transformation:

`d = wp + wa + wb * I_B * g(C_b)`

`R = (wp*P + wa*A + wb*I_B*g(C_b)*B) / d`

When B is unavailable, omit the behavioural term entirely and renormalize P/A; do not
interpret absent history as B=0 or zero risk. Effective weights are
`(wp/d, wa/d, wb*I_B*g(C_b)/d)`. Contributions multiply those weights by the corresponding
available signal, summing to R. Confidence zero removes behavioural influence; confidence
one permits its full base weight. Ground truth never determines live contributions.

The study compares linear g(C)=C and gated linear transforms above confidence 0.5 or 0.75.
Adaptation means explicit reliability weighting, not unbounded online learning or request-time
weight modification. Exactly one selected configuration freezes weights, confidence rule,
threshold and risk bands together. Zero component weights are allowed when validation does
not support inclusion. Required input/provenance verification remains enforced.

## Offline selection, versions and immutable history

The frozen Phase 4 Decision Tree/calibrator/threshold and Phase 5 detector/normalization/
threshold/behavioural engine are verified by hashes and versions. Only their checksum-bound
TRAIN/VALIDATION score sidecar is decoded. Original transaction files are read as opaque
checksum bytes; original temporal test rows are never decoded, tuned upon or scored.

A bounded simplex grid compares P-only, P+A, P+B and full three-signal ablations using the
same validation policies and strong-scenario protection floors. The frozen P threshold is
preserved separately from a fair policy-optimized P-only control. Every exact threshold's
metrics and all candidate-policy results are persisted. Exclusive run directories preserve
historical files; one active selection is published only after post-execution integrity checks.
The initial and interrupted trace-export attempts remain archived rather than overwritten.

BALANCED maximizes F1 with overall FPR<=1% and hard-negative FPR<=5%. HIGH_RECALL maximizes
recall with FPR<=2% and hard-negative FPR<=10%. STRICT minimizes false positives subject to
recall>=60%, FPR<=0.5% and hard-negative FPR<=3%. All constants are declared in the search
configuration. Candidate ties consider AP, precision/recall, false-positive rates and complexity.
Risk bands derive from TRAIN score quantiles without labels and remain distinct from the
validation investigation threshold. They are decision-support categories, not guilt labels.

## Input and explanation contracts

`FusionInput` accepts event ID, P/A scores and frozen predictions, nullable B, confidence,
availability and optional already-available structured evidence. Target, scenario and
hard-negative fields are rejected. Evaluation metadata lives separately in research artifacts.
`fuse` returns the exact inputs, base/effective weights, weighted contributions, R, risk level,
threshold, investigation flag and deterministic evidence codes. Unsupported novelty/velocity/
amount explanations are never inferred from B alone. A code records observed evidence; an
inactive component's code does not mean it influenced R. Its contribution remains explicit.

Subtractive crossing allocations are mathematical allocation diagnostics, not causal or SHAP
explanations. Classifier SHAP remains deferred and must not be relabelled as explanation of
the complete ABRF function. No final transaction prediction endpoint is implemented.

## Calibration, confidence and future acceptance

R remains an uncalibrated fused score. Strictly monotonic calibration cannot improve ranking
or threshold tradeoffs; no empirical probability-calibration benefit is claimed. Probability
claims would require a separately justified calibration protocol. Behavioural confidence is
an unchanged provisional count/availability proxy, not calibrated probability reliability.

Repeated validation reuse, synthetic generator rules, overlapping signals, small scenario
counts and dependent entities/episodes limit conclusions. Selected weights may omit anomaly
influence, and a modest gain over a threshold-optimized control must not be overstated.
Fresh future holdout status is PREPARATION_ONLY_NOT_GENERATED. Follow FINAL_HOLDOUT_PLAN.md
only after explicit authorization and reviewed stateful continuation support; do not generate,
inspect or score it during this phase. No SHAP, final inference, UI redesign or Phase 7 occurs.

## Legacy compatibility

The Phase 0 ABRFConfig/RiskFusion declarations remain import-compatible foundation types.
Their provisional equal defaults are never used by the Phase 6 research runner. The strict
contracts, deterministic engine and explicit selected JSON configuration define current use.
