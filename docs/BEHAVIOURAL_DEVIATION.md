# Phase 5 behavioural deviation

**Synthetic-data finding.** Dataset `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`; feature pipeline `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`; temporal split `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`. Development contains 70,000 TRAIN and 15,000 VALIDATION rows. The original temporal test partition was not decoded into records, scored, or used for Phase 5 decisions. Full files are read as opaque bytes for checksums; identity projection checks excluded numeric join keys without decoding excluded context fields.

## Entity baseline and strict-prior evidence

Version `behavioural-deviation-v1` is a deterministic engine, separate from Isolation
Forest and the supervised classifier. The unchanged frozen feature builder uses only
earlier event times for each simulated behavioural entity. Equal-time peers do not enter
each other's history. Prior labels never filter entity history; earlier attack observations
can contaminate baselines. Synthetic entities are not actual bank customers.

## Components and provisional configuration

Amount ratio severity is `abs(log(max(ratio, 1e-6)))` for nonnegative ratios. This considers
both decreases and increases. Z-score severity is its absolute value. Velocity uses 30-minute
and 24-hour prior counts. Short-gap severity is `1/(1+prior_gap_seconds)`. Failed attempts
are pre-outcome counts. Each continuous severity uses a strict-left empirical CDF fitted
on ALL TRAIN features, without target labels. Zero-mass/minimum components map to zero.
Merchant/device/billing-region/product novelty is `1 - seen_before`, but only where prior
context exists. Unknown context stays unavailable, never automatically novel or safe.

Ten components have explicit equal provisional weights in `ml/config/behaviour.example.json`.
Amount ratio/z-score and correlated velocity/gap components are not independent; equal weights
are a transparent starting definition, not calibrated risk contributions. Weights must ultimately
be validated/calibrated experimentally. No label-driven weight search occurs in Phase 5.
B is the weighted mean of AVAILABLE components. Missing components are excluded explicitly
from the denominator; coverage is reported separately. Individual contributions sum to B
when overall history is available. Structured explanations expose raw values, normalized
severity, contribution, direction, availability and window; no fabricated sample values.

## Confidence and cold start

Confidence is `n/(n+20) * available_weight_fraction`, where n is strictly prior transaction
count. Saturation 20 and minimum history 1 are configurable reliability assumptions, not
empirical fraud probabilities. Context missingness (including absent device evidence) reduces
coverage. History duration is explicitly NOT used because frozen features lack a prior-span
field. Future research should assess duration and independent context coverage before calling
confidence calibrated reliability.

First entity event: history_count=0, history_available=false, B=null, confidence=0.
Missing history is not zero risk. All unavailable scores are treated as abstentions; reports
retain full-row denominators for flag rates and separately disclose available-score AUC scope.
`DeviationEngine.explain` returns typed component records. The API exposes aggregate reports,
not raw development transactions.

## Measured validation findings

| Cohort            | Mean B   | Median B | Confidence mean | B flag rate |
| ----------------- | -------- | -------- | --------------- | ----------- |
| normal_legitimate | 0.203591 | 0.196223 | 0.669587        | 0.5410%     |
| hard_negative     | 0.368451 | 0.377618 | 0.607190        | 9.9125%     |
| fraud             | 0.495519 | 0.457549 | 0.580493        | 46.6165%    |

The analysis-only B threshold is **0.4823676452550990**, selected
under the same validation false-positive caps. Recall is
**46.6165%**; legitimate flag rate
**0.9773%**; hard-negative flag rate
**9.9125%**.
No B threshold is deployed as a banking fraud decision.
**B and confidence are not fraud probabilities.**

## Availability and limitations

International novelty is unavailable: the frozen feature contract lacks prior international
history. The anomaly engine can use raw is_international, while B does not silently relabel
it as entity-specific novelty. No GPS/distance feature exists. Time gap is global training
gap severity, not a learned per-entity normal-gap model. Confidence currently measures count
and component coverage, not prior duration or independently calibrated reliability.
Missing device identity, sparse baselines, correlated components and contaminated prior
history can limit evidence. Synthetic findings cannot prove production performance.

`behavioural_engine.joblib` stores only configuration and fitted references/provenance;
`behavioural_metadata.json` documents sources, definitions and confidence. The evaluation
sidecar `development_scores.csv` keeps P/A/B, confidence and outcome metadata OUTSIDE X.
GET `/api/v1/behaviour/{summary,components,validation-analysis}` serves read-only summaries.
No score fusion, ABRF, SHAP or frontend redesign occurs.
