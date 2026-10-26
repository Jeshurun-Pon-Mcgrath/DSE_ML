# IEEE-CIS leakage review

## Evidence state

Both local real CSVs are absent. Measured duplicate-target fields, ordering correlations,
bucket prevalences and suspicious identifier associations are **unavailable**. No actual
feature is accused of leakage based on synthetic test fixtures. Run the adapter to obtain
the fingerprint-bound measured report under `ml/artifacts/ieee_cis/IEEE_CIS_LEAKAGE_REVIEW.md`.

## Implemented investigation

The existing Phase 1 checks scan all joined fields for direct target copies, post-outcome
names, nearly perfect numeric target relationships and supported repeated categorical
partitions. Constant/small/single-class samples cannot justify correlation evidence.
Each candidate includes severity, reason, affected count where meaningful and inference flag.

The IEEE review adds `TransactionID` and `TransactionDT` file-order monotonicity, numeric
correlation with the configured binary target, up to ten quantile buckets with actual row/
fraud counts and prevalence, and rank correlation between transaction ID and elapsed time.
Equal elapsed values stay in the same quantile bin. These are descriptive diagnostics,
not a trained predictor, AUC, calibration score or proof of leakage. Invalid targets or
insufficient observations return unavailable reasons. Float summaries can approximate
extremely large numeric identifiers; exact source strings and joins remain unchanged.

## Candidates requiring future review

| Field / mechanism                                          | Risk and proposed policy                                                                                               |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `isFraud` or direct copies                                 | Target only, exclude from predictive feature contract                                                                  |
| `TransactionID`                                            | Audit/join key; never directly encode as a predictive measurement                                                      |
| Raw `TransactionDT`                                        | Acquisition-period/order drift; retain for chronological split/history, direct prediction requires explicit validation |
| Fraud status/final decision/chargeback/outcome-time fields | Verify availability at assessment time; candidate exclusion pending review                                             |
| High-cardinality identifiers/codes                         | Memorization and overlapping entities; grouped/temporal sensitivity checks                                             |
| C/D/M/id/V engineered fields                               | Construction/availability partly hidden; check source provenance, not just correlation                                 |
| Full-dataset history or future aggregates                  | Leakage by construction; prohibit in proposed history contract                                                         |
| Proxy groups constructed using target                      | Prohibited; candidates use approved non-outcome proxy fields only                                                      |

No column is automatically dropped. `EXCLUDED_CANDIDATE` is a reviewable contract tag, not
an ingestion deletion. Suspicious associations may be legitimate signals or collection bias;
unobserved weak/nonlinear leakage can escape these heuristics. Label availability/maturation
and opaque source feature generation require review beyond statistics.

Later experiments must split chronologically, build history from strictly earlier events,
fit transformations on training data only, exclude same-time peer information unless explicitly
available and compare grouped splits. Do not tune against future/test target data.
