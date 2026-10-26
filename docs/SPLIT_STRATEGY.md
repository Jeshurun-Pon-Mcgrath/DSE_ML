# RiskByte split and historical-availability strategy

**Temporal evaluation is primary.** The task predicts later transactions, and MEDIUM has
both classes and all configured fraud archetypes in each chronological partition. A seeded
random stratified split is saved as a comparison benchmark; it is not evidence of future
performance. Ratios default to 70% train, 15% validation and the remainder test.

Temporal rows sort by TransactionDT. Boundary ties move together so no equal-time peers
cross partitions; this can change exact row counts. The random comparator stratifies the
explicit `isFraud` target. Integer/fraction rounding can change a partition by one row.
Both strategies validate disjoint, exhaustive row positions, nonempty partitions, explicit
binary labels, finite nonnegative elapsed time and source-ID row order. The primary run
requires both classes in every partition. Usable prevalence is not a guarantee of statistical
power: review scenario-specific sample counts before future model comparisons.

Fingerprint-bound split JSON stores version, pipeline version, row-order hash, positions,
seed where applicable, ratios, prevalence, elapsed range, entity overlap, scenario shares and
hard-negative counts. Deterministic artifact names use split IDs. Identical runs verify and
reuse existing artifacts; incompatible content is rejected. Later experiments must load the
recorded artifact, not select a fresh holdout or tune against the test partition.

## Preprocessing and history policy

Imputation, scaling, rare grouping, frequencies and category dictionaries fit recorded
training rows only. Validation/test values cannot alter those states. Features use only
strictly earlier observed transaction context, never targets. At equal elapsed time, the
entire entity batch reads the same prior state before any member is committed. Velocity
windows are `[event_time-window, event_time)`. Current and simultaneous events are excluded.

In sequential validation/test, earlier transactions in that partition become observable
history for later transactions, without label feedback or preprocessor refitting. This is
an online transaction-availability policy. The random comparator also uses globally prior
unlabelled event context, including earlier observations assigned to another partition. It
is consequently a transductive historical-context comparator, not independent future-period
evaluation. Do not compare it as if every held-out record were completely unobserved.
Prior suspicious transactions remain history and can contaminate baselines; no target-aware
filtering is allowed. Group-overlap figures expose that existing entities recur in holdouts;
future unseen-entity evaluation requires a separately authorized grouped experiment.

## MEDIUM observed comparison

Measured values and complete scenario distributions are recorded in
[the preprocessing report](PHASE_2_PREPROCESSING_REPORT.md) and ignored split artifacts.
Temporal prevalence varies across periods because event compression and activity lifetimes
are part of this simulation. Random stratification preserves the global 2% by design.
This is simulated temporal variation, not evidence of real financial-market drift.

Example configurable rerun (does not regenerate source files):

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.preprocessing --split-strategy temporal --train-ratio 0.7 --validation-ratio 0.15 --seed 42
```

No resampling, target encoding, label-history features, model fitting or hyperparameter
selection occurs in Phase 2. The test period remains untouched by model selection.
