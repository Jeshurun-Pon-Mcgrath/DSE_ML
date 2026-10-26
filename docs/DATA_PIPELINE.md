# Phase 1 data intelligence

## Phase 3 frozen statistical execution

Run `python -m riskbyte_ml.statistics --config ml/config/synthetic_riskbyte.example.json`
from the repository root after Phase 2 artifacts exist. The command verifies the frozen
synthetic source/configuration fingerprint, evaluation sidecar hash, feature version and
recorded temporal split. It reuses the exact saved indexes and strictly prior features;
no source, split, preprocessing object or model is regenerated/fitted. Mismatches abort
with a sanitized domain error and exit 2 before publishing output.

Outputs under `ml/artifacts/statistics` contain provenance-bound descriptives, hypothesis
tests, correlations, prevalence intervals, hard-negative/scenario and elapsed-day analysis.
API reads a single atomically published typed bundle. No configuration returns
`dataset_not_configured`; a verified frozen source without a bundle returns
`statistics_not_generated`; other selected adapters return `statistics_unavailable` with
an explanation. Missing sources and invalid provenance are structured errors, not global
application startup failures. See [STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md).

## Authorized Phase 2 synthetic workflow

The project decision now explicitly permits generated academic application data, separate
from synthetic tests and absent real IEEE-CIS sources. See [SYNTHETIC_DATASET.md](SYNTHETIC_DATASET.md).
The generator writes transaction/identity CSVs, a generation manifest and evaluation-only
sidecars. Source byte hashes, generator configuration/version and sidecar verification precede
the shared audited LEFT join. Existing Phase 1 inspection/validation/statistics/EDA operate
on the joined frame unchanged, including all missingness/cardinality/imbalance warnings.
Calendar analyses remain unavailable because elapsed time has no real-world calendar origin.

Offline Phase 2 adds elapsed-day EDA, scenario/hard-negative aggregate EDA, recorded splits,
strictly prior features, allowlisted lineage, train-only preprocessing and serialization/reload
checks. No row/column is silently removed from ingestion. Preprocessing excludes approved
nonpredictive fields from model matrices while preserving originals and uses separate artifacts.
The `synthetic_riskbyte` API adapter never generates missing data or downloads originals.
Generated values are synthetic observations, not financial-market measurements.

## Scope and flow

Operator-owned JSON configuration → bounded CSV loader → schema inspection → quality
validation → descriptive statistics and EDA → proposed cleaning plan → local reports.
The API delegates to this same ML package. No model, scoring endpoint, automatic cleaning,
dataset download or frontend dashboard is implemented.

CSV ingestion preserves values as strings, including leading zeros, whitespace and literal
`NA`. Analysis creates separate numeric/datetime views; it never changes the source frame.
Blank/whitespace-only cells are considered missing during analysis. Additional missing
tokens require explicit `na_values`. Duplicate headers, blank headers, ragged records,
invalid encoding, empty/header-only files and exceeded resource limits fail explicitly.
No rows are skipped or truncated. Defaults: UTF-8 with optional BOM, comma delimiter,
256 MB source, 1,000,000 rows and 500 columns. Files must be local regular files.

## Configuration

Copy `ml/config/dataset.example.json` to ignored `ml/config/dataset.local.json` and set
`source_path` only after acquiring an approved dataset. Relative source paths resolve
against the configuration's directory. Inspection mode reports candidates without choosing
any target, identifier, amount or timestamp role. No source path means
`dataset_not_configured`; no fake records or reports are generated.

Switch to explicit mode after reviewing the data dictionary. This example describes
configuration syntax, not guaranteed dataset fields:

```json
{
  "name": "approved_transactions",
  "source_path": "../data/raw/approved.csv",
  "mode": "explicit",
  "encoding": "utf-8-sig",
  "target_column": "ACTUAL_LABEL_COLUMN",
  "positive_class": "ACTUAL_FRAUD_LABEL",
  "negative_class": "ACTUAL_LEGITIMATE_LABEL",
  "amount_column": "ACTUAL_AMOUNT_COLUMN",
  "timestamp_column": "ACTUAL_DATETIME_COLUMN",
  "timestamp_format": "ISO8601",
  "timezone": "UTC",
  "identifier_columns": ["ACTUAL_EVENT_KEY", "ACTUAL_ENTITY_KEY"],
  "unique_identifier_columns": ["ACTUAL_EVENT_KEY"]
}
```

Omit absent optional roles. Repeated entity identifiers are expected; only declared unique
identifiers fail validation when repeated. Target labels are exact string values and both
classes are required. `1` never implicitly means fraud. Configured missing targets fail
ingestion; absent optional fields produce warnings and unavailable analyses.

Text timestamps use the declared format and timezone (UTC by default); offset-bearing
values retain their instant. Ambiguous/nonexistent local times are invalid. Numeric times
require a unit and either `timestamp_kind: elapsed` or an explicitly supplied absolute
`timestamp_origin`. Elapsed times cannot produce real hour-of-day/week analyses without
a documented calendar origin. Do not invent one.

## Inspection and validation

Metadata reports storage dtype separately from inferred logical type, row/column counts,
unique counts, missingness, constant and near-constant columns, and high cardinality.
Numeric candidates require at least 95% numeric observations unless explicitly selected;
boolean candidates recognize textual true/false. Datetime inspection uses the first 1,000
non-missing ISO-shaped values. Identifier/target/time names and binary cardinality are
heuristics. Candidates carry `inferred: true` and an explanation; they are never committed
as roles. Configurable defaults include 98% dominance/uniqueness, 100 categorical values
and a 20-row minimum for small-sample-sensitive heuristics.

Quality rules cover missing cells and percentages, blank strings, duplicate rows and
identifiers, invalid/infinite numeric values, malformed configured timestamps, missing
configured fields, constants, near constants, high cardinality, suspicious uniqueness,
unsupported/missing target labels and class imbalance. Every warning has a code, severity,
explanation, affected column/count where relevant and inference flag. `ERROR` makes quality
`FAILED`; warnings do not automatically block the report.

Leakage candidates include exact target copies, outcome/post-outcome names, near-perfect
numeric target correlation and repeated low-cardinality categorical partitions that almost
determine the target. These are review signals, not causal findings or permission to delete
columns. Statistical association may be legitimate and small samples can mislead.

## Statistics and target intelligence

Numeric summaries include finite count, missing/invalid/infinite counts, mean, median,
range, sample variance/std (`ddof=1`), Q1/Q3/IQR, configured percentiles and adjusted sample
skewness. Undefined/overflowed results become JSON `null`, never NaN/Infinity. Missing,
invalid and infinite values are excluded explicitly. Floating-point analysis is descriptive,
not an accounting representation for exact currency arithmetic.

Target reports include both label counts/percentages, positive prevalence, unsupported and
missing counts, minority/majority classes and majority-to-minority imbalance ratio. Valid
labels are the percentage denominator, which is reported. Unsupported or missing labels
fail target validation and dependent EDA. A single observed class has a null imbalance
ratio and a warning; a tie has no unique minority/majority class.

## Structured EDA

Each analysis returns `AVAILABLE` with typed data or `UNAVAILABLE_WITH_REASON` with an
explicit reason. Missing prerequisites are never replaced with invented values.

| Analysis                           | Requirements / output                                        |
| ---------------------------------- | ------------------------------------------------------------ |
| Class distribution                 | Configured, valid binary target                              |
| Amount distribution                | Configured amount with finite observations                   |
| Fraud/legitimate amount comparison | Valid target and configured amount                           |
| Fraud frequency by hour            | Valid target and calendar timestamp                          |
| Fraud frequency by weekday         | Valid target and calendar timestamp                          |
| Correlation matrix                 | At least two suitable numeric fields; pairwise finite counts |
| Outliers                           | Suitable numeric fields; IQR and configurable z-score counts |
| Missing data                       | Per-column counts and percentages                            |
| Numeric distributions              | Reusable numeric statistics                                  |
| Categorical frequencies            | Top categories, missing and omitted-record counts            |

Configured targets and identifiers are excluded from continuous summaries/correlations.
Correlation is Pearson, requires three pairwise finite observations and defaults to a
40-column cap with exclusions listed. Category summaries default to 20 values and suppress
configured/candidate identifiers. IQR uses a configurable 1.5 multiplier; z-scores use
population standard deviation (`ddof=0`) and a default threshold of 3. Constant fields
report an unavailable z-score reason. Outliers are identified, never removed automatically;
fraudulent events may be genuine statistical outliers.

## Reproducibility and artifacts

The SHA-256 fingerprint hashes exact source bytes read for ingestion. Identical contents
at different paths have the same fingerprint; changes to encoding/newlines alter it.
Profiles also record a canonical configuration hash, pipeline version, dependency versions
and UTC generation time. Retain the operator configuration with an experiment manifest;
the hash alone cannot reconstruct it.

Default outputs are `ml/artifacts/data_profile.json` and
`ml/artifacts/data_quality_report.md`. Local JSON includes the source path; public responses
omit it. Each report is atomically replaced, but the pair is not a multi-file transaction.
Use one writer per artifact directory. Generated outputs are ignored by Git. An unconfigured
service does not read stale reports as a fallback.

## Reviewed cleaning

Profiling proposes a source-fingerprint-bound plan. To approve specific operations, save
the profile's `cleaning_plan` object to a separate reviewed JSON file and edit its operations.
Only duplicate removal, empty-string normalization, explicitly selected numeric coercion
and explicitly selected timestamp parsing exist. Conversions default to `errors: raise`;
`coerce` must be requested explicitly. Operations run on a deep copy and failure leaves the
source unchanged. No feature/column dropping, imputation, outlier deletion or label rewriting.

Each operation records the affected column, affected rows, before/after shape, dtype,
missing/unique counts, reason and error policy. Conversion row counts include converted
non-missing cells even when their displayed values appear unchanged. The audit also stores
source/processed hashes and generation time. Processed CSV and adjacent audit JSON must use
new paths; existing outputs and the source cannot be overwritten.

```powershell
# No dataset: documented state, no generated files.
.\.venv\Scripts\python.exe -m riskbyte_ml.data profile
# Approved real source, inspection only:
.\.venv\Scripts\python.exe -m riskbyte_ml.data profile --config ml/config/dataset.local.json
# Only after reviewing the separate cleaning plan:
.\.venv\Scripts\python.exe -m riskbyte_ml.data clean --config ml/config/dataset.local.json --plan ml/config/reviewed.local.json --output ml/data/processed/reviewed.csv
```

Profile exits 0 for passed quality or unconfigured state; validation/operational failures
exit 2. Quality failures still produce reviewable reports; fatal load errors do not.

## API

Set `RISKBYTE_DATASET_CONFIG_PATH=ml/config/dataset.local.json` in the root environment.
The API resolves configuration/artifact settings relative to the repository root. It loads
the dataset lazily, serializes concurrent first loads and caches one report
snapshot per process. Configuration/source stat changes invalidate the cache. Changing
bytes while deliberately preserving all file attributes requires a restart to force reload.

Read-only routes: `/api/v1/dataset/status`, `/profile`, `/quality`, `/statistics`, `/eda`,
`/schema` (the latter five share the same prefix). Every route returns exactly
`{"status":"dataset_not_configured"}` when no source exists in configuration.
Configured report routes return a typed `status` plus `data`. Quality failure uses
`dataset_validation_failed` with HTTP 200 and a failed quality report; malformed files,
missing files/targets and bad configuration return sanitized structured HTTP 422 errors
using the existing request-ID architecture. Unknown fields/nonfinite response numbers are
rejected by schemas, and OpenAPI describes the nested structures. No filesystem path or
raw Pandas object is exposed.

## IEEE-CIS adapter scope

Phase 1.5 adds a dataset-specific two-table adapter and reusable bounded batch loader.
It invokes the same Phase 1 profiling functions on a preloaded joined frame. Explicit
categorical roles override numeric-code inference. Generic CSV loading retains its existing
behavior. Read [IEEE_CIS_INTEGRATION.md](IEEE_CIS_INTEGRATION.md) before configuring this
adapter; the default application remains unconfigured and no source is downloaded.

## Generic pipeline limitations

One in-memory CSV is supported; no streaming, parquet, multi-table joins or chunked model
training. Memory can substantially exceed source size; raise bounds only after reviewing
available RAM. Correlations/categorical leakage checks are screening heuristics. No dataset
licensing/semantic validation can be automated completely. Aggregate categorical values may
still be sensitive: keep the unauthenticated development API on loopback; production access
control is future work. Synthetic fixtures live only under tests and smoke-test outputs.
