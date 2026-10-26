# Data dictionary

## Implemented Phase 2 synthetic schema

[The synthetic dictionary](SYNTHETIC_DATASET.md) defines the approved two-table fields,
units, simulated identities/devices/merchants, nullable context and evaluation-only sidecars.
`behavioural_entity_id` is BEHAVIOURAL_KEY, excluded from direct model input. `isFraud`
is TARGET; scenario/hard-negative fields are TARGET_METADATA / EXCLUDED. Transaction IDs
are audit-only, TransactionDT is elapsed seconds, and optional V1–V20 are ANONYMIZED.
Merchant/device IDs are partially interpretable synthetic identifiers, approved only for
prior novelty and training frequency encoding. The manifest lists every raw/derived/encoded
feature, dependencies, missingness policy, eligibility and temporal/leakage status.

Prior std uses population variance (`ddof=0`). Prior median uses exact earlier finite amounts.
Velocity windows include their lower boundary and exclude current/equal-time events.
Novelty is null when either current context or prior context evidence is absent, with an
availability flag. A first entity event has zero history count and unavailable behavioural
evidence. Relative time position divides elapsed seconds by the configured synthetic horizon,
never by a learned maximum. No calendar or actual GPS travel semantics are implied.

## Status

No real dataset is provided in Phase 1. The following fields are **proposed logical fields**
for a future transaction-risk domain, not existing columns, guarantees of data availability
or an implemented API schema. Map and approve them against the actual dataset before
implementing preprocessing. Do not invent real customer data for this scaffold.

| Proposed field      | Logical type           | Meaning / constraints                                            | Role                                        |
| ------------------- | ---------------------- | ---------------------------------------------------------------- | ------------------------------------------- |
| `event_id`          | String                 | Unique opaque event key; validate uniqueness                     | Audit/join; normally exclude from model     |
| `entity_id`         | Pseudonymous string    | Entity/account key; protected linkage                            | Past-context joins; normally exclude raw ID |
| `event_time`        | UTC timestamp          | Time event occurred; distinguish ingestion time                  | Temporal splitting and feature cutoff       |
| `ingested_at`       | UTC timestamp          | Time source event became available                               | Availability/leakage checks                 |
| `amount`            | Decimal                | Non-negative transaction amount; refunds need explicit semantics | Candidate feature                           |
| `currency`          | ISO currency string    | Unit of amount; conversions require event-time reference         | Candidate feature/context                   |
| `label`             | Nullable binary        | Approved fraud outcome; unknown is not negative                  | Offline target only                         |
| `label_observed_at` | Nullable UTC timestamp | When the outcome became available                                | Label maturation; exclude from model        |

Candidate derived features include prior event count over a defined window and deviation
from a past amount baseline. Exact windows, required history, missing-value encodings,
units and definitions remain undecided. Never derive features using later events or labels.

## Implemented Phase 1 dataset contracts

Actual names/types are discovered from each CSV; the proposed fields above are not required.
Storage string dtype is separate from inferred logical type. Candidate identifiers, targets
and timestamps carry `inferred: true` and a reason. Only explicit mappings enable role-dependent
EDA. Full authoritative fields/types live in `riskbyte_ml.data.models` and generated OpenAPI.

| Contract / field                                          | Meaning                                                                |
| --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `DatasetConfig.source_path`                               | Operator-only source; null means unconfigured                          |
| `mode`                                                    | `inspection` candidates only; `explicit` selected roles                |
| `positive_class`, `negative_class`                        | Exact string labels; no implicit 1/0 mapping                           |
| `identifier_columns`                                      | Entity/event identifiers; repetition may be expected                   |
| `unique_identifier_columns`                               | Explicitly unique event keys, subset of identifiers                    |
| Timestamp configuration                                   | Declared kind, unit, origin, format and timezone                       |
| Profile `fingerprint`                                     | SHA-256 of exact source contents                                       |
| `configuration_fingerprint`                               | Canonical policy hash excluding source path                            |
| `generated_at`, `pipeline_version`, `library_versions`    | Provenance                                                             |
| `schema`, `statistics`, `quality`, `eda`, `cleaning_plan` | Strict nested reports                                                  |
| Quality `validation_status`                               | `PASSED` or `FAILED`; warnings carry severity/reason                   |
| Target `status`                                           | `valid`, `invalid` or `not_configured`                                 |
| Analysis `status`                                         | `AVAILABLE` or `UNAVAILABLE_WITH_REASON`                               |
| Numeric null                                              | Undefined/unrepresentable statistic, never NaN/Infinity                |
| Audit                                                     | Operation, affected column/rows, before/after state, reason and policy |

Report routes return `status` plus typed `data`; all unconfigured routes return only
`{"status":"dataset_not_configured"}`. Configured status is `dataset_ready` or
`dataset_validation_failed`. Local JSON includes `source_path`; HTTP schemas omit it.
No raw records/dataframes are returned. Fatal ingestion failures reuse the existing error
envelope and request ID. Missing labels never become legitimate labels. Class percentages
use the reported valid-label denominator; a missing minority gives a null imbalance ratio.
Currency/amount validity needs a dataset dictionary: negative values may represent refunds.

## IEEE-CIS adapter roles and metadata

These names apply only to the dedicated IEEE adapter, not arbitrary datasets:

| Field                                                     | Role / interpretation                                              |
| --------------------------------------------------------- | ------------------------------------------------------------------ |
| `TransactionID`                                           | Exact string audit/join key; direct predictive exclusion candidate |
| `isFraud`                                                 | Dataset-specific `1` fraud / `0` legitimate target                 |
| `TransactionDT`                                           | Elapsed seconds from undisclosed origin; not calendar time         |
| `TransactionAmt`                                          | Readable transaction amount; source string precision retained      |
| Product/card/address/email/M/device/categorical id family | Explicit categories; context/codes need review                     |
| `V1`–`V339`                                               | Anonymized engineered fields; no asserted business meaning         |
| C/D/dist/id numeric fields                                | Inspect semantics/availability; not universally interpretable      |

Feature metadata has overlapping INTERPRETABLE, PARTIALLY_INTERPRETABLE, ANONYMIZED,
IDENTIFIER, TEMPORAL, TARGET, POTENTIAL_BEHAVIOURAL_KEY and EXCLUDED_CANDIDATE tags.
Join audit includes counts, duplicate/missing keys, coverage, added columns and target integrity.
Memory audit measures retained dataframe estimates, not peak RSS. Integration fingerprint
combines both content hashes and adapter/config versions. Entity metrics describe complete
proxy tuples, never real people. No raw entity values or filesystem paths are in integration
HTTP responses. [Detailed semantics](IEEE_CIS_INTEGRATION.md).

## Foundation inference contracts

`Record` is currently a generic mapping and `FeatureVector` is a mapping of named numeric
features. These are placeholders for module interoperability, not runtime data validation.
Future concrete schemas must enforce allowed fields, stable ordering, finite values,
units, nullability, valid ranges and feature-schema version.

`ComponentSignal` carries a classifier/anomaly/behavioural name, score and component
version. `RiskAssessment` carries fused score, signals, an optional explanation and
model-bundle/ABRF versions. These dataclasses are internal contracts only and are not
exposed through a production scoring endpoint. `Explanation` must name its target and
include baseline/output-space metadata when finalized.

## Implemented public responses

| Endpoint/field           | Type                   | Meaning                                                      |
| ------------------------ | ---------------------- | ------------------------------------------------------------ |
| `GET /health`: `status`  | Literal `ok`           | API process liveness                                         |
| `GET /health`: `service` | Literal `riskbyte-api` | Service identifier                                           |
| `GET /health`: `version` | String                 | Package version                                              |
| `error.code`             | String                 | Stable error category (`http_404`, `validation_error`, etc.) |
| `error.message`          | String                 | Sanitized human-readable summary                             |
| `error.request_id`       | String                 | Generated request correlation ID                             |
| `error.issues`           | Optional array         | Validation field locations and codes; no input values        |
| `X-Request-ID`           | HTTP header string     | Server-generated correlation ID                              |

Future dataset manifests must include source/license, owner, snapshot hash, row count,
schema version, observation period, label semantics, retention and access classification.
Raw data and model artifacts are excluded from Git; documentation is tracked.
