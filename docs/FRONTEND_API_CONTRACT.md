# RiskByte final frontend API contract

Use `/api/v1` on the configured API origin. Performance pages, report, PPT and viva must
use [FINAL_SYSTEM_RESULTS.md](FINAL_SYSTEM_RESULTS.md) and its locked `/evaluation/*`
artifacts. Older `/models`, `/statistics`, `/anomaly`, `/abrf` and `/explainability` routes
remain historical research views; their snapshot holdout statuses are historical, not the
current system state. Current status is `/system/model` and `/system/health`.

## Endpoints

| Page/action          | Method and path                                    | Contract                                                                                                                                                           |
| -------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dashboard            | GET /dashboard/summary                             | transaction_count, risk_level_distribution, investigation_count, average_risk_score, risk_trend, recent_transaction_activity, recent_high_risk_transactions, scope |
| Transactions         | GET /transactions                                  | total, page, page_size, items; each item has transaction, prediction, investigation                                                                                |
| Detail               | GET /transactions/{transaction_id}                 | transaction context, prediction, investigation review state                                                                                                        |
| Explanation          | GET /transactions/{transaction_id}/explanation     | strict UnifiedExplanation: risk_summary, supervised tree SHAP, behaviour components/confidence, anomaly diagnostics, fusion and formula trace, limitations         |
| Submit transaction   | POST /predict                                      | strict transaction object; returns PredictionResponse                                                                                                              |
| Submit bounded batch | POST /predict/batch                                | {transactions: [transaction objects]}; returns prediction array in input order                                                                                     |
| Review workflow      | PATCH /transactions/{transaction_id}/investigation | {state: UNREVIEWED, UNDER_REVIEW, CLEARED or ESCALATED}; returns transaction_id, state, scores_unchanged                                                           |
| Final performance    | GET /evaluation/final                              | frozen final and supervised metrics, fingerprint, counts, improvement accounting                                                                                   |
| Scenarios            | GET /evaluation/scenarios                          | each scenario: count, supervised/abrf/anomaly count, flagged, rate                                                                                                 |
| Hard negatives       | GET /evaluation/hard-negatives                     | each type: count, supervised/abrf count, flagged, rate                                                                                                             |
| Model/about          | GET /system/model                                  | dataset_type=synthetic, notice, model/model ID, calibration, versions, fingerprints, system status, minimum_elapsed_seconds, final_holdout_status                  |
| Readiness            | GET /system/health                                 | API/inference readiness, supervised/anomaly/behaviour/ABRF/SHAP loaded flags, database_ready, safe unavailable_reason                                              |
| Audit history        | GET /audit                                         | total, page, page_size, items; each event contains id, transaction_id, action, timestamp (UTC) and allowlisted review metadata                                     |

Transactions query: `page>=1`, `1<=page_size<=100`, optional `risk_level` in
LOW/GUARDED/ELEVATED/HIGH/CRITICAL, `investigation_required=true|false`,
`investigation_state=UNREVIEWED|UNDER_REVIEW|CLEARED|ESCALATED`, `sort=recent|risk`,
and ID substring `search` (maximum 100 characters). Sorting is descending elapsed time
or risk, with transaction ID as deterministic tie-breaker. No unlimited listing or batch.

Audit query: `page>=1`, `1<=page_size<=100`, optional exact `transaction_id` (maximum 100
characters). Events sort by descending event ID. Only previous/from/to/state/origin
metadata keys are exposed; credentials, amounts and arbitrary internal fields are excluded.
This Phase 9 read-only extension does not modify any scoring or final-evaluation artifact.

## Strict prediction input

Required fields: `TransactionID` (new ASCII ID, 1–100 letters/digits/underscore/hyphen),
`TransactionDT` (finite elapsed seconds, 0–1e12), `TransactionAmt` (finite 0–1e12),
`behavioural_entity_id` (ASCII grouping ID, 1–100 characters).

Optional nullable fields: merchant_id, merchant_category, transaction_type, product_code,
card_type, card_network, billing_region, shipping_region, P_emaildomain, R_emaildomain,
failed_attempts_before, account_age_days, is_international, is_card_present, DeviceType,
DeviceInfo, device_id, os_family, browser_family, identity_region, identity_age_days.
Strings are bounded at 80–100 characters. Failed attempts is a nonnegative integer up to
100,000; ages are finite nonnegative values up to 100,000; binary flags are 0/1. Missing
optional context remains unknown and uses frozen preprocessing, not invented defaults.
No target, fraud_scenario, hard_negative, risk score or caller-supplied history features
are accepted. Unknown fields and non-finite values fail validation.

New event times must be at least the committed global watermark, exposed as
`minimum_elapsed_seconds`. Equal-time peers are excluded from each other's history;
out-of-order requests receive 409. Historical demo records are read/explained through GET,
not reposted. IDs must be unique; the original numeric development namespace 1–100,000
is reserved. Duplicate retries receive 409; fetch the existing ID
to resolve a possibly committed retry. A batch is atomic and uses only strictly prior
observed history; target labels never enter inference.

## Prediction response

`transaction_id`; `risk={score,level,investigation_required}`;
`supervised={probability,prediction,model_version}`;
`anomaly={score,flag,diagnostic_only:true,abrf_weight:0}`;
`behaviour={score:null|number,confidence,history_count,available}`;
`abrf={version,base_weights,effective_weights,contributions,reason_codes}`;
`explanation={status,top_increasing_features,top_decreasing_features,behavioural_reasons}`;
`versions={feature,anomaly,behaviour,abrf,explainability,shap}`.

R is a fused risk score, not fraud probability. P is the frozen calibrated supervised
probability on synthetic data. A/B are anomaly/deviation scores, not probabilities.
Anomaly can flag diagnostically but has **zero influence** on active ABRF risk.
Cold starts retain null B and explicit confidence/availability; missing history is not safety.
Risk bands and the binary investigation threshold are independent and frozen.

POST responses include real local SHAP. Imported demo predictions whose explanation has
not been requested use `explanation.status=NOT_REQUESTED` and empty lists; this is not a
zero-attribution result. The explanation GET lazily computes and caches actual SHAP using
the stored strictly-prior feature vector and must reproduce its saved risk score.
Tree SHAP explains `underlying_decision_tree`, not sigmoid-calibrated P or full ABRF.
The unified contract exposes raw_tree_probability and calibrated_supervised_probability
separately. Interpretability and transformation metadata must be retained in UI wording.

## Display and workflow rules

All data is synthetic. Display the reusable `/system/model.notice` in dataset/about areas;
do not fabricate revenue, prevented losses or real customer statistics. Dashboard counts
are actual demo/live activity; performance denominators are the locked 30,000-row holdout.
Risk trends use synthetic elapsed_day_index, never fabricated dates or weekdays.
Human review flags do not establish guilt. SHAP is model behaviour, not causation or intent.
Clearing/escalating review does not change risk, model flags, final metrics or ground truth.

## Limits, security and errors

Maximum batch 500 (configurable downwards); maximum request body 2,000,000 bytes by default,
enforced before JSON parsing including chunked bodies. Pydantic violations return 422;
byte/configured batch limits return 413; duplicates/out-of-order requests 409; unknown IDs
404; unavailable runtime 503. Errors use `{error:{code,message,request_id,issues?}}`, with
sanitized details and `X-Request-ID`. CORS allows only configured origins and GET/POST/PATCH.

Development is a trusted local synthetic demo. If `RISKBYTE_WRITE_API_KEY` is configured,
POST/PATCH require `X-Api-Key`. Production inference requires an operator-provided key of
at least 32 characters. Do not embed server/operator secrets in public VITE_* configuration;
production needs a trusted proxy/session integration. This is not enterprise IAM.
Audit events persist timestamp, ID, action and safe metadata; no credentials or amounts
are logged. Read-only endpoints expose no local file paths.

## Backend setup

```powershell
$env:RISKBYTE_DATASET_CONFIG_PATH = 'ml/config/synthetic_riskbyte.example.json'
$env:RISKBYTE_DATASET_ADAPTER = 'synthetic_riskbyte'
.\.venv\Scripts\python.exe -m uvicorn riskbyte_api.main:create_app --factory --host 127.0.0.1 --port 8000
```

With the configured frozen synthetic source and locked final artifacts, startup loads
models/calibration/preprocessor/anomaly/behaviour/ABRF/SHAP once and imports actual demo
predictions into ignored `ml/artifacts/api.sqlite3`. Subsequent requests do not reload
artifacts. SQLite and PostgreSQL transaction locks protect sequential submissions across
writers; base checkpoint history stays fixed and later entity history is queried from
the indexed durable store. No growing duplicate in-memory live history is maintained.

PostgreSQL needs the optional `apps/api[database]` dependencies and the migration
`alembic -c infra/alembic.ini upgrade head` against an empty prepared database. Existing
development SQLite tables use SQLAlchemy bootstrap. Alembic was not installable here;
its migration files are provided, but remote PostgreSQL/migration execution was not tested.
No credential is committed. Model provenance and request handling are verified, but
production banking performance, input authenticity and enterprise deployment are not claimed.
