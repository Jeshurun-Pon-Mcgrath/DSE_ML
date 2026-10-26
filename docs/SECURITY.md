# Security foundation

Phase 2 generates only fictional academic records under ignored paths. Evaluation-only
scenario/hard-negative annotations are excluded from normal dataset API/model inputs.
Trusted local preprocessing joblib files are executable serialization: never accept them
as uploads or load untrusted artifacts. Artifact fingerprint/version/row-order checks
establish compatibility, not authenticity or protection from malicious pickle execution.
The ignored local development `.env` selects the generated synthetic dataset; no credentials
are stored. Production authentication and exposure controls remain future work.

## Current boundary

Phase 1 is a local development data-inspection API. There is no authentication,
authorization or production deployment. Future risk endpoints
must not be exposed publicly before those controls are designed and tested.

## Implemented controls

- Secret-free `.env.example`; local environment files, private keys, data and model artifacts
  are ignored by Git. Public `VITE_*` variables must never contain credentials.
- Typed startup configuration; exact-origin CORS allowlist with credentials disabled.
  CORS controls browser access and does not replace authorization.
- Server-generated request IDs; structured HTTP/validation/server-error envelopes.
  Validation responses omit supplied input, exception context and raw messages.
- JSON console logs omit request bodies, database URLs and traceback/exception messages.
  Access logging is disabled to avoid recording sensitive query strings. Use constant
  event messages; the formatter is not a general-purpose secret redactor.
- Debug responses are disabled; development API docs are disabled in production mode.
- Servers bind to loopback in documented development commands. PostgreSQL connections
  are optional and no tables or credentials are provisioned.

## Phase 1 data controls

Paths are server-owned configuration, never HTTP input. Strict bounded CSV ingestion does
not load remote URLs or execute serialized models. Domain errors sanitize paths/detail;
public schemas omit source paths. Raw/processed data, local configuration and reports are
ignored by Git. Aggregates and column names can still be sensitive: keep the unauthenticated
API on loopback and protect local artifacts. CORS is not access control.

Cleaning operates explicitly on copies and refuses source/existing-output overwrite.
CSV outputs are data, not spreadsheet-safe exports; opening untrusted values as spreadsheet
formulas requires a separate export policy. Resource bounds do not replace production
authentication, quotas or worker isolation. No online vulnerability audit is claimed.

## Before future deployment

Define identity/authentication and least-privilege authorization for assessments,
administration, history and explanations. Enforce request size/rate limits and bounded
inference workloads. Use TLS, protected network access, secret injection/rotation and
database roles with minimum privileges. Introduce reviewed migrations and connection
pool limits. Assess CSRF protection if adopting cookie authentication.

Minimize or pseudonymize identifiers, document lawful access, restrict histories and set
retention/deletion policies for events, features, scores, explanations and audit logs.
Do not log transaction payloads, identity documents, secrets or unnecessary sensitive fields.
Protect backups and test restoration/deletion workflows.

Accept only approved artifact sources with manifests, checksums and version compatibility.
Serialized Python model files can execute code when loaded; never load user-supplied
pickle/joblib artifacts. Training and notebooks must remain outside request handling.
Evaluate adversarial input, model abuse and explanation leakage before exposing scoring.

Review locked dependencies for vulnerabilities and licenses when updating or deploying.
No online vulnerability audit is claimed by Phase 0 checks. Future deployment needs
monitoring, incident response, audit access control and rollback plans.

# Phase 8 operational controls

Prediction input is strict, outcome-free and bounded. Batches are capped at 500; actual
request bytes are capped before JSON parsing. Duplicate/historical IDs and out-of-order
events are rejected. CORS uses an explicit allowlist. Production inference requires an
operator-provided write API key; POST/PATCH validate it without logging credentials.
Development remains a trusted local synthetic demo. No enterprise IAM is implemented.
Audit events record UTC timestamp, transaction ID, action and safe model/state metadata,
without financial amounts or raw request payloads. Reviews cannot alter scores or labels.
Final artifact hashes and component provenance are checked before runtime startup.
PostgreSQL/migration execution remains unverified until optional database dependencies
are installed. Real banking input authenticity and production assurance are not claimed.
