# Phase 9 frontend implementation

RiskByte now provides a React analyst workspace and public product pages. The ML stack,
calibration, feature pipeline, SHAP snapshots, ABRF configuration and final holdout results
are unchanged. FINAL_EVALUATED_LOCKED remains the current evaluation state.

## Run locally

From the repository root, start the API in one terminal:

```powershell
$env:RISKBYTE_DATASET_CONFIG_PATH = 'ml/config/synthetic_riskbyte.example.json'
$env:RISKBYTE_DATASET_ADAPTER = 'synthetic_riskbyte'
.\.venv\Scripts\python.exe -m uvicorn riskbyte_api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The Vite development proxy forwards `/api` to the validated
VITE_API_BASE_URL origin, avoiding a localhost/127.0.0.1 CORS mismatch. Production builds
use that configured API origin directly; configure the backend CORS allowlist for the
actual frontend origin. API startup verifies artifacts and imports actual saved demo
predictions once. First development EDA requests can be expensive while preparing the
cached profile; the interface shows loading/error/retry states rather than invented data.

Production POST/PATCH access requires the existing trusted operator proxy/key integration.
Never embed an operator secret in VITE_* settings. This phase does not add enterprise IAM,
an authentication screen or a misleading signed-in user profile.

## Routes

| Route                               | View                                                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `/`                                 | Product homepage with API-driven final metrics and scenario coverage                                         |
| `/technology`                       | Intelligence methodology, calibration/SHAP scope and fusion                                                  |
| `/privacy`, `/terms`                | Synthetic research environment policies                                                                      |
| unmatched routes                    | Useful 404 with return navigation                                                                            |
| `/app`                              | Actual current activity and separately labelled final performance                                            |
| `/app/transactions`                 | Search, filters, sorting, pagination and single/batch submission                                             |
| `/app/transactions/:id`             | Context, risk, evidence summary and review workflow                                                          |
| `/app/transactions/:id/explanation` | Four independent explanation layers                                                                          |
| `/app/explainability`               | Genuine global classifier SHAP and selected record attribution                                               |
| `/app/behaviour`                    | Development distributions, confidence and strictly-prior components                                          |
| `/app/anomaly`                      | Diagnostic-only evidence and labelled validation scenario metrics                                            |
| `/app/dataset`                      | Development class balance, missingness, amount quantiles, categorical frequencies, outliers and correlations |
| `/app/statistics`                   | Hypotheses, effect sizes, FDR p-values and descriptive summaries                                             |
| `/app/performance`                  | Canonical final comparison plus labelled development classifiers                                             |
| `/app/performance/scenarios`        | All final scenario recall, including weak velocity results                                                   |
| `/app/performance/hard-negatives`   | Final false-positive protection by subtype                                                                   |
| `/app/abrf`                         | Frozen configuration, confidence logic and exact transaction trace                                           |
| `/app/investigations`               | Flagged transaction queue with review-state filters                                                          |
| `/app/audit`                        | Paginated actual operational audit events                                                                    |
| `/app/system`                       | Current readiness, versions, fingerprints and final locked state                                             |
| `/app/settings`                     | Browser display density, page size, API connection and dataset notice                                        |

## Data and explanation contracts

`src/lib/api.ts` is the sole application fetch boundary. Zod validates operational DTOs,
score bounds, explicit unavailability and diagnostic-only anomaly semantics. Research
payloads use validated JSON and safe typed readers. `src/lib/queries.ts` owns TanStack
Query hooks, cancellation, retries, caching and review invalidation. No application `any`
casts, fabricated fallback metrics or raw Pandas objects are used.

Final headline performance always comes from `/evaluation/final`, `/scenarios` and
`/hard-negatives`, corresponding to [FINAL_SYSTEM_RESULTS.md](FINAL_SYSTEM_RESULTS.md).
Dashboard activity has a different population and never substitutes for final evaluation
denominators. Research, model-comparison and global SHAP views clearly label development
or validation populations. XGBoost remains unavailable, with no invented measurements.

Tree SHAP explains the underlying Decision Tree. Raw and calibrated probabilities are
separate; top factors are a truncated view. Anonymized feature metadata is retained without
invented business semantics. Behavioural evidence, anomaly diagnostics and exact ABRF
calculation are separate components. Anomaly effective weight/contribution is zero.
Fused risk is not a literal fraud probability. Cold starts retain explicit unavailable
behaviour and confidence; event-level entity history is not invented when the API does
not expose it. TransactionDT is elapsed time, never a fabricated calendar date.

Single input uses React Hook Form and Zod, with optional merchant/device/region and
pre-outcome signals. Blank optional fields remain unknown. The JSON batch editor accepts
the full API input schema, bounds count at 500 and relies on strict server validation.
Prediction and review writes invalidate actual affected views. Review decisions leave
model scores unchanged. The only backend extension is the bounded read-only `/api/v1/audit`
route, exposing UTC timestamps and an explicit safe metadata allowlist.

## Components, accessibility and performance

Reusable primitives include Button, IconButton, Dialog, Breadcrumb, Pagination, DataTable,
StatusIndicator, RiskBadge, RiskMeter, MetricDisplay, ChartContainer, PageHeader,
SectionHeader, EmptyState, ErrorState, Skeleton, Notice and Toast. Domain components own
transaction tables, SHAP factors, behavioural evidence and fusion traces. Native labelled
inputs, selects, checkboxes and expandable sections provide accessible controls without
unused component abstractions.

The desktop sidebar and mobile navigation share routes. Mobile navigation traps focus,
supports Escape and restores focus. Native dialogs provide modal semantics and focus
handling. One primary heading per route, skip links, semantic landmarks, visible focus,
explicit status text and reduced-motion support are implemented. Mobile tables become
labelled record cards rather than shrinking desktop rows. No formal WCAG certification
is claimed.

Route content is lazy loaded; Recharts, the prediction form and the React Three Fiber scene
load separately. Charts use genuine summaries, tooltips and legends. No GSAP or Zustand was
needed for these interactions. Transaction intelligence now maps actual recent transaction,
behavioural-entity, merchant and device relationships in a bounded 3D scene. Its equivalent
keyboard-operable 2D view handles compact screens, reduced motion, unavailable WebGL and
renderer failures. See [FINAL_FRONTEND_QA.md](FINAL_FRONTEND_QA.md) for the final audit.

## Verification and limitations

Frontend verification runs `npm run lint`, `npm run typecheck`, `npm run test`,
`npm run build` and `npm run format:check`. The focused suite includes 17 contract/graph
unit tests and 17 browser integration checks covering routes, final metric source,
transaction filtering/pagination, detail, review updates, explanation layers, zero anomaly
weight, fusion display, scenarios, 3D relationships, fallback behaviour, loading/error
states, prediction form, batch limits and mobile navigation. Browser tests intercept
explicitly labelled synthetic UI fixtures;
they do not fit models, generate holdouts or fabricate application fallback data.

The browser runner uses installed Chrome/Edge on Windows, RISKBYTE_BROWSER_PATH if set,
or Playwright-managed Chromium. On a machine without a supported browser, install one
or run `npx playwright install chromium`. Node and package versions follow the repository
lock; required packages were sourced from the existing local cache in this restricted
environment, with standard registry URLs retained in package-lock.json.

A separate live API audit checked 22 routes. Overview, transactions, transaction detail,
unified explanation, investigations and performance were checked at 1920, 1440, 1280,
1024, 768 and 430 pixels without document overflow. The Python suite passed 540 tests,
including bounded/redacted audit history. Chrome and software WebGL are verified;
cross-browser and formal assistive-technology certification remain
outside this phase. Synthetic metrics cannot prove production banking performance.
PostgreSQL migration execution and enterprise authorization remain prior backend limits.
