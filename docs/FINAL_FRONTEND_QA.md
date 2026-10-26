# Final frontend quality assurance

## Scope and frozen-system integrity

This final implementation pass adds transaction-network visualization and product polish to
the existing Phase 9 React application. It does not retrain, recalibrate or change the
supervised model, anomaly detector, behavioural engine, ABRF, SHAP logic, final holdout or
API contracts. `FINAL_SYSTEM_RESULTS.md` and the locked evaluation APIs remain the sole
sources for final performance values.

## 3D transaction intelligence

The Overview route now contains a React Three Fiber and Drei network. Transaction nodes
connect only to non-empty `behavioural_entity_id`, `merchant_id` and `device_id` fields in
the API response. Shared relation nodes are deduplicated. No inferred customer identity or
relationship is added. The dashboard supplies 10 recent transactions; the graph builder has
a hard ceiling of 24 transactions for future callers.

Transaction risk uses both node size and the existing risk-band tone. Entity, merchant and
device nodes use distinct geometry. Selecting a transaction adds a high-contrast ring,
emphasizes its observed edges, dims unrelated context and opens a compact inspector with
the actual fused score, calibrated supervised probability, behavioural score, anomaly
diagnostic, investigation state and relationship identifiers. Links open the existing
transaction detail and unified explanation routes. The anomaly value is labelled
diagnostic-only and its active ABRF contribution remains zero.

The camera has bounded orbit and zoom, no automatic orbit, and an explicit Reset view
control. The scene uses restrained lighting and a static analyst-workstation composition.
There are no particles, constant rotation or decorative motion.

## Fallback and performance

The Three.js scene is a separate lazy production chunk. An IntersectionObserver defers its
creation until the graph approaches the viewport. Rendering uses a bounded device-pixel
ratio, demand-driven frames and stops while the view or document is inactive. Geometry is
bounded and deterministic; relation edges are drawn as line segments rather than individual
animated objects.

A keyboard-operable SVG network uses the same actual transaction records when WebGL is
unavailable, reduced motion is requested, the viewport is compact, the device exposes very
limited resources, or the 3D component fails to load. Users can deliberately switch between
2D and 3D when WebGL is available. The 2D view preserves risk size/tone, selection, context
dimming and the evidence inspector. A failed canvas never replaces the underlying data with
fabricated content.

The production build keeps the 3D dependency out of the main application chunk. Measured
minified output is 876.85 kB / 236.18 kB gzip for the independently loaded Three.js scene;
the main application chunk is 344.33 kB / 106.15 kB gzip. Recharts is also independently
loaded. This is a development-machine bundle measurement, not a production latency claim.

## Responsive and visual verification

All 22 configured routes were loaded against the real local API at 1440 px. None produced a
blank page, route-level error state, document overflow or uncaught page exception. Overview,
transactions, transaction detail, risk explanation, investigation queue and final model
performance were additionally checked at 1920, 1440, 1280, 1024, 768 and 430 px. The 3D
canvas is used on capable desktop layouts; the useful 2D network is the default at 430 px.

The final pass checked alignment, panel hierarchy, long identifiers, responsive record
tables, chart containment, empty states and error states across public, analyst, research,
model, investigation and legal views. Chrome 155.0.8059.40 on Windows was used for the live
route and WebGL audit. Cross-browser device coverage remains a deployment-stage concern.

The locked evaluation endpoint was checked directly during the audit. Its precision,
recall, F1, PR-AUC, ROC-AUC, FPR and hard-negative FPR match the canonical final results.
The application source contains no copied headline metric constants; UI values continue to
come from the evaluation APIs.

## Accessibility checks

- Semantic headings, landmarks, links and buttons remain intact on every route.
- The canvas is presentation-only and has an equivalent keyboard-operable transaction list,
  SVG fallback and textual evidence inspector.
- Transaction status is communicated through labels, geometry/size and text as well as
  colour.
- Visible focus, mobile navigation focus trapping, Escape handling and native dialog focus
  behaviour remain covered.
- Reduced-motion preference selects the static analytical view and disables CSS motion.
- Loading, empty and error states remain explicit; raw exceptions are not rendered.

These checks do not claim formal WCAG certification or assistive-technology coverage.

## Automated verification

Frontend verification includes 17 Vitest contract/graph tests and 17 Chrome integration
checks. The browser suite covers real 3D canvas initialization with software WebGL, actual
relationship selection, the analytical 2D fallback, reduced motion, API-driven final
metrics, transaction workflows, four explanation layers, zero anomaly weight, ABRF,
scenarios, hard negatives, loading/error states and mobile navigation. The Python regression
suite covers the frozen backend/ML system separately.

Required lint, strict type checking, formatting, production build, Python lint/format,
compilation and dependency integrity checks are part of the final verification record. The
frozen holdout remains `FINAL_EVALUATED_LOCKED` and was not rescored.

## Known limitations

- The 3D network is a bounded relationship view, not a complete graph of all 30,000 holdout
  transactions.
- Device nodes are absent when identity context is unavailable; the UI does not infer them.
- 3D performance was measured with desktop Chrome software WebGL, not a representative set
  of mobile GPUs.
- Formal screen-reader, WCAG certification, Safari/Firefox testing and production network
  profiling remain outside this local QA pass.
- All displayed records and evaluation findings are synthetic academic data and do not
  establish production banking performance.
