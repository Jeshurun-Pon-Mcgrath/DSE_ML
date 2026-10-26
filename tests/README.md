# Foundation verification

Phase 4 adds small synthetic classifier fixtures, hand-calculated imbalanced metrics,
probability bounds, calibration/threshold contracts, hard-negative/scenario summaries,
reproducible training, frozen-input and completed-artifact tampering checks, serialization,
registry uniqueness, one-use test isolation, KNN benchmark fallback and typed model-report
API tests. Tests remain offline and never train on the real MEDIUM development dataset.

Phase 2 adds small scenario-generated fixtures, two-table integration, split determinism,
strictly prior amounts/velocity/novelty, equal-time batching, outcome/sidecar exclusion,
train-only numeric/category fitting, unseen category handling, sparse alignment,
serialization/reload, version/provenance guards and synthetic metadata API tests.
Tests are offline and never generate MEDIUM/LARGE. The MEDIUM offline pipeline is separate
development verification; it establishes simulated measurements, not model accuracy.

Install both Python packages in editable mode before running `python -m pytest` from
the repository root. API tests use independent test settings without a root `.env`,
database, artifacts or external services. Failure routes exist only in test fixtures.
ML tests cover package import safety and the configurable ABRF weight contract.

Frontend checks are root npm scripts: `typecheck`, `lint`, `build`, `format:check`.

Phase 1 covers strict CSV loading/errors, schema candidates, quality rules, configurable
labels, finite statistics, IQR/z-score reporting, EDA availability, deterministic hashes,
explicit cleaning/audits, CLI and six dataset API routes. All records are clearly labelled
synthetic fixtures; no network, real dataset, trained model or database is needed.
Full browser/domain/model tests will follow when those behaviors are implemented.

Phase 1.5 adds synthetic two-table join/cardinality/coverage tests, combined hashing,
lossless memory optimization, elapsed-time safeguards, proxy entity measurements, feature
classification/contracts and metadata API privacy/error tests. Real Kaggle files are unnecessary.
