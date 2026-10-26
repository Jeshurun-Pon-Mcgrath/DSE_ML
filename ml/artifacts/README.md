# Model artifact boundary

Phase 4 stores trained core classifiers, trusted executable joblib bundles, checksummed
metadata and one selected supervised model in `models/`. `evaluation/` holds the validation
comparison, curves, report-ready tables, selection freeze and one-time selected-model test
results. Frozen Phase 2 source, feature manifest, split and preprocessors remain unchanged.
XGBoost is explicitly unavailable in this workspace; no fabricated model artifact is saved.
See [measured supervised evaluation](../../docs/SUPERVISED_MODEL_EVALUATION.md).

Phase 2 adds synthetic dataset manifests, fingerprint-bound splits, reviewable feature
lineage, fitted train-only preprocessors and actual simulated-data reports. No fraud model
is trained. Default outputs are documented in [SYNTHETIC_DATASET.md](../../docs/SYNTHETIC_DATASET.md).
Joblib serialization is executable and restricted to trusted local artifacts. Source records
are not embedded in preprocessors; feature matrices are generated/reload-verified in memory.

Phase 1 writes ignored `data_profile.json` and `data_quality_report.md` here only when
a source is configured. Local profiles include provenance paths and potentially sensitive
aggregates. No fake reports are generated when unconfigured. Tests/synthetic smoke runs
use temporary/test directories instead.

No models are trained or bundled in Phase 0. Future immutable bundles should include
preprocessing, feature schema, classifier, anomaly detector, ABRF configuration,
explanation background reference, evaluation report, dependency lock and checksums.
Generated artifacts are ignored by Git and must be stored in controlled artifact storage.

Phase 1.5 real IEEE reports belong under `ieee_cis/` only after both local sources are supplied.
Reports include join audits, combined provenance, entity studies, leakage/missingness review
and a proposed feature contract. No model or processed joined CSV is generated.
