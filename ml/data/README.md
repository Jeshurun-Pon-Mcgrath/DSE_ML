# Local data boundary

Phase 2 explicitly authorizes synthetic application data under `synthetic_riskbyte/`.
This is an IEEE-CIS-inspired simulation, not competition data. Transaction/identity CSVs,
evaluation-only annotations, entity profiles and generation metadata are ignored. Generate
explicitly; no fake replacements are created by ingestion. See [methodology](../../docs/SYNTHETIC_DATASET.md).

No real dataset is supplied or approved in Phase 1. Keep authorized local datasets here;
contents are ignored by Git. Later use `raw/`, `interim/`, and `processed/` partitions,
with dataset manifests and access controls maintained outside sensitive payloads.
Do not place real customer or payment data in source control.

Configure approved CSVs with `ml/config/dataset.local.json`; see
[pipeline guide](../../docs/DATA_PIPELINE.md). No automatic downloads or source mutation.
Reviewed cleaning writes a new processed file and audit. Synthetic fixtures belong in tests.

Phase 1.5 expects approved IEEE-CIS `ieee_cis/train_transaction.csv` and
`ieee_cis/train_identity.csv`. These paths are ignored. Neither exists in this checkout;
the adapter reports a dataset configuration error, with no download/replacement or model fit.
