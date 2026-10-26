# IEEE-CIS integration — Phase 1.5

## Current local state

Checked on 2026-10-11: **neither real training CSV was found** at
`ml/data/ieee_cis/train_transaction.csv` or `ml/data/ieee_cis/train_identity.csv`.
Actual transaction/identity/joined counts, prevalence, coverage, missingness, memory usage,
ordering associations and entity statistics are **unavailable**. No fake replacement,
download or synthetic application profile was created. Test fixtures are explicitly synthetic.

The integration implementation is ready to inspect approved local files. A missing source
returns a sanitized dataset configuration error; application startup and `/health` remain
available. The generic CSV adapter and its default unconfigured state remain unchanged.

## Setup and commands

Manually obtain both labelled IEEE-CIS training CSVs after reviewing access/competition
terms. Put them at the expected ignored paths. Do not commit data, reports or credentials.
Copy `ml/config/ieee_cis.example.json` to `ml/config/ieee_cis.local.json`; inspect limits
against available RAM before running. Relative file paths resolve against the JSON directory.

```powershell
Copy-Item ml/config/ieee_cis.example.json ml/config/ieee_cis.local.json
.\.venv\Scripts\python.exe -m riskbyte_ml.datasets.ieee_cis --config ml/config/ieee_cis.local.json
```

CLI defaults to the checked-in example and `ml/artifacts/ieee_cis`. It exits 2 on dataset
errors/failed quality, or 0 on passed quality. Missing files generate no replacement reports.
Fatal duplicate/missing/conflicting join keys produce a rejected `join_audit.json` when
writable, not a joined profile. Existing artifact files are not evidence of a successful
new run: check fingerprints, generation time and command outcome.

For API access, set the following in the ignored root environment and restart the server:

```dotenv
RISKBYTE_DATASET_ADAPTER=ieee_cis
RISKBYTE_DATASET_CONFIG_PATH=ml/config/ieee_cis.local.json
RISKBYTE_DATASET_ARTIFACTS_DIR=ml/artifacts/ieee_cis
```

The existing six dataset endpoints use the joined source. New GET
`/api/v1/dataset/integration` returns typed join/coverage/memory/catalog/entity/feasibility/
leakage/contract metadata. It contains no local paths or raw entity keys. With the generic
adapter it returns unavailable integration metadata; when unconfigured it returns
`dataset_not_configured`. Configured missing IEEE files return the established structured
422 envelope with code `ieee_cis_files_missing`; process health still returns 200.

## Adapter boundary and join

IEEE-specific names and semantics live under `riskbyte_ml.datasets.ieee_cis`, separate from
generic `riskbyte_ml.data`. The generic pipeline gained a preloaded-frame entrypoint and
explicit categorical roles so numeric-looking card/address codes are not measurements.

The adapter declares `isFraud` labels `1`/`0`, `TransactionID`, elapsed `TransactionDT`
seconds and `TransactionAmt`. It requires those four transaction roles and the identity
join key; other columns are optional and inspected. Identity must not replace transaction
columns. Join is explicitly **LEFT**, validated **one-to-one**, preserving order, count,
identifier sequence and exact target sequence. Missing/duplicate keys in either table
reject the join; no automatic deduplication, row deletion, suffix overwrite or inner join.

Audit includes both source counts/unique keys, duplicate distinct keys and extra rows,
missing keys, joined count, matched identity records, unmatched transactions, unused identity
records, coverage, columns added and target integrity. Coverage is key-based, including
matched identity records whose optional attributes are all blank. Attribute completeness
is measured separately by the Phase 1 missing-value report. Identity orphans are reported.

## Memory and correctness

Strict CSV records stream through bounded batches, with bytes hashed as they are consumed.
This avoids holding an entire decoded file and every row list alongside a dataframe.
Source limits, record widths, duplicate headers, null characters and encoding are checked;
changed file attributes during reading fail explicitly. No silent skips or truncation.

All numeric and identifier lexemes remain exact strings; no float16/float32 downcasting,
integer reinterpretation or lossy amount coercion. Repeated string lexemes, including low-cardinality numeric lexemes, use categorical
dictionary storage only when measured smaller. Each batch is optimized before retention;
per-column dictionary union preserves values across batches. Logical numeric/category roles
remain separate from storage dtype. Category codes are never predictive numeric values. Analytical float64 views retain Phase 1 finite-value handling;
they are descriptive approximations, not exact financial arithmetic.

Defaults: 10,000 rows/batch, 1 GB/file, 1,000,000 rows/file, 500 columns/file and 3 GB retained
dataframe budget. Transactions load first, and identity uses the remaining retained budget.
Table before-optimization estimates sum the measured original string batches; after
optimization measures retained frames. Joined-frame before/after values are measured
directly. These are **dataframe estimates, not peak process RSS**. Chunk concatenation, joins, hashing/unique checks,
group studies and analysis allocate additional working memory. Category storage does not
make this out-of-core. Full-scale runtime/RAM have not been measured without the real files.
Use offline profiling with sufficient RAM; do not launch many API workers on a large source.
Column selection is limited to small views for entity/order studies, not silent ingestion drops.
Dictionary union uses the documented Pandas operation; values are preserved while storage
codes may change. [Pandas categorical union](https://pandas.pydata.org/docs/reference/api/pandas.api.types.union_categoricals.html).

## Fingerprint and outputs

SHA-256 source hashes represent exact transaction and identity bytes. The joined fingerprint
hashes a canonical manifest of both role-labelled source hashes, adapter/join version and
operator policy hash (excluding paths). File names alone cannot identify an experiment.
Changing either source or the policy changes it; moving identical files does not. Record
dependency and pipeline versions alongside the profile. API cache invalidation tracks both
files' size/mtime/inode plus configuration; preserving all attributes requires restart.

Generated, ignored artifacts include:

- `data_profile.json`, `data_quality_report.md` (existing Phase 1 profile of the joined frame).
- `ieee_cis_integration.json`, `join_audit.json`, `feature_contract.json`.
- `ENTITY_KEY_STUDY.md`, `IEEE_CIS_LEAKAGE_REVIEW.md`.
- `IEEE_CIS_BEHAVIOURAL_FEASIBILITY.md`, `IEEE_CIS_CLEANING_REVIEW.md`.

Reports are individually atomic, not a multi-file transaction; use one writer/directory.
Both sources are protected against report overwrite. Source paths exist only in local
profile provenance; the integration report/API has none. No processed joined CSV or model
is emitted. Source content is untouched. Synthetic smoke artifacts use `.tools`, not these
real-data artifact paths.

## Interpretation and temporal semantics

The catalog distinguishes readable context, partially understood codes, anonymized features,
identifiers, temporal roles, target, behavioural-key candidates and exclusion candidates.
Tags can overlap and every entry requires review; `V1–V339` are anonymized engineered fields,
not meaningful business explanations. C/D/M/id families are not universally interpretable.

`TransactionDT` is **elapsed time from an undisclosed reference**, not a real calendar
timestamp. Only relative hour/day indexes and chronological differences/windows are proposed.
No dates, weekdays, months or wall-clock hour-of-day are generated. The generic calendar
EDA results remain `UNAVAILABLE_WITH_REASON`.
[Official data description](https://www.kaggle.com/c/ieee-fraud-detection/data).

The competition host documents the target convention and describes label maturation.
Retain that context in later evaluation design; never treat outcome-time data as available
at the transaction instant. [Host's source discussion](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/100304).

## Next action

Place approved real files at the expected paths, review limits and run the offline command.
Review measured quality, coverage, entity-study and leakage outputs before approving any
Phase 2 contract. No training, fitted preprocessing, feature materialization or frontend
redesign is part of this phase.
