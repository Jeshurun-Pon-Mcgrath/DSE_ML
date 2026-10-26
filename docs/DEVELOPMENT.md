# Developer setup

## Phase 4 supervised evaluation

Use the frozen MEDIUM source and existing temporal feature/split/preprocessing artifacts;
do not regenerate them for this phase. Run:

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.models.supervised --config ml/config/synthetic_riskbyte.example.json --experiment-config ml/config/supervised.example.json
```

Four core classifiers use existing dependencies. XGBoost needs the optional extra
`python -m pip install -e "./ml[boosting]"` in an environment with registry access, or an
approved compatible offline wheel. It is absent in this workspace because dependency
downloads are blocked. No XGBoost measurements are claimed. Completed identical runs
reuse saved results; adding algorithms/settings after final evaluation requires a separately
declared experiment and must not silently reopen the test or change the frozen classifier.
See [measured evaluation and limitations](SUPERVISED_MODEL_EVALUATION.md).

## Phase 2 development

The approved synthetic source needs no Kaggle account or external dataset download.
Runtime preprocessing dependencies now include scikit-learn, SciPy and joblib, with their
tested versions/transitive dependencies in `requirements-dev.lock.txt`. Install the updated
lock and editable packages using the existing commands. This workspace used compatible
already-installed local Python 3.12 packages because shell network access was unavailable.
No dependency source changes or global environment writes were made.

Read [synthetic setup](SYNTHETIC_DATASET.md) before generating data. Existing MEDIUM files
need no regeneration: use `python -m riskbyte_ml.preprocessing` with the venv interpreter.
Always keep generated data and artifacts ignored; never substitute synthetic files under
the real `ml/data/ieee_cis` path. Joblib files are executable: load only trusted local outputs.

## Prerequisites and installation

Phase 7.1 pins SHAP 0.53.0 in the ML explainability extra and the existing lock, together
with slicer 0.0.8 and tqdm 4.70.1. Every previously verified version is preserved. The
lock was checked against `python -m pip freeze --exclude-editable`; 42 pins matched exactly.
Use the lock for fresh installation; no pip upgrade is needed. Local editable ML metadata
can be refreshed with `pip install --no-deps --no-build-isolation -e ./ml` when already-installed
locked dependencies are verified. See [SHAP completion](EXPLAINABLE_AI.md) for reproduction.

Use Node.js 24+ (recommended) or 22.13+ in the 22.x line, and Python 3.12+. npm workspaces are sufficient for
the current single web application; no additional monorepo task runner is required.
Internet access to npm and PyPI is needed on a fresh machine.
No cloud account, PostgreSQL instance, Docker, GPU or real dataset is required for Phase 1 tests.

Windows PowerShell, from the repository root:

```powershell
Copy-Item .env.example .env
npm ci
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock.txt -e ./ml -e ./apps/api
```

POSIX shell:

```sh
cp .env.example .env
npm ci
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock.txt -e ./ml -e ./apps/api
```

The Python lock records the tested development environment, not platform-specific ML
libraries. Package metadata carries compatible ranges; update the lock deliberately in
a clean environment when changing dependencies. Retest on the target OS and Python
version. Install editable packages together so pip can resolve `riskbyte-api`'s dependency
on `riskbyte-ml`. Activation is optional when using explicit virtualenv executable paths.

## Start and verify

```powershell
npm run dev
# In a second terminal:
.\.venv\Scripts\python.exe -m uvicorn riskbyte_api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 --no-access-log
```

`GET /health` returns `status`, `service` and `version`, with `X-Request-ID`.
This is process liveness only; it does not claim model or database readiness.
The browser shell performs no risk request. Vite binds to loopback and requires port 5173;
change the backend CORS allowlist intentionally if you change the browser origin.

Run the quality commands in the root README. Python tests import installed packages
and test health, configuration, CORS, error sanitization, request correlation and ML
configuration. Test-only routes exercise failures without adding production endpoints.
ESLint uses TypeScript project-aware rules. Prettier formats JS/TS/CSS/JSON/Markdown;
Ruff handles Python.

## Environment strategy

One ignored root `.env` serves both applications. Vite explicitly uses that root as
`envDir`; Zod validates the public API origin at server startup/build and again at browser
startup. Only `VITE_*` variables enter browser bundles. A missing API origin defaults to
`http://localhost:8000`; a supplied invalid value fails immediately.

The API factory validates Pydantic Settings on startup. `RISKBYTE_` environment variables
override root `.env`; production should inject them through a secret manager.
`RISKBYTE_ENVIRONMENT` accepts `development`, `test`, `production`.
`RISKBYTE_LOG_LEVEL` accepts standard uppercase levels.
`RISKBYTE_CORS_ORIGINS` is a JSON array of exact HTTP(S) origins, without paths or wildcards.
`RISKBYTE_DATABASE_URL` is optional; if provided, it must be a PostgreSQL psycopg URL.
Omit this variable instead of assigning an empty value.

Database engine/session helpers exist but are unused by health. Future lifespan code
must own engine creation/disposal; migrations must run separately from serving traffic.
Install the optional driver in that future phase:

```powershell
.\.venv\Scripts\python.exe -m pip install -e "./apps/api[database]"
```

## Phase 1 configuration and verification

See [DATA_PIPELINE.md](DATA_PIPELINE.md). The default is `dataset_not_configured`, with
no generated data. `RISKBYTE_DATASET_CONFIG_PATH` selects an operator JSON file;
`RISKBYTE_DATASET_ARTIFACTS_DIR` defaults to `ml/artifacts`. Both settings resolve relative
to the repository root; source paths inside JSON resolve relative to its directory.
Leave the configuration environment variable absent until a real dataset is approved.

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.data profile --config ml/config/dataset.local.json
.\.venv\Scripts\python.exe -m compileall -q apps/api/src ml/src tests
.\.venv\Scripts\python.exe -m pip check
```

All pipeline tests use clearly labelled synthetic fixtures and need no internet/database.
For large CSVs, review RAM and configured limits first; use one writer per artifact directory.
Never publish local profiles containing paths or sensitive category aggregates.

## Phase 1.5 IEEE-CIS

Read [setup, API settings and memory limits](IEEE_CIS_INTEGRATION.md). Both real files must
be supplied manually under ignored `ml/data/ieee_cis`; the checked-in config is a safe path/
resource declaration, not supplied data. CLI missing-files failure is expected in this checkout.
Use synthetic fixtures only in tests and separate `.tools`/temporary smoke directories.
No new third-party package, GPU or database is required.

## Source ownership boundaries

Keep HTTP parsing in routers, application orchestration in services, database concerns
in `db/`, and reusable modeling code in `riskbyte_ml`. Do not import notebooks into the
API or execute training on application startup. Both Python packages use a `src/` layout
to avoid accidental imports from the working directory.

No generated data, model files, credentials or virtual environments belong in commits.
Do not add later-phase UI or scoring implementations as part of foundation maintenance.
