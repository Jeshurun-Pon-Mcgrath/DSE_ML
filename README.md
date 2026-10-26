# RiskByte

RiskByte is an explainable financial-fraud intelligence platform built for academic research
with synthetic transaction data. It combines a calibrated supervised classifier, diagnostic
anomaly detection and confidence-aware behavioural evidence in a transparent Adaptive
Behavioural Risk Fusion (ABRF) score.

The repository contains the complete React analyst interface, FastAPI application, reusable
data-intelligence and feature pipelines, frozen experiment methodology, and offline tests.
All published performance measurements come from the one-use synthetic future holdout
documented in [FINAL_SYSTEM_RESULTS.md](docs/FINAL_SYSTEM_RESULTS.md).

> This environment uses synthetically generated financial transaction data for academic
> fraud-detection research and demonstration. Results do not establish production banking
> performance or determine fraud guilt.

## Final evaluation

The frozen system was evaluated once on a 30,000-transaction future synthetic holdout.

| Metric | Final ABRF |
| --- | ---: |
| Precision | 82.08% |
| Recall | 58.00% |
| F1 | 0.6797 |
| PR-AUC | 0.7173 |
| ROC-AUC | 0.9484 |
| False-positive rate | 0.2585% |
| Hard-negative false-positive rate | 1.47% |

Runtime pages read these values from the locked evaluation API rather than embedding them in
frontend source code.

## System flow

```text
Transaction
  → validation and leakage-safe preprocessing
  → strictly-prior behavioural features
  → calibrated Decision Tree probability (P)
  → Isolation Forest diagnostic score (A)
  → behavioural deviation score and confidence (B, Cb)
  → confidence-aware ABRF fusion
  → risk level, investigation decision and evidence trace
  → FastAPI
  → React analyst workspace
```

The active ABRF configuration uses base weights `P=0.9`, `A=0.0`, `B=0.1`. Isolation
Forest remains available as diagnostic evidence but does not influence the active fused
risk score. SHAP explains the underlying Decision Tree, not the calibration layer or the
complete fusion system.

## Repository structure

```text
apps/
  api/                 FastAPI application, schemas, services and persistence
  web/                 React, TypeScript and Vite analyst interface
ml/
  config/              Example and frozen pipeline configuration
  data/                Local ignored datasets; tracked usage notice only
  notebooks/           Exploratory notebook boundary
  src/riskbyte_ml/     Data, features, models, anomaly, behaviour, ABRF and XAI
docs/                  Current architecture, methodology, results and contracts
infra/migrations/      Alembic migrations for the PostgreSQL-ready data layer
tests/
  api/                 FastAPI contract and integration tests
  ml/                  Data, feature and model methodology tests
  fixtures/            Clearly labelled small synthetic CSV fixtures
```

Raw datasets, generated artifacts, local databases, model binaries, environments, build
outputs and credentials are excluded by `.gitignore`.

## Requirements

- Python 3.12
- Node.js 22.13+ or 24+
- npm 10+
- A browser with WebGL for the optional 3D network; a useful 2D fallback is included

The verified Python dependency set is locked in `requirements-dev.lock.txt`. Frontend
dependencies are locked in `package-lock.json`.

## Development setup

Create and install the Python environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock.txt
.\.venv\Scripts\python.exe -m pip install -e .\ml -e .\apps\api
```

Install the frontend dependencies:

```powershell
npm ci
```

Copy `.env.example` to `.env` and adjust only local, non-secret settings when needed. Never
commit `.env`, datasets, generated artifacts or credentials.

## Run the application

Start the API:

```powershell
$env:RISKBYTE_DATASET_CONFIG_PATH = 'ml/config/synthetic_riskbyte.example.json'
$env:RISKBYTE_DATASET_ADAPTER = 'synthetic_riskbyte'
.\.venv\Scripts\python.exe -m uvicorn riskbyte_api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
npm run dev --workspace @riskbyte/web -- --host 127.0.0.1 --port 5173
```

Open:

- Analyst application: `http://127.0.0.1:5173/app`
- API documentation: `http://127.0.0.1:8000/docs`

The API loads the frozen inference, anomaly, behaviour, ABRF and explainability artifacts
once at startup. If local ignored artifacts have not been generated, health endpoints return
an explicit unavailable state rather than fabricated output.

## Verification

Frontend:

```powershell
npm run lint
npm run typecheck
npm run test
npm run build
npm run format:check
```

Python:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q apps/api/src ml/src tests
.\.venv\Scripts\python.exe -m pip check
```

Tests use small synthetic fixtures and do not require network access or the real IEEE-CIS
competition files.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Developer setup](docs/DEVELOPMENT.md)
- [Data pipeline](docs/DATA_PIPELINE.md)
- [Synthetic dataset methodology](docs/SYNTHETIC_DATASET.md)
- [ML methodology](docs/ML_METHODOLOGY.md)
- [Statistical analysis](docs/STATISTICAL_ANALYSIS.md)
- [Supervised model evaluation](docs/SUPERVISED_MODEL_EVALUATION.md)
- [Anomaly detection](docs/ANOMALY_DETECTION.md)
- [Behavioural deviation](docs/BEHAVIOURAL_DEVIATION.md)
- [ABRF design](docs/ABRF_DESIGN.md)
- [Explainable AI](docs/EXPLAINABLE_AI.md)
- [Final system results](docs/FINAL_SYSTEM_RESULTS.md)
- [Frontend API contract](docs/FRONTEND_API_CONTRACT.md)
- [Frontend implementation](docs/FRONTEND_IMPLEMENTATION.md)
- [Final frontend QA](docs/FINAL_FRONTEND_QA.md)
- [Security](docs/SECURITY.md)
