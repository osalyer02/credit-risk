# credit-risk-platform

Local-first credit risk modeling and scoring system for binary default prediction.

## Features

- Local file ingestion (`.csv` and `.parquet`)
- Schema validation and deterministic data cleaning
- Reusable feature engineering + preprocessing pipeline
- Model candidates:
  - Logistic Regression
  - Random Forest
- Probability calibration:
  - Sigmoid (Platt)
  - Isotonic
- Explainability outputs (SHAP optional with fallback reason codes)
- FastAPI scoring API with routes:
  - `GET /` local web workbench
  - `GET /health`
  - `POST /predict`
  - `POST /predict_batch`
  - `GET /prediction/{request_id}`
- Local persistence:
  - model bundle artifacts on disk
  - batch prediction outputs on disk
  - prediction records in JSONL

## Repository Structure

```text
credit-risk-platform/
├── configs/
├── data/
├── notebooks/
├── scripts/
├── src/credit_risk/
│   ├── api/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── schemas/
│   ├── scoring/
│   ├── storage/
│   └── utils/
├── tests/
└── artifacts/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Configuration

The platform uses YAML config with environment-variable overrides.

- Base config: `configs/default.yaml`
- Local overrides: `configs/local.yaml`

Environment variables use `CRP_` prefix and `__` for nesting:

```bash
export CRP_PROJECT__MODEL_VERSION=v1.2.3
export CRP_API__PORT=8080
```

## Train Locally

1. Generate demo training data:

```bash
python scripts/generate_synthetic_data.py --rows 2500 --seed 42
```

2. Place training data at `data/raw/credit_train.csv` (or update config).
3. Run:

```bash
python -m credit_risk.models.train --config configs/default.yaml --env-config configs/local.yaml
```

Artifacts are saved under:

- `artifacts/models/<model_version>/model_bundle.joblib`
- `artifacts/metrics/<model_version>/metrics.json`
- `artifacts/metrics/<model_version>/validation_report.json`
- `artifacts/metrics/<model_version>/model_registry_record.json`

## Run API + Local UI

After training:

```bash
uvicorn credit_risk.api.app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the local web workbench.

The UI supports:

- Interactive single-applicant scoring
- Batch scoring from a local CSV/Parquet file path
- Service health checks and raw JSON response inspection

## Test

```bash
pytest
```

## Notes

- The project is local-only and does not require cloud services.
- SHAP is optional; fallback explanations are always available.
