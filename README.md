# credit-risk-platform

Production-style credit risk modeling system for binary default prediction with:

- modular `src/` Python package layout,
- local/AWS-compatible data and artifact persistence,
- calibrated probability modeling,
- explainability outputs,
- FastAPI scoring API,
- AWS SAM infrastructure template.

## Features

- Data ingestion from local paths or `s3://` URIs (CSV or Parquet)
- Schema validation and deterministic data cleaning
- Reusable feature engineering + preprocessing pipeline
- Model candidates:
  - Logistic Regression
  - Random Forest (tree-based)
- Probability calibration:
  - sigmoid (Platt)
  - isotonic
- Evaluation outputs:
  - ROC-AUC
  - PR-AUC
  - log loss
  - Brier score
  - KS statistic
  - confusion matrix summaries
  - calibration curve and expected calibration error (ECE)
- Explainability:
  - SHAP when available
  - fallback reason-code generation from model coefficients/importances
- API routes:
  - `GET /health`
  - `POST /predict`
  - `POST /predict_batch`
  - `GET /prediction/{request_id}`
- Persistence abstractions:
  - S3-compatible artifact store
  - DynamoDB-compatible prediction metadata store
- AWS SAM template provisioning:
  - Lambda
  - API Gateway
  - S3 bucket
  - DynamoDB table

## Repository Structure

```text
credit-risk-platform/
├── configs/
├── data/
├── notebooks/
├── scripts/
├── src/credit_risk/
│   ├── api/
│   ├── aws/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── schemas/
│   ├── scoring/
│   └── utils/
├── tests/
├── artifacts/
└── template.yaml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Configuration

The platform uses YAML config with environment variable overrides.

- Base config: `configs/default.yaml`
- Local overrides: `configs/local.yaml`
- AWS overrides: `configs/aws.yaml`

Environment variables use `CRP_` prefix and `__` for nesting:

```bash
export CRP_PROJECT__MODEL_VERSION=v1.2.3
export CRP_STORAGE__BACKEND=aws
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

## Run API Locally

After training:

```bash
uvicorn credit_risk.api.app:app --reload
```

### Sample `POST /predict` payload

```json
{
  "application_id": "app-001",
  "annual_income": 85000,
  "loan_amount": 12000,
  "dti": 18.7,
  "fico_range_low": 680,
  "fico_range_high": 684,
  "revolving_utilization": 42.1,
  "open_accounts": 8,
  "delinquencies_2y": 0
}
```

## Test

```bash
pytest
```

## Deploy with AWS SAM

```bash
sam build
sam deploy --guided
```

The template provisions Lambda + API Gateway + S3 + DynamoDB.

## Notes

- Use `storage.backend=local` for local development.
- Use `storage.backend=aws` with bucket/table configuration to run in AWS.
- SHAP is optional; fallback explanations are always available.
