"""FastAPI routes for health checks and credit risk scoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from credit_risk.data.load import load_dataset
from credit_risk.schemas.payloads import (
    ApplicantInput,
    BatchPredictionResponse,
    HealthResponse,
    PredictBatchRequest,
    PredictionResponse,
)
from credit_risk.scoring.predict import PredictionService

router = APIRouter()

HOME_PAGE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Credit Risk Workbench</title>
    <style>
      :root {
        --bg-top: #f3f8f1;
        --bg-bottom: #eef0ff;
        --ink: #173146;
        --muted: #4f5f75;
        --surface: rgba(255, 255, 255, 0.88);
        --line: rgba(23, 49, 70, 0.12);
        --primary: #1f6f5f;
        --primary-dark: #165145;
        --accent: #e89b2f;
        --danger: #8f2f2f;
        --radius: 16px;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
        color: var(--ink);
        background: linear-gradient(160deg, var(--bg-top), var(--bg-bottom));
        min-height: 100vh;
        padding: 28px 16px 48px;
      }

      .shell {
        max-width: 1080px;
        margin: 0 auto;
        display: grid;
        gap: 16px;
      }

      .hero {
        border: 1px solid var(--line);
        background: linear-gradient(125deg, rgba(31, 111, 95, 0.11), rgba(232, 155, 47, 0.13));
        border-radius: var(--radius);
        padding: 20px;
      }

      h1 {
        margin: 0 0 8px;
        font-size: clamp(1.5rem, 3vw, 2rem);
      }

      .subtitle {
        margin: 0;
        color: var(--muted);
      }

      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
      }

      .panel {
        border: 1px solid var(--line);
        background: var(--surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius);
        padding: 16px;
      }

      .panel h2 {
        margin: 0 0 10px;
        font-size: 1.1rem;
      }

      .status {
        margin: 0;
        color: var(--muted);
      }

      label {
        display: block;
        margin-bottom: 10px;
        font-size: 0.92rem;
      }

      input,
      textarea,
      button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--line);
        padding: 10px 12px;
        font: inherit;
      }

      textarea {
        min-height: 120px;
        resize: vertical;
      }

      button {
        border: none;
        background: var(--primary);
        color: #fff;
        font-weight: 600;
        cursor: pointer;
      }

      button:hover {
        background: var(--primary-dark);
      }

      .meta {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 8px;
      }

      pre {
        margin: 0;
        background: #12243a;
        color: #ecf5ff;
        border-radius: 12px;
        padding: 12px;
        overflow: auto;
        min-height: 180px;
      }

      .danger {
        color: var(--danger);
      }

      .caption {
        color: var(--muted);
        margin: 10px 0 0;
        font-size: 0.88rem;
      }

      .pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        background: rgba(232, 155, 47, 0.16);
        border: 1px solid rgba(232, 155, 47, 0.45);
        font-size: 0.8rem;
        margin-top: 8px;
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <h1>Credit Risk Workbench</h1>
        <p class="subtitle">Local-first scoring dashboard for one-off and batch predictions.</p>
        <span class="pill">Runs entirely on your machine</span>
      </section>

      <section class="grid">
        <article class="panel">
          <h2>Service Health</h2>
          <p class="status" id="healthStatus">Checking model status...</p>
          <button type="button" id="refreshHealth">Refresh health</button>
        </article>

        <article class="panel">
          <h2>Batch From Local File</h2>
          <form id="batchForm">
            <label>
              Input file path (`.csv` or `.parquet`)
              <input id="inputPath" type="text" placeholder="data/raw/credit_batch.csv" />
            </label>
            <button type="submit">Run Batch Prediction</button>
          </form>
          <p class="caption">File path is resolved on this local machine where the API is running.</p>
        </article>
      </section>

      <section class="panel">
        <h2>Single Applicant Prediction</h2>
        <form id="predictForm" class="grid">
          <label>
            Application ID
            <input name="application_id" type="text" value="app-ui-001" required />
          </label>
          <label>
            Annual income
            <input name="annual_income" type="number" step="0.01" value="85000" required />
          </label>
          <label>
            Loan amount
            <input name="loan_amount" type="number" step="0.01" value="12000" required />
          </label>
          <label>
            DTI
            <input name="dti" type="number" step="0.01" value="18.7" required />
          </label>
          <label>
            FICO low
            <input name="fico_range_low" type="number" value="680" required />
          </label>
          <label>
            FICO high
            <input name="fico_range_high" type="number" value="684" required />
          </label>
          <label>
            Revolving utilization
            <input name="revolving_utilization" type="number" step="0.01" value="42.1" required />
          </label>
          <label>
            Open accounts
            <input name="open_accounts" type="number" value="8" required />
          </label>
          <label>
            Delinquencies (2y)
            <input name="delinquencies_2y" type="number" value="0" required />
          </label>
          <div style="align-self: end;">
            <button type="submit">Predict Applicant</button>
          </div>
        </form>
      </section>

      <section class="panel">
        <h2>Latest API Output</h2>
        <pre id="output">Submit a request to see response JSON.</pre>
        <p class="meta" id="errorLine"></p>
      </section>
    </main>

    <script>
      const output = document.getElementById("output");
      const errorLine = document.getElementById("errorLine");
      const healthStatus = document.getElementById("healthStatus");

      async function requestJson(url, options) {
        const response = await fetch(url, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          const detail = payload.detail || response.statusText;
          throw new Error(detail);
        }
        return payload;
      }

      function showResult(payload) {
        output.textContent = JSON.stringify(payload, null, 2);
        errorLine.textContent = "";
        errorLine.className = "meta";
      }

      function showError(error) {
        errorLine.textContent = error.message;
        errorLine.className = "meta danger";
      }

      async function refreshHealth() {
        try {
          const payload = await requestJson("/health");
          const loaded = payload.model_loaded ? "loaded" : "not loaded";
          healthStatus.textContent = `Status: ${payload.status} | Model: ${payload.model_version} (${loaded})`;
          if (payload.startup_error) {
            healthStatus.textContent += ` | Startup error: ${payload.startup_error}`;
          }
        } catch (error) {
          healthStatus.textContent = `Health check failed: ${error.message}`;
          healthStatus.className = "status danger";
        }
      }

      document.getElementById("refreshHealth").addEventListener("click", refreshHealth);

      document.getElementById("predictForm").addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        const payload = {
          application_id: formData.get("application_id"),
          annual_income: Number(formData.get("annual_income")),
          loan_amount: Number(formData.get("loan_amount")),
          dti: Number(formData.get("dti")),
          fico_range_low: Number(formData.get("fico_range_low")),
          fico_range_high: Number(formData.get("fico_range_high")),
          revolving_utilization: Number(formData.get("revolving_utilization")),
          open_accounts: Number(formData.get("open_accounts")),
          delinquencies_2y: Number(formData.get("delinquencies_2y"))
        };

        try {
          const prediction = await requestJson("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          showResult(prediction);
        } catch (error) {
          showError(error);
        }
      });

      document.getElementById("batchForm").addEventListener("submit", async (event) => {
        event.preventDefault();
        const inputPath = document.getElementById("inputPath").value.trim();
        if (!inputPath) {
          showError(new Error("Please provide a local CSV or Parquet file path."));
          return;
        }

        try {
          const batchResult = await requestJson("/predict_batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input_path: inputPath })
          });
          showResult(batchResult);
        } catch (error) {
          showError(error);
        }
      });

      refreshHealth();
    </script>
  </body>
</html>
""".strip()


def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not initialized. Train or load artifacts first.",
        )
    return service


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return HTMLResponse(content=HOME_PAGE_HTML)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "prediction_service", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    config = getattr(request.app.state, "config", None)

    if service is not None:
        response = service.health()
        response.model_loaded = True
        response.startup_error = None
        return response

    model_version = "unavailable"
    if config is not None:
        model_version = config.project.model_version

    return HealthResponse(
        model_version=model_version,
        model_loaded=False,
        startup_error=startup_error,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    applicant: ApplicantInput,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return service.predict_one(applicant)


@router.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: PredictBatchRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    applicants: list[ApplicantInput] = []

    if payload.applicants:
        applicants = payload.applicants
    elif payload.input_path:
        frame = load_dataset(payload.input_path)
        expected_fields = set(ApplicantInput.model_fields)
        rows: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            rows.append({key: value for key, value in row.items() if key in expected_fields})
        applicants = [ApplicantInput.model_validate(row) for row in rows]

    if not applicants:
        raise HTTPException(status_code=400, detail="No applicants available for batch scoring")

    return service.predict_batch(applicants)


@router.get("/prediction/{request_id}")
def get_prediction(
    request_id: str,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, Any]:
    record = service.fetch_prediction(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record
