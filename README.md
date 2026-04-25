# Streaming IoT Intrusion Detection with Drift Monitoring

Production-style ML cybersecurity project for classifying IoT traffic as benign vs malicious (and attack type), then simulating deployment with stream monitoring.

## Resume-ready project description
Built an end-to-end IoT intrusion detection system using scikit-learn and FastAPI with leakage-aware splitting, imbalance-aware training, security-focused metrics (macro-F1, per-class recall, false positives on benign traffic), micro-batch simulation, and drift/alert monitoring (PSI, JSD, alert spikes).

## Why intrusion detection is different from generic classification
- **False positives are expensive**: flagging benign traffic as malicious creates alert fatigue.
- **Class imbalance is severe**: some attack classes are rare, so accuracy alone is misleading.
- **Data drift is expected**: device behavior and attack patterns evolve; monitoring is required.

## Project architecture
- **Data**: load, sample, validate, split (time/group aware when available).
- **Features**: sklearn preprocessing pipeline + label encoding.
- **Models**: Logistic Regression baseline and Random Forest stronger model.
- **Evaluation**: macro/weighted F1, per-class recall, confusion matrix, benign FPR, PR-AUC and ROC-AUC (when valid).
- **Serving**: FastAPI for single/batch inference with class probabilities.
- **Monitoring**: feature drift (PSI), prediction drift (JSD), alert spike detection.

## Repository structure
```text
app/                 # FastAPI app
configs/             # Config + logging
data/                # raw/interim/processed data
models/              # persisted model artifacts
reports/             # metrics + figures
scripts/             # pipeline runners
src/                 # modular source code
tests/               # lightweight tests
```

## Dataset setup (CICIoT2023)
1. Download CICIoT2023.
2. Place a CSV or Parquet file in `data/raw/`.
3. Update `configs/config.yaml`:
   - `paths.raw_data_path`
   - `schema.target_column` (**Set target column here**)
   - optionally `schema.timestamp_column` / `schema.group_column` (**Set group/time column here if available**)
   - benign label list in `schema.benign_labels`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline
```bash
python scripts/run_all.py
```

Or run step-by-step:
```bash
python scripts/run_data_validation.py
python scripts/run_training.py
python scripts/run_evaluation.py
python scripts/run_stream_simulation.py
```

## Launch FastAPI
```bash
uvicorn app.main:app --reload --port 8000
```

### API examples
Health:
```bash
curl http://127.0.0.1:8000/health
```

Single prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.12, "feature_2": 10.3}}'
```

Batch prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature_1": 0.12, "feature_2": 10.3}, {"feature_1": 0.2, "feature_2": 9.8}]}'
```

Response includes predicted label, confidence, and top class probabilities.

## Monitoring outputs
Generated under `reports/metrics/`:
- `data_validation_report.json`
- `training_summary.json`
- `baseline_test_metrics.json`
- `stronger_test_metrics.json`
- `model_comparison.csv`
- `stream_monitoring_report.json`
- `stream_predictions.csv`

## Key design choices
- **Random Forest** chosen as stronger model: robust tabular baseline with simple, reproducible training.
- **Shared preprocessing artifact** ensures train/inference consistency.
- **Leakage guardrails**: configured ID/time/group columns are excluded from model features during training.
- **Fallback split strategy**: if no timestamp/group metadata exists, stratified random split is used and documented.

## Interview explanation (simple language)
I treated this like a real SOC-facing pipeline, not just a notebook. First, I validate and sample the dataset so large CICIoT files are manageable and reproducible. I split data in a way that reduces leakage risk, then train a transparent baseline and a stronger model. I evaluate with metrics that actually matter to defenders (macro-F1, per-class recall, and false alarms on benign traffic). Finally, I expose an API for deployment-style inference and simulate streaming batches with drift and alert monitoring so we can catch model degradation early.

## Limitations
- CICIoT column names can vary across exports; configuration fields must be aligned by user.
- Current preprocessing is numeric-only for robustness; categorical support can be added if needed.
- Monitoring is intentionally lightweight (PSI/JSD/thresholds), not enterprise observability tooling.

## Future improvements
- Add calibrated probabilities and threshold tuning for SOC cost trade-offs.
- Add richer time-window drift dashboards.
- Support online/continual retraining triggers.
