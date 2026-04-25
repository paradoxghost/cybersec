"""FastAPI entrypoint for IoT IDS inference."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.models.registry import load_bundle
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.serving.service import InferenceService
from src.utils.io import load_yaml

app = FastAPI(title="Streaming IoT IDS API", version="0.1.0")

CFG = load_yaml("configs/config.yaml")
SERVICE: InferenceService | None = None
STARTUP_ERROR: str | None = None
LOGGER = logging.getLogger(__name__)


@app.on_event("startup")
def startup() -> None:
    global SERVICE, STARTUP_ERROR
    model_name = CFG.get("runtime", {}).get("active_model", "stronger_model")
    try:
        bundle = load_bundle(CFG["paths"]["models_dir"], model_name=model_name)
        SERVICE = InferenceService(bundle, top_k=CFG["serving"]["top_k_probabilities"])
        STARTUP_ERROR = None
    except Exception as exc:  # startup should not crash if artifacts are absent
        SERVICE = None
        STARTUP_ERROR = str(exc)
        LOGGER.warning("Model service not loaded at startup: %s", exc)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": SERVICE is not None,
        "startup_error": STARTUP_ERROR,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Model service not loaded. Run training first.")
    try:
        return SERVICE.predict_records([payload.features])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Model service not loaded. Run training first.")
    try:
        preds = SERVICE.predict_records(payload.records)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
