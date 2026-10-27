import os
import hmac
from threading import Lock
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "parkinson_ensemble_model.pkl"
)

DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "")
FRONTEND_ORIGIN = "https://neurodetect-parkinson.netlify.app"

FEATURE_COLUMNS = ["aX", "aY", "aZ", "gX", "gY", "gZ"]
MAX_BATCH_SIZE = 20


# =====================================================
# 2. APP AND CORS
# =====================================================

app = FastAPI(
    title="NeuroDetect Parkinson API",
    version="4.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# =====================================================
# 3. LOAD THE EXISTING MODEL
# =====================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load model: {MODEL_PATH}. "
        "Check the model file and its Python dependencies."
    ) from exc


# =====================================================
# 4. IN-MEMORY SESSION STATE
# IMPORTANT: RAM ONLY; LOST WHEN THE PROCESS RESTARTS.
# Run with ONE worker (--workers 1).
# =====================================================

state_lock = Lock()

current_session_id = None
latest_final_result = None


# =====================================================
# 5. REQUEST MODELS
# =====================================================

class SensorData(BaseModel):
    """Used by /predict: only the six model features."""
    aX: float
    aY: float
    aZ: float
    gX: float
    gY: float
    gZ: float


class SensorSample(SensorData):
    """Used by /predict/batch: ESP32 sample with extra metadata."""
    timestamp_ms: int
    # Display-only. NOT passed to the AI model.
    fsr_raw: Optional[int] = None


class BatchRequest(BaseModel):
    session_id: int
    samples: List[SensorSample] = Field(min_length=1)


class SessionStartRequest(BaseModel):
    session_id: int


class FinalResultRequest(BaseModel):
    session_id: int
    sample_count: int = Field(gt=0)
    healthy_count: int = Field(ge=0)
    parkinson_count: int = Field(ge=0)


# =====================================================
# 6. API KEY VALIDATION
# =====================================================

def verify_device_key(x_api_key: Optional[str]) -> None:
    if not DEVICE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="DEVICE_API_KEY is not configured on Render."
        )

    if not x_api_key or not hmac.compare_digest(
        x_api_key, DEVICE_API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )


# =====================================================
# 7. RESULT CALCULATION
# 0 = Healthy / Normal
# 1 = Parkinson's model prediction
# =====================================================

def calculate_final_result(
    healthy_count: int,
    parkinson_count: int
) -> str:
    if healthy_count == parkinson_count:
        return "Inconclusive (tie)"

    if healthy_count > parkinson_count:
        return "Healthy / Normal"

    return "Parkinson's Detected"


# =====================================================
# 8. HOME AND HEALTH
# =====================================================

@app.get("/")
def home():
    return {
        "message": "NeuroDetect Parkinson API is running.",
        "version": app.version,
        "endpoints": [
            "/health",
            "/predict",
            "/predict/batch",
            "/session/start",
            "/session/final-result"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device_api_key_configured": bool(DEVICE_API_KEY)
    }


# =====================================================
# 9. SINGLE-SAMPLE PREDICTION
# Accepts only the six IMU values (no timestamp needed).
# =====================================================

@app.post("/predict")
def predict_one(data: SensorData):
    features = pd.DataFrame(
        [[
            data.aX, data.aY, data.aZ,
            data.gX, data.gY, data.gZ
        ]],
        columns=FEATURE_COLUMNS
    )

    try:
        prediction = int(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Prediction failed. Check model input format."
        ) from exc

    if prediction not in (0, 1):
        raise HTTPException(
            status_code=500,
            detail="Unexpected prediction class."
        )

    return {
        "prediction": prediction,
        "result": (
            "Parkinson's Detected"
            if prediction == 1
            else "Healthy / Normal"
        )
    }


# =====================================================
# 10. ESP32 BATCH PREDICTION
# Response remains {"predictions": [0, 1, ...]}
# =====================================================

@app.post("/predict/batch")
def predict_batch(
    batch: BatchRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    verify_device_key(x_api_key)

    if len(batch.samples) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {MAX_BATCH_SIZE}."
        )

    rows = [
        {
            "aX": s.aX,
            "aY": s.aY,
            "aZ": s.aZ,
            "gX": s.gX,
            "gY": s.gY,
            "gZ": s.gZ
        }
        for s in batch.samples
    ]

    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    try:
        predictions = [
            int(v) for v in model.predict(features)
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="The model could not process this batch."
        ) from exc

    if (
        len(predictions) != len(batch.samples)
        or any(v not in (0, 1) for v in predictions)
    ):
        raise HTTPException(
            status_code=500,
            detail="The model returned invalid predictions."
        )

    # FSR is not an AI model input.
    # This endpoint does not store sensor batches.
    return {"predictions": predictions}


# =====================================================
# 11. START A NEW SESSION
# ESP32 must call this at the beginning of each test.
# It clears the previous result to avoid showing stale data.
# =====================================================

@app.post("/session/start")
def start_session(
    data: SessionStartRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    verify_device_key(x_api_key)

    global current_session_id, latest_final_result

    with state_lock:
        current_session_id = data.session_id
        latest_final_result = None

    return {
        "success": True,
        "message": "Session started.",
        "session_id": data.session_id
    }


# =====================================================
# 12. SUBMIT FINAL RESULT
# ESP32 sends its majority-vote counts here.
# =====================================================

@app.post("/session/final-result")
def submit_final_result(
    data: FinalResultRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    verify_device_key(x_api_key)

    global latest_final_result

    if (
        data.healthy_count + data.parkinson_count
        != data.sample_count
    ):
        raise HTTPException(
            status_code=400,
            detail="sample_count must equal healthy_count "
                   "+ parkinson_count."
        )

    result = {
        "session_id": data.session_id,
        "sample_count": data.sample_count,
        "healthy_count": data.healthy_count,
        "parkinson_count": data.parkinson_count,
        "result": calculate_final_result(
            data.healthy_count,
            data.parkinson_count
        )
    }

    with state_lock:
        if current_session_id is None:
            raise HTTPException(
                status_code=409,
                detail="No active session. Start a session first."
            )

        if data.session_id != current_session_id:
            raise HTTPException(
                status_code=409,
                detail="Session ID mismatch. Start the correct session."
            )

        # Repeated submission of the same result is safe.
        latest_final_result = result

    return {
        "success": True,
        "message": "Final result received.",
        **result
    }


# =====================================================
# 13. WEBSITE READS THE LATEST FINAL RESULT
# No API key is exposed to the browser.
#
# Always returns HTTP 200:
#   {"status": "waiting"}              -> no result yet
#   {"status": "ready", ...result...}  -> result available
# =====================================================

@app.get("/session/final-result")
def get_final_result():
    with state_lock:
        if latest_final_result is None:
            return {"status": "waiting"}

        return {"status": "ready", **latest_final_result}
