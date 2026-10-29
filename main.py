import os
import hmac
from threading import Lock
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
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

# RAM safety limit for ONE test (about 50 minutes at 20 samples/sec).
# When the limit is reached, extra samples are still predicted
# normally but are no longer stored for the website graph.
MAX_SESSION_SAMPLES = 60000


# =====================================================
# 2. APP AND CORS
# =====================================================

app = FastAPI(
    title="NeuroDetect Parkinson API",
    version="4.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# =====================================================
# 3. LOAD THE EXISTING MODEL (NO RETRAINING)
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

# Sensor data of the CURRENT test only.
# Cleared when a new session starts.
sensor_history = []


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
            "/session/final-result",
            "/session/sensor-data"
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
# Response remains {"predictions": [0, 1, ...]}.
# This endpoint NEVER rejects a batch because of session
# state, so the existing ESP32 firmware behaves exactly
# as before. Sensor data is additionally stored (RAM only)
# for the website graphs.
# =====================================================

@app.post("/predict/batch")
def predict_batch(
    batch: BatchRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    global current_session_id

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

    # Store sensor data for the website graphs.
    # FSR is display-only; it is not a model feature.
    with state_lock:
        # If the server restarted in the middle of a test, it
        # has forgotten the session. Re-adopt the ESP32's session
        # so history and the final result keep working.
        if current_session_id is None:
            current_session_id = batch.session_id

        if batch.session_id == current_session_id:
            room = MAX_SESSION_SAMPLES - len(sensor_history)

            if room > 0:
                for sample, prediction in list(
                    zip(batch.samples, predictions)
                )[:room]:
                    sensor_history.append({
                        "timestamp_ms": sample.timestamp_ms,
                        "aX": sample.aX,
                        "aY": sample.aY,
                        "aZ": sample.aZ,
                        "gX": sample.gX,
                        "gY": sample.gY,
                        "gZ": sample.gZ,
                        "fsr_raw": sample.fsr_raw,
                        "prediction": prediction
                    })

    return {"predictions": predictions}


# =====================================================
# 11. START A NEW SESSION
# ESP32 calls this at the beginning of each test.
# Clears the previous result AND previous sensor data,
# unless the same session is simply being re-registered
# (e.g. after a server restart) - then data is kept.
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

    global current_session_id, latest_final_result, sensor_history

    with state_lock:
        if data.session_id != current_session_id:
            sensor_history = []
            latest_final_result = None

        current_session_id = data.session_id

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
# Always HTTP 200:
#   {"status": "waiting"}              -> no result yet
#   {"status": "ready", ...result...}  -> result available
# =====================================================

@app.get("/session/final-result")
def get_final_result():
    with state_lock:
        if latest_final_result is None:
            return {"status": "waiting"}

        return {"status": "ready", **latest_final_result}


# =====================================================
# 14. WEBSITE READS SENSOR DATA OF THE CURRENT TEST
# (MPU6050 six values + FSR + prediction per sample)
#
# `since` = number of samples the website already has.
# First call: since=0 (everything). Later calls: only new ones.
# The website must compare `session_id`: when it changes, a new
# test has started, so the old graph must be cleared and
# `since` reset to 0.
# =====================================================

@app.get("/session/sensor-data")
def get_sensor_data(
    since: int = Query(default=0, ge=0)
):
    with state_lock:
        if current_session_id is None:
            return {
                "status": "waiting",
                "session_id": None,
                "sample_count": 0,
                "start_index": 0,
                "samples": []
            }

        total = len(sensor_history)
        start_index = min(since, total)

        return {
            "status": "ready",
            "session_id": current_session_id,
            "sample_count": total,
            "start_index": start_index,
            "samples": sensor_history[start_index:],
            "history_limit": MAX_SESSION_SAMPLES
        }
