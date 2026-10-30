import os
import hmac
import time
from datetime import datetime, timezone
from threading import Lock
from typing import List, Literal, Optional

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

# Secret PIN for the website's Start / Stop / Reset buttons.
# Set it in Render -> Environment. Use letters and digits only.
CONTROL_PIN = os.getenv("CONTROL_PIN", "")

FRONTEND_ORIGIN = "https://neurodetect-parkinson.netlify.app"

FEATURE_COLUMNS = ["aX", "aY", "aZ", "gX", "gY", "gZ"]
MAX_BATCH_SIZE = 20

# RAM safety limit for ONE test (about 50 minutes at 20 samples/sec).
# When the limit is reached, extra samples are still predicted
# normally but are no longer stored for the website graph.
MAX_SESSION_SAMPLES = 60000

# ---- Remote control settings ----
DEFAULT_DEVICE_ID = "esp32-parkinson-01"

# A command that the ESP32 does not pick up within this time is
# cancelled, so an old "Start" can never run later by surprise.
COMMAND_TTL_SECONDS = 20

# No request from the ESP32 for this long -> shown as offline.
DEVICE_OFFLINE_AFTER_SECONDS = 30

# Too many wrong PIN attempts lock the control buttons for a while.
PIN_MAX_FAILURES = 5
PIN_LOCK_SECONDS = 60

# ESP32 states in which Start / Reset may be requested.
IDLE_LIKE_STATUSES = ("idle", "completed", "error", "unknown")


# =====================================================
# 2. APP AND CORS
# =====================================================

app = FastAPI(
    title="NeuroDetect Parkinson API",
    version="5.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Control-PIN"],
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

# Remote-control state (also RAM only).
remote = {
    "device": {
        "device_id": DEFAULT_DEVICE_ID,
        "status": "unknown",        # last status reported by the ESP32
        "session_id": None,
        "message": "Waiting for the ESP32.",
        "last_seen_mono": None,     # any authenticated ESP32 request
        "last_seen_utc": None,
    },
    "pending": None,                # command waiting for the ESP32
    "last_command": None,           # what happened to the previous one
    "last_event": {                 # sticky message for the website
        "level": "info",
        "message": "No events yet.",
        "time_utc": None,
    },
    "acknowledged_command_id": 0,
    # Time-based start keeps command IDs unique even after a restart.
    "command_counter": int(time.time()),
}

# Wrong-PIN protection.
pin_guard = {"failures": [], "locked_until": 0.0}


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


class WebsiteCommandRequest(BaseModel):
    """Sent by the website when a control button is pressed."""
    command: Literal["start", "stop", "reset"]
    device_id: str = Field(
        default=DEFAULT_DEVICE_ID, min_length=1, max_length=80
    )


class DeviceStatusRequest(BaseModel):
    """Sent by the ESP32 to report its real state."""
    device_id: str = Field(
        default=DEFAULT_DEVICE_ID, min_length=1, max_length=80
    )
    status: Literal[
        "idle", "recording", "finalizing", "completed", "error"
    ]
    session_id: Optional[int] = None
    # Set when this report answers a website command.
    command_id: int = Field(default=0, ge=0)
    message: Optional[str] = Field(default=None, max_length=240)
    # True when the ESP32 refused the command (for example, it was busy).
    rejected: bool = False
    # True when the ESP32 has just cleared its summary (Reset button
    # or website Reset). The API then clears the test data too.
    reset_done: bool = False


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
# 6b. CONTROL PIN AND REMOTE-CONTROL HELPERS
# Functions ending in _locked must be called while
# holding state_lock.
# =====================================================

def verify_control_pin(x_control_pin: Optional[str]) -> None:
    """Check the PIN typed on the website. Locks after repeated failures."""
    if not CONTROL_PIN:
        raise HTTPException(
            status_code=503,
            detail="CONTROL_PIN is not configured on Render."
        )

    now = time.monotonic()

    with state_lock:
        locked_until = pin_guard["locked_until"]

    if now < locked_until:
        wait = int(locked_until - now) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Too many wrong PIN attempts. Try again in {wait} seconds."
        )

    supplied = (x_control_pin or "").encode("utf-8")
    expected = CONTROL_PIN.encode("utf-8")

    if not x_control_pin or not hmac.compare_digest(supplied, expected):
        with state_lock:
            recent = [
                t for t in pin_guard["failures"]
                if now - t < PIN_LOCK_SECONDS
            ]
            recent.append(now)
            pin_guard["failures"] = recent

            if len(recent) >= PIN_MAX_FAILURES:
                pin_guard["locked_until"] = now + PIN_LOCK_SECONDS
                pin_guard["failures"] = []

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing control PIN."
        )

    with state_lock:
        pin_guard["failures"] = []


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_event_locked(level: str, message: str) -> None:
    remote["last_event"] = {
        "level": level,
        "message": message,
        "time_utc": utc_now_iso(),
    }


def touch_device_locked(status: Optional[str] = None) -> None:
    """Remember that the ESP32 just contacted the API."""
    device = remote["device"]
    device["last_seen_mono"] = time.monotonic()
    device["last_seen_utc"] = utc_now_iso()

    if status is not None:
        device["status"] = status


def device_age_locked() -> Optional[float]:
    seen = remote["device"]["last_seen_mono"]

    if seen is None:
        return None

    return time.monotonic() - seen


def device_online_locked() -> bool:
    age = device_age_locked()
    return age is not None and age <= DEVICE_OFFLINE_AFTER_SECONDS


def expire_command_locked() -> None:
    """Cancel a command the ESP32 did not collect in time."""
    pending = remote["pending"]

    if pending is None:
        return

    if time.monotonic() - pending["created_mono"] > COMMAND_TTL_SECONDS:
        remote["last_command"] = {
            "command_id": pending["id"],
            "command": pending["command"],
            "result": "expired",
            "time_utc": utc_now_iso(),
        }
        remote["pending"] = None
        record_event_locked(
            "warning",
            f"The '{pending['command']}' command was cancelled because "
            f"the ESP32 did not respond within {COMMAND_TTL_SECONDS} seconds."
        )


def pending_command_locked(only: Optional[str] = None):
    """Return the waiting command (optionally only of one kind)."""
    expire_command_locked()
    pending = remote["pending"]

    if pending is None:
        return None

    if only is not None and pending["command"] != only:
        return None

    return pending


def resolve_pending_locked(
    command: str,
    result: str,
    event_message: str
) -> bool:
    """
    Close the waiting command when the ESP32's behaviour already proves
    it was carried out (for example a batch arrives after "start"), even
    if the ESP32's own confirmation message never reached the API.
    """
    pending = remote["pending"]

    if pending is None or pending["command"] != command:
        return False

    remote["acknowledged_command_id"] = pending["id"]
    remote["last_command"] = {
        "command_id": pending["id"],
        "command": pending["command"],
        "result": result,
        "time_utc": utc_now_iso(),
    }
    remote["pending"] = None
    record_event_locked("info", event_message)
    return True


def build_hint(
    online: bool,
    status: str,
    pending: Optional[dict],
    message: str
) -> str:
    """Plain-language instruction the website can show to the operator."""
    if not online:
        return (
            "The ESP32 is offline. Check its power and Wi-Fi, wait about "
            "30 seconds, and make sure this API is running."
        )

    if pending is not None:
        return "Command sent. Waiting for the ESP32 to confirm..."

    hints = {
        "idle": "Ready. Press Start to begin a new test.",
        "recording": (
            "Recording in progress. Press Stop when the drawing is finished."
        ),
        "finalizing": (
            "Recording stopped. Uploading the remaining data and "
            "calculating the final result. Please wait."
        ),
        "completed": (
            "Test completed. Review the result, or press Reset to "
            "prepare for the next participant."
        ),
        "error": (
            f"The ESP32 reported a problem: {message} "
            "You can try Start again or press Reset."
        ),
        "unknown": (
            "The ESP32 is connected. Waiting for its first status "
            "report (up to 10 seconds)."
        ),
    }

    return hints.get(status, "")


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
            "/session/sensor-data",
            "/session/command",
            "/session/status",
            "/device/command",
            "/device/status"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device_api_key_configured": bool(DEVICE_API_KEY),
        "control_pin_configured": bool(CONTROL_PIN)
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
# The response still contains {"predictions": [0, 1, ...]}.
# Two extra fields are added for remote control:
#   "command":    "stop" when the website asked to stop, else "none"
#   "command_id": ID of that command (0 when none)
# Older ESP32 firmware ignores the extra fields.
# This endpoint NEVER rejects a batch because of session
# state. Sensor data is additionally stored (RAM only)
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

        # A batch proves the ESP32 is alive and recording. The only
        # exception is the last partial batch after Stop ("finalizing").
        touch_device_locked()

        if remote["device"]["status"] != "finalizing":
            remote["device"]["status"] = "recording"
            remote["device"]["session_id"] = batch.session_id

            # Data is arriving, so a waiting "start" has clearly worked.
            resolve_pending_locked(
                "start", "done", "Recording started."
            )

        stop_request = pending_command_locked(only="stop")

    response = {
        "predictions": predictions,
        "command": "none",
        "command_id": 0
    }

    if stop_request is not None:
        response["command"] = "stop"
        response["command_id"] = stop_request["id"]

    return response


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
        touch_device_locked()

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
        touch_device_locked()

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


# =====================================================
# 15. WEBSITE SENDS A COMMAND (START / STOP / RESET)
# Needs the control PIN in the X-Control-PIN header.
# The command is only QUEUED here; the ESP32 collects it
# and confirms through /device/status.
# =====================================================

@app.post("/session/command")
def website_command(
    data: WebsiteCommandRequest,
    x_control_pin: Optional[str] = Header(
        default=None,
        alias="X-Control-PIN"
    )
):
    verify_control_pin(x_control_pin)

    with state_lock:
        if data.device_id != remote["device"]["device_id"]:
            raise HTTPException(
                status_code=404,
                detail="Unknown device_id."
            )

        expire_command_locked()

        if not device_online_locked():
            raise HTTPException(
                status_code=409,
                detail=(
                    "The ESP32 is offline. Check its power and Wi-Fi, "
                    "wait about 30 seconds, then try again."
                )
            )

        pending = remote["pending"]

        if pending is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"The '{pending['command']}' command is still waiting "
                    "for the ESP32. Please wait a few seconds."
                )
            )

        status = remote["device"]["status"]

        if data.command in ("start", "reset") and (
            status not in IDLE_LIKE_STATUSES
        ):
            if status == "recording":
                detail = "A test is already running. Press Stop first."
            else:
                detail = "The previous test is still finishing. Please wait."

            raise HTTPException(status_code=409, detail=detail)

        if data.command == "stop" and status != "recording":
            raise HTTPException(
                status_code=409,
                detail="The ESP32 is not recording."
            )

        remote["command_counter"] += 1

        command = {
            "id": remote["command_counter"],
            "command": data.command,
            "created_mono": time.monotonic(),
            "created_utc": utc_now_iso(),
        }

        remote["pending"] = command

        record_event_locked(
            "info",
            f"Website requested '{data.command}'. "
            "Waiting for the ESP32 to confirm."
        )

        return {
            "success": True,
            "command_id": command["id"],
            "command": data.command,
            "expires_in_seconds": COMMAND_TTL_SECONDS,
            "message": "Command queued. Waiting for the ESP32."
        }


# =====================================================
# 16. ESP32 CHECKS FOR A COMMAND
# Needs the device API key. Every call also tells the API
# that the ESP32 is alive.
# =====================================================

@app.get("/device/command")
def device_get_command(
    device_id: str = Query(default=DEFAULT_DEVICE_ID),
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    verify_device_key(x_api_key)

    with state_lock:
        if device_id != remote["device"]["device_id"]:
            raise HTTPException(
                status_code=404,
                detail="Unknown device_id."
            )

        touch_device_locked()
        pending = pending_command_locked()

        if pending is None:
            return {"command_id": 0, "command": "none"}

        return {
            "command_id": pending["id"],
            "command": pending["command"]
        }


# =====================================================
# 17. ESP32 REPORTS ITS REAL STATE
# Also confirms (acknowledges) a website command, and tells
# the API when the ESP32 has been reset.
# =====================================================

@app.post("/device/status")
def device_update_status(
    data: DeviceStatusRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    verify_device_key(x_api_key)

    global current_session_id, latest_final_result

    message = data.message or f"ESP32 status: {data.status}"

    with state_lock:
        device = remote["device"]

        if data.device_id != device["device_id"]:
            raise HTTPException(
                status_code=404,
                detail="Unknown device_id."
            )

        touch_device_locked(data.status)
        device["session_id"] = data.session_id or None
        device["message"] = message

        if data.reset_done:
            # The ESP32 cleared its summary: clear the website data too.
            sensor_history.clear()
            latest_final_result = None
            current_session_id = None
            device["session_id"] = None
            record_event_locked(
                "info",
                "Reset done. Previous test data was cleared."
            )

        expire_command_locked()

        acknowledged = False
        pending = remote["pending"]

        if (
            data.command_id > 0
            and pending is not None
            and pending["id"] == data.command_id
        ):
            acknowledged = True
            remote["acknowledged_command_id"] = data.command_id
            remote["last_command"] = {
                "command_id": pending["id"],
                "command": pending["command"],
                "result": "rejected" if data.rejected else "done",
                "time_utc": utc_now_iso(),
            }
            remote["pending"] = None

            record_event_locked(
                "warning" if data.rejected else "info",
                message
            )

        elif (
            not data.rejected
            and (
                (data.command_id == 0 and pending is not None)
                or (pending is not None and data.command_id != pending["id"])
            )
            and (
                (pending["command"] == "start" and data.status == "recording")
                or (pending["command"] == "stop"
                    and data.status in ("finalizing", "completed", "idle"))
                or (pending["command"] == "reset" and data.reset_done)
            )
        ):
            # The state itself proves the command was carried out.
            acknowledged = resolve_pending_locked(
                pending["command"], "done", message
            )

        elif (
            data.status == "error"
            and remote["last_event"]["message"] != message
        ):
            # Keep the reason visible instead of letting the next
            # heartbeat wipe it.
            record_event_locked("error", message)

        return {
            "success": True,
            "acknowledged": acknowledged,
            "acknowledged_command_id": remote["acknowledged_command_id"]
        }


# =====================================================
# 18. WEBSITE READS THE ESP32 STATE
# Public, read-only. No key is exposed to the browser.
#
# status is one of:
#   offline, unknown, idle, recording, finalizing,
#   completed, error
# can_start / can_stop / can_reset tell the website which
# buttons make sense right now.
# =====================================================

@app.get("/session/status")
def website_session_status():
    with state_lock:
        expire_command_locked()

        device = remote["device"]
        age = device_age_locked()
        online = age is not None and age <= DEVICE_OFFLINE_AFTER_SECONDS

        last_known_status = device["status"]
        status = last_known_status if online else "offline"

        message = device["message"]

        if not online:
            if age is None:
                message = "No signal from the ESP32 yet."
            else:
                message = (
                    f"No signal from the ESP32 for {int(age)} seconds."
                )

        pending = remote["pending"]
        pending_info = None

        if pending is not None:
            waited = time.monotonic() - pending["created_mono"]
            pending_info = {
                "command_id": pending["id"],
                "command": pending["command"],
                "waiting_seconds": round(waited, 1),
                "expires_in_seconds": max(
                    0.0, round(COMMAND_TTL_SECONDS - waited, 1)
                )
            }

        idle_like = status in IDLE_LIKE_STATUSES
        free = online and pending is None

        session_id = device["session_id"]

        if session_id is None:
            session_id = current_session_id

        return {
            "device_id": device["device_id"],
            "online": online,
            "status": status,
            "last_known_status": last_known_status,
            "session_id": session_id,
            "message": message,
            "hint": build_hint(online, status, pending, message),
            "last_seen_utc": device["last_seen_utc"],
            "last_seen_seconds_ago": (
                None if age is None else round(age, 1)
            ),
            "pending_command": pending_info,
            "last_command": remote["last_command"],
            "last_event": remote["last_event"],
            "can_start": free and idle_like,
            "can_stop": free and status == "recording",
            "can_reset": free and idle_like,
            "acknowledged_command_id": remote["acknowledged_command_id"],
            "api_version": app.version
        }
