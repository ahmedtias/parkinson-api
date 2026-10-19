from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Parkinson's Disease Detection API")

model = joblib.load('parkinson_ensemble_model.pkl')

class SensorData(BaseModel):
    aX: float
    aY: float
    aZ: float
    gX: float
    gY: float
    gZ: float

@app.get("/")
def home():
    return {"message": "Parkinson Detection API is Running Live!"}

@app.post("/predict")
def predict(data: SensorData):
    input_data = pd.DataFrame([{
        'aX': data.aX,
        'aY': data.aY,
        'aZ': data.aZ,
        'gX': data.gX,
        'gY': data.gY,
        'gZ': data.gZ
    }])
    
    prediction = model.predict(input_data)[0]
    result_text = "Parkinson's Detected" if prediction == 1 else "Healthy / Normal"
    
    return {
        "prediction": int(prediction),
        "result": result_text
    }
