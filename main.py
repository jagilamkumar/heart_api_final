from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize app
app = FastAPI()

# Load the trained model
model = joblib.load("xgboost_heart_disease_model.pkl")


# Define input schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalch: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: HeartData):
    # Convert input to 2D numpy array
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalch, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_label = int(prediction[0])

    # Map prediction to message
    result_message = "Heart Disease Detected" if prediction_label == 1 else "No Heart Disease"

    return {
        "prediction": prediction_label,
        "message": result_message
    }
