from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.model import get_model_response

model_name = "Model"
version = "v1.0.0"

app = FastAPI()

# Input for data validation
class Input(BaseModel):
    # Add features
    feature_one: float
    feature_two: float

# Ouput for data validation
class Output(BaseModel):
    proba_0: float
    proba_1: float


@app.get('/health')
def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response