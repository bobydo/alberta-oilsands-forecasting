from pydantic import BaseModel

class PredictionRequest(BaseModel):
    values: list[float]

class PredictionResponse(BaseModel):
    prediction: float