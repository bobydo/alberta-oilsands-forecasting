"""Mining production forecasting API endpoints."""
from fastapi import APIRouter
from src.st39.inference_st39 import ST39Predictor
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter()
predictor = None

@router.post("/predict", response_model=PredictionResponse)
def mining_predict(request: PredictionRequest):
    """Predict next mining production value."""
    global predictor
    if predictor is None:
        predictor = ST39Predictor("models")
    return {"prediction": predictor.predict(request.values)}
