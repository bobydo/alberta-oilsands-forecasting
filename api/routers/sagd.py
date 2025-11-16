"""SAGD bitumen production forecasting API endpoints."""
from fastapi import APIRouter
from src.st53.inference_st53 import ST53Predictor
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter()
predictor = None

@router.post("/predict", response_model=PredictionResponse)
def sagd_predict(request: PredictionRequest):
    """Predict next SAGD bitumen production value."""
    global predictor
    if predictor is None:
        predictor = ST53Predictor("models")
    return {"prediction": predictor.predict(request.values)}
