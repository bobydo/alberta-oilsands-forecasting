from fastapi import APIRouter
from src.st39.inference_st39 import load_st39_model, predict_st39
from api.schemas import PredictionRequest, PredictionResponse

router=APIRouter()
model=None
window=None

@router.post("/predict", response_model=PredictionResponse)
def mining_predict(request: PredictionRequest):
    global model, window
    if model is None:
        model, window = load_st39_model("models")
    return {"prediction":predict_st39(model,window,request.values)}
