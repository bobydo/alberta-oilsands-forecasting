from fastapi import APIRouter
from src.st53.inference_st53 import load_st53_model, predict_st53
from api.schemas import PredictionRequest, PredictionResponse

router=APIRouter()
model=None
window=None

@router.post("/predict", response_model=PredictionResponse)
def sagd_predict(request: PredictionRequest):
    global model, window
    if model is None:
        model, window = load_st53_model("models")
    return {"prediction":predict_st53(model,window,request.values)}
