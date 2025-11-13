from fastapi import APIRouter
from src.st39.inference_st39 import load_st39_model, predict_st39
model,window=load_st39_model("models")
router=APIRouter()
@router.post("/predict")
def mining_predict(values:list):
    return {"prediction":predict_st39(model,window,values)}
