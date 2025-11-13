from fastapi import APIRouter
from src.st53.inference_st53 import load_st53_model, predict_st53
model,window=load_st53_model("models")
router=APIRouter()
@router.post("/predict")
def sagd_predict(values:list):
    return {"prediction":predict_st53(model,window,values)}
