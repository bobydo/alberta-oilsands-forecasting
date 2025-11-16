"""SAGD bitumen production forecasting API endpoints."""
from fastapi import APIRouter
from src.st53.inference_st53 import ST53Predictor
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter()
predictor = None

@router.get("/")
def sagd_info():
    """SAGD endpoint information."""
    return {
        "endpoint": "/sagd/predict",
        "method": "POST",
        "description": "Predict next month's SAGD bitumen production",
        "required_data": "8 months of historical production values (m³)",
        "example_generic": {
            "values": [95000.0, 98000.0, 97500.0, 99000.0, 101000.0, 100500.0, 102000.0, 103500.0]
        },
        "real_data_examples": {
            "Cenovus_Christina_Lake": {
                "values": [38440.48, 38453.59, 38345.22, 23973.98, 40922.27, 40339.68, 38701.56, 37223.34],
                "description": "Largest Cenovus SAGD operation (~38,000-41,000 m³/month)"
            },
            "Cenovus_Foster_Creek": {
                "values": [30717.12, 32122.67, 30897.38, 30137.87, 29907.24, 31504.45, 30909.04, 30768.31],
                "description": "Consistent production (~30,000-32,000 m³/month)"
            },
            "Cenovus_Sunrise": {
                "values": [8232.27, 8086.21, 7690.36, 8227.58, 8423.98, 8405.0, 8477.05, 7892.54],
                "description": "Lower production (~7,700-8,500 m³/month)"
            }
        },
        "documentation": "Visit /docs for interactive API testing"
    }

@router.post("/predict", response_model=PredictionResponse)
def sagd_predict(request: PredictionRequest):
    """Predict next SAGD bitumen production value."""
    global predictor
    if predictor is None:
        predictor = ST53Predictor("models")
    return {"prediction": predictor.predict(request.values)}
