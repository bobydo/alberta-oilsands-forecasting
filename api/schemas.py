"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request body for production prediction."""
    values: List[float] = Field(..., description="Recent production values (window_size length)", min_length=8, max_length=8)
    
    class Config:
        json_schema_extra = {
            "example": {
                "values": [95000.0, 98000.0, 97500.0, 99000.0, 101000.0, 100500.0, 102000.0, 103500.0]
            }
        }

class PredictionResponse(BaseModel):
    """Response body containing predicted value."""
    prediction: float = Field(..., description="Predicted next production value")