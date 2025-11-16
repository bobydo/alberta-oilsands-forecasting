"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request body for production prediction."""
    values: List[float] = Field(..., description="Recent production values (window_size length)", min_length=6, max_length=6)
    
    class Config:
        json_schema_extra = {
            "example": {
                "values": [1500.0, 1550.0, 1600.0, 1650.0, 1700.0, 1750.0]
            }
        }

class PredictionResponse(BaseModel):
    """Response body containing predicted value."""
    prediction: float = Field(..., description="Predicted next production value")