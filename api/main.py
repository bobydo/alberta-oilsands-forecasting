"""FastAPI application for oil sands production forecasting."""
from fastapi import FastAPI
from api.routers import sagd, mining

app = FastAPI(
    title="Alberta Oil Sands Forecasting API",
    description="LSTM-based production forecasting for ST53 (SAGD) and ST39 (Mining)",
    version="1.0.0"
)

app.include_router(sagd.router, prefix="/sagd", tags=["SAGD"])
app.include_router(mining.router, prefix="/mining", tags=["Mining"])
