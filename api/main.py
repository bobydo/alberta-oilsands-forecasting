from fastapi import FastAPI
from api.routers import sagd, mining
app=FastAPI()
app.include_router(sagd.router,prefix="/sagd")
app.include_router(mining.router,prefix="/mining")
