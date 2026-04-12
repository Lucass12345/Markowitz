from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.optimize import router as optimize_router
from api.routers.runs import router as runs_router

app = FastAPI(
    title="Markowitz Platform API",
    version="0.1.0",
    description="REST API for portfolio optimization, run persistence, and monitoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(optimize_router, prefix="/optimize", tags=["optimize"])
app.include_router(runs_router, prefix="/runs", tags=["runs"])