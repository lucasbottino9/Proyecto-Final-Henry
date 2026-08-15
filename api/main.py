"""
API de scoring de intención de compra y motor de recomendación (Metric Mindset).

Correr con: `uvicorn api.main:app --reload` desde la raíz del repo.
Documentación interactiva: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import inference
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.cargar_artefactos()
    yield


app = FastAPI(
    title="Metric Mindset — Scoring de Intención de Compra",
    description=(
        "Sirve el modelo de scoring de intención de compra (`Revenue`) y el motor de "
        "recomendación de cross-selling / retención sobre sesiones de e-commerce."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
