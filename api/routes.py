"""Ruteo de la API: endpoints de salud y de predicción."""
from fastapi import APIRouter, HTTPException

from api import inference
from api.schemas import PredictResponse, SesionInput

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Liveness check simple, usado por la demo de Streamlit antes de llamar a `/predict`."""
    return {"status": "ok", "model_loaded": inference.artefactos_cargados()}


@router.post("/predict", response_model=PredictResponse)
def predict(sesion: SesionInput) -> PredictResponse:
    """Recibe los datos crudos de una sesión y devuelve la probabilidad de compra
    (`Revenue`) y la acción recomendada por el motor de reglas."""
    try:
        probabilidad, prescripcion = inference.predecir_sesion(
            sesion.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        purchase_probability=probabilidad,
        recommended_action=prescripcion["accion"],
        priority=prescripcion["prioridad"],
        reason=prescripcion["motivo"],
    )
