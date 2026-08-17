"""
Smoke test de la API: confirma que levanta, carga los artefactos del modelo
(`data/models/*.joblib`, versionados en el repo) y que `/predict` responde
sobre el ejemplo documentado en `SesionInput`. Pensado para correr en CI en
cada push, sin depender de reentrenar nada.
"""
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import SesionInput

EJEMPLO_SESION = SesionInput.model_config["json_schema_extra"]["example"]


def test_health_reporta_modelo_cargado():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_sobre_sesion_valida():
    with TestClient(app) as client:
        response = client.post("/predict", json=EJEMPLO_SESION)

    assert response.status_code == 200
    cuerpo = response.json()
    assert 0.0 <= cuerpo["purchase_probability"] <= 1.0
    assert cuerpo["recommended_action"] in {"cross_selling", "retencion", "sin_accion"}


def test_predict_rechaza_categoria_no_vista_en_entrenamiento():
    sesion_invalida = {**EJEMPLO_SESION, "TrafficType": 12}

    with TestClient(app) as client:
        response = client.post("/predict", json=sesion_invalida)

    assert response.status_code == 422
