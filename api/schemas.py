"""
Schemas Pydantic de la API de scoring de intención de compra.

`SesionInput` espera las mismas columnas crudas del dataset (`COLUMNAS_ESPERADAS`
en `src/preprocessing.py`, sin `Revenue`, que es lo que se predice), para poder
pasarlas directo a `src.features.crear_features_derivadas` sin una capa de
mapeo intermedia.
"""
from typing import Literal

from pydantic import BaseModel, Field

MESES_VISTOS_EN_ENTRENAMIENTO = Literal[
    "Aug", "Dec", "Feb", "Jul", "June", "Mar", "May", "Nov", "Oct", "Sep"
]
TIPOS_VISITANTE_VISTOS_EN_ENTRENAMIENTO = Literal[
    "New_Visitor", "Other", "Returning_Visitor"
]


class SesionInput(BaseModel):
    """Métricas de comportamiento de una sesión de e-commerce, en tiempo real."""

    Administrative: int = Field(ge=0, description="Cantidad de páginas administrativas visitadas en la sesión")
    Administrative_Duration: float = Field(ge=0, description="Tiempo total en páginas administrativas, en segundos")
    Informational: int = Field(ge=0, description="Cantidad de páginas informativas visitadas en la sesión")
    Informational_Duration: float = Field(ge=0, description="Tiempo total en páginas informativas, en segundos")
    ProductRelated: int = Field(ge=0, description="Cantidad de páginas de producto visitadas en la sesión")
    ProductRelated_Duration: float = Field(ge=0, description="Tiempo total en páginas de producto, en segundos")
    BounceRates: float = Field(ge=0, le=1, description="Tasa de rebote promedio de las páginas vistas (0 a 1)")
    ExitRates: float = Field(ge=0, le=1, description="Tasa de salida promedio de las páginas vistas (0 a 1)")
    PageValues: float = Field(ge=0, description="Valor promedio (económico) de las páginas vistas en la sesión")
    SpecialDay: float = Field(ge=0, le=1, description="Cercanía de la sesión a una fecha comercial especial (0 = lejos, 1 = el mismo día)")
    Month: MESES_VISTOS_EN_ENTRENAMIENTO = Field(
        description="Mes de la sesión. El dataset UCI no tiene sesiones de enero ni abril, por eso no son valores válidos."
    )
    OperatingSystems: int = Field(
        ge=1, le=8,
        description="Código de sistema operativo, entero de 1 a 8. Sin diccionario público que traduzca el código a un nombre real (limitación del dataset UCI).",
    )
    Browser: int = Field(
        ge=1, le=13,
        description="Código de navegador, entero de 1 a 13. Sin diccionario público (limitación del dataset UCI).",
    )
    Region: int = Field(
        ge=1, le=9,
        description="Código de región geográfica, entero de 1 a 9. Sin diccionario público (limitación del dataset UCI).",
    )
    TrafficType: int = Field(
        ge=1, le=20,
        description="Código de tipo de tráfico, entero de 1 a 20 salvo el 12 (no existe en el dataset de entrenamiento). Sin diccionario público.",
    )
    VisitorType: TIPOS_VISITANTE_VISTOS_EN_ENTRENAMIENTO = Field(description="Tipo de visitante: nuevo, recurrente u otro")
    Weekend: bool = Field(description="`true` si la sesión ocurrió en fin de semana, `false` en caso contrario")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 45.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 25,
                "ProductRelated_Duration": 620.5,
                "BounceRates": 0.01,
                "ExitRates": 0.02,
                "PageValues": 35.0,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    }

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    direction: str

class PredictResponse(BaseModel):
    """Respuesta del scoring + motor de prescripción."""

    purchase_probability: float = Field(
        ge=0,
        le=1,
        description="Probabilidad estimada de compra.",
    )

    recommended_action: str = Field(
        description="Acción comercial recomendada."
    )

    priority: str = Field(
        description="Prioridad de la intervención: alta, media o baja."
    )

    reason: str = Field(
        description="Motivo que explica la prescripción generada."
    )
    top_features: list[FeatureExplanation] = Field( 
        description="Principales variables que influyeron en la predicción."
    )   