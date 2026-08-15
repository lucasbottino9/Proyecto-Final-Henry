"""
Carga de artefactos (preprocesador + modelo final) y lógica de inferencia
para la API de scoring de intención de compra.

Reusa el mismo pipeline que los notebooks de modelado: `crear_features_derivadas`
(features derivadas) -> `preprocessor.transform` (encoding + escalado, ya
ajustado) -> `modelo.predict_proba` -> `asignar_accion` (motor de reglas).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import CARPETA_MODELOS_POR_DEFECTO, crear_features_derivadas
from src.preprocessing import COLUMNAS_CATEGORICAS
from src.recommender import asignar_accion

RUTA_PREPROCESSOR = CARPETA_MODELOS_POR_DEFECTO / "preprocessor.joblib"
RUTA_MODELO = CARPETA_MODELOS_POR_DEFECTO / "modelo_final.joblib"

_preprocessor = None
_modelo = None


def artefactos_cargados() -> bool:
    return _preprocessor is not None and _modelo is not None


def cargar_artefactos(
    ruta_preprocessor: Path = RUTA_PREPROCESSOR, ruta_modelo: Path = RUTA_MODELO
) -> None:
    """Carga `preprocessor.joblib` y `modelo_final.joblib` una sola vez (ver `api.main:lifespan`)."""
    global _preprocessor, _modelo
    _preprocessor = joblib.load(ruta_preprocessor)
    _modelo = joblib.load(ruta_modelo)


def categorias_conocidas() -> dict[str, list]:
    """Categorías que vio el `OneHotEncoder` al ajustarse, por columna categórica."""
    ohe = _preprocessor.named_transformers_["cat"].named_steps["onehot"]
    return dict(zip(COLUMNAS_CATEGORICAS, ohe.categories_))


def validar_categorias(fila: pd.Series) -> None:
    """
    Rechaza códigos categóricos no vistos en entrenamiento (ej. `TrafficType=12`,
    que no existe en el dataset UCI) en vez de dejar que `OneHotEncoder
    (handle_unknown="ignore")` los degrade en silencio a un vector all-zero.

    Raises:
        ValueError: si algún valor no es una categoría conocida.
    """
    errores = []
    for columna, valores_validos in categorias_conocidas().items():
        valor = fila[columna]
        if valor not in valores_validos:
            valor_legible = valor.item() if hasattr(valor, "item") else valor
            validos_legibles = sorted(str(v) for v in valores_validos)
            errores.append(f"{columna}={valor_legible!r} no es una categoría vista en entrenamiento (válidas: {validos_legibles})")
    if errores:
        raise ValueError("; ".join(errores))


def predecir_sesion(datos_sesion: dict) -> tuple[float, str]:
    """
    Predice la probabilidad de compra y la acción recomendada para una sesión.

    Args:
        datos_sesion: diccionario con las columnas crudas de la sesión
            (mismos nombres que `SesionInput` / `COLUMNAS_ESPERADAS` sin `Revenue`).

    Returns:
        `(probabilidad_compra, accion_recomendada)`.

    Raises:
        RuntimeError: si los artefactos no fueron cargados todavía.
        ValueError: si alguna columna categórica trae un valor no visto en entrenamiento.
    """
    if not artefactos_cargados():
        raise RuntimeError("Los artefactos del modelo no están cargados (ver api.main:lifespan).")

    df = pd.DataFrame([datos_sesion])
    validar_categorias(df.iloc[0])

    df = crear_features_derivadas(df)
    X = _preprocessor.transform(df)

    probabilidad = float(_modelo.predict_proba(X)[0, 1])
    accion = asignar_accion(np.array([probabilidad])).iloc[0]
    return probabilidad, accion
