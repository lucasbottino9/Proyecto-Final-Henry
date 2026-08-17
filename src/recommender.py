"""
Motor de recomendación: traduce la probabilidad de compra estimada por el
modelo en una acción de negocio.

- Alta propensión (> umbral_alto): cross-selling / upselling.
- Baja propensión (< umbral_bajo): incentivo de retención (ej. cupón).
- Resto: sin acción automática.
"""

import numpy as np
import pandas as pd

ACCION_CROSS_SELLING = "cross_selling"
ACCION_RETENCION = "retencion"
ACCION_SIN_ACCION = "sin_accion"


def asignar_accion(
    probabilidades: np.ndarray, umbral_alto: float = 0.7, umbral_bajo: float = 0.3
) -> pd.Series:
    """
    Mapea probabilidades de compra a la acción de recomendación correspondiente.

    Args:
        probabilidades: Probabilidad de compra (`Revenue=1`) por sesión.
        umbral_alto: A partir de este umbral, se dispara cross-selling/upselling.
        umbral_bajo: Por debajo de este umbral, se dispara el incentivo de retención.

    Returns:
        Serie con una de `ACCION_CROSS_SELLING`, `ACCION_RETENCION` o
        `ACCION_SIN_ACCION` por sesión.
    """
    condiciones = [probabilidades > umbral_alto, probabilidades < umbral_bajo]
    valores = [ACCION_CROSS_SELLING, ACCION_RETENCION]
    return pd.Series(
        np.select(condiciones, valores, default=ACCION_SIN_ACCION), name="accion"
    )

def prescribir_sesion(
    probabilidad: float,
    datos_sesion: dict,
    umbral_alto: float = 0.70,
    umbral_bajo: float = 0.30,
) -> dict:
    """
    Traduce la probabilidad de compra y las señales de navegación
    en una prescripción comercial interpretable.

    Los umbrales de comportamiento se basan en la distribución
    observada del dataset procesado.
    """

    page_values = float(datos_sesion.get("PageValues", 0))
    bounce_rates = float(datos_sesion.get("BounceRates", 0))
    exit_rates = float(datos_sesion.get("ExitRates", 0))
    product_related = int(datos_sesion.get("ProductRelated", 0))
    product_duration = float(datos_sesion.get("ProductRelated_Duration", 0))

    # Umbrales derivados del dataset procesado
    PRODUCT_RELATED_ALTO = 38
    PRODUCT_DURATION_ALTA = 1477.0
    BOUNCE_ALTO = 0.0167
    EXIT_ALTO = 0.0485
    PAGE_VALUE_ALTO = 35.0

    interes_producto_alto = (
        product_related >= PRODUCT_RELATED_ALTO
        or product_duration >= PRODUCT_DURATION_ALTA
    )

    valor_pagina_alto = page_values >= PAGE_VALUE_ALTO

    riesgo_abandono = (
        bounce_rates >= BOUNCE_ALTO
        or exit_rates >= EXIT_ALTO
    )

    # 1. Alta propensión + señales fuertes
    if probabilidad >= umbral_alto and (
        interes_producto_alto or valor_pagina_alto
    ):
        return {
            "accion": "cross_selling",
            "prioridad": "alta",
            "motivo": (
                "Alta probabilidad de compra combinada con fuerte "
                "interés en productos o PageValues elevado."
            ),
        }

    # 2. Alta propensión sin señales adicionales fuertes
    if probabilidad >= umbral_alto:
        return {
            "accion": "upselling",
            "prioridad": "media",
            "motivo": (
                "Alta probabilidad de compra; se propone aumentar "
                "el valor potencial de la conversión."
            ),
        }

    # 3. Baja propensión + señales de abandono
    if probabilidad < umbral_bajo and riesgo_abandono:
        return {
            "accion": "retencion",
            "prioridad": "alta",
            "motivo": (
                "Baja propensión de compra acompañada por señales "
                "elevadas de rebote o salida."
            ),
        }

    # 4. Baja propensión sin señales de abandono fuertes
    if probabilidad < umbral_bajo:
        return {
            "accion": "retencion_suave",
            "prioridad": "media",
            "motivo": (
                "Baja probabilidad de compra, pero sin señales "
                "críticas de abandono."
            ),
        }

    # 5. Propensión intermedia con interés comercial
    if interes_producto_alto or page_values > 0:
        return {
            "accion": "incentivo_suave",
            "prioridad": "media",
            "motivo": (
                "Propensión intermedia con señales de interés "
                "en productos."
            ),
        }

    return {
        "accion": "sin_accion",
        "prioridad": "baja",
        "motivo": (
            "No se detectan señales suficientes para justificar "
            "una intervención automática."
        ),
    }

def resumen_acciones(acciones: pd.Series, y_true: pd.Series) -> pd.DataFrame:
    """
    Cruza la acción asignada contra la compra real, para auditar el criterio.

    Args:
        acciones: Salida de `asignar_accion`.
        y_true: Etiqueta real de `Revenue` (mismo orden/índice que `acciones`).

    Returns:
        DataFrame con, por acción: cantidad de sesiones, tasa real de compra
        y proporción sobre el total.
    """
    y_true = pd.Series(np.asarray(y_true), name="Revenue")
    resumen = (
        pd.DataFrame({"accion": acciones.values, "Revenue": y_true.values})
        .groupby("accion")["Revenue"]
        .agg(n_sesiones="count", tasa_compra_real="mean")
    )
    resumen["pct_del_total"] = round(100 * resumen["n_sesiones"] / len(y_true), 2)
    return resumen.sort_values("n_sesiones", ascending=False)
