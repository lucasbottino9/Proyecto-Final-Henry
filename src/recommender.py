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
