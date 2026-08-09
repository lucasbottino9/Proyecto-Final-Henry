"""
Evaluación de modelos de clasificación para el scoring de intención de compra.

El dataset está desbalanceado (~85/15), por lo que se prioriza precision,
recall, F1 y AUC (ROC y precision-recall) de la clase positiva (`Revenue=1`)
por sobre accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluar_modelo(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Calcula las métricas relevantes para un dataset desbalanceado.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas (clase 0/1).
        y_prob: Probabilidad predicha de la clase positiva.

    Returns:
        Diccionario con `precision`, `recall`, `f1`, `roc_auc` y
        `average_precision` (PR-AUC), todas sobre la clase positiva.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }


def comparar_modelos(resultados: dict[str, dict]) -> pd.DataFrame:
    """
    Arma una tabla comparativa a partir de varios resultados de `evaluar_modelo`.

    Args:
        resultados: Diccionario `{nombre_modelo: metricas}`, donde `metricas`
            es la salida de `evaluar_modelo`.

    Returns:
        DataFrame con una fila por modelo, ordenado por `roc_auc` descendente.
    """
    tabla = pd.DataFrame(resultados).T
    return tabla.sort_values("roc_auc", ascending=False)
