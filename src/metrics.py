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


def precision_at_k(y_true: pd.Series, y_prob: np.ndarray, k: int) -> float:
    """
    Precision entre las `k` sesiones con mayor probabilidad de compra.

    Simula la situación real de negocio: solo hay presupuesto/capacidad para
    intervenir (cross-selling o retención) sobre las `k` sesiones más
    prometedoras, no sobre todo el tráfico.

    Args:
        y_true: Etiquetas reales (`Revenue`).
        y_prob: Probabilidad predicha de compra, mismo orden que `y_true`.
        k: Cantidad de sesiones top a considerar.

    Returns:
        Proporción de las top-`k` sesiones que efectivamente compraron.
    """
    orden = np.argsort(y_prob)[::-1][:k]
    y_true_arr = np.asarray(y_true)
    return float(y_true_arr[orden].mean())


def recall_at_k(y_true: pd.Series, y_prob: np.ndarray, k: int) -> float:
    """
    Recall entre las `k` sesiones con mayor probabilidad de compra.

    Args:
        y_true: Etiquetas reales (`Revenue`).
        y_prob: Probabilidad predicha de compra, mismo orden que `y_true`.
        k: Cantidad de sesiones top a considerar.

    Returns:
        Proporción de todas las compras reales que quedan capturadas
        dentro de las top-`k` sesiones.
    """
    orden = np.argsort(y_prob)[::-1][:k]
    y_true_arr = np.asarray(y_true)
    total_positivos = y_true_arr.sum()
    if total_positivos == 0:
        return 0.0
    return float(y_true_arr[orden].sum() / total_positivos)


def evaluar_ranking(
    y_true: pd.Series, y_prob: np.ndarray, k_pct: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
) -> pd.DataFrame:
    """
    Tabla de Precision@K / Recall@K para distintos tamaños de intervención.

    Args:
        y_true: Etiquetas reales (`Revenue`).
        y_prob: Probabilidad predicha de compra, mismo orden que `y_true`.
        k_pct: Fracciones del total de sesiones a evaluar como `k` (ej. `0.1`
            = top 10% de sesiones por probabilidad de compra).

    Returns:
        DataFrame indexado por `k_pct`, con columnas `k`, `precision_at_k`
        y `recall_at_k`.
    """
    n = len(y_true)
    filas = []
    for pct in k_pct:
        k = max(1, round(n * pct))
        filas.append(
            {
                "k_pct": pct,
                "k": k,
                "precision_at_k": precision_at_k(y_true, y_prob, k),
                "recall_at_k": recall_at_k(y_true, y_prob, k),
            }
        )
    return pd.DataFrame(filas).set_index("k_pct")


def comparar_modelos(resultados: dict[str, dict], criterio: str = "roc_auc") -> pd.DataFrame:
    """
    Arma una tabla comparativa a partir de varios resultados de `evaluar_modelo`.

    Args:
        resultados: Diccionario `{nombre_modelo: metricas}`, donde `metricas`
            es la salida de `evaluar_modelo`.
        criterio: Columna por la que ordenar (ej. `"roc_auc"` o
            `"average_precision"`). `average_precision` (PR-AUC) es preferible
            cuando la clase positiva es minoritaria, como en este dataset.

    Returns:
        DataFrame con una fila por modelo, ordenado por `criterio` descendente.
    """
    tabla = pd.DataFrame(resultados).T
    return tabla.sort_values(criterio, ascending=False)
