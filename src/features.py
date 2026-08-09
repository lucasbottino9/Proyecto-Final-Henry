"""
Feature engineering y preparación de la matriz de modelado.

Toma el dataset ya limpio (ver `src.preprocessing.cargar_datos_procesados`),
agrega features derivadas, separa train/test de forma estratificada y
ajusta el encoding (One-Hot) + escalado (StandardScaler) solo con train,
para evitar fuga de datos hacia el conjunto de test.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_MODELOS_POR_DEFECTO = RUTA_PROYECTO / "data" / "models"

COLUMNAS_FEATURES_DERIVADAS = [
    "duracion_total",
    "paginas_totales",
    "duracion_promedio_pagina",
    "proporcion_paginas_producto",
    "es_visitante_recurrente",
]
COLUMNAS_NUMERICAS_MODELADO = COLUMNAS_NUMERICAS + [
    "duracion_total",
    "paginas_totales",
    "duracion_promedio_pagina",
    "proporcion_paginas_producto",
]


def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivadas del comportamiento de navegación de la sesión.

    Args:
        df: DataFrame ya limpio (ver `src.preprocessing.limpiar_dataset`).

    Returns:
        Copia de `df` con `duracion_total`, `paginas_totales`,
        `duracion_promedio_pagina`, `proporcion_paginas_producto` y
        `es_visitante_recurrente`.
    """
    df = df.copy()
    df["duracion_total"] = (
        df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    )
    df["paginas_totales"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]

    hay_paginas = df["paginas_totales"] > 0
    df["duracion_promedio_pagina"] = np.where(
        hay_paginas, df["duracion_total"] / df["paginas_totales"], 0.0
    )
    df["proporcion_paginas_producto"] = np.where(
        hay_paginas, df["ProductRelated"] / df["paginas_totales"], 0.0
    )
    df["es_visitante_recurrente"] = df["VisitorType"] == "Returning_Visitor"
    return df


def construir_column_transformer(
    columnas_numericas: list[str], columnas_categoricas: list[str]
) -> ColumnTransformer:
    """
    Arma el `ColumnTransformer` de encoding + escalado para modelado.

    Args:
        columnas_numericas: Columnas a escalar con `StandardScaler`.
        columnas_categoricas: Columnas a codificar con `OneHotEncoder`.

    Returns:
        `ColumnTransformer` sin ajustar.
    """
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), columnas_numericas),
            ("cat", Pipeline(steps=[("onehot", ohe)]), columnas_categoricas),
        ]
    )


def dividir_train_test(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa `df` en train/test de forma estratificada por `target_col`.

    El dataset está desbalanceado (~85/15), por lo que el split se
    estratifica para preservar esa proporción en ambos conjuntos.

    Args:
        df: DataFrame con features y la columna objetivo.
        target_col: Nombre de la columna objetivo.
        test_size: Proporción del conjunto de test.
        random_state: Semilla para reproducibilidad.

    Returns:
        `(X_train, X_test, y_train, y_test)`.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preparar_datos_modelado(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """
    Orquesta el pipeline completo de feature engineering para modelado.

    Aplica `crear_features_derivadas`, descarta columnas que no aportan a
    un modelo (flags de outliers y etiquetas legibles para dashboard),
    separa train/test estratificado y ajusta el `ColumnTransformer` solo
    con train.

    Args:
        df: DataFrame limpio (ver `src.preprocessing.cargar_datos_procesados`).
        target_col: Nombre de la columna objetivo.
        test_size: Proporción del conjunto de test.
        random_state: Semilla para reproducibilidad.

    Returns:
        `(X_train, X_test, y_train, y_test, preprocessor)`, donde
        `X_train`/`X_test` ya están codificadas y escaladas, y
        `preprocessor` es el `ColumnTransformer` ajustado.
    """
    df = crear_features_derivadas(df)
    columnas_a_descartar = [c for c in df.columns if c.startswith("outlier_")]
    columnas_a_descartar += ["Weekend_label", "Revenue_label"]
    df = df.drop(columns=[c for c in columnas_a_descartar if c in df.columns])

    X_train, X_test, y_train, y_test = dividir_train_test(
        df, target_col=target_col, test_size=test_size, random_state=random_state
    )

    columnas_numericas = COLUMNAS_NUMERICAS_MODELADO + ["Weekend", "es_visitante_recurrente"]
    preprocessor = construir_column_transformer(columnas_numericas, COLUMNAS_CATEGORICAS)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


def guardar_artefactos_modelado(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    carpeta: Path | str = CARPETA_MODELOS_POR_DEFECTO,
) -> Path:
    """
    Persiste el split train/test y el preprocesador ajustado con `joblib`.

    Args:
        X_train, X_test, y_train, y_test: Salida de `preparar_datos_modelado`.
        preprocessor: `ColumnTransformer` ya ajustado.
        carpeta: Carpeta de destino. Por defecto, `data/models/`.

    Returns:
        Ruta resuelta de la carpeta donde se guardaron los artefactos.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        carpeta / "train_test_split.joblib",
    )
    joblib.dump(preprocessor, carpeta / "preprocessor.joblib")
    return carpeta.resolve()
