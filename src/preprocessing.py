"""
Limpieza y validación del dataset de intención de compra.

Expone las funciones que arman el pipeline de ETL: validación de esquema,
corrección de tipos, eliminación de duplicados exactos, detección (no
eliminación) de outliers, chequeos de consistencia, y persistencia del
dataset limpio en ``data/processed/``. `limpiar_dataset` orquesta todo el
pipeline en un solo llamado.

Los duplicados se eliminan (no solo se marcan) para que el dataset
persistido sea único y coherente tanto para el modelo como para el
dashboard de Power BI.

No se realiza imputación (el dataset no tiene valores faltantes), ni
encoding/escalado para modelado, ni generación de datos sintéticos: esas
decisiones quedan para la etapa de feature engineering / modelado.
"""

from pathlib import Path

import pandas as pd

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_PROCESSED_POR_DEFECTO = (
    RUTA_PROYECTO / "data" / "processed" / "online_shoppers_intention_procesado.csv"
)

COLUMNAS_ESPERADAS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend", "Revenue",
]
COLUMNAS_BOOLEANAS = ["Weekend", "Revenue"]
COLUMNAS_CATEGORICAS = [
    "Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType",
]
COLUMNAS_NUMERICAS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
PARES_CONTEO_DURACION = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
]


def validar_esquema(df: pd.DataFrame) -> None:
    """
    Valida que el DataFrame tenga exactamente las columnas esperadas y
    ningún valor faltante.

    Args:
        df: DataFrame a validar.

    Raises:
        ValueError: Si faltan columnas, sobran columnas, o hay nulos.
    """
    columnas_actuales = set(df.columns)
    columnas_esperadas = set(COLUMNAS_ESPERADAS)

    faltantes = columnas_esperadas - columnas_actuales
    sobrantes = columnas_actuales - columnas_esperadas
    if faltantes or sobrantes:
        raise ValueError(
            f"Esquema inesperado. Faltan columnas: {sorted(faltantes)}. "
            f"Columnas no esperadas: {sorted(sobrantes)}."
        )

    nulos_por_columna = df.isna().sum()
    total_nulos = int(nulos_por_columna.sum())
    if total_nulos > 0:
        columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0].to_dict()
        raise ValueError(f"Se encontraron valores nulos: {columnas_con_nulos}")


def corregir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige los tipos de dato de las columnas booleanas y categóricas.

    Args:
        df: DataFrame de entrada (no se modifica in-place).

    Returns:
        Copia de `df` con `Weekend`/`Revenue` como `bool` y las columnas
        categóricas como dtype `category`.
    """
    df = df.copy()
    for columna in COLUMNAS_BOOLEANAS:
        df[columna] = df[columna].astype(bool)
    for columna in COLUMNAS_CATEGORICAS:
        df[columna] = df[columna].astype("category")
    return df


def detectar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca las filas que son duplicados exactos, sin eliminarlas todavía.

    Se usa antes de `eliminar_duplicados` para poder reportar (ver
    `resumen_duplicados`) cuántas filas había antes de quitarlas.

    Args:
        df: DataFrame de entrada.

    Returns:
        Copia de `df` con una columna booleana adicional `es_duplicado_exacto`.
    """
    df = df.copy()
    df["es_duplicado_exacto"] = df.duplicated(keep=False)
    return df


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados exactos, conservando la primera aparición de cada fila.

    Args:
        df: DataFrame de entrada. Si tiene la columna `es_duplicado_exacto`
            (ver `detectar_duplicados`), se descarta del resultado.

    Returns:
        Copia de `df` sin filas duplicadas, con el índice reiniciado.
    """
    columnas = [c for c in df.columns if c != "es_duplicado_exacto"]
    return df[columnas].drop_duplicates(keep="first").reset_index(drop=True)


def resumen_duplicados(df: pd.DataFrame) -> dict:
    """
    Resume la cantidad y proporción de duplicados exactos.

    Args:
        df: DataFrame con la columna `es_duplicado_exacto` (ver `detectar_duplicados`).

    Returns:
        Diccionario con `n_filas`, `n_duplicados_exactos` y `pct_duplicados`.
    """
    n_filas = len(df)
    n_duplicados = int(df["es_duplicado_exacto"].sum())
    return {
        "n_filas": n_filas,
        "n_duplicados_exactos": n_duplicados,
        "pct_duplicados": round(100 * n_duplicados / n_filas, 2) if n_filas else 0.0,
    }


def detectar_outliers_iqr(
    df: pd.DataFrame, columnas: list[str] | None = None, factor: float = 1.5
) -> pd.DataFrame:
    """
    Marca outliers univariados en columnas numéricas usando el rango
    intercuartílico (IQR), sin modificar los valores originales.

    Args:
        df: DataFrame de entrada.
        columnas: Columnas numéricas a evaluar. Por defecto, `COLUMNAS_NUMERICAS`.
        factor: Multiplicador del IQR para definir las cercas (1.5 = regla estándar).

    Returns:
        Copia de `df` con una columna booleana `outlier_<columna>` por cada
        columna evaluada.
    """
    columnas = columnas if columnas is not None else COLUMNAS_NUMERICAS
    df = df.copy()
    for columna in columnas:
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - factor * iqr
        limite_superior = q3 + factor * iqr
        df[f"outlier_{columna}"] = (df[columna] < limite_inferior) | (
            df[columna] > limite_superior
        )
    return df


def resumen_outliers(
    df_con_flags: pd.DataFrame, columnas: list[str] | None = None
) -> pd.DataFrame:
    """
    Resume la cantidad y proporción de outliers detectados por columna.

    Args:
        df_con_flags: DataFrame con columnas `outlier_<columna>` (ver `detectar_outliers_iqr`).
        columnas: Columnas numéricas a resumir. Por defecto, `COLUMNAS_NUMERICAS`.

    Returns:
        DataFrame con una fila por columna: `columna`, `n_outliers`, `pct_outliers`.
    """
    columnas = columnas if columnas is not None else COLUMNAS_NUMERICAS
    n_filas = len(df_con_flags)
    filas = []
    for columna in columnas:
        n_outliers = int(df_con_flags[f"outlier_{columna}"].sum())
        filas.append(
            {
                "columna": columna,
                "n_outliers": n_outliers,
                "pct_outliers": round(100 * n_outliers / n_filas, 2) if n_filas else 0.0,
            }
        )
    return pd.DataFrame(filas)


def validar_consistencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta chequeos de consistencia lógica sobre el dataset.

    Chequeos duros (levantan `ValueError` si se violan, porque indicarían
    corrupción de datos real): conteo=0 con duración>0 en los pares
    conteo/duración, y valores negativos en columnas numéricas.

    Chequeos informativos (se reportan, no son errores): conteo>0 con
    duración≈0 (plausible en visitas muy breves), `BounceRates`/`ExitRates`/
    `SpecialDay` fuera de `[0, 1]`.

    Args:
        df: DataFrame ya tipado (ver `corregir_tipos`).

    Returns:
        DataFrame resumen con columnas `chequeo`, `n_violaciones`, `severidad`.

    Raises:
        ValueError: Si se viola algún chequeo duro.
    """
    resultados = []

    for columna in COLUMNAS_NUMERICAS:
        n_negativos = int((df[columna] < 0).sum())
        if n_negativos > 0:
            raise ValueError(
                f"Se encontraron {n_negativos} valores negativos en '{columna}'."
            )
        resultados.append(
            {"chequeo": f"sin_negativos_{columna}", "n_violaciones": 0, "severidad": "dura"}
        )

    for conteo, duracion in PARES_CONTEO_DURACION:
        conteo_cero_duracion_positiva = int(((df[conteo] == 0) & (df[duracion] > 0)).sum())
        if conteo_cero_duracion_positiva > 0:
            raise ValueError(
                f"Se encontraron {conteo_cero_duracion_positiva} filas con "
                f"{conteo}=0 pero {duracion}>0."
            )
        resultados.append(
            {
                "chequeo": f"{conteo}_cero_implica_{duracion}_cero",
                "n_violaciones": 0,
                "severidad": "dura",
            }
        )

        conteo_positivo_duracion_cero = int(((df[conteo] > 0) & (df[duracion] == 0)).sum())
        resultados.append(
            {
                "chequeo": f"{conteo}_positivo_con_{duracion}_cero",
                "n_violaciones": conteo_positivo_duracion_cero,
                "severidad": "informativa",
            }
        )

    for columna in ["BounceRates", "ExitRates", "SpecialDay"]:
        fuera_de_rango = int(((df[columna] < 0) | (df[columna] > 1)).sum())
        resultados.append(
            {
                "chequeo": f"{columna}_fuera_de_[0,1]",
                "n_violaciones": fuera_de_rango,
                "severidad": "informativa",
            }
        )

    return pd.DataFrame(resultados)


def agregar_etiquetas_legibles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas con etiquetas legibles para consumo en dashboards
    (ej. Power BI), sin modificar ni reemplazar las columnas originales.

    `OperatingSystems`, `Browser`, `Region` y `TrafficType` quedan como
    códigos categóricos: el dataset original (UCI) no incluye el
    diccionario de valores para traducirlos.

    Args:
        df: DataFrame ya tipado (ver `corregir_tipos`).

    Returns:
        Copia de `df` con `Weekend_label` ("Sí"/"No") y `Revenue_label`
        ("Compra"/"No compra").
    """
    df = df.copy()
    df["Weekend_label"] = df["Weekend"].map({True: "Sí", False: "No"})
    df["Revenue_label"] = df["Revenue"].map({True: "Compra", False: "No compra"})
    return df


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquesta el pipeline completo de limpieza sobre el dataset crudo.

    Args:
        df: DataFrame crudo (ver `src.data_loader.cargar_datos`).

    Returns:
        DataFrame limpio, tipado, sin duplicados exactos, con outliers
        marcados (no eliminados) y etiquetas legibles agregadas.
    """
    df = df.copy()
    validar_esquema(df)
    df = corregir_tipos(df)
    df = detectar_duplicados(df)
    df = eliminar_duplicados(df)
    df = detectar_outliers_iqr(df)
    df = agregar_etiquetas_legibles(df)
    return df


def guardar_datos_procesados(
    df: pd.DataFrame, ruta_salida: Path | str = RUTA_PROCESSED_POR_DEFECTO
) -> Path:
    """
    Persiste el dataset limpio como CSV.

    Args:
        df: DataFrame a guardar.
        ruta_salida: Ruta de destino. Por defecto, `data/processed/online_shoppers_intention_procesado.csv`.

    Returns:
        Ruta resuelta del archivo escrito.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    return ruta_salida.resolve()


def cargar_datos_procesados(
    ruta_csv: Path | str = RUTA_PROCESSED_POR_DEFECTO,
) -> pd.DataFrame:
    """
    Recarga el dataset procesado desde CSV, reaplicando la corrección de
    tipos (el CSV no preserva los dtypes `category`/`bool`).

    Args:
        ruta_csv: Ruta al CSV procesado. Por defecto, la ruta de salida
            estándar del ETL.

    Returns:
        DataFrame con los mismos dtypes que produce `limpiar_dataset`.

    Raises:
        FileNotFoundError: Si no existe ningún archivo en `ruta_csv`.
    """
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos procesados en: {ruta_csv}"
        )
    df = pd.read_csv(ruta_csv)
    return corregir_tipos(df)


if __name__ == "__main__":
    from src.data_loader import cargar_datos

    datos_crudos = cargar_datos()
    print(f"Dataset crudo: {datos_crudos.shape[0]} filas x {datos_crudos.shape[1]} columnas")

    validar_esquema(datos_crudos)
    datos_tipados = corregir_tipos(datos_crudos)
    datos_marcados = detectar_duplicados(datos_tipados)

    print("\nResumen de duplicados (antes de eliminarlos):")
    print(resumen_duplicados(datos_marcados))

    datos_limpios = eliminar_duplicados(datos_marcados)
    datos_limpios = detectar_outliers_iqr(datos_limpios)
    datos_limpios = agregar_etiquetas_legibles(datos_limpios)
    print(
        f"\nDataset limpio: {datos_limpios.shape[0]} filas x {datos_limpios.shape[1]} columnas"
    )

    print("\nResumen de outliers:")
    print(resumen_outliers(datos_limpios).to_string(index=False))

    print("\nChequeos de consistencia:")
    print(validar_consistencia(datos_limpios).to_string(index=False))

    ruta_final = guardar_datos_procesados(datos_limpios)
    print(f"\nDataset procesado guardado en: {ruta_final}")
