"""
Carga del dataset de intención de compra (online_shoppers_intention.csv).

Expone `cargar_datos`, que localiza y lee el CSV del proyecto como un
DataFrame de pandas, validando que el archivo exista antes de intentar
leerlo.
"""

from pathlib import Path
import pandas as pd

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CSV_POR_DEFECTO = RUTA_PROYECTO / "data" / "csv" / "online_shoppers_intention.csv"


def cargar_datos(ruta_csv: Path | str = RUTA_CSV_POR_DEFECTO) -> pd.DataFrame:
    """
    Carga el dataset de intención de compra desde un archivo CSV.

    Args:
        ruta_csv: Ruta al CSV a cargar. Por defecto, el dataset del proyecto
            en ``data/csv/online_shoppers_intention.csv``.

    Returns:
        DataFrame con el contenido del CSV.

    Raises:
        FileNotFoundError: Si no existe ningún archivo en `ruta_csv`.
    """
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")
    return pd.read_csv(ruta_csv)


if __name__ == "__main__":
    datos = cargar_datos()
    print(f"Dataset cargado: {datos.shape[0]} filas x {datos.shape[1]} columnas\n")
    print("Primeras filas:")
    print(datos.head())
    print("\nColumnas:")
    print(datos.columns.tolist())
