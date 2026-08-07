# Proyecto-Final-Henry
Proyecto final de Ciencia de Datos de Henry. Desarrollo de un sistema de recomendación inteligente para comercio electrónico que combina la predicción de intención de compra con recomendaciones de productos personalizadas, utilizando Machine Learning, FastAPI, Streamlit y las mejores prácticas de MLOps.

## Estructura del proyecto

```
data/
  csv/        # Datasets en formato CSV
  raw/        # Datos crudos sin procesar
  processed/  # Datos procesados listos para modelado
  models/     # Modelos entrenados serializados
models/       # Artefactos de modelos
notebooks/    # Notebooks de EDA, feature engineering y modelado
src/          # Código fuente (carga de datos, preprocesamiento, métricas)
```

## Uso

Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Cargar el dataset y ver un resumen (filas, columnas, primeras filas):

```bash
python src/data_loader.py
```

## Notebooks

### `notebooks/EDA.ipynb` — Análisis Exploratorio de Datos

EDA de referencia del proyecto sobre `online_shoppers_intention.csv` (12.330 sesiones, 18 variables), con la variable objetivo `Revenue` (si la sesión terminó en compra). Carga los datos mediante `src/data_loader.py` y recorre:

1. **Estructura y calidad de datos**: tipos, `describe()`, valores faltantes y registros duplicados.
2. **Variable objetivo**: distribución y balance de clases de `Revenue`.
3. **Variables numéricas y categóricas**: distribución individual (histogramas, countplots).
4. **Outliers**: boxplots y conteo de atípicos por rango intercuartílico (IQR).
5. **Análisis bivariado**: cada variable vs. `Revenue` (boxplots y countplots).
6. **Correlaciones**: matriz de correlación numérica y correlación de todas las variables (incluidas las categóricas codificadas) contra `Revenue`.

**Principales hallazgos:**

- No hay valores faltantes en ninguna columna, por lo que no se requiere imputación.
- Hay ~1% de registros duplicados; no se eliminan automáticamente porque el dataset no tiene un identificador único de sesión/usuario, y podrían ser sesiones distintas con comportamiento similar.
- `Revenue` está fuertemente desbalanceada (~85% no compra vs. ~15% compra), lo que obliga a priorizar métricas como precision/recall/F1 por sobre accuracy al entrenar modelos.
- Las variables numéricas tienen distribuciones muy sesgadas y bastantes outliers, especialmente `PageValues`, `Informational` e `Informational_Duration`; se conservan porque pueden reflejar comportamiento real de navegación.
- `PageValues`, `ExitRates`, `BounceRates` y `ProductRelated_Duration` son las variables más asociadas con `Revenue`, y son las mejores candidatas como features para el modelo.

### `notebooks/modeling_mvp.ipynb`

Notebook de modelado (MVP) posterior al EDA.
