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

- `notebooks/eda_final.ipynb`: EDA de referencia del proyecto (estructura, calidad de datos, distribución del target, análisis univariado/bivariado, outliers y correlaciones), consolidado a partir de los EDAs exploratorios previos.
- `notebooks/01_eda_by_julian.ipynb`: EDA exploratorio original, con un primer modelo baseline (Random Forest) sobre el dataset.
