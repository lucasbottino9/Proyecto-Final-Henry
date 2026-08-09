# 🚀 Metric Mindset — Scoring de Intención de Compra y Motor de Recomendaciones

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Methodology](https://img.shields.io/badge/Methodology-Scrum-green.svg)](#)

Desarrollo de un sistema de analítica predictiva y recomendación inteligente para comercio electrónico. Combina la clasificación en tiempo real de la intención de compra (`Revenue`) con la activación automática de estrategias de retención y recomendación de productos, aplicando Machine Learning, FastAPI, Streamlit y mejores prácticas de MLOps.

> *"Transformando sesiones anónimas de e-commerce en conversiones reales mediante analítica predictiva."*

---

## 👥 Equipo de Trabajo — Metric Mindset

| Integrante | Rol en el Proyecto |
| :--- | :--- |
| **Ema Camila** | Líder de Equipo / Product Owner & Scrum Master |
| **Julián Culzoni** | Data Analyst |
| **Luis** | Data Engineer |
| **Juan Manuel** | Data Scientist |
| **Lucas Bottino** | Data Scientist |

---

## 📌 Contexto de Negocio

En el comercio electrónico, aproximadamente el **85% de las sesiones terminan sin realizar una compra**, representando un elevado costo de adquisición de tráfico no monetizado.

**Metric Mindset** resuelve este problema mediante una arquitectura dual:
1. **Scoring de Intención (ML):** Asigna a cada sesión un porcentaje de propensión a la compra basándose en sus métricas de navegación.
2. **Motor de Recomendación:** Dispara acciones personalizadas en tiempo real:
   * **Alta Propensión (>70%):** Muestra recomendaciones cruzadas (*Cross-selling / Upselling*).
   * **Baja Propensión / Riesgo (<30%):** Despliega incentivos dinámicos de retención (ej. cupones de descuento por tiempo limitado).

---

## 📂 Estructura del Proyecto

```text
Proyecto-Final-Henry/
├── data/
│   ├── csv/          # Datasets en formato CSV (online_shoppers_intention.csv)
│   ├── raw/          # Datos crudos sin procesar
│   ├── processed/    # Datos limpios y matrices procesadas para modelado
│   └── models/       # Artefactos y binarios de modelos (.pkl / .joblib)
├── docs/             # Registro formal de decisiones técnicas (ADR)
├── notebooks/        # Cuadernos Jupyter del ciclo de vida analítico
│   ├── EDA.ipynb                 # Análisis Exploratorio de Datos (EDA)
│   ├── ETL.ipynb                 # Limpieza, validación y persistencia del dataset
│   ├── feature_enginering.ipynb  # Feature engineering y preparación para modelado
│   └── modeling_mvp.ipynb        # Baseline, desbalance, ensembles y recomendador
├── src/              # Código fuente modular y reutilizable
│   ├── data_loader.py               # Módulo de ingesta de datos
│   ├── preprocessing.py             # Pipeline de ETL (limpieza, validación)
│   ├── features.py                  # Feature engineering, split, OHE y escalado
│   ├── metrics.py                   # Evaluación de modelos (precision/recall/F1/AUC)
│   └── recommender.py               # Motor de recomendación (cross-selling/retención)
├── requirements.txt  # Dependencias del proyecto
└── README.md         # Documentación principal del repositorio
```

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/lucasbottino9/Proyecto-Final-Henry.git
cd Proyecto-Final-Henry
```

### 2. Crear y activar el entorno virtual

```bash
# En Windows:
python -m venv venv
.\venv\Scripts\activate

# En Mac/Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 💻 Uso del Pipeline de Datos (src/)

Cargar el dataset y ver un resumen (filas, columnas, primeras filas):

```bash
python src/data_loader.py
```

Correr el pipeline de ETL (limpieza, validación y persistencia del dataset procesado):

```bash
python -m src.preprocessing
```

`src/features.py` arma la matriz de modelado a partir del dataset ya limpio (features derivadas, split train/test estratificado, encoding y escalado — ver detalle en `feature_enginering.ipynb`):

```python
from src.preprocessing import cargar_datos_procesados, COLUMNAS_CATEGORICAS
from src.features import preparar_datos_modelado, guardar_artefactos_modelado

df = cargar_datos_procesados()
X_train, X_test, y_train, y_test, preprocessor = preparar_datos_modelado(df)
guardar_artefactos_modelado(X_train, X_test, y_train, y_test, preprocessor)
```

## 📓 Notebooks

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

### `notebooks/ETL.ipynb` — Limpieza y persistencia del dataset

Toma el dataset crudo (el mismo que carga `EDA.ipynb`) y actúa sobre las decisiones que el EDA dejó pendientes, apoyándose en las funciones de `src/preprocessing.py`.

**¿Qué devuelve y para qué se usa?** El ETL devuelve un único dataset limpio (`data/processed/online_shoppers_intention_procesado.csv`), y ese mismo archivo se usa **para ambos destinos**, no uno u otro:

- **`feature_enginering.ipynb` (modelado)**: lo recarga con `cargar_datos_procesados()` como punto de partida para features derivadas, split train/test, encoding y escalado.
- **Dashboard de Power BI**: se conecta directamente a ese mismo CSV como fuente de datos. Lo necesita porque ya viene tipado, validado, sin duplicados y con las columnas `Weekend_label`/`Revenue_label` legibles para negocio, sin tener que replicar la limpieza en Power BI.

Es un único dataset para ambos casos (no dos archivos separados) para que modelo y dashboard vean exactamente los mismos datos: la misma validación de esquema, tipos, duplicados y filas vale tanto para entrenar el modelo como para lo que ve un analista en el dashboard.

Pasos del notebook:

1. **Validación de esquema**: confirma, como invariante forzada, que las 18 columnas esperadas están presentes y que no hay valores nulos (por lo tanto no se requiere imputación).
2. **Corrección de tipos**: `Weekend`/`Revenue` como `bool`; `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` como `category`.
3. **Duplicados**: se detectan y marcan (`es_duplicado_exacto`) para auditar cuántos había, y luego **se eliminan** (conservando la primera aparición) — ver "Principales decisiones" abajo.
4. **Outliers**: se detectan y marcan con columnas booleanas (`outlier_<columna>`), pero **no se capan**.
5. **Chequeos de consistencia**: invariantes duras (sin negativos, conteo=0 ⇒ duración=0) y hallazgos informativos (conteo>0 con duración≈0, rangos `[0,1]`).
6. **Verificación de balance de clases** de `Revenue` (~85/15), sin corregirlo.
7. **Etiquetas legibles** (`Weekend_label`, `Revenue_label`) pensadas para un dashboard de Power BI.
8. **Persistencia**: guarda el resultado en `data/processed/online_shoppers_intention_procesado.csv`.

**Principales decisiones:**

- Los duplicados exactos (~1,6% de las filas, 125 sobre 12.330) se marcan (`es_duplicado_exacto`) y luego se eliminan (conservando la primera aparición): modelo y dashboard deben partir del mismo dataset para que las métricas de tráfico/conversión sean coherentes entre sí, en vez de que cada consumidor vea un conteo de filas distinto.
- Los outliers (regla IQR) se marcan pero no se capan ni se recortan: el EDA ya concluyó que probablemente reflejan comportamiento real de navegación; capar depende del modelo que se use, así que esa decisión se deja para `feature_enginering.ipynb`.
- El dataset procesado se guarda en **CSV** (no Parquet, ya que el proyecto no tiene `pyarrow` como dependencia) y es la fuente pensada para conectar directamente a un **dashboard de Power BI**.
- El split train/test y el encoding/escalado de variables se hacen en `feature_enginering.ipynb`, no acá, para evitar fuga de datos al ajustar encoders/scalers con información del conjunto de test.
- No se agregan filas ni datos sintéticos en ninguna etapa: tanto el dataset de modelado como el del dashboard deben reflejar datos reales.

### `notebooks/feature_enginering.ipynb` — Feature Engineering

Parte del dataset limpio de `ETL.ipynb` y lo deja listo para entrenar, usando `src/features.py`:

- Descarta las columnas `outlier_*` (no capa los valores: el EDA/ETL ya concluyó que reflejan comportamiento real) y las etiquetas pensadas para el dashboard.
- Agrega cinco features derivadas: `duracion_total`, `paginas_totales`, `duracion_promedio_pagina`, `proporcion_paginas_producto` y `es_visitante_recurrente`.
- Separa train/test de forma estratificada por `Revenue` (dataset desbalanceado ~85/15), antes de ajustar cualquier transformación.
- Codifica (`OneHotEncoder`) y escala (`StandardScaler`) mediante un `ColumnTransformer` ajustado solo con train.
- Persiste `X_train`, `X_test`, `y_train`, `y_test` y el preprocesador ajustado en `data/models/` (`train_test_split.joblib`, `preprocessor.joblib`).

### `notebooks/modeling_mvp.ipynb` — Modelado y MVP

Carga los artefactos que deja `feature_enginering.ipynb` desde `data/models/`, entrena y compara modelos usando `src/metrics.py` y `src/recommender.py`:

- **Baseline:** Regresión Logística sin balancear — buena precision, recall bajo en la clase positiva (evidencia del desbalance).
- **Manejo del Desbalance:** `class_weight="balanced"` y SMOTE (`imbalanced-learn`, aplicado solo sobre train), comparados contra el baseline.
- **Comparativa:** Random Forest y XGBoost (`scale_pos_weight`) frente a las variantes de Regresión Logística, evaluados por precision, recall, F1, ROC-AUC y PR-AUC.
- **Selección y persistencia:** el mejor modelo por ROC-AUC/PR-AUC se guarda en `data/models/modelo_final.joblib`.
- **Lógica del Recomendador:** `asignar_accion` traduce la probabilidad de compra en `cross_selling` (>70%), `retencion` (<30%) o `sin_accion`, auditado contra la compra real con `resumen_acciones`.

