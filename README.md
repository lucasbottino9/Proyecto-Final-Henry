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
├── models/           # Exportación de modelos y pipelines finalizados
├── notebooks/        # Cuadernos Jupyter del ciclo de vida analítico
│   ├── EDA.ipynb                 # Análisis Exploratorio de Datos (EDA)
│   ├── ETL.ipynb                 # Limpieza, validación y persistencia del dataset
│   ├── feature_enginering.ipynb  # Feature engineering y preparación para modelado
│   └── modeling_mvp.ipynb        # Entrenamiento Baseline, SMOTE y MVP
├── src/              # Código fuente modular y reutilizable
│   ├── data_loader.py               # Módulo de ingesta de datos
│   ├── preprocessing.py             # Pipeline de ETL, OHE y Escalado
│   └── metrics.py                   # Evaluación de métricas de negocio y ML
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

`src/preprocessing.py` también expone `procesar_etl` / `preparar_datos_modelo`, el pipeline de encoding (One-Hot) y escalado (`StandardScaler`) que usa `feature_enginering.ipynb` para armar la matriz de modelado:

```python
from src.data_loader import cargar_datos
from src.preprocessing import preparar_datos_modelo

df = cargar_datos()
X_processed, y, preprocessor = preparar_datos_modelo(df)
print(f"Matriz procesada lista para entrenamiento: {X_processed.shape}")
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

- **`feature_enginering.ipynb` (modelado)**: lo recarga con `cargar_datos_procesados()` como punto de partida para encoding, escalado, features derivadas y split train/test. Lo necesita porque ahí es donde se decide qué hacer con los duplicados y outliers ya marcados (dropear o no, capar o no) según el modelo a entrenar — decisiones que el ETL deja documentadas pero no aplica.
- **Dashboard de Power BI**: se conecta directamente a ese mismo CSV como fuente de datos. Lo necesita porque ya viene tipado, validado y con las columnas `Weekend_label`/`Revenue_label` legibles para negocio, sin tener que replicar la limpieza en Power BI.

Es un único dataset para ambos casos (no dos archivos separados) para no mantener dos pipelines de limpieza que puedan divergir: la misma validación de esquema, tipos y filas vale tanto para entrenar el modelo como para lo que ve un analista en el dashboard. Cada consumidor aplica sus propias transformaciones adicionales (dropear duplicados, capar outliers, encoding) sobre su propia copia en memoria, sin modificar el archivo compartido — por eso el ETL marca en vez de eliminar (ver "Principales decisiones" abajo).

Pasos del notebook:

1. **Validación de esquema**: confirma, como invariante forzada, que las 18 columnas esperadas están presentes y que no hay valores nulos (por lo tanto no se requiere imputación).
2. **Corrección de tipos**: `Weekend`/`Revenue` como `bool`; `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` como `category`.
3. **Duplicados y outliers**: se detectan y marcan con columnas booleanas (`es_duplicado_exacto`, `outlier_<columna>`), pero **no se eliminan ni se capan** — ver "Principales decisiones" abajo.
4. **Chequeos de consistencia**: invariantes duras (sin negativos, conteo=0 ⇒ duración=0) y hallazgos informativos (conteo>0 con duración≈0, rangos `[0,1]`).
5. **Verificación de balance de clases** de `Revenue` (~85/15), sin corregirlo.
6. **Etiquetas legibles** (`Weekend_label`, `Revenue_label`) pensadas para un dashboard de Power BI.
7. **Persistencia**: guarda el resultado en `data/processed/online_shoppers_intention_procesado.csv`.

**Principales decisiones:**

- Los duplicados exactos (~1,6% de las filas) se marcan (`es_duplicado_exacto`) pero no se eliminan en el ETL: el dataset no tiene un identificador único de sesión/usuario, por lo que podrían ser sesiones reales distintas con el mismo comportamiento. Cada consumidor del dataset procesado decide qué hacer con ellos sobre su propia copia, sin tocar la fuente compartida:
  - **Dashboard (Power BI)**: conviene conservarlos. Al no haber ID de sesión, eliminarlos subestimaría el tráfico/las conversiones reales que ve un analista de negocio.
  - **Modelo (`feature_enginering.ipynb`)**: es razonable dropearlos antes de entrenar, usando la columna `es_duplicado_exacto`, para que filas idénticas no le den peso artificial extra a un mismo patrón durante el ajuste.
- Los outliers (regla IQR) se marcan pero no se capan ni se recortan: el EDA ya concluyó que probablemente reflejan comportamiento real de navegación; capar depende del modelo que se use, así que esa decisión se deja para `feature_enginering.ipynb`.
- El dataset procesado se guarda en **CSV** (no Parquet, ya que el proyecto no tiene `pyarrow` como dependencia) y es la fuente pensada para conectar directamente a un **dashboard de Power BI**.
- El split train/test y el encoding/escalado de variables se hacen en `feature_enginering.ipynb`, no acá, para evitar fuga de datos al ajustar encoders/scalers con información del conjunto de test.
- No se agregan filas ni datos sintéticos en ninguna etapa: tanto el dataset de modelado como el del dashboard deben reflejar datos reales.

### `notebooks/feature_enginering.ipynb` — Feature Engineering

Validación de la limpieza de datos y creación del `ColumnTransformer` modular (One-Hot Encoding + `StandardScaler`) para la automatización del preprocesamiento previo al modelado.

### `notebooks/modeling_mvp.ipynb` — Modelado y MVP

Entrenamiento de modelos de clasificación en iteraciones:

- **Manejo del Desbalance:** Integración de SMOTE y balanceo por pesos (`class_weight='balanced'`).
- **Comparativa:** Evaluación del Modelo Baseline (Regresión Logística) frente a modelos ensemble avanzados (Random Forest / XGBoost).
- **Lógica del Recomendador:** Transformación de la probabilidad estimada por el modelo en acciones de recomendación automática.

---

## 📝 Registro de Decisiones Técnicas

Las justificaciones de arquitectura, tratamiento de duplicados, codificación y escalado se encuentran documentadas en detalle en el archivo:
👉 `docs/DECISIONES_TECNICAS.md`
