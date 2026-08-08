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
| **[Tu Nombre]** | Líder de Equipo / Product Owner & Scrum Master |
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
│   ├── 01_eda.ipynb                 # Análisis Exploratorio de Datos (EDA)
│   ├── 02_feature_engineering.ipynb  # Validación de ETL y Pipelines
│   └── 03_modeling_mvp.ipynb        # Entrenamiento Baseline, SMOTE y MVP
├── src/              # Código fuente modular y reutilizable
│   ├── data_loader.py               # Módulo de ingesta de datos
│   ├── preprocessing.py             # Pipeline modular de ETL, OHE y Escalado
│   └── metrics.py                   # Evaluación de métricas de negocio y ML
├── requirements.txt  # Dependencias del proyecto
└── README.md         # Documentación principal del repositorio

Instalación y Configuración
1. Clonar el repositorio
Bash
git clone [https://github.com/lucasbottino9/Proyecto-Final-Henry.git](https://github.com/lucasbottino9/Proyecto-Final-Henry.git)
cd Proyecto-Final-Henry
2. Crear y activar el entorno virtual
Bash
# En Windows:
python -m venv venv
.\venv\Scripts\activate

# En Mac/Linux:
python3 -m venv venv
source venv/bin/activate
3. Instalar dependencias
Bash
pip install -r requirements.txt
💻 Uso del Pipeline de Datos (src/)
Carga rápida de datos base:
Bash
python src/data_loader.py
Ejecución del Pipeline Modular de Preprocesamiento:
El módulo src/preprocessing.py limpia el dataset (eliminación de duplicados exactos e imputación) y aplica transformaciones de One-Hot Encoding y StandardScaler mediante Pipelines reproducibles de Scikit-Learn:

Python
from src.data_loader import cargar_datos
from src.preprocessing import preparar_datos_modelo

# 1. Carga de dataset
df = cargar_datos()

# 2. Ejecutar ETL + Pipeline de transformación
X_processed, y, preprocessor = preparar_datos_modelo(df)

print(f"Matriz procesada lista para entrenamiento: {X_processed.shape}")
# Salida esperada: (12205, 80)
📓 Resumen de Notebooks
1. notebooks/01_eda.ipynb — Análisis Exploratorio de Datos
Análisis integral sobre online_shoppers_intention.csv (12.330 sesiones, 18 variables) centrado en la variable objetivo Revenue.

Principales hallazgos técnicos y de negocio:

Calidad de Datos: No existen valores faltantes en el conjunto base.

Duplicados: Se identificó ~1% de registros duplicados exactos (125 filas), cuyo tratamiento se delegó a la fase de preprocesamiento/ETL.

Sparsity & Desbalance: Revenue está fuertemente desbalanceado (~84.5% no compra vs. ~15.5% compra), lo que exige priorizar métricas como Precision, Recall, F1-Score y ROC-AUC por sobre el Accuracy.

Cold Start & Long Tail: PageValues, ExitRates, BounceRates y ProductRelated_Duration se identificaron como los predictores más asociados a la compra, permitiendo caracterizar sesiones de usuarios nuevos (New_Visitor) frente a recurrentes.

2. notebooks/02_feature_engineering.ipynb — Pipeline y ETL
Validación de la limpieza de datos y la creación del ColumnTransformer modular para la automatización del preprocesamiento.

3. notebooks/03_modeling_mvp.ipynb — Modelado y MVP
Entrenamiento de modelos de clasificación en iteraciones:

Manejo del Desbalance: Integración de SMOTE y balanceo por pesos (class_weight='balanced').

Comparativa: Evaluación del Modelo Baseline (Regresión Logística) frente a modelos ensemble avanzados (Random Forest / XGBoost).

Lógica del Recomendador: Transformación de la probabilidad estimada por el modelo en acciones de recomendación automática.

📝 Registro de Decisiones Técnicas
Las justificaciones de arquitectura, tratamiento de duplicados, codificación y escalado se encuentran documentadas en detalle en el archivo:
👉 docs/DECISIONES_TECNICAS.md