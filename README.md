# Metric Mindset 🚀

## Scoring de Intención de Compra y Motor de Recomendaciones para E-Commerce

> **Transformando sesiones anónimas de e-commerce en conversiones reales mediante analítica predictiva.**

---

## Descripción del proyecto 📌

**Metric Mindset** es una solución de analítica predictiva aplicada al comercio electrónico que busca transformar el comportamiento de navegación de los usuarios en **predicciones accionables y recomendaciones de negocio**.

El proyecto parte de una problemática habitual en e-commerce: una gran proporción de las sesiones de navegación no termina en una compra.

A partir del análisis de comportamiento de las sesiones, desarrollamos un sistema capaz de:

1. Analizar el comportamiento de navegación de cada sesión.
2. Estimar la probabilidad de que dicha sesión termine en una compra.
3. Clasificar la intención de compra utilizando umbrales de negocio.
4. Activar una recomendación de acción según el nivel de intención.
5. Exponer la predicción mediante una API.
6. Visualizar e interactuar con el sistema mediante una aplicación desarrollada en Streamlit.

La solución combina:

- Data Analytics
- Machine Learning
- Feature Engineering
- Model Selection
- Hyperparameter Tuning
- Probability Calibration
- Recommendation Logic
- FastAPI
- Streamlit
- MLflow
- Testing y CI/CD

---

## Objetivo de negocio 🎯

El objetivo principal es **identificar la intención de compra de una sesión antes de que finalice**, permitiendo que el negocio pueda adaptar su estrategia de intervención.

La solución busca responder dos preguntas:

### 1. ¿Qué probabilidad tiene esta sesión de convertirse en una compra?

El modelo genera un score de probabilidad de compra entre 0 y 1.

### 2. ¿Qué acción debería tomar el negocio?

A partir de ese score se activa una estrategia:

| Probabilidad estimada | Acción | Objetivo |
|---|---|---|
| > 70% | Cross-selling | Aprovechar una alta intención de compra |
| 30% – 70% | Sin acción | Evitar intervenciones innecesarias |
| < 30% | Retención | Intentar reducir el riesgo de abandono |

De esta manera, el proyecto busca transformar una predicción de Machine Learning en una **decisión accionable de negocio**.

---

## Dataset 📊

El proyecto utiliza el dataset **Online Shoppers Intention**, compuesto originalmente por:

- **12.330 sesiones**
- **18 variables**
- Variable objetivo: `Revenue`

La variable `Revenue` indica si una sesión terminó (`True`) o no (`False`) en una compra.

### Distribución inicial del target

| Resultado | Sesiones | Proporción |
|---|---:|---:|
| No compra | 10.422 | 84,53% |
| Compra | 1.908 | 15,47% |

El fuerte desbalance entre clases fue uno de los principales desafíos técnicos del proyecto.

---

## Análisis Exploratorio de Datos — EDA 🔎

El análisis exploratorio permitió comprender la estructura, calidad y comportamiento del dataset antes de avanzar al modelado.

Entre los principales controles realizados se encuentran:

- Dimensiones del dataset.
- Tipos de datos.
- Valores faltantes.
- Duplicados.
- Distribución de la variable objetivo.
- Distribución de variables numéricas.
- Variables categóricas.
- Detección de outliers mediante IQR.
- Análisis univariado.
- Análisis bivariado.
- Correlaciones.
- Relación entre variables y `Revenue`.

### Calidad de los datos

El dataset original no presenta valores faltantes en sus 18 columnas.

Se detectaron:

- **0 valores nulos**
- **125 duplicados efectivos**
- Diversos valores considerados outliers mediante la regla IQR.

Los outliers no fueron eliminados automáticamente porque el análisis determinó que podían representar comportamientos reales de navegación y no necesariamente errores de captura.

Entre las variables con mayor relación con `Revenue` se destacan:

- `PageValues`
- `ExitRates`
- `ProductRelated`
- `Month`
- `ProductRelated_Duration`
- `BounceRates`
- `Administrative`
- `VisitorType`

---

## ETL y calidad de datos 🧹

Luego del EDA se desarrolló un pipeline de limpieza y validación.

El proceso incluye:

1. Validación del esquema.
2. Control de valores nulos.
3. Corrección de tipos de datos.
4. Detección y eliminación de duplicados.
5. Detección de outliers.
6. Validación de reglas de consistencia.
7. Verificación del balance de clases.
8. Generación de etiquetas legibles para negocio.
9. Persistencia del dataset procesado.

El dataset pasa de **12.330** a **12.205 sesiones**.

Luego del proceso de limpieza se obtiene:

**12.205 filas × 30 columnas**

El mismo dataset procesado es utilizado tanto para el modelado como para el Dashboard de Power BI, garantizando consistencia entre los indicadores analíticos y los datos utilizados durante el entrenamiento.

---

## Feature Engineering 🧠

A partir de las variables originales se generaron nuevas características orientadas a representar mejor el comportamiento de navegación.

Entre ellas:

### `duracion_total`

Suma del tiempo dedicado a páginas administrativas, informativas y de producto.

### `paginas_totales`

Cantidad total de páginas visitadas durante la sesión.

### `duracion_promedio_pagina`

Tiempo promedio de navegación por página.

### `proporcion_paginas_producto`

Proporción de la sesión dedicada a páginas relacionadas con productos.

### `es_visitante_recurrente`

Variable derivada de `VisitorType` que identifica si el usuario es recurrente.

---

## Preprocesamiento ⚙️

El pipeline de preprocesamiento fue diseñado para evitar data leakage.

El proceso realiza:

- Split train/test estratificado.
- Encoding de variables categóricas mediante `OneHotEncoder`.
- Escalado de variables numéricas mediante `StandardScaler`.
- Persistencia del `ColumnTransformer`.

El split utilizado es:

- **80% Train**
- **20% Test**
- `random_state = 42`
- Estratificación sobre `Revenue`

Resultado:

| Dataset | Sesiones |
|---|---:|
| Train | 9.764 |
| Test | 2.441 |

El preprocesador se ajusta exclusivamente sobre Train y posteriormente se aplica sobre Test.

---

## Modelado 🤖

### Sprint 1 — Modelado MVP

Durante el primer Sprint se evaluaron diferentes alternativas:

- Regresión Logística
- Regresión Logística + SMOTE
- Random Forest
- XGBoost

El problema de desbalance fue tratado mediante:

- `class_weight="balanced"`
- `scale_pos_weight`
- SMOTE

La evaluación priorizó:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

Accuracy no resulta suficiente para evaluar correctamente un problema con una clase positiva minoritaria.

### Resultado Sprint 1

Random Forest fue el mejor modelo inicial según ROC-AUC y PR-AUC:

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,658 | 0,709 | 0,683 | **0,924** | **0,723** |
| XGBoost | 0,646 | 0,678 | 0,662 | 0,922 | 0,717 |
| Logistic Regression Balanced | 0,503 | 0,806 | 0,620 | 0,911 | 0,664 |
| Logistic Regression + SMOTE | 0,502 | 0,791 | 0,614 | 0,910 | 0,665 |
| Logistic Regression Baseline | 0,766 | 0,419 | 0,541 | 0,901 | 0,654 |

---

## Sprint 2 — Modelado Avanzado 🚀

Durante el segundo Sprint se profundizó la optimización del modelo.

Se incorporaron:

- LightGBM
- CatBoost
- Optuna
- SMOTE
- `class_weight`
- Calibración de probabilidades
- `CalibratedClassifierCV`
- Brier Score
- Curvas de calibración
- Precision@K
- Recall@K
- MLflow

La selección final se realizó utilizando **PR-AUC** como métrica principal.

Esto se debe a que el sistema no solamente necesita ordenar correctamente las sesiones, sino producir probabilidades suficientemente confiables para tomar decisiones mediante umbrales absolutos.

---

## Modelo final 🏆

El modelo definitivo seleccionado fue:

### LightGBM tuneado con Optuna + calibración sigmoid

La calibración se realizó mediante:

```python
CalibratedClassifierCV(
    method="sigmoid",
    cv=3
)
```

### Métricas sobre el conjunto de Test

| Métrica | Resultado |
|---|---:|
| Precision | 0,718 |
| Recall | 0,665 |
| F1-Score | 0,690 |
| ROC-AUC | 0,938 |
| PR-AUC | 0,756 |
| Brier Score | 0,068 |

La calibración tuvo un impacto especialmente importante.

El modelo tuneado sin calibrar presentaba:

- PR-AUC = **0,752**
- Brier = **0,101**

Luego de aplicar calibración sigmoid:

- PR-AUC = **0,756**
- Brier = **0,068**

Esto representa una mejora de aproximadamente **33% en Brier Score**.

La calibración resulta especialmente relevante porque el motor de recomendaciones utiliza umbrales absolutos de probabilidad.

---

## Evaluación de Ranking 🎯

Además de las métricas tradicionales de clasificación, se implementó una evaluación orientada a escenarios donde el negocio posee una capacidad limitada para intervenir sobre las sesiones.

Se utilizaron:

- Precision@K
- Recall@K

El modelo ordena las sesiones según su probabilidad estimada de compra y permite seleccionar solamente el porcentaje de usuarios más prometedor.

### Resultados

| Top K | Precision@K | Recall@K |
|---:|---:|---:|
| 5% | 86% | 27% |
| 10% | 80% | 51% |
| 15% | 70% | 68% |
| 20% | 60% | 77% |
| 25% | 54% | 87% |
| 30% | 47% | 91% |

### Insight de negocio

Si el negocio tuviera capacidad para intervenir solamente sobre el **10% de las sesiones**, el modelo permitiría alcanzar aproximadamente:

- **80% de Precision@10%**
- **51% de Recall@10%**

Es decir, aproximadamente **8 de cada 10 sesiones seleccionadas son compradores reales**, mientras que se captura alrededor de la mitad de todas las compras del período.

Esto permite dimensionar campañas según presupuesto y capacidad operativa.

---

## Motor de Recomendaciones 🧩

El modelo predictivo se conecta con un motor de reglas de negocio.

La función principal es:

```python
asignar_accion()
```

La lógica actual utiliza dos umbrales:

```text
Probabilidad > 70%  → CROSS-SELLING
Probabilidad < 30%  → RETENCIÓN
30% – 70%           → SIN ACCIÓN
```

### Resultado del modelo final

| Acción | Sesiones | % del total | Tasa real de compra |
|---|---:|---:|---:|
| Retención | 1.987 | 81,40% | 4,88% |
| Sin acción | 256 | 10,49% | 44,92% |
| Cross-selling | 198 | 8,11% | 85,86% |

El segmento clasificado como `cross_selling` presenta una tasa real de compra del **85,86%**, mostrando una fuerte coherencia entre el score predictivo y el comportamiento observado.

La calibración permitió además corregir el desajuste existente en el modelo sin calibrar: la tasa real de compra del grupo `cross_selling` pasó de aproximadamente **65% a 85,86%**.

---

## Dashboard de Analytics 📊

El proyecto incorpora un Dashboard desarrollado en **Power BI** para facilitar la interpretación de los principales hallazgos del EDA.

Entre los principales indicadores se encuentran:

- Total de sesiones.
- Total de compras.
- Tasa de conversión.
- Distribución por tipo de visitante.
- Comportamiento temporal.
- Variables relacionadas con la conversión.
- Indicadores de comportamiento de navegación.

### Principales KPIs

Sobre el dataset procesado:

- **12.205 sesiones**
- **1.908 compras**
- **15,63% de conversión**

Esto significa que aproximadamente **16 de cada 100 sesiones terminan en una compra**.

El dashboard permite analizar estos indicadores de forma interactiva y segmentarlos según diferentes dimensiones del comportamiento.

---

## API — FastAPI 🌐

El modelo final se expone mediante una API REST desarrollada con FastAPI.

La API permite:

### Health Check

```http
GET /health
```

### Predicción

```http
POST /predict
```

La solicitud recibe las variables originales de una sesión y devuelve:

```json
{
  "purchase_probability": 0.83,
  "recommended_action": "cross_selling"
}
```

La API valida los datos de entrada mediante schemas de Pydantic.

También rechaza categorías no válidas mediante respuestas HTTP `422`, evitando que valores desconocidos sean procesados silenciosamente.

---

## Aplicación Streamlit 🖥️

El proyecto incorpora una interfaz interactiva desarrollada con Streamlit.

La aplicación permite:

1. Ingresar las características de una sesión.
2. Enviar la información a la API.
3. Obtener la probabilidad estimada de compra.
4. Visualizar la intención detectada.
5. Mostrar la acción recomendada.

La aplicación permite visualizar especialmente las variables con mayor relevancia para la predicción:

- Page Values
- Exit Rate
- Bounce Rate
- Month
- Visitor Type
- Weekend

El resto de las variables se encuentran disponibles dentro de una sección expandible.

---

## Arquitectura de la solución 🔄

```text
Sesión E-Commerce
       │
       ▼
Data Processing
ETL + Validation
       │
       ▼
Feature Engineering
Encoding + Scaling
       │
       ▼
LightGBM
+ Calibration Sigmoid
       │
       ▼
Purchase Probability
       │
       ▼
Recommendation Engine
       │
 ┌─────┼───────────┐
 ▼     ▼           ▼
Cross  Sin acción  Retención
>70%   30-70%      <30%
```

---

## Estructura del repositorio 🗂️

```text
Proyecto-Final-Henry/
│
├── api/
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── inference.py
│
├── data/
│   ├── csv/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── ETL.ipynb
│   ├── feature_enginering.ipynb
│   ├── modeling_mvp.ipynb
│   └── modeling_avanzado.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── metrics.py
│   └── recommender.py
│
├── tests/
│   └── test_api_smoke.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── demo_app.py
├── requirements.txt
└── README.md
```

---

## Notebooks 📓

### `EDA.ipynb`

Análisis exploratorio y diagnóstico de calidad de datos.

Incluye:

- Estructura del dataset.
- Calidad.
- Target.
- Distribuciones.
- Outliers.
- Análisis bivariado.
- Correlaciones.
- Conclusiones.

### `ETL.ipynb`

Pipeline de limpieza y persistencia.

Incluye:

- Validación de esquema.
- Corrección de tipos.
- Duplicados.
- Outliers.
- Validaciones de consistencia.
- Balance de clases.
- Etiquetas para Power BI.

### `feature_enginering.ipynb`

Preparación del dataset para Machine Learning.

Incluye:

- Feature engineering.
- Train/test split.
- Encoding.
- Scaling.
- Persistencia de artefactos.

### `modeling_mvp.ipynb`

Modelado correspondiente al Sprint 1.

Incluye:

- Baseline.
- Manejo de desbalance.
- Comparación de modelos.
- Métricas.
- Primer motor de recomendaciones.

### `modeling_avanzado.ipynb`

Modelado correspondiente al Sprint 2.

Incluye:

- LightGBM.
- CatBoost.
- Optuna.
- SMOTE.
- `class_weight`.
- Calibración.
- Brier Score.
- Precision@K.
- Recall@K.
- MLflow.
- Selección del modelo final.

---

## Testing y CI 🧪

El proyecto incluye tests automatizados para validar el funcionamiento básico de la API.

El smoke test verifica:

- `GET /health`
- Predicción válida.
- Respuesta del modelo.
- Validación de categorías desconocidas.
- Código HTTP `422` ante datos inválidos.

Los tests se ejecutan mediante:

```bash
pytest tests/ -v
```

Además, GitHub Actions ejecuta automáticamente los tests ante cambios mediante Push o Pull Request.

---

## MLflow 📈

Durante el Sprint 2 se incorporó MLflow para realizar tracking de experimentos.

Cada experimento puede registrar:

- Modelo utilizado.
- Hiperparámetros.
- Métricas.
- Precision@K.
- Recall@K.
- Artefactos.

Para acceder a la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir:

```text
http://localhost:5000
```

---

## Tecnologías utilizadas 🛠️

### Data Analytics

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Power BI

### Machine Learning

- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- Imbalanced-learn
- Optuna

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend / Demo

- Streamlit

### MLOps

- MLflow
- Joblib
- Git
- GitHub
- GitHub Actions

---

## Instalación ⚙️

### 1. Clonar el repositorio

```bash
git clone https://github.com/lucasbottino9/Proyecto-Final-Henry.git
cd Proyecto-Final-Henry
```

### 2. Crear entorno virtual

#### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Los artefactos del modelo final y el preprocesador se encuentran versionados en el repositorio, por lo que no es necesario ejecutar previamente los notebooks para probar la API y la aplicación.

---

## Ejecutar la API 🚀

Con el entorno virtual activo:

```bash
uvicorn api.main:app --reload
```

La API estará disponible en:

```text
http://localhost:8000
```

La documentación interactiva de Swagger se encuentra en:

```text
http://localhost:8000/docs
```

---

## Ejecutar Streamlit 🖥️

Con la API ejecutándose en otra terminal:

```bash
streamlit run demo_app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

---

## Ejemplo de predicción 🔌

```json
{
  "Administrative": 2,
  "Administrative_Duration": 45.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 25,
  "ProductRelated_Duration": 620.5,
  "BounceRates": 0.01,
  "ExitRates": 0.02,
  "PageValues": 35.0,
  "SpecialDay": 0.0,
  "Month": "Nov",
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

Respuesta esperada:

```json
{
  "purchase_probability": 0.83,
  "recommended_action": "cross_selling"
}
```

---

## Variables de entrada 📋

La API recibe las 17 variables originales utilizadas como features, excluyendo `Revenue`, que es la variable objetivo.

| Variable | Tipo |
|---|---|
| Administrative | int |
| Administrative_Duration | float |
| Informational | int |
| Informational_Duration | float |
| ProductRelated | int |
| ProductRelated_Duration | float |
| BounceRates | float |
| ExitRates | float |
| PageValues | float |
| SpecialDay | float |
| Month | string |
| OperatingSystems | int |
| Browser | int |
| Region | int |
| TrafficType | int |
| VisitorType | string |
| Weekend | boolean |

---

## Principales Insights de Negocio 💡

### 1. Conversión global

Luego del proceso de limpieza:

- **12.205 sesiones**
- **1.908 compras**
- **15,63% de conversión**

### 2. Usuarios nuevos vs. recurrentes

Los usuarios nuevos presentan una tasa de conversión superior a la de los usuarios recurrentes.

Sin embargo, los recurrentes representan la mayor parte del tráfico.

Esto genera una oportunidad clara para trabajar estrategias de retención y conversión sobre dicho segmento.

### 3. Temporalidad

El volumen de compras presenta un crecimiento hacia el último trimestre del año, con los mayores niveles observados durante noviembre y diciembre.

Esto puede utilizarse para anticipar:

- Campañas comerciales.
- Capacidad operativa.
- Infraestructura.
- Stock.
- Presupuesto de marketing.

### 4. Variables relacionadas con intención

Entre las variables con mayor relación con `Revenue` aparecen:

- `PageValues`
- `ExitRates`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `VisitorType`
- `Month`

Estas variables fueron especialmente relevantes para comprender y modelar el comportamiento de las sesiones.

---

## Conclusiones 🎯

**Metric Mindset** demuestra cómo un problema de negocio puede abordarse mediante un flujo completo de Data Science:

```text
Datos
  ↓
EDA
  ↓
ETL
  ↓
Feature Engineering
  ↓
Modelado
  ↓
Optimización
  ↓
Calibración
  ↓
Scoring
  ↓
Recomendación
  ↓
Acción de negocio
```

El resultado final no es solamente un modelo predictivo.

Es una solución integrada que permite:

- Comprender el comportamiento de los usuarios.
- Estimar intención de compra.
- Priorizar sesiones.
- Asignar acciones.
- Exponer predicciones mediante una API.
- Interactuar con el modelo mediante Streamlit.
- Monitorear experimentos.
- Automatizar pruebas.

De esta manera, **Metric Mindset** busca convertir la analítica predictiva en una herramienta concreta para mejorar la toma de decisiones en e-commerce.

---

## Próximos pasos 🔮

Como evolución futura de la solución se plantean:

- Incorporar datos de usuarios identificados para mejorar el tratamiento del cold start.
- Incorporar feedback de las acciones recomendadas.
- Optimizar dinámicamente los umbrales de intervención.
- A/B testing de estrategias de retención y cross-selling.
- Incorporar nuevas variables de comportamiento.
- Monitorear drift del modelo.
- Automatizar el reentrenamiento.
- Integrar métricas de negocio como ROI de las intervenciones.
- Llevar el sistema a un entorno cloud productivo.

---

## Equipo — Metric Mindset 👥

| Integrante | Rol |
|---|---|
| Camila Conde | Líder de Equipo / Product Owner & Scrum Master |
| Luis | Data Engineer & Data Analytics |
| Juan Manuel | Data Scientist |
| Lucas Bottino | Data Scientist |

---

## Fuente de datos 📚

**Dataset:** Online Shoppers Intention Dataset

**Fuente original:** UCI Machine Learning Repository.

El dataset contiene información sobre sesiones de navegación de usuarios de un sitio de comercio electrónico y una variable objetivo `Revenue` que indica si la sesión terminó en una compra.

---

## Licencia / Uso académico 📄

Proyecto desarrollado con fines académicos como parte del proyecto final de Ciencia de Datos de Henry.

---

## Metric Mindset 🚀

**De los datos a la decisión.**  
**De la intención a la conversión.**
