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

EDA de referencia del proyecto sobre `online_shoppers_intention.csv`, cargado con `src.data_loader.cargar_datos()`: **12.330 sesiones × 18 variables** (10 numéricas, 6 categóricas de código, 2 booleanas), variable objetivo `Revenue` (si la sesión terminó en compra). Recorre, en este orden:

1. **Configuración y carga**: `cargar_datos()`, vista previa (`df.head(10)`).
2. **Estructura y calidad de datos**: `df.info()` / `df.describe()`, tabla de valores faltantes por columna y detección de duplicados exactos con `df.duplicated()`.
3. **Variable objetivo `Revenue`**: conteo y proporción de clases.
4. **Variables numéricas y categóricas**: se separan 9 numéricas (`Administrative` … `PageValues`) y 8 tratadas como categóricas para graficar (`Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`, `SpecialDay`) — histogramas para las numéricas, countplots para las categóricas. `SpecialDay` se visualiza acá como categórica porque solo toma 6 valores discretos (`0, 0.2, 0.4, 0.6, 0.8, 1.0`); en `src/preprocessing.py` (`COLUMNAS_NUMERICAS`) se trata como numérica continua para outliers/escalado — es una elección de representación distinta según el propósito (visual vs. pipeline), documentada explícitamente en el propio notebook, no una inconsistencia.
5. **Análisis univariado y outliers** (boxplots + regla IQR 1.5×RIC) sobre las 9 numéricas, calculado sobre las 12.330 filas crudas.
6. **Análisis bivariado**: cada numérica vs. `Revenue` (boxplot) y cada categórica vs. `Revenue` (countplot con `hue`).
7. **Correlaciones**: heatmap de correlación numérica, y correlación de **todas** las variables (categóricas codificadas con `pd.get_dummies`, solo para este análisis — no se usa para modelar) contra `Revenue`.
8. **Conclusiones**.

**Cifras clave:**

| Chequeo | Resultado |
|---|---|
| Filas / columnas | 12.330 × 18 |
| Valores faltantes | 0 en las 18 columnas — no requiere imputación |
| Duplicados exactos (`df.duplicated()`, cuenta repeticiones más allá de la 1ª aparición) | 125 filas (1,01%) |
| Balance de `Revenue` | 10.422 `False` (84,53%) vs. 1.908 `True` (15,47%) |

Outliers por columna (regla IQR, sobre las 12.330 filas crudas): `PageValues` 22,14% (2.730), `Informational` 21,34% (2.631), `Informational_Duration` 19,51% (2.405), `BounceRates` 12,58% (1.551), `Administrative_Duration` 9,51% (1.172), `ExitRates` 8,91% (1.099), `ProductRelated` 8,00% (987), `ProductRelated_Duration` 7,79% (961), `Administrative` 3,28% (404). No se eliminan: el EDA concluye que reflejan comportamiento real de navegación (sesiones muy cortas o muy largas), no errores de captura.

Top variables correlacionadas con `Revenue` (incluye categóricas codificadas): `PageValues` **+0,49**, `ExitRates` **−0,21**, `ProductRelated` **+0,16**, `Month_Nov` **+0,15**, `ProductRelated_Duration` **+0,15**, `BounceRates` **−0,15**, `Administrative` **+0,14**, `VisitorType_Returning_Visitor` **−0,10** — la base para elegir features en las etapas siguientes.

Los duplicados no se eliminan en este notebook (el dataset no tiene identificador único de sesión/usuario, podrían ser sesiones distintas con comportamiento similar) — la decisión de eliminarlos se toma recién en `ETL.ipynb`.

### `notebooks/ETL.ipynb` — Limpieza y persistencia del dataset

Toma el mismo dataset crudo que `EDA.ipynb` y actúa sobre las decisiones que el EDA dejó pendientes (duplicados, outliers), apoyándose en `src/preprocessing.py`. No imputa (no hay nulos) ni genera datos sintéticos.

**¿Qué devuelve y para qué se usa?** Un único dataset limpio (`data/processed/online_shoppers_intention_procesado.csv`), usado **para ambos destinos**, no uno u otro:

- **`feature_enginering.ipynb` (modelado)**: lo recarga con `cargar_datos_procesados()` como punto de partida.
- **Dashboard de Power BI**: se conecta directamente a ese mismo CSV — ya viene tipado, validado, sin duplicados y con columnas `Weekend_label`/`Revenue_label` legibles para negocio.

Pasos del notebook, con resultado real de cada uno:

1. **Validación de esquema** (`validar_esquema`): invariante forzada (no un supuesto) — si el CSV fuente cambiara y apareciera un nulo o una columna inesperada, esta celda falla explícitamente en vez de propagar el problema. Confirma las 18 columnas esperadas y 0 nulos.
2. **Corrección de tipos** (`corregir_tipos`): `Weekend`/`Revenue` → `bool` (defensivo, para cuando se recargue el CSV, que no preserva booleanos); `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` → `category`.
3. **Duplicados** (`detectar_duplicados` → `resumen_duplicados` → `eliminar_duplicados`): marca **201 filas** (1,63%) como parte de algún grupo duplicado — esta cifra usa `keep=False` (marca *todas* las filas del grupo, incluida la primera aparición), por eso no coincide con las 125 de `EDA.ipynb` (que cuenta solo repeticiones *además* de la primera, con `keep="first"`). Ambas miden el mismo fenómeno con convenciones distintas: las filas **efectivamente eliminadas** son 125 en los dos casos. Resultado: **12.330 → 12.205 filas**.
4. **Outliers** (`detectar_outliers_iqr` → `resumen_outliers`): mismo criterio IQR que el EDA, recalculado sobre las 12.205 filas ya sin duplicados (por eso los porcentajes difieren levemente de los del EDA — los cuartiles se recalculan sin esas 125 filas) y ahora sobre 10 columnas numéricas (agrega `SpecialDay`, que en el EDA se había graficado como categórica): `PageValues` 22,37% (2.730), `Informational` 21,56% (2.631), `Informational_Duration` 19,71% (2.405), `BounceRates` 11,70% (1.428), `ExitRates` 10,86% (1.325), `SpecialDay` 10,23% (1.249), `Administrative_Duration` 9,41% (1.149), `ProductRelated` 8,25% (1.007), `ProductRelated_Duration` 7,79% (951), `Administrative` 3,31% (404). Se marcan con columnas `outlier_<columna>`, **no se capan**.
5. **Chequeos de consistencia** (`validar_consistencia`): invariantes duras (sin valores negativos; conteo de páginas = 0 ⇒ duración = 0) — **0 violaciones**. Hallazgos informativos (conteo > 0 con duración ≈ 0, plausible en visitas muy breves): 135 filas en `Administrative`, 226 en `Informational`, 592 en `ProductRelated`. `BounceRates`/`ExitRates`/`SpecialDay` siempre dentro de `[0, 1]`.
6. **Verificación de balance de clases**: 84,37% `False` / 15,63% `True` (levemente distinto al 84,53/15,47 del EDA, ya sin los duplicados eliminados) — se deja constancia, no se corrige acá.
7. **Etiquetas legibles** (`agregar_etiquetas_legibles`): agrega `Weekend_label` ("Sí"/"No") y `Revenue_label` ("Compra"/"No compra") sin tocar las columnas originales, pensadas para Power BI. `OperatingSystems`/`Browser`/`Region`/`TrafficType` quedan como códigos numéricos: el UCI no publica el diccionario para traducirlos.
8. **Resultado final**: **12.205 filas × 30 columnas** (18 originales + 10 `outlier_*` + 2 `_label`).
9. **Persistencia** (`guardar_datos_procesados`): CSV de **1.973,2 KB** en `data/processed/online_shoppers_intention_procesado.csv`.

**Principales decisiones:**

- Duplicados exactos: se marcan (auditoría, 201 filas/1,63% con `keep=False`) y luego se eliminan de verdad (125 filas, conservando la primera aparición de cada grupo) — modelo y dashboard deben partir del mismo dataset para que las métricas de tráfico/conversión sean coherentes entre sí.
- Outliers (IQR): se marcan pero no se capan ni recortan — el EDA ya concluyó que probablemente reflejan comportamiento real de navegación; capar depende del modelo, así que esa decisión se deja para `feature_enginering.ipynb`.
- CSV (no Parquet, el proyecto no tiene `pyarrow` como dependencia), pensado para conectar directo a Power BI.
- Split train/test y encoding/escalado se hacen en `feature_enginering.ipynb`, no acá, para evitar fuga de datos.
- No se agregan filas ni datos sintéticos en ninguna etapa.

### `notebooks/feature_enginering.ipynb` — Feature Engineering

Parte de las **12.205 filas × 30 columnas** que deja `ETL.ipynb` (`cargar_datos_procesados()`) y lo deja listo para entrenar, usando `src/features.py`. No entrena ningún modelo — persiste artefactos para `modeling_mvp.ipynb`.

1. **Outliers**: descarta las 10 columnas `outlier_*` y las etiquetas `Weekend_label`/`Revenue_label` (no aportan como feature) — vuelve a 18 columnas. Los valores numéricos originales **no se capan**: incluso con >20% de "outliers" en `PageValues`/`Informational`, EDA y ETL ya concluyeron que reflejan comportamiento real.
2. **Features derivadas** (`crear_features_derivadas`), cinco en total:
   - `duracion_total` (suma de las 3 duraciones; media 1.323,5s, máx. 69.921,6s)
   - `paginas_totales` (suma de los 3 conteos de páginas; media 34,9, máx. 746)
   - `duracion_promedio_pagina` (proxy de interés/lectura por página, no solo volumen; media 38,2s)
   - `proporcion_paginas_producto` (qué fracción de la sesión fue a páginas de producto; media 0,90, mediana 0,96)
   - `es_visitante_recurrente` (booleano derivado de `VisitorType == "Returning_Visitor"`)
3. **Split train/test estratificado** (`dividir_train_test`, `test_size=0.2`, `random_state=42`, estratificado por `Revenue`): **Train 9.764 filas** (balance 15,63%), **Test 2.441 filas** (balance 15,65%) — se separa antes de ajustar cualquier encoder/scaler, para que el test no influya en los parámetros aprendidos.
4. **Encoding + escalado** (`construir_column_transformer`): `StandardScaler` sobre 16 columnas numéricas (10 originales + 4 derivadas numéricas + `Weekend` + `es_visitante_recurrente`) y `OneHotEncoder(handle_unknown="ignore")` sobre 6 categóricas (`Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`), ajustados **solo con train**. `handle_unknown="ignore"` para que una categoría nueva en producción no rompa la inferencia. Resultado: `X_train_proc` **(9.764, 78)**, `X_test_proc` **(2.441, 78)**.
5. **Persistencia** (`guardar_artefactos_modelado`): `data/models/train_test_split.joblib` (`X_train`/`X_test`/`y_train`/`y_test`) y `data/models/preprocessor.joblib` (el `ColumnTransformer` ya ajustado).

### `notebooks/modeling_mvp.ipynb` — Modelado y MVP (Sprint 1)

Carga `train_test_split.joblib`/`preprocessor.joblib` desde `data/models/`, entrena y compara 5 modelos usando `src/metrics.py` (`evaluar_modelo`, `comparar_modelos`) y `src/recommender.py` (`asignar_accion`, `resumen_acciones`). Métricas siempre sobre la clase positiva (`Revenue=1`), priorizando precision/recall/F1/ROC-AUC/PR-AUC por sobre accuracy dado el desbalance ~85/15.

**Comparativa de modelos (test set, 2.441 sesiones):**

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** (`n_estimators=300, class_weight="balanced"`) | 0,658 | 0,709 | 0,683 | **0,924** | 0,723 |
| XGBoost (`n_estimators=300, scale_pos_weight≈5,40`) | 0,646 | 0,678 | 0,662 | 0,922 | 0,717 |
| Regresión Logística (`class_weight="balanced"`) | 0,503 | 0,806 | 0,620 | 0,911 | 0,664 |
| Regresión Logística + SMOTE (solo sobre train: 9.764→16.476 filas, balance 50/50) | 0,502 | 0,791 | 0,614 | 0,910 | 0,665 |
| Regresión Logística (baseline, sin balancear) | 0,766 | 0,419 | 0,541 | 0,901 | 0,654 |

1. **Baseline**: Regresión Logística sin balancear — precision alta pero recall bajo (0,419) en la clase positiva: entrenada sobre train desbalanceado, tiende a predecir la clase mayoritaria.
2. **Manejo del desbalance**: `class_weight="balanced"` (penaliza más los errores sobre la clase minoritaria, sin tocar los datos) y SMOTE (`imbalanced-learn`, sobresamplea la clase minoritaria por interpolación, aplicado **solo sobre train**) — ambos mejoran mucho el recall frente al baseline, a costa de precision.
3. **Modelos ensemble**: Random Forest y XGBoost (con `scale_pos_weight` = razón negativos/positivos en train, el equivalente de `class_weight="balanced"` para XGBoost) superan a las variantes de Regresión Logística en las 5 métricas.
4. **Selección** (`comparar_modelos`, ordenado por `roc_auc`): **Random Forest** gana por ROC-AUC (0,924) y PR-AUC (0,723, el criterio secundario, más informativo que ROC-AUC con clase minoritaria) → persistido en `data/models/modelo_final.joblib`.
5. **Motor de recomendación** (`asignar_accion` sobre las probabilidades del modelo ganador en test, umbrales 0,7/0,3; auditado con `resumen_acciones`):

   | Acción | Sesiones | % del total | Tasa real de compra |
   |---|---|---|---|
   | `retencion` (prob. < 30%) | 1.794 | 73,49% | 2,68% |
   | `sin_accion` | 422 | 17,29% | 39,34% |
   | `cross_selling` (prob. > 70%) | 225 | 9,22% | 74,67% |

   `cross_selling` concentra la mayor tasa real de compra y `retencion` la menor — coherencia de negocio confirmada antes de dar el modelo por válido.
6. **Limitación conocida** (documentada en las conclusiones del propio notebook): el modelo se elige por ROC-AUC/PR-AUC (métricas agregadas sobre todos los umbrales posibles), no por el desempeño específico en los umbrales de negocio 0,7/0,3 que usa el recomendador. Es exactamente el gap que cierra Precision@K/Recall@K, planificado para Sprint 2.

