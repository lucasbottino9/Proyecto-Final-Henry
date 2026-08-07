# Proyecto-Final-Henry
Proyecto final de Ciencia de Datos de Henry. Desarrollo de un sistema de recomendación inteligente para comercio electrónico que combina la predicción de intención de compra con recomendaciones de productos personalizadas, utilizando Machine Learning, FastAPI, Streamlit y las mejores prácticas de MLOps.

# Avances del Proyecto - Modulo de Prediccion y Recomendacion 

### Analisis Exploratorio (EDA) y Limpieza
**Calidad de Datos:** Se eliminaron 125 registros duplicados para evitar sobreajuste. El data no presenta valores nulos.

### Preprocesamiento
* Se aplico ('pd.get_dummies') para transformar las variables cualitativas ('Mes', 'Tipo de Visitante') y booleanos a formato numerico.

### Modelado Predictivo 
* Se entreno un modelo estandar de ('RandomForestClassifier') con una division de datos de 80/20.
* El rendimiento se evaluo mediante la metrica de **Exactitud** para un entendimiento directo y claro de los resultados.

### Motor de Recomendacion
* Se diseño una funcion logica ('if/else') que simula la accion comercial en tiempo real: si el modelo predice que el usuario abandonara la pagina ('0'), el sistema de dispara automaticamente un incentivo (descuento del 10% o envio gratis) segun si el visitante es nuevo o recurrente.