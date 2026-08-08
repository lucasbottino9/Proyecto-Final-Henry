import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def procesar_etl(df: pd.DataFrame, eliminar_duplicados: bool = True) -> pd.DataFrame:
    """
    Realiza la limpieza inicial de datos: tratamiento de nulos y duplicados.
    
    Args:
        df (pd.DataFrame): Dataframe original.
        eliminar_duplicados (bool): Si es True, elimina filas duplicadas exactas.
        
    Returns:
        pd.DataFrame: Dataframe limpio.
    """
    df_clean = df.copy()
    
    # 1. Tratamiento de duplicados
    if eliminar_duplicados:
        filas_antes = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        filas_despues = len(df_clean)
        print(f"[ETL] Registros duplicados eliminados: {filas_antes - filas_despues}")
    
    # 2. Verificación e Imputación de Nulos (por seguridad)
    if df_clean.isnull().sum().sum() > 0:
        # Para numéricas se imputa mediana, para categóricas la moda
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        print("[ETL] Se completó la imputación de valores faltantes.")
    else:
        print("[ETL] Sin valores nulos detectados.")
        
    # 3. Normalización del Target (Revenue) a binario entero (1 / 0)
    if 'Revenue' in df_clean.columns and df_clean['Revenue'].dtype == bool:
        df_clean['Revenue'] = df_clean['Revenue'].astype(int)
        
    return df_clean


def obtener_pipeline_preprocesamiento():
    numerical_columns = [
        "Administrative", "Administrative_Duration", 
        "Informational", "Informational_Duration", 
        "ProductRelated", "ProductRelated_Duration", 
        "BounceRates", "ExitRates", "PageValues"
    ]
    
    categorical_columns = [
        "Month", "OperatingSystems", "Browser", 
        "Region", "TrafficType", "VisitorType", "Weekend", "SpecialDay"
    ]
    
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Compatibilidad para versiones viejas y nuevas de Scikit-Learn
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipeline = Pipeline(steps=[
        ('onehot', ohe)
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_columns),
        ('cat', cat_pipeline, categorical_columns)
    ])
    
    return preprocessor, numerical_columns, categorical_columns


def preparar_datos_modelo(df: pd.DataFrame, target_col: str = 'Revenue'):
    """
    Aplica el ETL y el Pipeline de transformación devolviendo matriz X e y.
    
    Returns:
        X_processed (np.ndarray / pd.DataFrame), y (pd.Series)
    """
    # Ejecutar ETL
    df_clean = procesar_etl(df)
    
    # Separar X e y
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Obtener preprocesador
    preprocessor, num_cols, cat_cols = obtener_pipeline_preprocesamiento()
    
    # Transformar características
    X_processed = preprocessor.fit_transform(X)
    
    print(f"[PREPROCESSING] Matriz procesada lista. Forma: {X_processed.shape}")
    return X_processed, y, preprocessor