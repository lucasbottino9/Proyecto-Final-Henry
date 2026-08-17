import numpy as np
import shap


def limpiar_nombre_feature(nombre: str) -> str:
    """Elimina prefijos técnicos generados por ColumnTransformer."""
    return (
        nombre
        .replace("num__", "")
        .replace("cat__", "")
    )


def explicar_sesion(
    modelo_calibrado,
    X_sesion,
    feature_names,
    top_n: int = 3,
) -> list[dict]:
    """
    Obtiene las principales contribuciones SHAP de una sesión.

    El modelo productivo es un CalibratedClassifierCV que contiene
    varios LightGBM internos. Para obtener una explicación más estable,
    se calculan valores SHAP para cada estimador y se promedian.

    Importante:
    SHAP explica los estimadores LightGBM base. La probabilidad final
    mostrada por la API corresponde al modelo calibrado.
    """

    contribuciones_folds = []

    for calibrado in modelo_calibrado.calibrated_classifiers_:
        estimador = calibrado.estimator

        explainer = shap.TreeExplainer(estimador)
        explicacion = explainer(X_sesion)

        valores = np.asarray(explicacion.values)

        # Una única sesión
        if valores.ndim == 2:
            valores = valores[0]

        contribuciones_folds.append(valores)

    # Promedio de contribuciones entre los folds
    shap_promedio = np.mean(contribuciones_folds, axis=0)

    nombres_limpios = [
        limpiar_nombre_feature(nombre)
        for nombre in feature_names
    ]

    indices = np.argsort(np.abs(shap_promedio))[::-1][:top_n]

    resultados = []

    for idx in indices:
        valor = float(shap_promedio[idx])

        resultados.append(
            {
                "feature": nombres_limpios[idx],
                "shap_value": round(valor, 4),
                "direction": "aumenta" if valor > 0 else "reduce",
            }
        )

    return resultados