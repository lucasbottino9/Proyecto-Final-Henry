import os

import requests
import streamlit as st

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Metric Mindset - E-commerce Conversion Demo",
    page_icon="🚀",
    layout="wide"
)

def _resolver_api_url() -> str:
    """Streamlit Community Cloud inyecta configuración vía `st.secrets`
    (`.streamlit/secrets.toml`), no variables de entorno tradicionales; Render
    y el uso local sí usan `os.environ`. Se prueban ambas fuentes."""
    try:
        return st.secrets["API_URL"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_URL", "http://localhost:8000")


API_URL = _resolver_api_url()

# --- CABECERA ---
st.title("🚀 Demo Interactiva: Predicción de Conversión en Tiempo Real")
st.markdown("""
Esta demo simula la telemetría de navegación de un usuario en un e-commerce.
El modelo asigna una probabilidad de compra y el **Motor de Prescripción** gatilla incentivos dinámicos instantáneos.
---
""")

# --- BARRA LATERAL (ENTRADA DE DATOS DE SESIÓN) ---
st.sidebar.header("📊 Telemetría de Sesión Actual")
st.sidebar.markdown("Simulá el comportamiento del usuario:")

# Variables Claves del Dataset
st.sidebar.subheader("Indicadores de Navegación")
page_values = st.sidebar.slider("Page Values (Valor de Página)", 0.0, 300.0, 20.0, help="Valor promedio de la página antes del abandono.")
exit_rates = st.sidebar.slider("Exit Rate (Tasa de Salida)", 0.0, 1.0, 0.10, help="Porcentaje de sesiones que terminaron en esta página.")
bounce_rates = st.sidebar.slider("Bounce Rate (Tasa de Rebote)", 0.0, 1.0, 0.05, help="Porcentaje de sesiones de una sola página.")

st.sidebar.subheader("Información de Usuario/Tiempo")
month = st.sidebar.selectbox("Mes", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
visitor_type = st.sidebar.radio("Tipo de Visitante", ["Returning_Visitor", "New_Visitor", "Other"])
weekend = st.sidebar.checkbox("Sesión en fin de semana")

# El modelo real necesita estas variables además de las de arriba (mismo esquema
# que `api/schemas.py:SesionInput`); se dejan colapsadas con valores típicos por
# defecto para no saturar la demo rápida.
with st.sidebar.expander("⚙️ Más variables de la sesión"):
    administrative = st.number_input("Páginas administrativas", min_value=0, value=0)
    administrative_duration = st.number_input("Duración administrativa (s)", min_value=0.0, value=0.0)
    informational = st.number_input("Páginas informativas", min_value=0, value=0)
    informational_duration = st.number_input("Duración informativa (s)", min_value=0.0, value=0.0)
    product_related = st.number_input("Páginas de producto", min_value=0, value=20)
    product_related_duration = st.number_input("Duración en producto (s)", min_value=0.0, value=500.0)
    special_day = st.slider("Cercanía a fecha especial", 0.0, 1.0, 0.0, step=0.2)
    st.caption("Códigos sin diccionario público (ver dataset UCI Online Shoppers Purchasing Intention)")
    operating_systems = st.number_input("Sistema operativo (código)", min_value=1, max_value=8, value=2)
    browser = st.number_input("Navegador (código)", min_value=1, max_value=13, value=2)
    region = st.number_input("Región (código)", min_value=1, max_value=9, value=1)
    traffic_type = st.number_input("Tipo de tráfico (código)", min_value=1, max_value=20, value=2)

st.sidebar.divider()
st.sidebar.caption(f"API: `{API_URL}`")
st.sidebar.caption("Si no responde: `uvicorn api.main:app --reload`")

# --- LÓGICA DE LA DEMO: llamado real a la API (api/main.py) ---


def predecir_via_api(payload: dict) -> dict:
    """POST a `/predict`. Lanza `requests.RequestException` (sin conexión) o
    `ValueError` (la API rechazó la sesión, ej. categoría no vista en entrenamiento)."""
    # timeout alto: el free tier de Render duerme tras ~15 min de inactividad y
    # tarda hasta ~60s en despertar en el primer pedido.
    respuesta = requests.post(f"{API_URL}/predict", json=payload, timeout=70)
    if respuesta.status_code != 200:
        detalle = respuesta.json().get("detail", respuesta.text)
        raise ValueError(str(detalle))
    return respuesta.json()


# Traduce la acción del motor de recomendación (`src/recommender.py`, vía la API)
# al panel visual de la demo. El umbral real (probabilidad > 0.7 / < 0.3) vive en
# el backend — acá solo se traduce a texto/color la etiqueta que ya decidió la API.
INFO_POR_ACCION = {
    "cross_selling": {
        "estado": "Alta Intención",
        "emoji": "🟢",
        "mensaje": "Sesión con alta propensión y señales fuertes de interés.",
        "accion": "🔗 Mostrar recomendaciones de Cross-Selling.",
    },

    "upselling": {
        "estado": "Alta Intención",
        "emoji": "🟢",
        "mensaje": "Sesión con alta probabilidad de compra.",
        "accion": "⬆️ Proponer una alternativa premium o de mayor valor.",
    },

    "retencion": {
        "estado": "Riesgo de Abandono",
        "emoji": "🟠",
        "mensaje": "Sesión con baja propensión y señales fuertes de abandono.",
        "accion": "🎁 Mostrar incentivo de retención: cupón, envío gratis u oferta.",
    },

    "retencion_suave": {
        "estado": "Baja Intención",
        "emoji": "🟡",
        "mensaje": "Sesión con baja propensión, pero sin señales críticas de abandono.",
        "accion": "💡 Aplicar una intervención suave de retención.",
    },

    "incentivo_suave": {
        "estado": "Intención Intermedia",
        "emoji": "🔵",
        "mensaje": "Sesión con intención intermedia y señales de interés en productos.",
        "accion": "✨ Mostrar un incentivo comercial moderado.",
    },

    "sin_accion": {
        "estado": "Sin Intervención",
        "emoji": "⚪",
        "mensaje": "No se detectan señales suficientes para justificar una acción.",
        "accion": "Sin acción automática.",
    },
}

# --- PANEL PRINCIPAL DE VISUALIZACIÓN ---

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuración de Sesión")
    st.markdown(f"**Mes:** `{month}`")
    st.markdown(f"**Visitante:** `{visitor_type}`")

    st.divider()
    st.markdown("📈 *Variables de Impacto:*")
    st.write(f"- Page Values: {page_values}")
    st.write(f"- Exit Rate: {exit_rates:.2f}")
    st.write(f"- Bounce Rate: {bounce_rates:.2f}")

    # Botón para correr la simulación
    predecir_btn = st.button("🚀 Predecir y Prescribir", type="primary", use_container_width=True)

with col2:
    st.subheader("Resultado del Motor Inteligente")

    if predecir_btn:
        with st.spinner("Procesando telemetría y consultando el modelo... (puede tardar ~1 min si la API estaba dormida)"):
            payload = {
                "Administrative": administrative,
                "Administrative_Duration": administrative_duration,
                "Informational": informational,
                "Informational_Duration": informational_duration,
                "ProductRelated": product_related,
                "ProductRelated_Duration": product_related_duration,
                "BounceRates": bounce_rates,
                "ExitRates": exit_rates,
                "PageValues": page_values,
                "SpecialDay": special_day,
                "Month": month,
                "OperatingSystems": int(operating_systems),
                "Browser": int(browser),
                "Region": int(region),
                "TrafficType": int(traffic_type),
                "VisitorType": visitor_type,
                "Weekend": weekend,
            }

            try:
                resultado = predecir_via_api(payload)
            except requests.RequestException as exc:
                st.error(
                    f"No se pudo conectar a la API en `{API_URL}`. "
                    f"¿Está corriendo `uvicorn api.main:app --reload`?\n\n{exc}"
                )
            except ValueError as exc:
                st.error(f"La API rechazó la sesión: {exc}")
            else:
                prob_final = resultado["purchase_probability"]
                accion_recomendada = resultado["recommended_action"]
                prioridad = resultado["priority"]
                motivo = resultado["reason"]
                top_features = resultado.get("top_features", [])
                
                info_incentivo = INFO_POR_ACCION[accion_recomendada]

                # 1. Score del modelo real (vía API)
                st.metric(label="Puntaje de Propensión a Compra (`Revenue`)", value=f"{prob_final * 100:.1f}%")
                st.progress(prob_final)

                # 2. Panel de resultado del motor de prescripción
                st.warning(
                    f"### {info_incentivo['emoji']} Estado: {info_incentivo['estado']}"
                )

                st.markdown(info_incentivo["mensaje"])
                st.markdown(f"**Prioridad de intervención:** `{prioridad.upper()}`")

                st.markdown("**Motivo de la prescripción:**")
                st.info(motivo)

                st.markdown("### 🧠 ¿Qué influyó en la predicción?")

                st.caption(
                    "Principales variables identificadas mediante SHAP "
                    "sobre los estimadores LightGBM."
                )

                for feature in top_features:
                    nombre = feature["feature"]
                    valor = feature["shap_value"]
                    direccion = feature["direction"]

                    if direccion == "aumenta":
                        icono = "⬆️"
                        efecto = "Aumenta la propensión"
                    else:
                        icono = "⬇️"
                        efecto = "Reduce la propensión"

                    st.markdown(
                    f"{icono} **{nombre}** — {efecto} "
                    f"(`SHAP: {valor:+.3f}`)"
                    )

                st.divider()

                st.success(
                    f"### 🎯 Acción Recomendada: {info_incentivo['accion']}"
                )          
    else:
        st.info("Ingresá los datos de telemetría en la barra lateral y hacé clic en 'Predecir y Prescribir' para simular la sesión.")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Metric Mindset - Demo Sprint 2 | Conectada a la API FastAPI en `{API_URL}` (`/predict`)")
