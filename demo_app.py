import streamlit as st
import random
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Metric Mindset - E-commerce Conversion Demo",
    page_icon="🚀",
    layout="wide"
)

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

# Variables Claves del Dataset (Mocks)
st.sidebar.subheader("Indicadores de Navegación")
page_values = st.sidebar.slider("Page Values (Valor de Página)", 0.0, 300.0, 20.0, help="Valor promedio de la página antes del abandono.")
exit_rates = st.sidebar.slider("Exit Rate (Tasa de Salida)", 0.0, 1.0, 0.10, help="Porcentaje de sesiones que terminaron en esta página.")
bounce_rates = st.sidebar.slider("Bounce Rate (Tasa de Rebote)", 0.0, 1.0, 0.05, help="Porcentaje de sesiones de una sola página.")

st.sidebar.subheader("Información de Usuario/Tiempo")
month = st.sidebar.selectbox("Mes", ["Feb", "Mar", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"])
visitor_type = st.sidebar.radio("Tipo de Visitante", ["Returning_Visitor", "New_Visitor"])

# --- LÓGICA DE LA DEMO (MOCK DATA & REGLAS) ---

# Función Mock para simular la API (Mientras Luis termina FastAPI)
def simular_prediccion(values, exits, bounces):
    """Lógica simplista para generar una probabilidad basada en la telemetría."""
    # En un e-commerce real: PageValues alto -> Compra; ExitRate alto -> Abandono.
    score_base = (values / 150.0) - exits - bounces
    
    # Asegurar que esté entre 0 y 1, con ruido aleatorio
    score_final = max(0.0, min(1.0, 0.5 + score_base * 0.4 + random.uniform(-0.1, 0.1)))
    return round(score_final, 3)

# Función para traducir el score a Incentivos (Tus Reglas de Negocio)
def obtener_incentivo(probabilidad):
    if probabilidad < 0.30:
        return {
            "estado": "Baja Intención",
            "emoji": "🔴",
            "mensaje": "Usuario sin intención de compra detectable.",
            "accion": "Sin acción (Ahorro de presupuesto).",
            "color": "error"
        }
    elif 0.30 <= probabilidad <= 0.70:
        return {
            "estado": "Usuario Indeciso / En Riesgo",
            "emoji": "🟠",
            "mensaje": "**TRIGGER ACTIVO**: Momento crítico de intervención.",
            "accion": "🎁 **¡MOSTRAR CUPÓN 15% OFF O ENVÍO GRATIS!**",
            "color": "warning"
        }
    else: # > 0.70
        return {
            "estado": "Alta Intención",
            "emoji": "🟢",
            "mensaje": "Usuario con alta propensión a la compra.",
            "accion": "🔗 **¡Mostrar Recomendaciones de Cross-Selling / Upselling!** (Sin descuento).",
            "color": "success"
        }

# --- PANEL PRINCIPAL DE VISUALIZACIÓN ---

# Contenedores para la visualización
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
        with st.spinner("Procesando telemetría y consultando modelo..."):
            time.sleep(1.2) # Simular latencia de red
            
            # 1. Llamada Simulada al Modelo
            prob_final = simular_prediccion(page_values, exit_rates, bounce_rates)
            
            # 2. Llamada al Motor de Prescripción
            info_incentivo = obtener_incentivo(prob_final)
            
            # 3. Visualización
            st.metric(label="Puntaje de Propensión a Compra (`Revenue`)", value=f"{prob_final*100:.1f}%")
            st.progress(prob_final)
            
            # Panel de Resultado con color
            st.warning(f"### {info_incentivo['emoji']} Estado: {info_incentivo['estado']}")
            st.markdown(info_incentivo['mensaje'])
            
            st.divider()
            
            st.success(f"### 🎯 Acción Recomendada: {info_incentivo['accion']}")
            
    else:
        st.info("Ingresá los datos de telemetría en la barra lateral y hacé clic en 'Predecir y Prescribir' para simular la sesión.")

# --- FOOTER ---
st.markdown("---")
st.caption("Metric Mindset - Demo Sprint 2 | Prototipo Visual (Mock Data / Inferencia Local)")