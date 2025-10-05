import streamlit as st
import joblib
import pandas as pd
import time

# ---- Configuración de la Página ----
st.set_page_config(
    page_title="Detector de Exoplanetas",
    page_icon="🪐",
    layout="wide" # Usamos un layout más ancho
)

# ---- Cargar Artefactos del Modelo ----
try:
    model = joblib.load('modelo_exoplanetas.joblib')
    feature_importances = joblib.load('feature_importances.joblib')
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo. Asegúrate de ejecutar 'entrenar_modelo.py' primero.")
    st.stop()

# ---- Layout de la App con Columnas ----
col1, col2 = st.columns([1, 1]) # Dividir la pantalla en dos columnas

with col1:
    # ---- Interfaz de Usuario ----
    st.title('🪐 Detector de Exoplanetas')
    st.markdown("Introduce los datos de la observación para predecir si podría ser un exoplaneta.")

    # ---- Explicaciones con st.expander ----
    with st.expander("💡 ¿Qué significan estos parámetros?"):
        st.markdown("""
        - **Período Orbital (días):** Es el tiempo que tarda un planeta en dar una vuelta completa a su estrella. Períodos cortos (pocos días) son más fáciles de detectar, ya que vemos tránsitos repetidos con más frecuencia.
        - **Duración del Tránsito (horas):** El tiempo que el objeto tarda en pasar por delante de su estrella. Depende de la velocidad orbital y el tamaño de la estrella.
        - **Profundidad del Tránsito (ppm):** Cuánto disminuye el brillo de la estrella (medido en partes por millón). **Una mayor profundidad sugiere un objeto más grande**, lo que aumenta la probabilidad de que sea un planeta.
        - **Magnitud de la Estrella:** Es una medida del brillo de la estrella. Valores más altos indican estrellas más tenues, lo que puede dificultar la detección.
        """)

    st.header('Ingresa los datos:')
    pl_orbper = st.slider('Período Orbital (días)', 0.1, 100.0, 10.4, 0.1)
    pl_trandurh = st.slider('Duración del Tránsito (horas)', 0.1, 20.0, 3.2, 0.1)
    pl_trandep = st.slider('Profundidad del Tránsito (ppm)', 100.0, 50000.0, 4500.0, 100.0)
    st_tmag = st.slider('Magnitud de la Estrella', 4.0, 20.0, 11.5, 0.1)

    # ---- Predicción ----
    if st.button('✨ Predecir ahora', type="primary"):
        input_data = pd.DataFrame({
            'pl_orbper': [pl_orbper],
            'pl_trandurh': [pl_trandurh],
            'pl_trandep': [pl_trandep],
            'st_tmag': [st_tmag]
        })

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        with st.spinner('Analizando los cielos...'):
            time.sleep(2) # Simulación de carga

        st.subheader('Resultado del Análisis:')
        if prediction[0] == 1:
            st.success('**¡Posible Exoplaneta Detectado!**')
            st.markdown(f"""
            El modelo tiene una confianza del **{prediction_proba[0][1]*100:.2f}%**.
            Esto significa que las características que ingresaste son **muy similares** a las de planetas ya confirmados en nuestro set de datos.
            """)
        else:
            st.warning('**Probablemente no es un Exoplaneta.**')
            st.markdown(f"""
            El modelo tiene una confianza de solo el **{prediction_proba[0][1]*100:.2f}%** de que sea un exoplaneta.
            Esto indica que los valores **no coinciden** con los patrones típicos de un planeta confirmado.
            """)

with col2:
    # ---- Explicación del Modelo ----
    st.header("🧠 ¿Cómo 'piensa' el modelo?")
    st.markdown("""
    Este modelo es un *Random Forest*, que funciona como un comité de expertos. Analizó miles de ejemplos de planetas confirmados y falsas alarmas para aprender los patrones.

    Abajo puedes ver qué característica le parece más importante a la hora de tomar una decisión.
    """)
    st.subheader('Importancia de cada Característica:')
    
    # Renombramos las características para que sean más legibles en el gráfico
    feature_importances_display = feature_importances.copy()
    feature_importances_display['feature'] = feature_importances_display['feature'].replace({
        'pl_trandep': 'Profundidad Tránsito',
        'pl_orbper': 'Período Orbital',
        'pl_trandurh': 'Duración Tránsito',
        'st_tmag': 'Magnitud Estrella'
    })
    
    st.bar_chart(feature_importances_display.set_index('feature')['importance'])

    st.info("""
    **Interpretación del gráfico:**
    Una barra más alta significa que el modelo presta **mucha más atención** a esa característica. Como puedes ver, la **profundidad del tránsito** es, por mucho, el indicador más decisivo para este modelo. Un tránsito profundo es una señal muy fuerte.
    """, icon="🧐")