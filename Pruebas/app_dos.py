import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Para guardar y cargar el modelo si se desea persistencia
import altair as alt # Para gráficos interactivos
import time

# ---- Configuración de la Página ----
st.set_page_config(
    page_title="Detector de Exoplanetas por Tránsito",
    page_icon="🪐",
    layout="wide"
)

st.title('🪐 Detector de Exoplanetas por Método de Tránsito')
st.markdown("""
Esta aplicación permite cargar un dataset, entrenar un modelo de Machine Learning y luego utilizarlo para predecir si una observación **potencialmente** es un exoplaneta, todo en un solo lugar.
""")

# ---- Carga y Procesamiento de Datos (Parte del Entrenador) ----
@st.cache_data # Cachea los datos para no recargarlos y procesarlos cada vez
def load_and_train_model(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Archivo CSV cargado exitosamente.")
        except Exception as e:
            st.error(f"Error al cargar el archivo CSV: {e}")
            return None, None, None, None

        # ---- Selección de Características y Limpieza ----
        features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'st_tmag']
        target = 'tfopwg_disp'

        # Asegúrate de que todas las columnas necesarias existan
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            st.error(f"El archivo CSV cargado no contiene todas las columnas necesarias. Faltan: {', '.join(missing_cols)}")
            st.warning("Asegúrate de que tu CSV contenga las columnas: 'pl_orbper', 'pl_trandurh', 'pl_trandep', 'st_tmag', 'tfopwg_disp'.")
            return None, None, None, None

        df_model = df[features + [target]].copy()
        initial_rows = len(df_model)
        df_model.dropna(inplace=True)
        dropped_rows = initial_rows - len(df_model)
        if dropped_rows > 0:
            st.info(f"Se eliminaron {dropped_rows} filas con valores faltantes para el entrenamiento.")

        if df_model.empty:
            st.error("Después de limpiar, el DataFrame está vacío. No se puede entrenar el modelo.")
            return None, None, None, None

        # ---- Creación de la Variable Objetivo ----
        df_model['is_exoplanet'] = df_model[target].apply(lambda x: 1 if x == 'CP' else 0)

        # Verificar si hay suficientes etiquetas para entrenar
        if df_model['is_exoplanet'].nunique() < 2:
            st.warning("Solo se encontraron una categoría en la columna 'tfopwg_disp' después de la limpieza. El modelo podría no entrenarse correctamente.")
            return None, None, None, None

        X = df_model[features]
        y = df_model['is_exoplanet']

        # ---- División de Datos en Entrenamiento y Prueba ----
        # Asegurarse de que haya al menos dos muestras para dividir y estratificar
        if len(y[y == 0]) < 2 or len(y[y == 1]) < 2:
            st.warning("No hay suficientes ejemplos para ambas clases ('CP' y 'otras') para dividir los datos de forma estratificada.")
            st.info("El modelo se entrenará con todos los datos disponibles, pero la evaluación podría no ser representativa.")
            X_train, y_train = X, y
            X_test, y_test = pd.DataFrame(), pd.Series() # No hay datos para test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            st.info(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

        # ---- Entrenamiento del Modelo ----
        st.subheader("🛠️ Entrenamiento del Modelo Random Forest...")
        with st.spinner('Entrenando un comité de clasificadores...'):
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            st.success("Modelo Random Forest entrenado exitosamente.")

        # ---- Evaluación del Modelo (si hay datos de prueba) ----
        if not X_test.empty:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Precisión del modelo en datos de prueba: {accuracy:.2f}**")
            st.text("Reporte de Clasificación:")
            st.code(classification_report(y_test, y_pred, target_names=['No Exoplaneta', 'Exoplaneta']))
        else:
            st.warning("No se pudieron evaluar el modelo porque no había suficientes datos para crear un conjunto de prueba.")

        # ---- Importancia de las Características ----
        importances = model.feature_importances_
        feature_importances_df = pd.DataFrame({'feature': features, 'importance': importances})
        feature_importances_df.sort_values(by='importance', ascending=False, inplace=True)
        
        st.success("Modelo y análisis de características listos.")
        return model, feature_importances_df, df_model, features # df_model para los gráficos

    return None, None, None, None

# ---- Interfaz de Carga de Archivo ----
st.sidebar.header("Cargar Dataset")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo 'datos_Tess.csv'", type=["csv"])

model, feature_importances, df_processed, features_list = None, None, None, None

if uploaded_file is not None:
    model, feature_importances, df_processed, features_list = load_and_train_model(uploaded_file)
else:
    st.info("Por favor, sube tu archivo 'datos_Tess.csv' en la barra lateral para empezar.")

# Solo mostrar el resto de la interfaz si el modelo se entrenó correctamente
if model is not None and feature_importances is not None and df_processed is not None:
    col1, col2 = st.columns([1, 1]) # Dividir la pantalla en dos columnas

    with col1:
        st.header('🔍 Predicción Interactiva')
        st.markdown("""
        Utiliza los controles deslizantes para simular una nueva observación astronómica y ver la predicción del modelo.
        """)

        with st.expander("💡 ¿Qué significan estos parámetros?"):
            st.markdown("""
            - **Período Orbital (días):** El tiempo que tarda un planeta en dar una vuelta completa a su estrella. Períodos más cortos suelen ser más fáciles de detectar por tránsito.
            - **Duración del Tránsito (horas):** El tiempo que el objeto permanece frente a su estrella. Depende del tamaño del planeta y la estrella, y de la velocidad orbital.
            - **Profundidad del Tránsito (ppm):** La disminución en el brillo de la estrella cuando el planeta pasa por delante. **Una mayor profundidad (un valor más grande) es un fuerte indicador de un objeto transitante más grande.**
            - **Magnitud de la Estrella (brillo):** Una medida de cuán brillante es la estrella desde la Tierra. Estrellas más brillantes (valores más bajos de magnitud) facilitan la detección.
            """)

        st.subheader('Ingresa los datos de la observación:')
        
        # Obtener valores min/max/media para los sliders desde los datos procesados
        # Manejo de errores en caso de que df_processed no tenga todos los features
        slider_values = {}
        for feature in features_list:
            if feature in df_processed.columns:
                min_val = df_processed[feature].min()
                max_val = df_processed[feature].max()
                mean_val = df_processed[feature].mean()
                slider_values[feature] = {'min': min_val, 'max': max_val, 'value': mean_val}
            else: # Valores por defecto si la columna no existe
                slider_values[feature] = {'min': 0.1, 'max': 100.0, 'value': 10.0}
                if feature == 'pl_trandurh': slider_values[feature] = {'min': 0.1, 'max': 20.0, 'value': 3.0}
                if feature == 'pl_trandep': slider_values[feature] = {'min': 100.0, 'max': 50000.0, 'value': 4500.0}
                if feature == 'st_tmag': slider_values[feature] = {'min': 4.0, 'max': 20.0, 'value': 11.0}


        pl_orbper = st.slider('Período Orbital (días)', float(slider_values['pl_orbper']['min']), float(slider_values['pl_orbper']['max']), float(slider_values['pl_orbper']['value']), step=0.1)
        pl_trandurh = st.slider('Duración del Tránsito (horas)', float(slider_values['pl_trandurh']['min']), float(slider_values['pl_trandurh']['max']), float(slider_values['pl_trandurh']['value']), step=0.1)
        pl_trandep = st.slider('Profundidad del Tránsito (ppm)', float(slider_values['pl_trandep']['min']), float(slider_values['pl_trandep']['max']), float(slider_values['pl_trandep']['value']), step=100.0)
        st_tmag = st.slider('Magnitud de la Estrella', float(slider_values['st_tmag']['min']), float(slider_values['st_tmag']['max']), float(slider_values['st_tmag']['value']), step=0.1)

        if st.button('✨ Predecir si es un Exoplaneta', type="primary"):
            input_data = pd.DataFrame({
                'pl_orbper': [pl_orbper],
                'pl_trandurh': [pl_trandurh],
                'pl_trandep': [pl_trandep],
                'st_tmag': [st_tmag]
            })

            with st.spinner('Analizando patrones en los datos...'):
                time.sleep(1.5) # Simulación de carga
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)

            st.subheader('Resultado de la Predicción:')
            if prediction[0] == 1:
                st.success('**¡Posible Exoplaneta Detectado!**')
                st.markdown(f"""
                El modelo predice con una confianza del **{prediction_proba[0][1]*100:.2f}%** que esta observación corresponde a un exoplaneta.
                Esto se basa en que los parámetros ingresados coinciden fuertemente con los patrones de tránsitos de exoplanetas confirmados.
                """)
            else:
                st.warning('**Probablemente no es un Exoplaneta.**')
                st.markdown(f"""
                El modelo tiene una confianza de solo el **{prediction_proba[0][1]*100:.2f}%** de que sea un exoplaneta.
                Los parámetros de esta observación no se alinean con los patrones típicos observados en exoplanetas confirmados.
                """)

    with col2:
        st.header("📈 Visualizaciones y Lógica del Modelo")

        st.subheader('Importancia de cada Característica para el Modelo:')
        st.markdown("""
        Este gráfico muestra qué características son las más influyentes para el modelo al tomar una decisión.
        Una barra más alta indica una mayor importancia.
        """)
        # Renombramos para mejor visualización
        feature_importances_display = feature_importances.copy()
        feature_importances_display['feature'] = feature_importances_display['feature'].replace({
            'pl_trandep': 'Profundidad Tránsito',
            'pl_orbper': 'Período Orbital',
            'pl_trandurh': 'Duración Tránsito',
            'st_tmag': 'Magnitud Estrella'
        })
        st.bar_chart(feature_importances_display.set_index('feature')['importance'])

        st.subheader('Relaciones Clave del Método de Tránsito:')
        st.markdown("""
        Estos gráficos interactivos te ayudan a entender la distribución de los datos de exoplanetas confirmados (`is_exoplanet = 1`)
        frente a otras observaciones (`is_exoplanet = 0`).
        """)

        # Gráfico 1: Profundidad del Tránsito vs Período Orbital
        chart1 = alt.Chart(df_processed).mark_circle(size=60).encode(
            x=alt.X('pl_orbper', title='Período Orbital (días)', scale=alt.Scale(type="log")),
            y=alt.Y('pl_trandep', title='Profundidad del Tránsito (ppm)', scale=alt.Scale(type="log")),
            # color=alt.Color('is_exoplanet:N', legend=alt.Legend(title="Es Exoplaneta", format=".0f"), scale=alt.Scale(range=['#FFD700', '#00BFFF']),
            #                 # Personalizar etiquetas de la leyenda
            #                 legend=alt.Legend(
            #                     title="Clase",
            #                     labelExpr="datum.value == 1 ? 'Exoplaneta Confirmado' : 'No Exoplaneta'"
            #                 )),
            tooltip=[
                alt.Tooltip('pl_orbper', title='Período'),
                alt.Tooltip('pl_trandep', title='Profundidad'),
                alt.Tooltip('is_exoplanet', title='Es Exoplaneta')
            ]
        ).properties(
            title='Profundidad del Tránsito vs. Período Orbital'
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)
        st.caption("Los exoplanetas confirmados suelen mostrar tránsitos con una profundidad y período consistentes.")

        # Gráfico 2: Duración del Tránsito vs Magnitud de la Estrella
        chart2 = alt.Chart(df_processed).mark_circle(size=60).encode(
            x=alt.X('st_tmag', title='Magnitud de la Estrella (brillo)', scale=alt.Scale(reverse=True)), # Estrellas más brillantes tienen magnitud menor
            y=alt.Y('pl_trandurh', title='Duración del Tránsito (horas)'),
            # color=alt.Color('is_exoplanet:N', legend=alt.Legend(title="Es Exoplaneta", format=".0f"), scale=alt.Scale(range=['#FFD700', '#00BFFF']),
            #                 legend=alt.Legend(
            #                     title="Clase",
            #                     labelExpr="datum.value == 1 ? 'Exoplaneta Confirmado' : 'No Exoplaneta'"
            #                 )),
            tooltip=[
                alt.Tooltip('st_tmag', title='Magnitud Estrella'),
                alt.Tooltip('pl_trandurh', title='Duración'),
                alt.Tooltip('is_exoplanet', title='Es Exoplaneta')
            ]
        ).properties(
            title='Duración del Tránsito vs. Magnitud de la Estrella'
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)
        st.caption("La duración del tránsito puede variar, y las estrellas más brillantes (menor magnitud) facilitan su detección.")

else:
    st.info("Sube un archivo CSV para cargar los datos y entrenar el modelo.")
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3d6bGt0M2x2bDJ6bnhiaGV0dmVob2ZqazB2a29xNnZjc2RjOGxoeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L9g7L2GD9eY2Q/giphy.gif") # Gif de un planeta