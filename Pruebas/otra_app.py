import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Configuración de la página de Streamlit
st.set_page_config(page_title="Detector de Exoplanetas", layout="wide")

# Título y descripción
st.title("🚀 Detector de Exoplanetas con el Método de Tránsito")
st.write("""
Esta aplicación utiliza un modelo de Machine Learning para predecir si un objeto de interés (TOI) del satélite TESS es un exoplaneta, basándose en sus características de tránsito.
""")

# --- Carga y Procesamiento de Datos ---
@st.cache_data
def load_data(filepath):
    """Carga y procesa el dataset de exoplanetas."""
    df = pd.read_csv(filepath)
    # Seleccionamos características relevantes para el método de tránsito
    # pl_orbper: Período Orbital (días)
    # pl_trandurh: Duración del Tránsito (horas)
    # pl_trandep: Profundidad del Tránsito (ppm - partes por millón)
    # st_tmag: Magnitud de la estrella en la banda de TESS
    features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'st_tmag']
    
    # La columna 'tfopwg_disp' indica la disposición del objeto:
    # CP = Planeta Confirmado, PC = Planeta Candidato, FP = Falso Positivo
    target = 'tfopwg_disp'
    
    # Nos quedamos solo con las columnas que nos interesan
    df_filtered = df[features + [target]].copy()
    
    # Eliminamos filas donde la disposición es desconocida
    df_filtered.dropna(subset=[target], inplace=True)
    
    # Creamos nuestra variable objetivo: 1 si es exoplaneta, 0 si es falso positivo
    df_filtered['is_exoplanet'] = df_filtered[target].apply(lambda x: 1 if x in ['CP', 'PC'] else 0)
    
    # Nos quedamos solo con los datos que son planetas confirmados o falsos positivos para un entrenamiento más claro
    df_final = df_filtered[df_filtered[target].isin(['CP', 'FP'])]

    X = df_final[features]
    y = df_final['is_exoplanet']
    
    return X, y, features

# Cargamos los datos
try:
    X, y, features = load_data('datos_Tess.csv')

    # --- Entrenamiento del Modelo ---
    st.sidebar.header("Configuración del Modelo")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Manejar valores faltantes en los datos con un imputador
    # Rellenará los valores NaN con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Inicializar y entrenar el clasificador de Bosque Aleatorio (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_imputed, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, y_pred)

    st.sidebar.success(f"Modelo entrenado con una precisión del {accuracy*100:.2f}%")
    st.sidebar.info(f"El modelo se entrenó con **{len(X_train)}** muestras y se probó con **{len(X_test)}**.")


    # --- Interfaz de Usuario para Predicción ---
    st.header("Predice si un Objeto es un Exoplaneta")
    st.write("Ajusta los valores de las características en la barra lateral para obtener una predicción.")

    st.sidebar.header("Parámetros de Entrada del Usuario")

    def user_input_features():
        """Recoge los inputs del usuario desde la barra lateral."""
        pl_orbper = st.sidebar.slider('Período Orbital (días)', float(X['pl_orbper'].min()), float(X['pl_orbper'].max()), float(X['pl_orbper'].mean()))
        pl_trandurh = st.sidebar.slider('Duración del Tránsito (horas)', float(X['pl_trandurh'].min()), float(X['pl_trandurh'].max()), float(X['pl_trandurh'].mean()))
        pl_trandep = st.sidebar.slider('Profundidad del Tránsito (ppm)', float(X['pl_trandep'].min()), float(X['pl_trandep'].max()), float(X['pl_trandep'].mean()))
        st_tmag = st.sidebar.slider('Magnitud de la Estrella (TESS mag)', float(X['st_tmag'].min()), float(X['st_tmag'].max()), float(X['st_tmag'].mean()))
        
        data = {'pl_orbper': pl_orbper,
                'pl_trandurh': pl_trandurh,
                'pl_trandep': pl_trandep,
                'st_tmag': st_tmag}
        
        features_df = pd.DataFrame(data, index=[0])
        return features_df

    # Obtener los datos del usuario
    input_df = user_input_features()

    # Mostrar los parámetros de entrada
    st.subheader('Parámetros de Entrada:')
    st.write(input_df)

    # Realizar la predicción
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Mostrar el resultado de la predicción
    st.subheader('Resultado de la Predicción:')
    if prediction[0] == 1:
        st.success('**¡Es muy probable que sea un Exoplaneta!** 🪐')
        st.write(f"Confianza de la predicción: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.error('**Es poco probable que sea un Exoplaneta.** 🔭')
        st.write(f"Confianza de la predicción de que NO es un exoplaneta: **{prediction_proba[0][0]*100:.2f}%**")

    # --- Visualización de la Importancia de las Características ---
    st.header("¿Qué características son más importantes?")
    st.write("El siguiente gráfico muestra qué tan influyente fue cada característica para la predicción del modelo.")
    
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    st.bar_chart(feature_importance.set_index('feature'))

except FileNotFoundError:
    st.error("Error: El archivo `datos_Tess.csv` no se encontró. Asegúrate de que el archivo esté en la misma carpeta que `app.py`.")
except Exception as e:
    st.error(f"Ocurrió un error al procesar los datos: {e}")