import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar los datos
try:
    df = pd.read_csv('datos_Tess.csv')
except FileNotFoundError:
    print("Error: El archivo 'datos_Tess.csv' no se encontró.")
    print("Asegúrate de que el archivo esté en la misma carpeta que este script.")
    exit()

# ---- 1. Selección de Características y Limpieza ----
features = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'st_tmag']
target = 'tfopwg_disp'
df_model = df[features + [target]].copy()
df_model.dropna(inplace=True)

# ---- 2. Creación de la Variable Objetivo ----
df_model['is_exoplanet'] = df_model[target].apply(lambda x: 1 if x == 'CP' else 0)
X = df_model[features]
y = df_model['is_exoplanet']

# ---- 3. División de Datos ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Datos listos para el entrenamiento.")

# ---- 4. Entrenamiento del Modelo ----
print("\nEntrenando el modelo...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Modelo entrenado.")

# ---- 5. Evaluación del Modelo ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Exoplaneta', 'Exoplaneta']))

# ---- 6. Guardar el Modelo y la Importancia de las Características ----
# Guardamos el modelo
joblib.dump(model, 'modelo_exoplanetas.joblib')
print("\nModelo guardado como 'modelo_exoplanetas.joblib'")

# ---- ¡NUEVO! Guardar la importancia de las características ----
# Extraemos la importancia que el modelo le dio a cada característica
importances = model.feature_importances_
# Creamos un DataFrame para que sea fácil de leer y visualizar
feature_importances = pd.DataFrame({'feature': features, 'importance': importances})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
# Guardamos este DataFrame
joblib.dump(feature_importances, 'feature_importances.joblib')
print("Importancia de las características guardada como 'feature_importances.joblib'")