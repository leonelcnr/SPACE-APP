import streamlit as st
import pandas as pd
import requests, io

st.set_page_config(page_title="PLUTONITA", layout="wide")

st.markdown(
    "<h1 style='text-align: center'>🚀 PLUTONITA 🚀</h1>"
    "<p style='text-align: center'>En busca de <strong>Exoplanetas</strong> 🪐</p>",
    unsafe_allow_html=True
)

dict_tables = {
    "Seleccione una opcion": "",
    "Kepler": "cumulative",     # si en TAP no existe, cambia a 'koi'
    "TESS": "toi",
    "Planetary Systems": "ps"
}

# (opcional) solo para jugar
x = st.slider('Hyperparameter Tweaking', max_value=10)

left_column, right_column = st.columns(2)
option = st.selectbox("Dataset", list(dict_tables.keys()))

# --- función de consulta (con caché) ---
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_df_from_tap(table: str, limit: int | None = None, timeout: int = 60):
    """
    Llama a TAP con SELECT * FROM <table> (opcional TOP)
    y devuelve (DataFrame, CSV_string).
    """
    LINK = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    top = f"top {limit} " if limit else ""
    q = f"select {top} * from {table}"
    r = requests.get(LINK, params={"query": q, "format": "csv"}, timeout=timeout)
    r.raise_for_status()
    csv_text = r.text
    df = pd.read_csv(io.StringIO(csv_text))
    return df, csv_text

# Placeholders para tabla y descarga
tabla_ph = st.empty()
dl_ph = st.empty()

if option != "Seleccione una opcion":
    table = dict_tables[option]

    with st.status("Ejecutando consulta…", expanded=True) as status:
        try:
            status.write(f"Solicitando datos de **{table}** a la API…")
            # si la tabla es muy grande, usa limit=2000 por ejemplo
            df, csv_text = fetch_df_from_tap(table)  # limit=2000

            status.write("Procesando CSV con pandas…")
            # aquí podrías filtrar/transformar df si hace falta

            status.update(label="¡Listo! ✅", state="complete")
            tabla_ph.dataframe(df, use_container_width=True)
            dl_ph.download_button(
                "Descargar resultados (CSV)",
                data=csv_text.encode("utf-8"),
                file_name=f"{table}.csv",
                mime="text/csv",
            )
            st.toast(f"{len(df):,} filas cargadas", icon="✅")

        except requests.exceptions.HTTPError as e:
            status.update(label="Error HTTP en la API", state="error")
            st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except requests.exceptions.Timeout:
            status.update(label="Timeout", state="error")
            st.error("La API tardó demasiado en responder. Probá de nuevo.")
        except Exception as e:
            status.update(label="Error inesperado", state="error")
            st.exception(e)
else:
    st.info("Elegí una opción del menú para ver la tabla. Mostraremos un indicador de carga durante la consulta.")
