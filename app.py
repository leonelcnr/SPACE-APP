import streamlit as st
import pandas as pd
import requests, io

st.set_page_config(page_title="PLUTONITA", layout="wide", page_icon="logo/logo.jpeg")

st.markdown(
    "<h1 style='text-align: center'>🚀 PLUTONITA 🚀</h1>"
    "<p style='text-align: center'>In search of <strong>Exoplanets</strong> 🪐</p>",
    unsafe_allow_html=True
)

dict_tables = {
    "Select an option": "",
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

if option != "Select an option":
    table = dict_tables[option]

    with st.status("Running query…", expanded=True) as status:
        try:
            status.write(f"Requesting data from **{table}** to the API…")
            # si la tabla es muy grande, usa limit=2000 por ejemplo
            df, csv_text = fetch_df_from_tap(table)  # limit=2000

            status.write("Processing CSV with pandas…")
            # aquí podrías filtrar/transformar df si hace falta

            status.update(label="Ready! ✅", state="complete")
            tabla_ph.dataframe(df, use_container_width=True)
            dl_ph.download_button(
                "Download results (CSV)",
                data=csv_text.encode("utf-8"),
                file_name=f"{table}.csv",
                mime="text/csv",
            )
            st.toast(f"{len(df):,} loaded rows", icon="✅")

        except requests.exceptions.HTTPError as e:
            status.update(label="HTTP Error in the API", state="error")
            st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except requests.exceptions.Timeout:
            status.update(label="Timeout", state="error")
            st.error("The API took too long to respond. Please try again.")
        except Exception as e:
            status.update(label="Unexpected error", state="error")
            st.exception(e)
else:
    st.info("Choose an option from the menu to view the table. We'll display a loading indicator during the query.")


# Add Link to your repo
'''
    [![Repo](https://img.icons8.com/?size=50&id=4MhUS4CzoLbx&format=png&color=000000)](https://github.com/leonelcnr/SPACE-APP/) 

'''
st.markdown("<br>",unsafe_allow_html=True)

