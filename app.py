import streamlit as st
import pandas as pd
import requests, io

st.markdown("<h1 style='text-align: center'>🚀 PLUTONITA 🚀</h1><p style='text-align: center'>En busca de <strong>Exoplanetas</strong> 🪐</p>", unsafe_allow_html = True)

dict_tables = {
    "Seleccione una opcion": '',
    "Kepler": "cumulative",
    "Tess": "toi",
    "Planetary Systems": "ps"
}

x = st.slider('Hyperparameter Tweaking', max_value=10)

left_column, right_column = st.columns(2)
option = st.selectbox("Dataset", dict_tables.keys())
st.write(dict_tables[option])
if option != "Seleccione una opcion":


    LINK = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    q = f"SELECT top 10 * FROM {dict_tables[option]}"
    r = requests.get(LINK, params={"query": q, "format": "csv"})

    # @st.cache_data
    def cargar_datos(nfilas):
        data = pd.read_csv(
            io.StringIO(r.text), 
            nrows=nfilas
        )
        return data

    data = cargar_datos(10)

    st.write(data)

# Add Link to your repo
'''
    [![Repo](https://img.icons8.com/?size=50&id=4MhUS4CzoLbx&format=png&color=000000)](https://github.com/leonelcnr/SPACE-APP/) 

'''
st.markdown("<br>",unsafe_allow_html=True)

