"""
mini prototipo de uso de librerias
- lightkurve (descarga/gestión de curvas TESS/Kepler) en combinacion con TOI oficial de la NASA
- astropy (BLS)
- numpy/pandas/matplotlib (manejo y plots)
"""
# ---------- Importes ----------
import os
import io
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# ---------- UI base ----------
st.set_page_config(
    page_title="Plutonita",
    page_icon="logo/logo.jpeg",  
)
st.set_page_config(page_title="PLUTONITA", layout="wide")
st.markdown(
    "<h1 style='text-align: center'>🚀 PLUTONITA 🚀</h1>"
    "<p style='text-align: center'>En busca de <strong>Exoplanetas</strong> 🪐</p>",
    unsafe_allow_html=True
)
st.title("Datasets de Objetos TESS de Interés (TOI)")

# left_column, right_column = st.columns(2)

# ---------- Helpers ----------
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OUT_DIR = "graficos_producidos"
os.makedirs(OUT_DIR, exist_ok=True)

@st.cache_data(show_spinner=False, ttl=1800)
def cargar_datos(nfilas: int) -> pd.DataFrame:
    """Carga TOP nfilas de TOI (PC/KP) ordenado por TID desde el Exoplanet Archive (CSV)."""
    q = (
        f"SELECT TOP {nfilas} * FROM toi "
        f"WHERE tfopwg_disp IN ('PC','KP') "
        f"ORDER BY tid"
    )
    r = requests.get(TAP, params={"query": q, "format": "csv"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), nrows=nfilas)
    return df

@st.cache_data(show_spinner=False, ttl=600)
def consultar_toi_por_tid(tid: int) -> pd.DataFrame:
    """Devuelve filas de TOI para un TID específico, solo PC/KP."""
    q = (
        "SELECT * FROM toi "
        f"WHERE (tfopwg_disp IN ('PC','KP')) AND (tid = {tid}) "
        "ORDER BY toi"
    )
    r = requests.get(TAP, params={"query": q, "format": "csv"}, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def descarga_tess_lk(tic_id, sector=None, author=None):
    """Descarga curva de luz TESS para un TIC; si hay varios sectores, los une."""
    res = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS", sector=sector, author=author)
    if len(res) == 0:
        return None
    try:
        lcc = res.download_all()
        if hasattr(lcc, "stitch"):
            return lcc.stitch()  # una sola curva de luz unificada
        return lcc[0]
    except Exception:
        try:
            return res[0].download()
        except Exception:
            return None

def suavizado_curva(flux, ventana_largo=401):
    """Detrende suave por mediana deslizante (medfilt), devuelve flujo normalizado."""
    f = np.asarray(flux, dtype=np.float64, order="C")
    if not np.isfinite(f).all():
        med = np.nanmedian(f)
        f = np.where(np.isfinite(f), f, med)
    wl = int(ventana_largo)
    if wl % 2 == 0:
        wl += 1
    base = medfilt(f, kernel_size=wl)
    if not np.isfinite(base).all():
        bmed = np.nanmedian(base[np.isfinite(base)])
        base = np.where(np.isfinite(base), base, bmed)
    base[base == 0] = np.nanmedian(base[base != 0])
    return f / base

def grafico_curva_luz(t, f, exoplaneta, direccion_salida):
    f_det = suavizado_curva(f, 401)
    plt.figure()
    plt.plot(t, f_det, ".", ms=1)
    plt.xlabel("Tiempo [días]"); plt.ylabel("Flujo (suavizado)")
    plt.title(f"TIC {exoplaneta.tid} — suavizado")
    plt.tight_layout()
    path = os.path.join(direccion_salida, f"TIC{exoplaneta.tid}_suavizado.png")
    plt.savefig(path); plt.close()
    return path

def grafico_curva_luz_plegado(t, f, exoplaneta, direccion_salida):
    f_det = suavizado_curva(f, 401)
    t0 = t[0]  # si tenés epoch del TOI, úsalo aquí
    periodo = exoplaneta['pl_orbper'] if np.isfinite(exoplaneta['pl_orbper']) else 1.0

    fase = ((t - t0) / periodo) % 1.0
    fase = fase - 0.5
    order = np.argsort(fase)
    fase, f_fold = fase[order], f_det[order]

    bins = np.linspace(-0.5, 0.5, 61)
    idx = np.digitize(fase, bins) - 1
    xb, yb = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if np.any(m):
            xb.append(0.5 * (bins[b] + bins[b + 1]))
            yb.append(np.nanmedian(f_fold[m]))
    xb, yb = np.array(xb), np.array(yb)

    plt.figure()
    plt.plot(fase, f_fold, ".", ms=1, alpha=0.4, label="datos")
    plt.plot(xb, yb, "-", lw=2, label="mediana por bins")
    plt.xlabel("Fase (ciclos)"); plt.ylabel("Flujo (suavizado)")
    plt.title(f"Plegado con P≈{periodo:.4f} d")
    plt.legend(); plt.tight_layout()
    path = os.path.join(direccion_salida, f"TIC{exoplaneta.tid}_plegado.png")
    plt.savefig(path); plt.close()
    return path

# ---------- Tabla inicial con indicador de carga ----------
cantidad_filas = st.number_input(
    "Cantidad de filas a cargar:", min_value=1, max_value=1000, value=10, step=1
)

with st.status("Cargando tabla inicial…", expanded=True) as status:
    try:
        status.write("Consultando TOI (PC/KP) en el Exoplanet Archive…")
        df_inicial = cargar_datos(cantidad_filas)
        tabla_ph = st.empty()  # placeholder para la tabla
        status.update(label="¡Tabla lista! ✅", state="complete")
        tabla_ph.dataframe(df_inicial, use_container_width=True)
        st.toast(f"{len(df_inicial):,} filas cargadas", icon="✅")
    except requests.exceptions.Timeout:
        status.update(label="Timeout consultando la API", state="error")
        st.error("La API tardó demasiado en responder. Probá nuevamente.")
    except requests.exceptions.HTTPError as e:
        status.update(label="Error HTTP al consultar", state="error")
        st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        status.update(label="Error inesperado al cargar", state="error")
        st.exception(e)

# ---------- Generador de gráficos ----------
st.title("Generador de gráficos de curvas de luz TESS")
st.write("Este prototipo genera gráficos con Lightkurve para exoplanetas candidatos (TOI PC/KP).")

tid_input = st.text_input(
    "Ingrese el TIC ID (tid) del exoplaneta candidato (por ejemplo, 16288184):",
    value="16288184"
)

if st.button("Generar gráficos", type="primary"):
    with st.status("Generando gráficos…", expanded=True) as status:
        try:
            # 1) Validar TID
            if not tid_input.isdigit():
                status.update(label="TID inválido", state="error")
                st.error("Por favor, ingrese un TID numérico válido.")
                st.stop()
            tid_val = int(tid_input)

            status.write("Buscando el TOI (PC/KP) para ese TID…")
            data_set = consultar_toi_por_tid(tid_val)
            muestra = data_set[data_set["tfopwg_disp"].isin(["PC", "KP"])]

            if muestra.empty:
                status.update(label="Sin resultados para ese TID", state="error")
                st.error(f"No se encontró ningún exoplaneta PC/KP con TIC ID {tid_val}.")
                st.stop()

            exoplaneta = muestra.iloc[0]

            # 2) Descargar curva de luz
            status.write(f"Descargando curva de luz TESS para TIC {tid_val}…")
            curva = descarga_tess_lk(exoplaneta["tid"])
            if curva is None:
                status.update(label="No se pudo descargar la curva", state="error")
                st.error("No se encontró curva de luz para ese TIC en TESS.")
                st.stop()

            # 3) Preparar arrays
            status.write("Procesando y suavizando curva…")
            t = curva.time.value if hasattr(curva.time, "value") else curva.time
            f = curva.flux.value if hasattr(curva.flux, "value") else curva.flux

            # 4) Generar gráficos
            status.write("Creando gráficos (suavizado y plegado)…")
            path_suav = grafico_curva_luz(t, f, exoplaneta, OUT_DIR)
            path_pleg = grafico_curva_luz_plegado(t, f, exoplaneta, OUT_DIR)

            # 5) Mostrar resultados
            status.update(label="¡Listo! ✅", state="complete")
            st.success("Gráficos generados exitosamente:")
            left_column, right_column = st.columns(2)


            with left_column:
                st.subheader("Curva de Luz Suavizada")
                st.image(path_suav, use_column_width=True)

            with right_column:
                st.subheader("Curva de Luz Plegada")
                st.image(path_pleg, use_column_width=True)

            st.toast(f"Imágenes guardadas en ./{OUT_DIR}", icon="🖼️")

        except requests.exceptions.Timeout:
            status.update(label="Timeout de la API", state="error")
            st.error("La API tardó demasiado en responder. Probá de nuevo.")
        except requests.exceptions.HTTPError as e:
            status.update(label="Error HTTP en la API", state="error")
            st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            status.update(label="Error generando gráficos", state="error")
            st.exception(e)

# ---------- Link a repo ----------
st.markdown(
    """
    <br>
    <a href="https://github.com/leonelcnr/SPACE-APP/" target="_blank">
        <img src="https://img.icons8.com/?size=50&id=4MhUS4CzoLbx&format=png&color=000000" alt="Repo"/>
    </a>
    """,
    unsafe_allow_html=True
)
