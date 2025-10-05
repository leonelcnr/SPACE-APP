"""
mini prototipo de uso de librerias
- lightkurve (descarga/gestión de curvas TESS/Kepler) en combinacion con TOI oficial de la NASA
- astropy (BLS)
- numpy/pandas/matplotlib (manejo y plots)
"""
# ---------- Importes ----------
import os
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import requests, io
# ---------- CONFIGURACION ----------
path_csv_exoplanetas_tess = "csv_temp/TOI_2025.10.04_12.24.10.csv" #ver con el dominio de streamlit donde crear las carpetas
out_dir = "graficos_producidos" #donde se alojararn los graficos
os.makedirs(out_dir, exist_ok=True)
# ---------- funcionamiento ----------
def descarga_tess_lk(tic_id, sector=None, autor=None):
    """Descarga curva de luz TESS para un TID; si hay varios sectores, los une."""
    resul = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS", sector=sector, author=autor)
    if len(resul) == 0:
        return None
    try:
        lcc = resul.download_all()
        if hasattr(lcc, "stitch"):
            return lcc.stitch()# una sola curva de luz unificada
        return lcc[0]
    except Exception:
        return resul[0].download()

def suavizado_curva(flux, ventana_largo=401): #flux es el brillo medido de la estrella a lo largo del tiempo
    f = np.asarray(flux, dtype=np.float64, order="C")
    if not np.isfinite(f).all(): #rellena los campos vacios que tienen Nan para que no se modifique la mediana
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
    plt.figure(); plt.plot(t, f_det, ".", ms=1)
    plt.xlabel("Tiempo [días]"); plt.ylabel("Flujo (suavizado)"); plt.title(f"TIC {exoplaneta.tid} — suavizado")
    plt.tight_layout(); plt.savefig(os.path.join(direccion_salida, f"TIC{exoplaneta.tid}_suavizado.png")); plt.close()
    print("hecho")

def grafico_curva_luz_plegado(t, f, exoplaneta, direccion_salida):#t (días), f_det (detrendido), periodo (días)
    f_det = suavizado_curva(f, 401)
    t0 = t[0]  # o el epoch del TOI si lo tenés
    periodo = exoplaneta['pl_orbper'] if np.isfinite(exoplaneta['pl_orbper']) else 1.0

    fase = ((t - t0)/periodo) % 1.0 #formula de plegado de la curva de luz
    fase = fase - 0.5
    order = np.argsort(fase)
    fase, f_fold = fase[order], f_det[order]

    # binning
    bins = np.linspace(-0.5, 0.5, 61)
    idx = np.digitize(fase, bins) - 1
    xb, yb = [], []
    for b in range(len(bins)-1):
        m = idx == b
        if np.any(m):
            xb.append(0.5*(bins[b]+bins[b+1]))
            yb.append(np.nanmedian(f_fold[m]))
    xb, yb = np.array(xb), np.array(yb)
    #armado del grafico
    plt.figure()
    plt.plot(fase, f_fold, ".", ms=1, alpha=0.4, label="datos")
    plt.plot(xb, yb, "-", lw=2, label="mediana por bins")
    plt.xlabel("Fase (ciclos)"); plt.ylabel("Flujo (suavizado)")
    plt.title(f"Plegado con P≈{periodo:.4f} d")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(direccion_salida, f"TIC{exoplaneta.tid}_plegado.png")); plt.close()
    print("hecho")

def generamiento_parametros_exoplaneta(tid_id, directorio_csv):
    LINK = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    q = f"SELECT * FROM {"toi"}"
    r = requests.get(LINK, params={"query": q, "format": "csv"})
    data_set = pd.read_csv(
            io.StringIO(r.text)
        )
    muestra_candidatos = data_set[data_set['tfopwg_disp'].isin(["PC", "KP"])] #filtro solo planetas candidatos sera hecho en el frontend
    aux = muestra_candidatos.loc[muestra_candidatos["tid"].astype("Int64") == tid_id]
    exoplaneta = aux.iloc[0]
    print(f"Descargando TIC {int(exoplaneta['tid'])} - TOI {exoplaneta['toi']}")
    curva_luz = descarga_tess_lk(exoplaneta['tid'])
    print(curva_luz)
    t = curva_luz.time.value if hasattr(curva_luz.time, "value") else curva_luz.time
    f = curva_luz.flux.value if hasattr(curva_luz.flux, "value") else curva_luz.flux 
    grafico_curva_luz(t, f, exoplaneta,out_dir)
    grafico_curva_luz_plegado(t, f, exoplaneta,out_dir)

generamiento_parametros_exoplaneta(231663901,path_csv_exoplanetas_tess)
