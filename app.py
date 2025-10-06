import streamlit as st
import pandas as pd
import os
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import json
from pathlib import Path
from exoplanet_model import (
    load_and_prepare, cross_validate, train_full_model,
    score_candidates, save_artifacts, load_model,
    predict_single, compute_full_metrics, load_medians, FEATURE_COLS
)

st.set_page_config(page_title="Exoplanet Predictor", layout="wide")
st.title("Exoplanet Candidate Ranking – Binary Model")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lgbm_exoplanet_model.joblib"
CANDS_PATH = ARTIFACTS_DIR / "candidates_scored.csv"
META_PATH = ARTIFACTS_DIR / "metadata.json"
USER_INPUTS = ARTIFACTS_DIR / "user_inputs.csv"

default_params = {
    "n_estimators": 400,
    "learning_rate": 0.04,
    "num_leaves": 48,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "objective": "binary"
}

with st.sidebar:
    st.header("Training")
    uploaded = st.file_uploader("Upload CSV TOI", type=["csv"])
    calibrate = st.checkbox("Calibrate probabilities (isotonic)", value=False)
    train_btn = st.button("Train / Re-train")

    st.markdown("---")
    st.header("Manual Prediction")
    threshold = st.slider("Classification threshold (planet-like)", 0.0, 1.0, 0.5, 0.01)

# Cargar modelo y medianas si existen
model = None
medians = {}
if MODEL_PATH.exists():
    try:
        model = load_model(MODEL_PATH)
        st.success("Loaded model.")
        medians = load_medians(META_PATH)
    except Exception as e:
        st.error(f"The model could not be loaded: {e}")

# Entrenamiento
if train_btn:
    if uploaded is None:
        st.error("First upload a CSV.")
        st.stop()
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    temp_csv = ARTIFACTS_DIR / "uploaded_training.csv"
    with open(temp_csv, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Preparing data...")
    train_df = load_and_prepare(str(temp_csv))
    st.write(f"Training rows: {len(train_df)} | PC candidates: {len(train_df)}")

    st.info("Cross-validation...")
    cv = cross_validate(train_df, default_params, n_splits=5, calibrate=calibrate)
    st.subheader("Average CV Metrics")
    st.json(cv)

    st.info("Training final model...")
    model = train_full_model(train_df, default_params, calibrate=calibrate)

    st.info("Metrics on the entire training dataset...")
    full_metrics = compute_full_metrics(train_df, model)
    st.json(full_metrics)

    st.info("Ranking candidates (PC)...")
    scored = score_candidates(model, train_df)
    if not scored.empty:
        st.dataframe(scored.head(20))
    else:
        st.warning("There were no PC rows to rank.")

    st.info("Storing artifacts...")
    save_artifacts(model, scored, cv, full_metrics, train_df, outdir=str(ARTIFACTS_DIR))

    st.success("Complete training. Reload to use updated medians.")
    st.stop()

# Tabs
tab_home, tab_ranking, tab_manual, tab_metadata, tab_transition_method= st.tabs(["Home", "PC Ranking", "Manual entry", "Metadata", "Transition Method"])
with tab_home:
    st.markdown("""
        <style>
            .main { background-color: #181a1b !important; }
            .logo { width: 200px; margin-bottom: 12px; }
            .libs-row img { height: 48px; margin: 0 18px 8px 0;}
            .title { font-size: 2.8em; font-weight: bold; letter-spacing: 2px;}
            .subtitle { font-size: 1.3em; color: #eee; margin-bottom: 35px;}
            .ml-highlight { color: #f39c12; font-weight: bold;}
            .btn-start { background: #f39c12; color: white; padding: 12px 30px; border-radius: 8px; font-size: 1.1em; text-decoration: none; }
            .libs-row { margin: 22px 0 18px 0; display: flex; flex-wrap: wrap; justify-content: center;}
            .divider { height:2px; background: #444; margin:28px 0;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title" style="text-align:center;">PLUTONITA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="text-align:center;">'
                'Unlock the Universe with <span class="ml-highlight">Machine Learning</span> 🚀<br>'
                'NASA Space Apps Project · Exoplanet Exploration'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to **Plutonita** – a modern web platform powered by **Machine Learning** and the latest data science technologies for exoplanet discovery and space exploration!

    - Explore interactive datasets
    - Visualize light curves and astrophysical data
    - Experiment with AI-powered model

    """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="libs-row" style="text-align:center;">'
        '<a href="https://streamlit.io/" title="Streamlit"><img src="https://streamlit.io/images/brand/streamlit-mark-color.png"/></a>'
        '<a href="https://numpy.org/" title="NumPy"><img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg"/></a>'
        '<a href="https://pandas.pydata.org/" title="Pandas"><img src="https://pandas.pydata.org/static/img/pandas_mark.svg"/></a>'
        '<a href="https://scikit-learn.org/" title="Scikit-learn"><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png"/></a>'
        '<a href="https://matplotlib.org/" title="Matplotlib"><img src="https://matplotlib.org/_static/images/logo2.svg"/></a>'
        '<a href="https://lightkurve.org/" title="Lightkurve">Lightkurve - </a>'
        '<a href="https://docs.astropy.org/" title="Astropy">Astropy - </a>'
        '<a href="https://lightgbm.readthedocs.io/" title="LightGBM">LightGBM</a>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; color: #aaa; font-size:0.95em;">'
                'Powered by Python · Open Source · <a href="https://github.com/leonelcnr/SPACE-APP/" style="color:#f39c12;">View on GitHub</a>'
                '</div>', unsafe_allow_html=True)
    
with tab_ranking:
    st.subheader("Candidate ranking (PC)")
    if CANDS_PATH.exists():
        df_cands = pd.read_csv(CANDS_PATH)
        st.dataframe(df_cands.head(50), use_container_width=True)
        st.download_button(
            label="Download full ranking (CSV)",
            data=df_cands.to_csv(index=False),
            file_name="candidates_scored.csv",
            mime="text/csv"
        )
    else:
        st.info("There's no ranking yet. Train first.")

with tab_manual:
    st.subheader("Enter new object / planet")
    if model is None:
        st.warning("First train or load a model.")
    else:
        with st.form("manual_form"):
            cols = st.columns(4)
            input_data = {}
            for i, feat in enumerate(FEATURE_COLS):
                with cols[i % 4]:
                    input_data[feat] = st.number_input(feat, value=0.0, format="%.5f", key=f"in_{feat}")
            submit_btn = st.form_submit_button("Calculate probability")

        if submit_btn:
            with open(META_PATH, "r") as f:
                data = json.load(f)
                st.write("ROC AUC (full):", data["full_metrics"]["roc_auc_full"])
            prob = predict_single(model, input_data, medians=medians)
            decision = "PLANET-LIKE" if prob >= threshold else "NOT PLANET-LIKE"
            if prob > 0.66:
                st.markdown("**Your object is an exoplanet!**")
            else:
                if prob > 0.33:
                    st.markdown("**The entered object is a candidate planet.**")
                else:
                    st.markdown("**False positive!**")
            st.markdown(f"**Probability:**{prob:.6f}")
##            st.markdown(f"**Decision (umbral {threshold:.2f}):** {decision}")

            # Opción guardar
            save_choice = st.checkbox("Save this object to local registry", value=True)
            if save_choice:
                row = input_data.copy()
                row.update({
                    "prob_planet": prob,
                    "decision": decision,
                    "threshold_used": threshold,
                    "timestamp": pd.Timestamp.utcnow().isoformat()
                })
                if USER_INPUTS.exists():
                    df_prev = pd.read_csv(USER_INPUTS)
                    df_new = pd.concat([df_prev, pd.DataFrame([row])], ignore_index=True)
                else:
                    df_new = pd.DataFrame([row])
                df_new.to_csv(USER_INPUTS, index=False)
                st.success("Saved in artifacts/user_inputs.csv")

            if USER_INPUTS.exists():
                with st.expander("View last saved"):
                    st.dataframe(pd.read_csv(USER_INPUTS).tail(10))

with tab_metadata:
    st.subheader("Metadata / Metrics")
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        st.json(meta)
        imps = meta.get("feature_importances")
        if imps:
            st.markdown("### Importance of features")
            imp_df = (pd.DataFrame(list(imps.items()), columns=["feature","importance"])
                        .sort_values("importance", ascending=False))
            st.bar_chart(imp_df.set_index("feature"))
    else:
        st.info("No metadata yet.")
        

with tab_transition_method:
    st.markdown(
        "<h1 style='text-align: center'>🚀 PLUTONITA 🚀</h1>"
        "<p style='text-align: center'>In search of <strong>Exoplanets</strong> 🪐</p>",
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
                    st.image(path_suav)

                with right_column:
                    st.subheader("Curva de Luz Plegada")
                    st.image(path_pleg)

                st.toast(f"Imágenes guardadas en ./{OUT_DIR}", icon="🖼️")

            except requests.exceptions.Timeout:
                status.update(label="Timeout de la API", state="error")
                st.error("La API tardó demasiado en responder. Probá de nuevo.")
            except requests.exceptions.HTTPError as e:
                status.update(label="HTTP Error in the API", state="error")
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
        

    st.caption("Prototype – Binary model (CP/KP vs FP) with PC ranking and manual entry form.")
