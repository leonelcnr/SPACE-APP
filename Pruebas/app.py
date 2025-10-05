import streamlit as st
import pandas as pd
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
tab_ranking, tab_manual, tab_metadata = st.tabs(["PC Ranking", "Manual entry", "Metadata"])

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

st.caption("Prototype – Binary model (CP/KP vs FP) with PC ranking and manual entry form.")
