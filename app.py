from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

BASE = Path(__file__).resolve().parent
TARGETS = ["Rocking_index", "Sliding_index", "Diagonal_index", "Toe_crushing_index"]
NAMES = ["Rocking", "Bed-joint sliding", "Diagonal cracking", "Toe crushing"]
COLORS = ["#2F5BEA", "#F97316", "#00A878", "#D12C7A"]


@st.cache_resource
def load_models():
    folder = BASE / "saved_models"
    metadata = joblib.load(folder / "model_metadata.joblib")
    if metadata.get("schema_version") != 2 or metadata["target_cols"] != TARGETS:
        raise ValueError("Export the matching models and metadata using export_website_files.py.")
    models = {t: joblib.load(folder / f"{t}_model.joblib") for t in TARGETS}
    return models, metadata


@st.cache_data
def load_rules():
    return json.loads((BASE / "saved_models" / "hysteresis_rules.json").read_text())


def classify(rule, values):
    score = float(sum(w * values[f] for w, f in zip(rule["weights"], rule["features"])) + rule["intercept"])
    return (rule["positive_class"] if score >= 0 else rule["negative_class"]), score


def self_centering(return_cap, return_ult, drift_cap, drift_ult, strength_cap, strength_ult, stiffness):
    values = [return_cap, return_ult, drift_cap, drift_ult, strength_cap, strength_ult, stiffness]
    if not np.isfinite(values).all() or stiffness <= 0:
        raise ValueError("Use finite inputs and a stiffness greater than zero.")
    elastic_cap = drift_cap - strength_cap / stiffness
    elastic_ult = drift_ult - strength_ult / stiffness
    if abs(elastic_cap) < 1e-12 or abs(elastic_ult) < 1e-12:
        raise ValueError("An elastic-return denominator is zero; self-centering cannot be calculated.")
    sc_cap, sc_ult = return_cap / elastic_cap, return_ult / elastic_ult
    # Match the supplied notebook's feature-extraction corrections exactly.
    if -0.5 < sc_cap < 0 or return_cap < 0.01:
        sc_cap = 0.0
    if -0.5 < sc_ult < 0:
        sc_ult = 0.0
    return sc_cap, sc_ult, elastic_cap, elastic_ult


def equation(rule):
    terms = " + ".join(f"({w:.6g}) × {f}" for w, f in zip(rule["weights"], rule["features"]))
    return f"{terms} + ({rule['intercept']:.6g}) = 0"


st.set_page_config(page_title="URM Wall Seismic Behavior Prediction", page_icon="🏗️", layout="wide")
st.title("Seismic Behavior Prediction of URM Walls")
st.write("Evaluate wall failure-mechanism contributions from design properties or detect a failure mode from hysteresis-test parameters.")
about, predict, detect = st.tabs(["About the Model", "Predict Seismic Behavior", "Failure Mode Detection from Hysteresis Test"])

with about:
    st.header("About the Model")
    st.write("The prediction module uses four XGBoost regressors to estimate rocking, sliding, diagonal cracking, and toe-crushing Hybridity Indices. Negative predictions are clipped to zero, then the four outputs are normalized to percentages.")
    st.write("The hysteresis module applies the linear SVM decision boundaries fitted to the notebook's two-stage clustering labels. The first stage determines force- or deformation-controlled behavior; the second stage assigns the failure mechanism.")
    st.write("Inputs for prediction: aspect ratio, axial load ratio, prism strength (MPa), cross-sectional area (m²), and boundary condition.")
    st.write("Initial stiffness K is obtained from the initial response preceding approximately 50% of maximum lateral resistance, following the notebook's extraction procedure.")

with predict:
    try:
        models, metadata = load_models()
    except Exception as error:
        st.error(f"Prediction models could not be loaded: {error}")
    else:
        with st.form("wall_inputs"):
            a, b = st.columns(2)
            with a:
                aspect = st.number_input("Wall aspect ratio", min_value=0.0001, value=1.0, step=0.01)
                fmc = st.number_input("Prism strength (MPa)", min_value=0.0001, value=5.0, step=0.1)
                boundary = st.selectbox("Boundary condition", ["Fixed-fixed", "Fixed-free"])
            with b:
                axial = st.number_input("Axial load ratio", min_value=0.0, value=0.1, step=0.01)
                an = st.number_input("Cross-sectional area (m²)", min_value=0.0001, value=1.0, step=0.01)
            submitted = st.form_submit_button("Predict Seismic Behavior")
        if submitted:
            row = {"aspect": aspect, "axial": axial, "fmc": fmc, "an": an,
                   "alpha": metadata["boundary_codes"][boundary]}
            x = pd.DataFrame([row], columns=metadata["feature_cols"])
            try:
                raw = np.array([models[t].predict(x)[0] for t in TARGETS], dtype=float)
                if not np.isfinite(raw).all():
                    raise ValueError("The model produced a non-finite output.")
                clipped = np.clip(raw, 0, None)
                if clipped.sum() <= 0:
                    raise ValueError("All model outputs are nonpositive; normalized indices are undefined.")
                percentages = 100 * clipped / clipped.sum()
            except Exception as error:
                st.error(f"Prediction could not be calculated: {error}")
            else:
                for column, name, value in zip(st.columns(4), NAMES, percentages):
                    column.metric(name, f"{value:.1f}%")
                winners = [NAMES[i] for i in np.flatnonzero(percentages == percentages.max())]
                st.success("Dominant predicted behavior: " + ", ".join(winners))
                st.dataframe(pd.DataFrame({"Failure mechanism": NAMES, "Hybridity Index (%)": percentages.round(2)}))
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(percentages, labels=NAMES, colors=COLORS, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Displayed values are rounded; calculations use unrounded percentages.")

with detect:
    st.header("Failure Mode Detection Using Hysteresis Testing Results")
    st.info("Enter drift and return values on the same numerical scale as the notebook database. Strength and K must be consistent so that strength / K has those same drift units. Do not mix drift ratios, percentages, and displacements.")
    st.caption("The notebook's SC_cap correction sets it to zero when return_cap < 0.01 in the database's drift units. Small negative self-centering values between −0.5 and 0 are also set to zero.")
    try:
        rules = load_rules()
    except Exception as error:
        st.error(f"Decision boundaries could not be loaded: {error}")
    else:
        image_path = BASE / "failure_mode_detection.png"
        if image_path.exists():
            try:
                with Image.open(image_path) as diagram:
                    st.image(diagram.copy(), width=650)
            except (OSError, ValueError):
                st.warning("The optional failure-mode diagram could not be opened.")
        with st.expander("Decision boundaries used by this application"):
            for label, rule in rules.items():
                st.write(f"{label}: {equation(rule)}")
                st.caption(f"Negative score: {rule['negative_class']}; nonnegative score: {rule['positive_class']}. Full precision is used internally.")
        with st.form("hysteresis_inputs"):
            a, b = st.columns(2)
            with a:
                return_cap = st.number_input("Return drift at capping", value=0.0, step=0.01)
                return_ult = st.number_input("Return drift at ultimate", value=0.0, step=0.01)
                drift_cap = st.number_input("Capping drift", value=0.0, step=0.01)
                drift_ult = st.number_input("Ultimate drift", value=0.0, step=0.01)
            with b:
                strength_cap = st.number_input("Strength at capping", value=0.0, step=0.01)
                strength_ult = st.number_input("Strength at ultimate", value=0.0, step=0.01)
                stiffness = st.number_input("Initial stiffness K", min_value=0.0, value=1.0, step=0.01)
            submitted = st.form_submit_button("Detect Failure Mode")
        if submitted:
            try:
                sc_cap, sc_ult, elastic_cap, elastic_ult = self_centering(
                    return_cap, return_ult, drift_cap, drift_ult, strength_cap, strength_ult, stiffness)
                values = {"SC_cap": sc_cap, "SC_ult": sc_ult,
                          "cap_point_drift": drift_cap, "ult_point_drift": drift_ult}
                control, score1 = classify(rules["first"], values)
                stage = "force" if control == "Force controlled" else "deformation"
                mode, score2 = classify(rules[stage], values)
            except (ValueError, KeyError) as error:
                st.error(str(error))
            else:
                a, b = st.columns(2)
                a.metric("Behavior control", control)
                b.metric("Detected failure mode", mode)
                st.dataframe(pd.DataFrame({"Parameter": ["Elastic return at capping", "Elastic return at ultimate", "SC_cap", "SC_ult", "First-stage score", "Second-stage score"],
                                           "Value": [elastic_cap, elastic_ult, sc_cap, sc_ult, score1, score2]}))
