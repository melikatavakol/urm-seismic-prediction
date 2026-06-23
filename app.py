# ---------------------------------------------------------------
# STREAMLIT WEBSITE FOR URM WALL SEISMIC BEHAVIOR PREDICTION
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------

st.set_page_config(
    page_title="URM Wall Seismic Behavior Prediction",
    page_icon="🏗️",
    layout="wide"
)


# ---------------------------------------------------------------
# LOAD SAVED MODELS
# ---------------------------------------------------------------

@st.cache_resource
def load_models():

    metadata = joblib.load("saved_models/model_metadata.joblib")

    feature_cols = metadata["feature_cols"]
    target_cols = metadata["target_cols"]

    models = {}

    for target in target_cols:
        model_path = f"saved_models/{target}_model.joblib"
        models[target] = joblib.load(model_path)

    return models, feature_cols, target_cols


models, feature_cols, target_cols = load_models()


# ---------------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------------

def predict_failure_indices(input_data):

    input_df = pd.DataFrame([input_data], columns=feature_cols)

    predictions = {}

    for target in target_cols:
        predictions[target] = models[target].predict(input_df)[0]

    return predictions


# ---------------------------------------------------------------
# WEBSITE TITLE
# ---------------------------------------------------------------

st.title("Seismic Behavior Prediction of URM Walls")

st.markdown(
    """
    This web application predicts the seismic behavior of unreinforced masonry walls 
    using an XGBoost-based machine learning model. The model estimates the normalized 
    contribution of rocking, sliding, diagonal cracking, and toe crushing mechanisms.
    """
)


# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------

tab1, tab2 = st.tabs(["Predict Seismic Behavior", "About the Model"])


# ---------------------------------------------------------------
# TAB 1: PREDICTION PAGE
# ---------------------------------------------------------------

with tab1:

    st.header("Enter Masonry Wall Inputs")

    col1, col2 = st.columns(2)

    with col1:

        aspect = st.number_input(
            "Wall aspect ratio",
            min_value=0.0,
            value=1.00,
            step=0.01,
            help="Wall height-to-length ratio. This input is dimensionless."
        )

        fmc = st.number_input(
            "Prism strength (MPa)",
            min_value=0.0,
            value=5.00,
            step=0.10,
            help="Masonry prism compressive strength in MPa."
        )

        boundary_condition = st.selectbox(
            "Boundary condition",
            options=["Fixed-fixed", "Fixed-free"],
            help="Fixed-fixed is coded as 1, and fixed-free is coded as 0."
        )

        # Convert selected boundary condition to numerical alpha value
        if boundary_condition == "Fixed-fixed":
            alpha = 1
        else:
            alpha = 0

    with col2:

        axial = st.number_input(
            "Axial load ratio",
            min_value=0.0,
            value=0.10,
            step=0.01,
            help="Normalized axial load ratio. This input is dimensionless."
        )

        an = st.number_input(
            "Cross-sectional area (m²)",
            min_value=0.0,
            value=1.00,
            step=0.01,
            help="Wall cross-sectional area in square meters."
        )

    input_data = {
        "aspect": aspect,
        "fmc": fmc,
        "alpha": alpha,
        "axial": axial,
        "an": an
    }

    st.markdown("### Numerical values used by the model")

    st.write(
        {
            "aspect": aspect,
            "fmc": fmc,
            "alpha": alpha,
            "axial": axial,
            "an": an
        }
    )

    st.divider()

    if st.button("Predict Seismic Behavior"):

        predictions = predict_failure_indices(input_data)

        results_df = pd.DataFrame({
            "Failure mechanism": [
                "Rocking",
                "Sliding",
                "Diagonal cracking",
                "Toe crushing"
            ],
            "Raw model output": [
                predictions["Rocking_index"],
                predictions["Sliding_index"],
                predictions["Diagonal_index"],
                predictions["Toe_crushing_index"]
            ]
        })

        # Remove negative values if any
        results_df["Raw model output"] = results_df["Raw model output"].clip(lower=0)

        # Calculate normalized Hybridity Index
        total_index = results_df["Raw model output"].sum()

        if total_index > 0:
            results_df["Hybridity Index (%)"] = (
                results_df["Raw model output"] / total_index * 100
            ).round(0).astype(int)
        else:
            results_df["Hybridity Index (%)"] = 0

        dominant_mode = results_df.loc[
            results_df["Hybridity Index (%)"].idxmax(),
            "Failure mechanism"
        ]

        st.header("Prediction Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rocking HI",
            f"{results_df.loc[0, 'Hybridity Index (%)']}%"
        )

        c2.metric(
            "Sliding HI",
            f"{results_df.loc[1, 'Hybridity Index (%)']}%"
        )

        c3.metric(
            "Diagonal HI",
            f"{results_df.loc[2, 'Hybridity Index (%)']}%"
        )

        c4.metric(
            "Toe Crushing HI",
            f"{results_df.loc[3, 'Hybridity Index (%)']}%"
        )

        st.success(f"Dominant predicted behavior: **{dominant_mode}**")

        st.subheader("Detailed Results")

        st.dataframe(
            results_df[
                [
                    "Failure mechanism",
                    "Hybridity Index (%)"
                ]
            ],
            use_container_width=True
        )

        # -------------------------------------------------------
        # CIRCULAR PIE CHART
        # -------------------------------------------------------

        st.subheader("Predicted Failure-Mechanism Portions")

        # Plot font settings
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 15
        })

        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        ax.pie(
            results_df["Hybridity Index (%)"],
            labels=results_df["Failure mechanism"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={
                "fontsize": 15,
                "fontname": "Times New Roman"
            }
        )

        ax.set_title(
            "Hybridity Index (%)",
            fontsize=15,
            fontname="Times New Roman"
        )

        ax.axis("equal")

        st.pyplot(fig)

        st.info(
            """
            The reported values represent the normalized contribution of each failure mechanism 
            to the overall cyclic response of the wall. These values are expressed as 
            Hybridity Index (%).
            """
        )


# ---------------------------------------------------------------
# TAB 2: ABOUT THE MODEL
# ---------------------------------------------------------------

with tab2:

    st.header("About the Model")

    st.markdown(
        """
        This web application is developed to predict the seismic behavior of 
        unreinforced masonry walls using a machine learning model.

        The predictive model is based on **XGBoost**, which stands for 
        **Extreme Gradient Boosting**. XGBoost is a tree-based machine learning algorithm 
        that builds an ensemble of decision trees sequentially. Each new tree attempts 
        to reduce the prediction error of the previous trees, allowing the model to capture 
        nonlinear relationships between input variables and target responses.

        In this application, the XGBoost model is used to predict the contribution of different
        failure modes to the overal wall behavior.Notably, the model requires only the wall design
        characteristics as input variables.
        The input parameters are:

        | Input variable | Unit / coding | Description |
        |---|---|---|
        | Wall aspect ratio | Dimensionless | Geometric slenderness of the wall |
        | Prism strength | MPa | Masonry prism compressive strength |
        | Boundary condition | Fixed-fixed = 1, Fixed-free = 0 | Support condition of the wall |
        | Axial load ratio | Dimensionless | Normalized axial load applied to the wall |
        | Cross-sectional area | m² | Wall sectional area |

        The predicted outputs are:

        | Output | Meaning |
        |---|---|
        | Rocking Hybridity Index (%) | contribution of rocking behavior |
        | Sliding Hybridity Index (%) | contribution of sliding behavior |
        | Diagonal Hybridity Index (%) | contribution of diagonal cracking behavior |
        | Toe Crushing Hybridity Index (%) | contribution of toe crushing behavior |

        A larger Hybridity Index indicates a larger contribution of that failure mechanism 
        to the predicted cyclic behavior of the URM wall.
        """
    )
