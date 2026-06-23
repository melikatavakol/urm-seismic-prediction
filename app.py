# ---------------------------------------------------------------
# STREAMLIT WEBSITE FOR URM WALL SEISMIC BEHAVIOR PREDICTION
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, UnidentifiedImageError


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
# XGBOOST PREDICTION FUNCTION
# ---------------------------------------------------------------

def predict_failure_indices(input_data):

    input_df = pd.DataFrame([input_data], columns=feature_cols)

    predictions = {}

    for target in target_cols:
        predictions[target] = models[target].predict(input_df)[0]

    return predictions


# ---------------------------------------------------------------
# HYSTERESIS-BASED FAILURE MODE DETECTION FUNCTION
# ---------------------------------------------------------------

def detect_failure_mode_from_hysteresis(
    delta_return_cap,
    delta_return_ultimate,
    drift_cap,
    drift_ultimate,
    strength_cap,
    strength_ultimate,
    K
):

    # Elastic return displacement at capping point
    delta_elastic_return_cap = drift_cap - (strength_cap / K)

    # Elastic return displacement at ultimate point
    delta_elastic_return_ultimate = drift_ultimate - (strength_ultimate / K)

    # Self-centering factors
    SC_cap = delta_return_cap / delta_elastic_return_cap

    # This follows your provided formula:
    # SC_ultimate = delta_return_ultimate / delta_elastic_return_cap
    SC_ultimate = delta_return_ultimate / delta_elastic_return_cap

    # First-level control criterion
    control_check = drift_ultimate - 0.92

    if control_check < 0:

        behavior_control = "Force-controlled"

        # Force-controlled failure-mode criterion
        force_classifier_value = (
            2.74 * drift_cap
            + 3.95 * drift_ultimate
            - 3.35
        )

        if force_classifier_value < 0:
            failure_mode = "Diagonal"
        else:
            failure_mode = "Toe crushing"

        deformation_classifier_value = None

    else:

        behavior_control = "Deformation-controlled"

        # Deformation-controlled failure-mode criterion
        deformation_classifier_value = (
            3.19 * SC_cap
            + SC_ultimate
            - 2.13
        )

        if deformation_classifier_value < 0:
            failure_mode = "Rocking"
        else:
            failure_mode = "Sliding"

        force_classifier_value = None

    results = {
        "delta_elastic_return_cap": delta_elastic_return_cap,
        "delta_elastic_return_ultimate": delta_elastic_return_ultimate,
        "SC_cap": SC_cap,
        "SC_ultimate": SC_ultimate,
        "control_check": control_check,
        "force_classifier_value": force_classifier_value,
        "deformation_classifier_value": deformation_classifier_value,
        "behavior_control": behavior_control,
        "failure_mode": failure_mode
    }

    return results


# ---------------------------------------------------------------
# WEBSITE TITLE
# ---------------------------------------------------------------

st.title("Seismic Behavior Prediction of URM Walls")

st.markdown(
    """
    This web application predicts and detects the seismic behavior of unreinforced masonry 
    walls using two complementary approaches: an XGBoost-based machine learning model based 
    on wall design characteristics, and a rule-based failure-mode detection procedure based 
    on hysteresis testing results.
    """
)


# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------

tab_about, tab_predict, tab_detect = st.tabs(
    [
        "ℹ️ About the Model",
        "🔮 Predict Seismic Behavior",
        "📈 Failure Mode Detection from Hysteresis Test"
    ]
)


# ---------------------------------------------------------------
# TAB 1: ABOUT THE MODEL
# ---------------------------------------------------------------

with tab_about:
    
    st.header("About the Model")

    st.markdown(
        """
        This web application is developed to evaluate the seismic behavior of 
        unreinforced masonry walls using two complementary approaches.

        The first approach is an **XGBoost-based machine learning model**. XGBoost stands for 
        **Extreme Gradient Boosting**. It is a tree-based machine learning algorithm that builds 
        an ensemble of decision trees sequentially. Each new tree attempts to reduce the prediction 
        error of the previous trees, allowing the model to capture nonlinear relationships between 
        input variables and target responses.

        In this application, the XGBoost model is used to predict the contribution of different 
        failure modes to the overall wall behavior. Notably, the model requires only the wall design 
        characteristics as input variables.

        The input parameters for the XGBoost-based prediction module are:

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
        | Rocking Hybridity Index (%) | Contribution of rocking behavior |
        | Sliding Hybridity Index (%) | Contribution of sliding behavior |
        | Diagonal Hybridity Index (%) | Contribution of diagonal cracking behavior |
        | Toe Crushing Hybridity Index (%) | Contribution of toe crushing behavior |

        A larger Hybridity Index indicates a larger contribution of that failure mechanism 
        to the predicted cyclic behavior of the URM wall.

        The second approach is a **hysteresis-based failure mode detection procedure**. This module 
        uses parameters extracted from hysteresis testing results, including return displacement, 
        drift capacity, ultimate drift, strength values, and initial elastic stiffness. The procedure 
        calculates elastic-return and self-centering parameters, then classifies the wall response as 
        either force-controlled or deformation-controlled. Based on this classification, the dominant 
        failure mode is detected as diagonal cracking, toe crushing, rocking, or sliding.

        For the hysteresis-based module, the initial elastic stiffness **K** is defined as the slope 
        of the data points preceding 50% of the maximum lateral resistance.
        """
    )


# ---------------------------------------------------------------
# TAB 2: XGBOOST-BASED PREDICTION PAGE
# ---------------------------------------------------------------

with tab_predict:

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
        plt.close(fig)

        st.info(
            """
            The reported values represent the normalized contribution of each failure mechanism 
            to the overall cyclic response of the wall. These values are expressed as 
            Hybridity Index (%).
            """
        )


# ---------------------------------------------------------------
# TAB 3: HYSTERESIS-BASED FAILURE MODE DETECTION
# ---------------------------------------------------------------

with tab_detect:
    
    st.header("Failure Mode Detection Using Hysteresis Testing Results")

    st.markdown(
        """
        This module detects the dominant failure mode of a URM wall using parameters extracted 
        from hysteresis testing results. The procedure first determines whether the wall response 
        is force-controlled or deformation-controlled, and then assigns the corresponding failure mode.
        """
    )

    # ---------------------------------------------------------------
    # FAILURE MODE DETECTION IMAGE
    # ---------------------------------------------------------------

    image_path = Path(__file__).parent / "failure_mode_detection.png"

    try:
        if image_path.exists():

            with Image.open(image_path) as img:
                failure_mode_image = img.copy()

            st.image(
                failure_mode_image,
                caption="Decision boundaries for hysteresis-based failure mode detection",
                width=650
            )

        else:
            st.error(
                f"Image file not found at: {image_path}. "
                "Make sure 'failure_mode_detection.png' is uploaded to GitHub "
                "in the same folder as app.py."
            )

    except UnidentifiedImageError:
        st.error(
            "The file 'failure_mode_detection.png' exists, but it is not a valid PNG image. "
            "Convert the original BMP image to PNG instead of only renaming the file extension."
        )

    st.subheader("Enter Hysteresis-Based Parameters")

    col1, col2 = st.columns(2)

    with col1:

        delta_return_cap = st.number_input(
            "δ_return,cap",
            value=0.00,
            step=0.01,
            help="Return displacement/drift at the capping point."
        )

        delta_return_ultimate = st.number_input(
            "δ_return,ultimate",
            value=0.00,
            step=0.01,
            help="Return displacement/drift at the ultimate point."
        )

        drift_cap = st.number_input(
            "drift_cap",
            value=0.00,
            step=0.01,
            help="Drift at the capping point."
        )

        drift_ultimate = st.number_input(
            "drift_ultimate",
            value=0.00,
            step=0.01,
            help="Drift at the ultimate point."
        )

    with col2:

        strength_cap = st.number_input(
            "Strength_cap",
            value=0.00,
            step=0.01,
            help="Lateral strength at the capping point."
        )

        strength_ultimate = st.number_input(
            "Strength_ultimate",
            value=0.00,
            step=0.01,
            help="Lateral strength at the ultimate point."
        )

        K = st.number_input(
            "Initial elastic stiffness, K",
            min_value=0.0,
            value=1.00,
            step=0.01,
            help=(
                "Initial elastic stiffness defined as the slope of the data points "
                "preceding 50% of the maximum lateral resistance."
            )
        )

    st.divider()

    if st.button("Detect Failure Mode from Hysteresis Test"):

        if K <= 0:
            st.error("Initial elastic stiffness K must be greater than zero.")

        else:

            delta_elastic_return_cap_check = drift_cap - (strength_cap / K)

            if delta_elastic_return_cap_check == 0:
                st.error(
                    "δ_elastic return,cap is zero. SC_cap and SC_ultimate cannot be calculated "
                    "because division by zero would occur."
                )

            else:

                hysteresis_results = detect_failure_mode_from_hysteresis(
                    delta_return_cap=delta_return_cap,
                    delta_return_ultimate=delta_return_ultimate,
                    drift_cap=drift_cap,
                    drift_ultimate=drift_ultimate,
                    strength_cap=strength_cap,
                    strength_ultimate=strength_ultimate,
                    K=K
                )

                st.header("Failure Mode Detection Results")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Behavior control type",
                    hysteresis_results["behavior_control"]
                )

                c2.metric(
                    "Detected failure mode",
                    hysteresis_results["failure_mode"]
                )

                if hysteresis_results["failure_mode"] in ["Diagonal", "Toe crushing"]:
                    st.warning(
                        f"The wall is classified as **{hysteresis_results['behavior_control']}**, "
                        f"and the detected failure mode is **{hysteresis_results['failure_mode']}**."
                    )
                else:
                    st.success(
                        f"The wall is classified as **{hysteresis_results['behavior_control']}**, "
                        f"and the detected failure mode is **{hysteresis_results['failure_mode']}**."
                    )

                st.subheader("Calculated Parameters")

                calculated_df = pd.DataFrame({
                    "Parameter": [
                        "δ_elastic return,cap",
                        "δ_elastic return,ultimate",
                        "SC_cap",
                        "SC_ultimate",
                        "drift_ultimate - 0.92"
                    ],
                    "Value": [
                        hysteresis_results["delta_elastic_return_cap"],
                        hysteresis_results["delta_elastic_return_ultimate"],
                        hysteresis_results["SC_cap"],
                        hysteresis_results["SC_ultimate"],
                        hysteresis_results["control_check"]
                    ]
                })

                st.dataframe(
                    calculated_df,
                    use_container_width=True
                )

                st.subheader("Decision Function Values")

                decision_rows = []

                if hysteresis_results["force_classifier_value"] is not None:
                    decision_rows.append(
                        {
                            "Decision function": "2.74 drift_cap + 3.95 drift_ultimate - 3.35",
                            "Value": hysteresis_results["force_classifier_value"],
                            "Rule": "< 0 → Diagonal, otherwise → Toe crushing"
                        }
                    )

                if hysteresis_results["deformation_classifier_value"] is not None:
                    decision_rows.append(
                        {
                            "Decision function": "3.19 SC_cap + SC_ultimate - 2.13",
                            "Value": hysteresis_results["deformation_classifier_value"],
                            "Rule": "< 0 → Rocking, otherwise → Sliding"
                        }
                    )

                decision_df = pd.DataFrame(decision_rows)

                st.dataframe(
                    decision_df,
                    use_container_width=True
                )

                st.info(
                    """
                    The classification is based on threshold functions calculated from 
                    hysteresis-response parameters. The same unit system should be used 
                    consistently for drift, displacement, strength, and stiffness.
                    """
                )
