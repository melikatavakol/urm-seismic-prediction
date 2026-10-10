# URM Seismic Prediction

Streamlit application for estimating failure-mechanism Hybridity Indices from
wall properties and detecting failure modes from hysteresis-response parameters.

## Prediction

The five inputs are aspect ratio, axial load ratio, prism strength (MPa),
cross-sectional area (m²), and boundary condition. Four saved XGBoost regressors
predict rocking, sliding, diagonal cracking, and toe-crushing indices. Negative
outputs are clipped to zero and normalized to percentages. An all-zero result
is reported as undefined rather than assigned a dominant failure mode.

## Hysteresis detection

The application uses the linear SVM boundaries fitted to the notebook's
two-stage clustering labels. The first stage uses ultimate drift; the
force-controlled branch uses capping and ultimate drift; the
deformation-controlled branch uses capping and ultimate self-centering.

SC_cap = return_cap / (drift_cap - strength_cap / K)

SC_ult = return_ult / (drift_ult - strength_ult / K)

The notebook corrections for small negative self-centering values and
return_cap < 0.01 are retained. Use exactly the notebook database's drift scale.
K must use the matching strength-per-drift units.

## Export and run

1. Regenerate `Results.xlsx` and `augDataSet.xlsx` using the notebook.
2. Run `export_website_files.py` in the same Python environment and working
   directory as those two workbooks. It creates `website_export/`.
3. Copy this README and `app.py` into `website_export/`.
4. In `website_export/`, run `python -m pip install -r requirements.txt`, then
   `python -m streamlit run app.py`.
5. Compare predictions with the generated `prediction_check.csv`.

The export uses learning_rate=0.01, matching the supplied cross-validation
code. Its SHAP cells use 0.1 and therefore describe a differently configured
model. Reconcile those cells before interpreting SHAP as an explanation of
this deployed model.

## Render

Build command: `pip install -r requirements.txt`

Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`

Upload the generated `.python-version` and matching `requirements.txt`.
If the service has a `PYTHON_VERSION` environment variable, keep it consistent
with the version in the generated file.

## Runtime files

| Path | Content |
| --- | --- |
| app.py | Streamlit inputs, predictions, self-centering calculations, results |
| requirements.txt | Package versions generated from the export environment |
| .python-version | Python version used for model export |
| saved_models/model_metadata.joblib | Feature/target order, boundary coding, training settings, package versions |
| saved_models/Rocking_index_model.joblib | Trained rocking regressor |
| saved_models/Sliding_index_model.joblib | Trained sliding regressor |
| saved_models/Diagonal_index_model.joblib | Trained diagonal regressor |
| saved_models/Toe_crushing_index_model.joblib | Trained toe-crushing regressor |
| saved_models/hysteresis_rules.json | Full-precision SVM coefficients, intercepts, feature order, class mapping |
| failure_mode_detection.png | Optional diagram; include only after updating its boundaries |

Training workbooks and SHAP/plotting routines are not needed by the running app.
