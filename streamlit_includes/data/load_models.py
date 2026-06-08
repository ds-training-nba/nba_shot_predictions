"""
Thin loader — reads pre-trained artefacts saved by train_and_save.py.
All Streamlit pages import from here; nothing trains at runtime.
"""

import os
import pickle
import joblib
import streamlit as st

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_MODEL_FILES = {
    "XGBoost":             "xgboost_pipeline.joblib",
    "Random Forest":       "random_forest_pipeline.joblib",
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "KNN":                 "knn_pipeline.joblib",
}


def _models_exist() -> bool:
    return os.path.isfile(os.path.join(_MODELS_DIR, "metrics.pkl"))


def _missing_models_error():
    st.error(
        "⚠️ Pre-trained models not found.\n\n"
        "Run the training script **once** before launching the app:\n\n"
        "```bash\n"
        "python streamlit_includes/data/train_and_save.py\n"
        "```"
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def load_pipelines() -> dict:
    if not _models_exist():
        _missing_models_error()
    return {
        name: joblib.load(os.path.join(_MODELS_DIR, fname))
        for name, fname in _MODEL_FILES.items()
    }


@st.cache_resource(show_spinner=False)
def load_metrics() -> dict:
    """Returns dict with keys: results_df, roc_data, cm_data"""
    if not _models_exist():
        _missing_models_error()
    with open(os.path.join(_MODELS_DIR, "metrics.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_splits() -> dict:
    """Returns dict with keys: X_test, y_test"""
    if not _models_exist():
        _missing_models_error()
    with open(os.path.join(_MODELS_DIR, "splits.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_feature_importance():
    """Returns DataFrame with columns: feature, importance"""
    if not _models_exist():
        _missing_models_error()
    with open(os.path.join(_MODELS_DIR, "feature_importance.pkl"), "rb") as f:
        return pickle.load(f)
