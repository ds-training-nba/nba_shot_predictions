import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.load_models import load_metrics, load_splits, load_feature_importance
from streamlit_includes.data.results import load_results


def render():
    df = load_results()
    st.markdown(
        '<div class="section-title">📉 Model Evaluation & Analysis</div>',
        unsafe_allow_html=True
    )
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.dataframe(df)
    with col_right:
        st.markdown(
            "## Conclusion"

        )
        st.markdown(

            "Simple Lookup Table made from Player and Main Action Type already delivers comparable results to"
            " complex ML Models"
        )