import pandas as pd
import streamlit as st
@st.cache_data
def load_results():
    df = pd.read_csv("streamlit_includes/data/result_metrics.csv")
    return df