import pandas as pd
import streamlit as st

@st.cache_data
def load_overview():
    df = pd.read_csv("streamlit_includes/data/Data Audit NBA - DataOverview_v2.csv", skiprows=2).iloc[:85]
    return df