import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datasets import load_dataset

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.data_overview import load_overview


def render():
    df = load_overview()

    """Data Presentation & Architecture"""
    st.markdown('<div class="section-title">📊 Data Presentation</div>', unsafe_allow_html=True)

    st.markdown("### Dataset Overview")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Dataset Specifications")

    st.markdown("""
              **Volume & Scale**
              - Total Shots: 8,208,626
              - Date Range: 1996 to 2025 years
              - Unique Players: 2,742
              - Unique Games: 36,656
              - Unique Teams: 32
              """)
