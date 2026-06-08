import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players


def render():
    """Feature Engineering"""
    st.markdown('<div class="section-title">⚡ Feature Engineering</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["New Features", "Importance", "Rationale"])

    with tab1:
        st.markdown("""
        ### Engineered Features (25 new features created)

        #### Game Situation Features
        - `IS_HOME`: Binary home/away indicator
        - `scoreMargin`: Difference before shot
        - `IsClutchTime`: Last 5 min + score within 5
        - `TimeRemainingInGame`: Time in seconds left
        - `IsOvertime`: Overtime indicator


        #### Shot Geometry Features
        - `shot_angle`: Angle to basket (degrees)
        - `ANGLE_SECTOR`: Categorized angle (front/side/behind)
        - `SHOT_DISTANCE`: Euclidean distance

        #### Defensive Features
        - `OPPONENT_INTERFERED`: Boolean contact

        """)

    with tab2:
        st.markdown("### Feature Importance Ranking")

    with tab3:
        st.markdown("""
        ### Rationale for Feature Engineering

        """)
