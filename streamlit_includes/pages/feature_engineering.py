import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players


def render():
    """Feature Engineering"""
    st.markdown('<div class="section-title">⚡ Feature Engineering</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["New Features", "Target Leak"])

    with tab1:
        st.markdown("""
        ### Used original Features 
        - `ACTION_TYPE`: shot technique category
        - `SHOT_TYPE`: 1pt/2pt/3pt category
        - `SHOT_DISTANCE`: euclidean distance in ft
        - `PLAYER_NAME`: The acting player category
        
        ### Engineered Features 

        #### Game Situation Features
        - `IS_HOME`: Binary home/away indicator
        - `scoreMarginBeforeShot`: Difference before shot (from player's perspective)
        - `IsClutchTime`: Last 5 min + score within 5
        - `TimeRemainingInGame`: Time in seconds left
        - `TimeRemainingInPeriod`: Time in seconds left
        - `IsOvertime`: Overtime indicator

        
        #### Shot Related Features
        - `ABS_ANGLE`: absolute angle in degrees (0 = center line through baskets)
        - `ANGLE_SECTOR`: Binned angle (front/side/behind)
        - `MAIN_ACTION_TYPE`: Grouped ACTION_TYPE variable
        
        #### Defensive Features
        - `OPPONENT_INTERFERED`: Boolean contact (not used because of target variable leak)
        
        #### Player Related Features
        - `player_age` from year and publicly available birth dates
        - `best_age` best average year in shot precision derived from training data

        """)

    with tab2:
        st.markdown("### Target Leak")
        st.markdown("""
        #### Suspicious Columns
        The following columns raised our suspicion, because using them improved our metrics too well: 
        
        - `OPPONENT_INTERFERED` identified clearly: No _misses_ with this flag
        - `ACTION_TYPE` suspicious modalities with inexplicably low hit rate were randomly re-distributed
        
        """)

