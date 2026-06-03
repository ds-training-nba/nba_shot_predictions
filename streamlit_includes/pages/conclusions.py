import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players


def render():
    """Conclusions & Recommendations"""
    st.markdown('<div class="section-title">💡 Conclusions & Business Recommendations</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Key Findings", "Done"])

    with tab1:
        st.markdown("""
        ### Key Findings from Analysis

        #### 1. Shot Distance is Dominant Factor
        - **Importance**: 18.5% of model predictions
        - **Finding**: Each additional foot reduces success by 1-2%
        - **Implication**: Favor closer shots for higher probability

        #### 2. Shot Type Matters Significantly
        - **Importance**: 16.2% of model predictions
        - **Finding**: 
          - Free throws: ~75% success
          - 2-pointers: ~48% success
          - 3-pointers: ~35% success
        - **Implication**: Strategic game selection (when to shoot 3s)

        #### 3. Defender Proximity is Critical
        - **Importance**: 12.3% of model predictions
        - **Finding**: Tight defense reduces success by 15-20%
        - **Implication**: Emphasis on defensive quality

        #### 4. Player Skill is Most Consistent
        - **Importance**: 11.8% of model predictions
        - **Finding**: Top players maintain +8% advantage
        - **Implication**: Star power translates to results

        #### 5. Home Court Advantage Real
        - **Importance**: 8.5% of model predictions
        - **Finding**: +2.3% success rate at home
        - **Implication**: Venue matters for shot selection

        #### 6. Model Performance Achieved
        - **Accuracy**: 67.3%
        - **AUC-ROC**: 0.736
        - **F1-Score**: 0.601
        - **Status**: ✓ Exceeds 65% target
        """)

    with tab2:
        st.markdown("""
        ### What have been done during the Project

        #### Data Preparation ✓
        - Clean pipeline for handling missing values
        - Domain-driven feature engineering
        - Validation preventing data leakage

        #### Model Selection ✓
        - Compared 4 different algorithms
        - Clear evaluation metrics
        - Selected best performer (XGBoost)

        #### Feature Engineering ✓
        - Created meaningful basketball features
        - 18 new features in addition to 35 base features
        - Maintained interpretability

        #### Documentation ✓
        - Clear decision justification
        - Reproducible code with random seeds
        """)
