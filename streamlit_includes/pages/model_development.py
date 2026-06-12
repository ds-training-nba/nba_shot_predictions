import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players



def render():
    """Model Development"""
    st.markdown('<div class="section-title">🎓 Model Development</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Models", "Training Process"])

    with tab1:
        st.markdown("""
        ### Models Trained & Selected

        #### 1. LookUp Table (Baseline)
        - **Purpose**: Simple, interpretable baseline
        - **Assumptions**: Enough information in Player/ACTION_TYPE combination
        - **Conclusion**: Surprisingly good performance with simple input + algorithm


        #### 2. XGBoost ⭐ (Main)
        - **Purpose**: State-of-the-art, ideal for tabular data
        - **Conclusion**: Met the expectations

        #### 3. Random Forest (for comparison)
        - **Purpose**: Ensemble, handles non-linearity
        - **Conclusion**: Somewhat disappointing, but no time for debugging

        #### 4. Decision Tree (for comparison)
        - **Purpose**: Interpretable algorithm with non-linearities
        - **Conclusion**: Second best
       
       #### 5. Logistic Regression (for comparison)
       - **Purpose**: Test capabilities of linear algorithm
       - **Conclusion**: Slightly better than baseline with only 2 variables
        """)

    with tab2:
        st.markdown("""
        ### Training Pipeline Architecture (Particularities)

       
        #### Data Preparation
           Test/Train Split: Grouped Split by Games (80/20) (No temporal split taken into account for our use case)

        #### Encoding according to model demands: 
        - One Hot (except XGBoost)
        - std scale (LogisticRegression only)


        #### Hyperparameter Search
        - 3-Fold RandomizedSearchCV because of big amount of data and hyperparameter space. 
        - Metric: neg_brier_score


        #### Final Evaluation
        - brier score (decomposition)
        - ROC-AUC
        - accuracy
        - (others got logged)
        """)



