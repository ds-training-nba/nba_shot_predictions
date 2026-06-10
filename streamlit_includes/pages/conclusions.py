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

    tab1, tab2 = st.tabs(["Key Findings", "Recommendations"])
    with tab1:
        st.markdown("""
        ### Key Findings
    
        #### 1. NBA shot prediction has its intrinsic difficulties
        - **Accurate data** on how the game and especially defender situation was is **hard** to get 
        - There is a high amount of **noise** in the target data
    
        #### 2. Quantitative metrics gain little for complex ML models
        - XGBoost achieves only 2% better accuracy and a brier score difference of 0.008 than a **simple baseline model**
        - Nevertheless, a complex XGBoost model might help **better in making decisions**, because it recognizes connections,
        even if it **doesn't look much more accurate through the noise**
    
        #### 3. Focus should be on Probabilities
        - **The shift in probabilities** when changing variables should reflect the reality of the game
        - not the individual "SHOT_MADE" prediction
        - That way, our virtual clients can **plan meaningful alternatives**
    
        #### 4. Results confirm scientific literature
        - We hit a scientifically estimated ceiling of 63-65% accuracy (considering field goals only)
        
   
        """)
    with tab2:
        st.markdown("""
        ### Recommendations
        #### 1. Further analysis and improvement of model behaviour. 
        - Regularizing rare ACTION_TYPES
        - Finding more semantic errors like this
        - Might not improve metrics, but reflect the behaviour more realistically
        - investigate role of player age (as showing in the demo app)

        #### 2. Further development of demo app.
        - it's a **DEMO**
        - real business decisions can only be concluded with the generation of **realistic alternatives**
        - further develop a generator of realistic alternatives, be it **data** driven or driven by **domain knowledge**
        """)


