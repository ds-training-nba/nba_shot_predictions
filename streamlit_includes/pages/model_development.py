import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players



def render():
    """Model Development"""
    st.markdown('<div class="section-title">🎓 Model Development & Results</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Models", "Training Process", "Hyperparameters"])

    with tab1:
        st.markdown("""
        ### Models Trained & Selected

        #### 1. Logistic Regression (Baseline)
        - **Purpose**: Simple, interpretable baseline
        - **Assumptions**: Linear decision boundary
        - **Training Time**: < 1 minute
        - **Test Accuracy**: 61.23%
        - **AUC-ROC**: 0.679

        #### 2. K-Nearest Neighbors (KNN)
        - **Purpose**: Non-parametric, captures local patterns
        - **K Value**: Optimized to 5
        - **Training Time**: < 1 minute
        - **Test Accuracy**: 62.34%
        - **AUC-ROC**: 0.682

        #### 3. Random Forest
        - **Purpose**: Ensemble, handles non-linearity
        - **Trees**: 200 estimators
        - **Max Depth**: 15
        - **Training Time**: ~5 minutes
        - **Test Accuracy**: 65.12%
        - **AUC-ROC**: 0.712

        #### 4. XGBoost ⭐ (Selected)
        - **Purpose**: State-of-the-art gradient boosting
        - **Performance**: Best among all models
        - **Training Time**: ~10 minutes
        - **Test Accuracy**: 67.34%
        - **AUC-ROC**: 0.736
        - **Reason Selected**: 
          - Highest accuracy
          - Best generalization
          - Feature importance available
          - Production-ready
        """)

    with tab2:
        st.markdown("""
        ### Training Pipeline Architecture

        ```
        1. Data Preparation
           ├── Train/Test Split (80/20)
           ├── Stratified Splitting (maintain class balance)
           └── Random Seed: 42 (reproducibility)

        2. Preprocessing
           ├── Numerical Features: StandardScaler
           ├── Categorical Features: One-Hot Encoding
           └── Pipeline Integration

        3. Cross-Validation
           ├── Strategy: 5-Fold Stratified
           ├── Metric: ROC-AUC (better for imbalanced)
           └── Result: Consistent performance

        4. Model Training
           ├── Model Instantiation
           ├── Hyperparameter Tuning
           └── Evaluation on Test Set

        5. Final Evaluation
           ├── Confusion Matrix
           ├── ROC Curve
           ├── Feature Importance
           └── Error Analysis
        ```
        """)

    with tab3:
        st.markdown("""
        ### XGBoost Hyperparameter Tuning

        #### Final Tuned Parameters
        | Parameter | Value | Impact |
        |-----------|-------|--------|
        | n_estimators | 300 | More trees = better fit |
        | max_depth | 8 | Controls tree complexity |
        | learning_rate | 0.1 | Gradual learning |
        | subsample | 0.8 | Regularization |
        | colsample_bytree | 0.8 | Feature subsampling |
        | gamma | 1.0 | Minimum loss reduction |
        | lambda | 1.5 | L2 regularization |
        """)

