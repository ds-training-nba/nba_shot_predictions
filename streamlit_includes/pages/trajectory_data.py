import streamlit as st
import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.utils import estimator_html_repr
import joblib
import streamlit.components.v1 as components
from  streamlit_includes.pages.trajectory_demo import render as render_trajectory_demo

import base64

@st.cache_resource
def load_xgb_pipeline():
    return joblib.load("models/xgboost_pipeline_tracking.pkl")

def show_sklearn_pipeline():
    st.markdown("**XGBoost Pipeline:**")

    pipeline = load_xgb_pipeline()

    set_config(display="diagram")

    html = estimator_html_repr(pipeline)

    components.html(
        html,
        height=300,
        scrolling=True
    )


def render():
    """Trajectory Data Page"""

    st.markdown(
        '<div class="section-title">Trajectory Data</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data overview",
        "Preprocessing + Feature Engineering",
        "Models + Results",
        "Demo"
    ])

    # -----------------------------------------------
    # TAB 1: DATA OVERVIEW
    # -----------------------------------------------

    with tab1:
        st.markdown("""
        ### 🏀 Tracking Dataset Overview

        ---

        #### Motivation
                    
        Experiments with the original dataset showed an underfitting of the target data. 
                    
        Tests with different feature combinations and model types did not seem to provide big improvements.
                    
        The biggest potential for improvements seemed to be to add additional features to better represent the game situation.

        The goal of this dataset is to capture information that is not fully represented by standard play-by-play features, such as:
                    
        ##### Why Tracking Data?

        Traditional shot data usually contains:
        """)
        st.info("""
        - shot distance
        - shot angle
        - shot type
        - player
        - game context
        """)

        st.markdown("""

        Tracking data adds spatial and temporal context:
        """)
        st.info("""
        - how close the defender is
        - whether the defender is closing out
        - how teammates are spaced
        - whether the shooter is moving
        - whether multiple defenders are near the shooter
        """)

        st.markdown("""            
        ---

        #### Available Tracking Data

        The trajectory data consists of **SportVU data** of the **2015/2016 NBA season** (Last season data was publicly available)
                    
        SportVU: **Optical player tracking system**, to cature **spatial coordinates of all players** (x/y) and the ball (x/y/z).
                    
        Data is available in a frequency of **25 Hz**
        """)

        st.info("""            
        **Improvements**:
        - high frequency data (25 Hz) of the whole game situation
        - positions of all players, not just the shooter
        - player movements
                    
        **Limitations**:
        - Only one season
        - Not all shots (e.g. no free throws)

        """)

        st.markdown("""
        ---

        #### Basic Format
                    
        ##### File Structure
                    
        Avalailable Data:
        - **1 json File per game** with tracking data of all plays
        - **1 csv File per game** with tabular **play-by-play** data (same as in original dataset)
        - **1 csv File** with shotdetails (same as in original dataset) + **exact shot times**
            - (shot time was calculated by dataset creator via the time of highest ball acceleration)
                    
        ##### Data structure   

        A **game event** consists of some player information (ids, names, position) and a list of **tracking moments** (sampled in 25 Hz)
                    
        A raw tracking moment contains:

        ```python
        [
            quarter,
            game_clock,
            shot_clock,
            ...
            [
                ball,
                player1,
                player2,
                ...
                player10
            ]
        ]
        ```

        Each entity contains coordinates:

        ```python
        ball = [x, y, z]
        player = [team_id, player_id, x, y, z]
        ```

        For the model, we transform the raw tracking data into a structured tabular representation.

        
        """)

    # -----------------------------------------------
    # TAB 2: PREPROCESSING + FEATURE ENGINEERING
    # -----------------------------------------------

    with tab2:
        st.markdown("""
        ## ⚙️ Preprocessing + Feature Engineering

        ---
                    
        ### 1. Extraction of tracking moments

        #### a. Match Shot Events to Tracking Moments

        For each shot event, we search for tracking moments around the **shot timestamp**.

        We use a **one-second window** before the shot to capture movement leading into the shot.

        ---

        #### b. Normalize to half court coordinates

        All shots are **normalized to one half court** so that the offensive team is attacking the same basket.

        ```python
        if x < 47:
            x = 94 - x
            y = 50 - y
        ```

        This ensures that the model does not need to learn separate patterns for left-side and right-side possessions.

        ---

        #### c. Sample Frames Before the Shot

        Instead of using every raw tracking frame, we sample six frames:

        ```text
        t0 = shot frame
        t1 = 0.2 seconds before
        t2 = 0.4 seconds before
        t3 = 0.6 seconds before
        t4 = 0.8 seconds before
        t5 = 1.0 seconds before
        ```

        This reduces dimensionality while preserving the most important movement information.

        ---

        #### d. Canonical Player Ordering

        Player order in raw tracking data is not directly useful for modeling.

        We therefore define a stable representation:

        - **shooter** is stored separately
        - **defenders** are **ordered by distance** to the shooter at the shot frame
        - **teammates** are **ordered by distance** to the shooter at the shot frame

        Example:

        ```text
        shooter
        defender1 = closest defender at shot release
        defender2 = second closest defender at shot release
        ...
        attacker1 = closest teammate at shot release
        ...
        ```

        This ordering is then kept consistent across all previous frames.

        ---

        #### e. Relative Coordinates

        Defenders and teammates are represented relative to the shooter:

        ```python
        dx = player_x - shooter_x
        dy = player_y - shooter_y
        dist = sqrt(dx² + dy²)
        ```

        The intention is to help the model learn basketball-relevant spatial relationships instead of absolute court positions only.

        ---

        #### f. Generated Tracking Features

        For every time frame, we generate:

        ```text
        shooter_x_t{i}
        shooter_y_t{i}

        defender1_dx_t{i}
        defender1_dy_t{i}
        defender1_dist_t{i}
        ...

        attacker1_dx_t{i}
        attacker1_dy_t{i}
        attacker1_dist_t{i}
        ...
        ```

        for all six frames:

        ```text
        i = 0, 1, 2, 3, 4, 5
        ```

        ---

        ### 2. Additional Feature generation

        Besides raw trajectory features, we also use **engineered basketball features** such as:

        - Spatial features:
            - shooter speed
            - shot angle
            - distance to basket
            - number of players in the paint
        - Defender features:
            - nearest defender distance
            - average defender distance
            - number of defenders within 3 / 5 / 7 feet
            - defender closing speed
            - indicator whether defenders are in a line between shooter and basket
        - Attacker features:
            - indicator whether a teammate is screening
            - nearest teammate distance
        - Other features:
            - Main action type
            - Period

        These features provide strong basketball priors and help the model learn from limited tracking data.
        """)

        st.markdown("### Final Feature Categories")

        final_features = pd.DataFrame({
            "Category": [
                "Temporal tracking features",
                "Shot geometry",
                "Defensive pressure",
                "Offensive support",
                "Player movement",
                "Categorical context",
                "Player identity"
            ],
            "Examples": [
                "defender1_dx_t0, defender1_dx_t5, attacker1_dist_t3",
                "shot_angle, distance_to_basket_tracking",
                "nearest_defender_dist, defenders_within_5ft, defender_closing_speed",
                "nearest_teammate_distance, teammate_between_defender",
                "shooter_speed, tracking features",
                "PERIOD, MAIN_ACTION_TYPE",
                "PLAYER_ID"
            ],
            "Purpose": [
                "Describe player locations and movement before the shot",
                "Capture shot location and geometry",
                "Measure contest and defensive positioning",
                "Capture spacing and teammate support",
                "Describe shooter motion before release",
                "Add game and action context",
                "Allow the model to learn shooter-specific tendencies"
            ]
        })

        st.dataframe(
            final_features,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
            #### Data Amount
            Since we only have one season the data amount is a bit limited
            **With top 20 players:**
            - 8149 rows
            
            **With all players:**
            - 57510 rows
                     
            We therefore, mainly trained on all players for this dataset
        """)

    # -----------------------------------------------
    # TAB 3: MODELS + RESULTS
    # -----------------------------------------------

    with tab3:
        st.markdown("""
        ### 🤖 Models + Results

        ---
        #### Model overview

        The evaluated models are the following:
                    
        **Baseline:** Simple Lookup Table
                    
        **Classical Machine Learning:** XgBoost model
                    
        **Deep Learning:** Deep Neural Network


        """)

        st.info("**Note:** The models were trained and evaluated on all players, " \
        "otherwise the data amount would be very limited for this dataset")

        st.markdown("""

        --- 

        #### Classical Machine Learning: XGBoost on Shot-Frame Features

        The first baseline model uses only features from the shot frame. And computed speeds over the 1 second window.
        """)

        with st.expander("**Selected features**", expanded=False):
            st.dataframe({
                'Features': ['shooter_x', 'shooter_y', 'shot_angle', 'distance_to_basket_tracking', 'nearest_defender_dist', 
                           'avg_defender_dist', 'defenders_within_3ft', 'defenders_within_5ft', 'defenders_within_7ft', 
                           'shooter_speed', 'defender_closing_speed', 'nearest_teammate_distance', 'defenders_between', 
                           'has_screen', 'offensive_spacing', 'teammate_between_defender', 'players_in_paint']
            })
        
        show_sklearn_pipeline()

        st.markdown("""**Model parameters:**
                    
        Optimized, using random search with cross validation.
        """)
        st.dataframe(pd.DataFrame({
            'Parameters:': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'min_child_weight', 'gamma', 'eval_metric'],
            'values': ["500", "3", "0.01", "0.7", "1", "0", "logloss"],
        }))

        st.markdown("""
        ---

        #### Deep Learning Model

        We also train a neural network using:

        - temporal tracking features from `t0` to `t5`
        - engineered shot-quality features
        - categorical context features
        - player embeddings

        The model structure is approximately:

        ```text
        continuous tracking + engineered features
            ↓
        dense neural network

        PLAYER_ID
            ↓
        player embedding

        PERIOD / MAIN_ACTION_TYPE
            ↓
        categorical embeddings

        concatenate
            ↓
        final dense layers
            ↓
        predicted make probability
        ```
                    
                    """)
        
        with st.expander("**Model structure**", expanded=False):
            # Pfad zu deinem Bild
            image_path = "streamlit_includes/data/deepNN-trajectory-data-js.keras.png"

            # Bild einlesen und in Base64 konvertieren
            with open(image_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode()

            scale = 0.7
            html_code = f"""
                <div style="
                    overflow: auto;
                    max-height: 800px;
                    max-width: 100%;
                    border: 1px solid #ddd;
                    padding: 10px;
                    background: white;
                ">
                    <img
                        src="data:image/png;base64,{b64_string}"
                        style="
                            width: {int(1200 * scale)}px;
                        "
                    >
                </div>
                """

            components.html(html_code, height=850)
            
        st.markdown("""

        ---

        ## Interpretation of Results
                    
        #### Comparison of feature sets
        
        **First test:** Comparison of different feature sets using the XgBoost model
                
        Three feature sets were tested here (**only for top 20 players**)
        - **Set1**: Only new trajectory features
        - **Set2**: Only features from original dataset
        - **Set3**: Combination of both
                    
        """)

        df_setComp = pd.DataFrame({
            'Feature Set': ['Set 1', 'Set 2', 'Set 3'],
            'Accuracy': [0.55, 0.60, 0.60],
            'ROC-AUC': [0.5797, 0.6156, 0.6141],
            'Brier Score': [0.2458, 0.2347, 0.2379],
        })
        df_setComp = df_setComp.set_index('Feature Set')

        st.dataframe(df_setComp)

        st.markdown("""   
        
        Experiments show that:

        - shot type and player identity are very strong predictors
        - tracking data adds some spatial context
        - predicting individual shot outcomes remains noisy

        #### Comparison of models

        Comparison of the different models on all players.
                    
        Tested models:
        - Simple Lookup Table
        - XgBoost model
        - Neural Network
                
        """)

        model_results = pd.DataFrame({
            "Model": [
                "Lookup table",
                "XGBoost",
                "Neural Network"
            ],
            "Accuracy": [
                0.61,
                0.63,
                0.63
            ],
            "ROC-AUC": [
                0.621,
                0.641,
                0.641
            ],
            "Brier Score": [
                0.2354,
                0.2275,
                0.2277
            ]
        })

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        ---

        #### Conclusion on trajectory data

        The tracking models do not completely transform predictive performance, 
        but it enables a much richer interpretation of shot quality.

        Instead of only asking:

        ```text
        Was this shot made or missed?
        ```

        we can ask:

        ```text
        How does the predicted probability change if the defender is closer?
        How does spacing affect shot quality?
        How does the model react to different player configurations?
        ```

        This motivates the interactive demo.
        """)



    # -----------------------------------------------
    # TAB 4: DEMO
    # -----------------------------------------------

    with tab4:
         render_trajectory_demo(prefix="trajectory_demo_page")