import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players


def render():
    """Data Preprocessing"""
    st.markdown('<div class="section-title">🔧 Data Preprocessing & Feature Engineering</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Cleaning Steps", "Feature Engineering", "Results"])

    with tab1:
        st.markdown("""
        ### 🧹 Data Cleaning Pipeline

        ---

        #### 0. Column Selection (Initial Filtering)
        We start by keeping only relevant columns and removing obvious noise / duplicates.

        ```python
        CLEAN_SOURCE_COLUMNS = [...] # 35/84 columns
        df = df[CLEAN_SOURCE_COLUMNS]
        ```

        **Justification:**
        - Removes redundant merge artifacts
        - Reduces memory usage
        - Keeps only modeling-relevant signals

        ---

        #### 1. Remove Inconsistent Target Labels
        ```python
        shot_result_conv = df['shotResult'].map({'Made': 1, 'Missed': 0})
        df = df[~(shot_result_conv.notna() & 
                  (df['SHOT_MADE_FLAG'] != shot_result_conv))]
        ```

        - Removed conflicting labels between sources
        - Prevents noisy supervision signal

        ---

        #### 2. Drop Critical Missing Values
        ```python
        df = df.dropna(subset=[
            'SHOT_DISTANCE', 'SHOT_TYPE',
            'SHOT_ZONE_RANGE', 'SHOT_ZONE_BASIC',
            'LOC_X', 'LOC_Y'
        ])
        ```

        - Ensures spatial + shot context completeness
        - These features are essential for shot modeling

        ---

        #### 3. Fill Game-Level Constants
        ```python
        df['HTM'] = df.groupby('GAME_ID_x')['HTM'].transform('first')
        df['VTM'] = df.groupby('GAME_ID_x')['VTM'].transform('first')
        df['GAME_DATE'] = df.groupby('GAME_ID_x')['GAME_DATE'].transform('first')
        ```

        - Game metadata is constant within GAME_ID
        - Fills 100% missing values safely

        ---

        #### 4. Remove Duplicate / Broken Records
        ```python
        df = df[~(
            df.duplicated(subset=['GAME_ID_x', 'GAME_EVENT_ID'], keep=False) &
            df['shotResult'].isna()
        )]

        df = df.drop_duplicates()
        ```

        - Removes merge artifacts
        - Prevents duplicate bias in training

        ---

        #### 5. Fill Team Abbreviation Gaps
        ```python
        df['PLAYER1_TEAM_ABBREVIATION'] = df['PLAYER1_TEAM_ABBREVIATION'].fillna(
            df.groupby(['PLAYER_ID', 'TEAM_ID'])['PLAYER1_TEAM_ABBREVIATION']
            .transform('first')
        )
        df = df.dropna(subset=['PLAYER1_TEAM_ABBREVIATION'])
        ```

        - Ensures home/away logic works correctly

        ---

        #### 6. Fix Time Information
        ```python
        extracted = df['clock'].str.extract(r'PT(\\d+)M(\\d+)\\.')
        df['MINUTES_REMAINING'] = df['MINUTES_REMAINING'].fillna(extracted[0])
        df['SECONDS_REMAINING'] = df['SECONDS_REMAINING'].fillna(extracted[1])
        ```

        - Recovers missing game clock data

        """)

    with tab2:
        st.markdown("""
        ### ⚙️ Feature Engineering Pipeline

        All features are generated in `add_computed_feature_columns(df)`:

        ```python
        for func in COMPUTED_FEATURES_FUNCTIONS:
            df = func(df)
        ```

        ---

        #### 1. Home/Away Indicator
        ```python
        IS_HOME = (HTM == PLAYER1_TEAM_ABBREVIATION)
        ```

        - Encodes home advantage signal

        ---

        #### 2. Score & Game Flow Features
        ```python
        scoreHome, scoreAway
        scoreMargin
        scoreMarginBeforeShot
        ```

        - Captures momentum and game pressure
        - Built via cumulative scoring per GAME_ID

        ---

        #### 3. Time Features
        ```python
        TimeRemainingInPeriod
        TotalPlayedTime
        TimeRemainingInGame
        IsOvertime
        OvertimeNumber
        IsClutchTime
        ```

        - Captures fatigue + pressure context
        - Clutch defined as:
          - ≤ 5 min remaining
          - ≤ 5 point margin

        ---

        #### 4. Opponent Interaction
        ```python
        OPPONENT_INTERFERED
        ```

        - Detects defensive contest situations

        ---

        #### 5. Shot Geometry Features
        ```python
        ANGLE
        ANGLE_SECTOR
        ABS_ANGLE
        ANGLE_SIN
        ANGLE_COS
        ```

        - Converts court position into polar representation
        - Captures shooting direction bias

        ---

        #### 6. Shot Type Simplification

        ```python
        MAIN_ACTION_TYPE
        ```
        
        We compress 72 fine-grained ACTION_TYPE labels into high-level semantic categories.
        
        Categories:
        
        Dunk
        Layup
        Hook
        Jump
        Other
        
        Why this transformation is needed:
        
        Original ACTION_TYPE contains ~72 rare and highly specific shot variations
        Many of them have very low frequency → increases noise and sparsity
        Grouping improves generalization ability of the model
        """)

    with tab3:
        st.markdown("### 📊 Data Quality Improvement")

        quality_data = {
            'Metric': [
                'Total Records',
                'Missing Values',
                'Duplicates',
            ],
            'Before Cleaning': [
                '8,208,626',
                '17%',
                '0.007%',
            ],
            'After Cleaning + Filtering + Feature Pipeline': [
                '537,909',
                '4.9%',
                '0%',
            ],
            'Improvement': [
                '-93.45%',
                '-12.1%',
                '-0.007%',
            ]
        }

        df_quality = pd.DataFrame(quality_data)
        st.dataframe(df_quality, use_container_width=True, hide_index=True)