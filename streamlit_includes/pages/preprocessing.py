import streamlit as st
import pandas as pd


def render():
    """Data Preprocessing"""
    st.markdown('<div class="section-title">🔧 Data Preprocessing & Feature Engineering</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Cleaning Steps", "Feature Engineering", "Data Quality Results"])

    # ── TAB 1: Cleaning ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        ### 🧹 Data Cleaning Pipeline

        All steps are implemented in `src/processing/preprocessing.py` and applied
        via `src/app/data_providers.py`.

        ---

        #### 0. Column Selection (Initial Filtering)
        We start by keeping only the 35 relevant columns (out of 84) and removing
        obvious noise and duplicates.

        ```python
        CLEAN_SOURCE_COLUMNS = [...] # 35 / 84 columns
        df = df[CLEAN_SOURCE_COLUMNS]
        ```

        **Justification:** reduces memory usage, removes redundant merge artifacts,
        keeps only modeling-relevant signals.

        ---

        #### 1. Remove Inconsistent Target Labels

        ```python
        shot_result_conv = df['shotResult'].map({'Made': 1, 'Missed': 0})
        df = df[~(shot_result_conv.notna() &
                  (df['SHOT_MADE_FLAG'] != shot_result_conv))]
        ```

        `SHOT_MADE_FLAG` (from *shotdetail*) and `shotResult` (from *nbastatsv3*) both
        carry the target label. Rows where they disagree are inconsistent and are dropped.

        ---

        #### 2. Drop Critical Missing Values

        ```python
        df = df.dropna(subset=[
            'SHOT_DISTANCE', 'SHOT_TYPE',
            'SHOT_ZONE_RANGE', 'SHOT_ZONE_BASIC',
            'LOC_X', 'LOC_Y'
        ])
        ```

        These columns are essential for shot modeling. Rows missing any of them cannot
        be imputed meaningfully.

        ---

        #### 3. Fill Game-Level Constants

        ```python
        df['HTM']       = df.groupby('GAME_ID_x')['HTM'].transform('first')
        df['VTM']       = df.groupby('GAME_ID_x')['VTM'].transform('first')
        df['GAME_DATE'] = df.groupby('GAME_ID_x')['GAME_DATE'].transform('first')
        ```

        Game metadata is constant within a `GAME_ID`. Filling from the first valid row
        within each game recovers ~21% missing values safely.

        ---

        #### 4. Remove Duplicate / Broken Records

        ```python
        # Rows that are duplicated on game+event but have no shotResult (block artifacts)
        df = df[~(
            df.duplicated(subset=['GAME_ID_x', 'GAME_EVENT_ID'], keep=False) &
            df['shotResult'].isna()
        )]
        df = df.drop_duplicates()
        ```

        Blocks and similar defensive events sometimes create a second row for the same
        event ID. These are identified by the absence of `shotResult` and removed.

        ---

        #### 5. Fill Team Abbreviation Gaps

        ```python
        df['PLAYER1_TEAM_ABBREVIATION'] = (
            df['PLAYER1_TEAM_ABBREVIATION']
            .fillna(
                df.groupby(['PLAYER_ID', 'TEAM_ID'])['PLAYER1_TEAM_ABBREVIATION']
                .transform('first')
            )
        )
        df = df.dropna(subset=['PLAYER1_TEAM_ABBREVIATION'])
        ```

        Required for the `IS_HOME` feature (home team abbreviation vs. player team
        abbreviation).

        ---

        #### 6. Fix Time Information

        ```python
        extracted = df['clock'].str.extract(r'PT(\\d+)M(\\d+)\\.')
        df['MINUTES_REMAINING'] = df['MINUTES_REMAINING'].fillna(extracted[0])
        df['SECONDS_REMAINING'] = df['SECONDS_REMAINING'].fillna(extracted[1])
        ```

        Recovers missing game clock data from the raw `clock` string field
        (format: `PT11M08.00S`).

        ---

        #### 7. Target Leakage Prevention ⚠️

        Two columns raised suspicion for **post-event labelling** (data entered only after
        the shot outcome was known):

        **OPPONENT_INTERFERED** — derived from `PLAYER2_TEAM_ABBREVIATION`. Investigation
        showed that *every* time an opponent was recorded as interfering, the shot was made.
        This column was excluded entirely from the feature set.

        **ACTION_TYPE subtypes** — within each `MAIN_ACTION_TYPE` group, the most generic
        label (e.g., `"Dunk Shot"`, `"Layup Shot"`) had a systematically lower hit rate than
        specific variants (e.g., `"Driving Dunk Shot"`). No semantic reason exists for
        `"Dunk Shot"` to be harder than `"Driving Dunk Shot"`. The likely cause is a
        post-event labelling artifact. Fix:

        ```python
        # Randomly redistribute the problematic category to other subtypes
        # within the same MAIN_ACTION_TYPE group (random seed = 42)
        df = fix_action_type_target_leak(df)
        ```
        """)

    # ── TAB 2: Feature Engineering ─────────────────────────────────────────────
    with tab2:
        st.markdown("""
        ### ⚙️ Feature Engineering Pipeline

        All engineered features are computed in `src/processing/compute_columns.py` and
        applied via `add_computed_feature_columns(df)`.

        ---

        #### 1. Home/Away Indicator
        ```python
        IS_HOME = (HTM == PLAYER1_TEAM_ABBREVIATION)  # 1 = home, 0 = away
        ```

        ---

        #### 2. Score & Game Flow Features
        ```python
        scoreHome, scoreAway            # cumulative per game
        scoreMarginBeforeShot           # score diff from shooter's perspective
        scoreHomeBeforeShot             # score state before each shot
        scoreAwayBeforeShot
        ```

        Built via cumulative sum per `GAME_ID`, shifted by 1 to avoid look-ahead.
        Captures momentum and pressure context.

        ---

        #### 3. Time Features
        ```python
        TimeRemainingInPeriod   # seconds left in current quarter
        TotalPlayedTime         # total seconds elapsed in game
        TimeRemainingInGame     # total seconds left (accounts for OT)
        IsOvertime              # 1 if PERIOD_x > 4
        OvertimeNumber          # how many OT periods played
        IsClutchTime            # (TimeRemainingInGame ≤ 300) & (|scoreMargin| ≤ 5)
        ```

        **Clutch time** definition: last 5 minutes of regulation/OT with score within
        5 points — the situations where pressure is highest.

        ---

        #### 4. Shot Geometry Features
        ```python
        ANGLE        = atan2(LOC_X, LOC_Y)   # symmetric around basket axis
        ANGLE_SECTOR                          # 0=front, 1=side, 2=far-side, 3=behind
        ABS_ANGLE    = abs(ANGLE)
        ANGLE_SIN    = sin(ANGLE)
        ANGLE_COS    = cos(ANGLE)
        ```

        The angle sectors capture non-linear relationships between angle and success
        probability that vary by distance range (e.g., behind-the-basket shots at <8 ft
        have surprisingly high success rates).

        ---

        #### 5. Shot Type Simplification

        ```python
        def main_category(action_type):
            for keyword in ['Dunk', 'Layup', 'Hook', 'Jump']:
                if keyword in action_type: return keyword
            return 'Other'

        df['MAIN_ACTION_TYPE'] = df['ACTION_TYPE'].apply(main_category)
        ```

        Compresses 72 fine-grained `ACTION_TYPE` labels into 5 semantic categories.
        Reduces sparsity and improves generalization.

        ---

        #### 6. Player Context Features *(external data)*

        ```python
        player_age        # shooter's age in the game year
        years_experience  # years since NBA debut
        best_age          # age at which player had highest FG% (from training data)
        year              # season year
        ```

        Merged from a separate player database. These capture career stage effects —
        a player at their peak age vs. early career vs. late career shows different
        shot profiles.
        """)

    # ── TAB 3: Results ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 Data Quality Improvement")

        quality_data = {
            "Metric": [
                "Total records (all players)",
                "Records after player filter (20 players)",
                "Records after full cleaning pipeline",
                "Missing values in key columns",
                "Duplicate rows",
                "Target leakage columns removed",
            ],
            "Before": [
                "8,208,626",
                "~538,000",
                "~538,000",
                "~21% (HTM/VTM/GAME_DATE)",
                "0.007%",
                "0",
            ],
            "After": [
                "8,208,626",
                "~538,000",
                "~537,909 (train + test)",
                "<5% (residual after fill)",
                "0%",
                "2 (OPPONENT_INTERFERED, raw ACTION_TYPE subtypes)",
            ],
        }

        st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)

        st.markdown("""
        ### Train / Test Split

        ```python
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y         # maintains class balance in both sets
        )
        ```

        The dataset is split 80 / 20. Stratified splitting ensures that the roughly
        50/50 hit/miss ratio (57% hit on the full dataset including free throws) is
        preserved in both train and test.

        The train/test split is **pre-computed and hosted on HuggingFace** so that all
        model experiments use an identical, reproducible split.
        """)