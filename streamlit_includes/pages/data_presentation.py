import streamlit as st
import pandas as pd
import numpy as np

from streamlit_includes.data.data_overview import load_overview


def render():
    """Data Presentation & Architecture"""

    st.markdown('<div class="section-title">📊 Data Presentation</div>', unsafe_allow_html=True)

    # ── 1. Data sources overview ──────────────────────────────────────────────
    st.markdown("### Data Sources")

    st.markdown("""
    The dataset was assembled by merging **three complementary data sources** from
    [Kaggle](https://www.kaggle.com/datasets/brains14482/nba-playbyplay-and-shotdetails-data-19962021),
    which itself combined play-by-play data from **stats.nba.com**, **data.nba.com**, and **pbpstats.com**.
    """)

    sources = pd.DataFrame({
        "Dataset / prefix": ["cdnnba", "nbastatsv3", "nbastats", "pbpstats", "shotdetail", "datanba"],
        "Start year": [2020, 1996, 1996, 2000, 1996, 2016],
        "End year":   [2024, 2024, 2024, 2024, 2024, 2024],
        "Info": [
            "Coordinates, play-by-play, shot distance, situational game stats",
            "Play-by-play description, coordinates, shot distance, shot classification",
            "Play-by-play split home/neutral/visitor; info for up to 3 involved players",
            "Play-by-play, game-specific stats, video URL",
            "Coordinates, shot classification, shot position — has SHOT_MADE_FLAG",
            "Play-by-play, coordinates, points",
        ],
        "Target variable": ["shotResult", "shotResult", "—", "—", "SHOT_MADE_FLAG", "—"],
        "Used": ["❌", "✅", "✅", "❌", "✅", "❌"],
    })
    st.dataframe(sources, use_container_width=True, hide_index=True)

    st.markdown("""
    **Selected combination: `shotdetail` + `nbastatsv3` + `nbastats`**

    Other datasets were excluded due to limited time frames or lack of additional
    information not already present in the selected three.

    **Merge strategy:** Left-merge based on **nbastatsv3** (the most complete play-by-play
    source), joined on shared `GAME_ID` and `GAME_EVENT_ID`. After merging and
    filtering to shots only, the result contains **8,208,626 rows across 84 columns**.
    """)

    # ── 2. Merging & initial preprocessing ───────────────────────────────────
    st.markdown("### Merging & Initial Preprocessing")

    st.markdown("""
    Because the raw sources are play-by-play data (every game action, not just shots),
    several initial steps were needed **before** the main preprocessing pipeline:

    - **Free throw extraction** — free throws were not flagged in any feature; they had
      to be identified from play-by-play descriptions and shot-type fields.
    - **Coordinate correction** — free throw coordinates were recorded as `(0, 0)`
      instead of the correct `(0, 15 ft)` and had to be fixed.
    - **Initial shot filter** — non-shot events (fouls, timeouts, …) were removed,
      leaving only shot attempts.

    This produced the working dataset used for exploration and, after further cleaning,
    for model training.
    """)

    # ── 3. General dataset characteristics ───────────────────────────────────
    st.markdown("### General Dataset Characteristics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows (raw)", "8,208,626", "all events, all players")
    col2.metric("Columns (raw)", "84", "after merge")
    col3.metric("Date range", "1996 – 2024", "nbastatsv3 coverage")
    col4.metric("Unique players", "2,742", "in raw dataset")

    st.markdown("""
    After filtering to the **20 selected players** and applying the full preprocessing
    and cleaning pipeline, the working dataset shrinks to **~537,909 shots**,
    split 80/20 into train/test sets.
    """)

    # ── 4. Column-level data audit ────────────────────────────────────────────
    st.markdown("### Column-Level Data Audit")
    st.markdown("""
    The table below is based on an audit of 1,000 sampled rows (representative sample).
    It covers all 84 columns with their types, missing-value rates, and descriptions.
    """)

    df = load_overview()
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── 5. Key column groups ──────────────────────────────────────────────────
    st.markdown("### Key Column Groups (Selected for Modeling)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Spatial / Shot geometry**
        - `LOC_X`, `LOC_Y` — court coordinates
        - `SHOT_DISTANCE` — distance to rim in ft
        - `SHOT_ZONE_BASIC`, `SHOT_ZONE_AREA`, `SHOT_ZONE_RANGE` — zone classifications
        - `ANGLE`, `ABS_ANGLE`, `ANGLE_SECTOR` *(engineered)*

        **Shot classification**
        - `ACTION_TYPE` — 72 fine-grained shot subtypes
        - `MAIN_ACTION_TYPE` *(engineered)* — Jump / Layup / Dunk / Hook / Other
        - `SHOT_TYPE` — 1PT / 2PT / 3PT

        **Target**
        - `SHOT_MADE_FLAG` — 0 = miss, 1 = hit
        """)

    with col2:
        st.markdown("""
        **Game context**
        - `PERIOD_x` — quarter (1–4+)
        - `TimeRemainingInPeriod`, `TimeRemainingInGame` *(engineered)*
        - `IsClutchTime` *(engineered)* — last 5 min & score within 5
        - `IS_HOME` *(engineered)* — home/away indicator
        - `is_playoffs` — playoff flag

        **Score**
        - `scoreMarginBeforeShot` *(engineered)*
        - `scoreHomeBeforeShot`, `scoreAwayBeforeShot` *(engineered)*

        **Player**
        - `PLAYER_ID`, `PLAYER_NAME`
        - `player_age`, `best_age`, `year` *(enriched from external player DB)*
        """)