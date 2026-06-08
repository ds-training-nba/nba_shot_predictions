import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players


def render():
    """Introduction & Problem Definition"""

    st.markdown(
        '<div class="main-header">🏀 NBA Shot Success Prediction</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Players Analyzed",
            "20",
            "Modern NBA Core"
        )

    with col2:
        st.metric(
            "Total Shots",
            "500,000+",
            "Historical NBA Data"
        )

    with col3:
        st.metric(
            "Model Accuracy",
            "67.3%",
            "Best: XGBoost"
        )

    st.markdown(
        '<div class="section-title">Project Motivation & Objectives</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 2])

    with col1:

        st.markdown("""
        ### Project Background

        American sports are highly driven by data analytics, and the NBA provides one of the richest
        sports datasets in the world.

        This project is based on a curated selection of 20 elite NBA players from the modern era
        (2000s–present), representing superstars, franchise leaders, and championship-level contributors.

        The selection follows ESPN's "Top NBA Players of the 21st Century" ranking as a foundation,
        extended with key championship contributors who defined the winning teams of this era —
        players like Tony Parker, Manu Ginobili, and Pau Gasol, whose impact goes beyond
        individual statistics.

        ### Player Selection Rationale

        The 20 players were chosen to capture three distinct archetypes:

        - **Franchise superstars** — players who carried teams and defined eras (LeBron, Kobe, Durant)
        - **Championship contributors** — elite role players and co-stars on title-winning rosters
          (Parker, Ginobili, Gasol, Garnett)
        - **Modern playmakers** — next-generation stars reshaping how the game is played
          (Curry, Jokic, Doncic, Giannis)

        ### Project Objectives

        1. Compare shooting frequency across different court locations.
        2. Compare shooting efficiency under various game situations.
        3. Estimate shot success probability using machine learning.
        4. Identify the most influential factors affecting shot outcomes.
        5. Evaluate shot-selection patterns across different player archetypes.
        6. Provide data-driven insights for coaching and player development.

        ### Why This Matters

        - **Player Evaluation**: distinguish efficiency from volume scoring.
        - **Game Strategy**: optimize shot selection schemes.
        - **Player Development**: identify individual strengths and weaknesses.
        - **Basketball Analytics**: compare players across different roles and systems.
        - **Coaching Decisions**: support data-driven in-game decision making.
        """)

    with col2:

        st.markdown("""
        ### Player Groups

        #### Modern NBA Superstars (21st Century Core)
        - LeBron James
        - Kobe Bryant
        - Stephen Curry
        - Kevin Durant
        - Giannis Antetokounmpo
        - Nikola Jokic
        - Luka Doncic
        - James Harden
        - Kawhi Leonard
        - Anthony Davis
        - Russell Westbrook
        - Chris Paul
        - Dwyane Wade

        #### Championship Core Players
        - Tim Duncan
        - Dirk Nowitzki
        - Kevin Garnett
        - Pau Gasol
        - Manu Ginobili
        - Tony Parker

        #### Two-Way & Versatile Stars
        - Draymond Green

        #### Research Focus
        This selection covers:
        - franchise-level superstars
        - elite championship contributors
        - modern playmakers and system-defining players
        """)

    st.markdown(
        '<div class="section-title">Problem Statement</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        Predict the probability that a shot results in a made basket based on
        player characteristics, shot location, and game context.

        All 20 players are included both for comparative statistical analysis
        and for predictive modeling — covering active players, recently retired
        stars, and era-defining contributors of the 21st century NBA.
        """
    )

    st.markdown(
        '<div class="section-title">Challenges & Complexities</div>',
        unsafe_allow_html=True
    )

    challenges = {
        "Challenge": [
            "Data Quality",
            "Multiple Hidden Factors",
            "Model Selection",
            "Feature Engineering"
        ],
        "Description": [
            "Missing values, inconsistencies and duplicate records",
            "Many important variables are difficult to quantify",
            "Balance predictive power and interpretability",
            "Extract meaningful signals from high-dimensional data"
        ],
        "Solution Applied": [
            "Validation pipeline and preprocessing",
            "Focus on measurable basketball metrics",
            "Comparison of multiple ML algorithms",
            "Domain-driven feature construction"
        ]
    }

    df_challenges = pd.DataFrame(challenges)

    st.dataframe(
        df_challenges,
        use_container_width=True,
        hide_index=True
    )
