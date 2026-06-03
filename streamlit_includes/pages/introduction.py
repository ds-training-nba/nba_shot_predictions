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
        
        The selection is NOT strictly limited to ESPN rankings, but is instead a balanced analytical
        dataset designed to capture different player archetypes and roles across the modern NBA.

        ### Extended Player Selection

        To broaden the scope of the analysis, several all-time legends were added
        to the player pool by our own initiative. These include players such as:

        - Michael Jordan
        - Kareem Abdul-Jabbar
        - Magic Johnson
        - Larry Bird
        - Charles Barkley

        While these players are not part of ESPN's "Top NBA Players of the
        21st Century" ranking, they represent historical benchmarks whose
        performance remains highly relevant when evaluating modern stars.

        ### Project Objectives

        1. Compare shooting frequency across different court locations.
        2. Compare shooting efficiency under various game situations.
        3. Analyze differences between modern stars and historical legends.
        4. Estimate shot success probability for active players using machine learning.
        5. Identify the most influential factors affecting shot outcomes.
        6. Evaluate whether modern players exhibit different shot-selection patterns
           compared to previous generations.

        ### Why This Matters

        - **Player Evaluation**: distinguish efficiency from volume scoring.
        - **Game Strategy**: optimize shot selection schemes.
        - **Player Development**: identify strengths and weaknesses.
        - **Basketball Analytics**: compare players across different eras.
        - **Coaching Decisions**: support data-driven decision making.
        """)

    with col2:

        st.markdown("""
        ### Player Groups
        
        #### Modern NBA Superstars (21st Century Core)
        - LeBron James
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
        
        #### NBA Championship Core Players
        - Kobe Bryant
        - Tim Duncan
        - Dirk Nowitzki
        - Kevin Garnett
        - Pau Gasol
        - Manu Ginobili
        - Tony Parker
        
        #### Modern Role & Two-Way Stars
        - Draymond Green
        
        #### Research Focus
        This selection includes both:
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

        The focus is placed on active players for predictive modeling, while
        historical legends are included for comparative statistical analysis
        across eras.
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
