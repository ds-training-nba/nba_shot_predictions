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
            "537,909",
            "After filtering & cleaning"
        )
    with col3:
        st.metric(
            "Best Model (LightGBM)",
            "Acc: 68.0% | AUC: 0.733",
            "Brier Score: 0.203"
        )

    st.markdown(
        '<div class="section-title">Project Motivation & Objectives</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Project Background

        The aim of the project was given as:
        - *Compare the shots (frequency and shooting efficiency by game situation and location
          on the court) of **20 of the best NBA players** of the 21st century (according to ESPN).*
        - *For each of these 20 players, **estimate using a model the probability** that their shot
          goes into the basket, according to different metrics.*

        We decided to put ourselves into the shoes of a **basketball team manager/coach**.
        There are numerous questions someone concerned with training, coaching and/or management may have:

        - Which players to sell/buy (performance or fit with other players)
        - Which players to put in the starting team of the next match
        - Define a necessary practice routine fitting to individual strengths and weaknesses
        - Strategic/tactical decisions during the match to optimize shot efficiency

        ### Economic Point of View

        The team manager/coach seeks to optimize business decisions through our model.
        Finding underrated players on the market and decisions concerning training and game
        strategy are economically crucial.

        For our model to be helpful, we must maximize **interpretability** — a mere prediction,
        however precise, will not offer economic advantage: the full explanatory data
        (shot type, position, etc.) is available only milliseconds before the result is
        visible publicly on TV.

        One relevant use case is providing **counterfactual tools**: for example, which
        player from the market would have performed better in that match, according to the
        shot probability model.

        ### Scientific Point of View

        A paper (*Quantifying Shot Quality in the NBA*, SSAC 2014) estimates a maximum
        predictability of a field goal at **63.3%**. With a naive baseline of ~52–57%
        (always predict miss/hit), the margin is disappointingly narrow. Our model reaches
        63.5% on field goals, which aligns with this theoretical ceiling.

        The takeaway: **noise is fundamental**. Shot outcomes are influenced by unmeasurable
        variables (the ball's spin, sweat on the floor, the player's mental state). The goal
        is not perfect prediction, but **meaningful probability estimation** that reflects
        physics and rules of the game.

        ### Project Objectives

        1. Compare shooting frequency across different court locations and game situations.
        2. Estimate shot success **probability** using machine learning (not just binary prediction).
        3. Identify the most influential factors affecting shot outcomes.
        4. Provide data-driven tools for coaching and player development.
        5. Maximize model **interpretability** for practical use by a coaching team.
        """)

    with col2:
        st.markdown("""
        ### Main Performance Metric

        Our primary metric is **Brier Score** (squared error of probability), not
        plain accuracy. This choice reflects that:

        - A coach wants to know *why* a shot missed, not just if
        - Correct probabilities are more informative than correct binary labels
        - Noise in the target means probability calibration matters most

        ROC-AUC is used as a secondary metric for discrimination ability.

        ### Player Groups

        #### Modern NBA Superstars
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
        """)

    st.markdown(
        '<div class="section-title">Problem Statement</div>',
        unsafe_allow_html=True
    )

    st.info("""
        Estimate the **probability** that a shot results in a made basket based on
        player characteristics, shot location, and game context.

        This is primarily an analytical tool for basketball coaches — not a real-time
        prediction system. The value lies in understanding which variables drive shot
        success, not in predicting individual shots with certainty.
    """)

    st.markdown(
        '<div class="section-title">Challenges & Complexities</div>',
        unsafe_allow_html=True
    )

    challenges = {
        "Challenge": [
            "Fundamental Noise Floor",
            "Target Leakage Risk",
            "Data Merging Complexity",
            "Interpretability vs. Accuracy",
        ],
        "Description": [
            "Shot outcomes depend on unmeasurable factors; theoretical max accuracy ~63.3% for field goals",
            "Several columns (OPPONENT_INTERFERED, ACTION_TYPE subtypes) introduced post-event labels",
            "Three datasets (shotdetail, nbastatsv3, nbastats) merged on game/event IDs — 80+ columns to assess",
            "A simple Lookup Table baseline was nearly as accurate as LightGBM — meaning complexity must earn its place",
        ],
        "Solution Applied": [
            "Shifted focus to probability calibration (Brier Score) rather than accuracy",
            "Excluded OPPONENT_INTERFERED; randomly redistributed ambiguous ACTION_TYPE values",
            "Left-merge on nbastatsv3; careful deduplication and NaN-filling per game ID",
            "Kept Lookup Table as baseline; selected LightGBM for best probability calibration + interpretability",
        ]
    }

    st.dataframe(pd.DataFrame(challenges), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="section-title">Model Results Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Full dataset (incl. free throws)**")
        results_full = pd.DataFrame({
            "Model": ["Lookup Table (baseline)", "LightGBM (main)", "Logistic Regression", "Decision Tree"],
            "Accuracy": [0.665, 0.680, 0.666, 0.672],
            "ROC-AUC": [0.709, 0.733, 0.718, 0.720],
            "Brier Score": [0.211, 0.203, 0.210, 0.208],
        })
        st.dataframe(results_full, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Field goals only**")
        results_fg = pd.DataFrame({
            "Model": ["Lookup Table (baseline)", "LightGBM (main)", "Logistic Regression", "Decision Tree"],
            "Accuracy": [0.616, 0.635, 0.618, 0.627],
            "ROC-AUC": [0.638, 0.673, 0.651, 0.655],
            "Brier Score": [0.229, 0.221, 0.227, 0.224],
        })
        st.dataframe(results_fg, use_container_width=True, hide_index=True)

    st.caption(
        "Note: the biggest accuracy gain came from the naive Lookup Table (52% → 66.5%). "
        "Advanced ML adds ~1-2% on top — but improves probability calibration meaningfully."
    )