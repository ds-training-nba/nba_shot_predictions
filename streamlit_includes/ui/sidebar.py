import streamlit as st
from streamlit_includes.pages import (
    introduction,
    data_presentation,
    preprocessing,
    data_analysis,
    feature_engineering,
    model_development,
    model_analysis,
    trajectory_data,
    player_insights,
    deep_dive,
    shot_prediction,
    coach_app,
    conclusions
)

PAGE_MAP = {
    "🎯 Introduction & Problem": introduction.render,
    "📊 Data Presentation": data_presentation.render,
    "🔧 Data Preprocessing": preprocessing.render,
    "📈 Data Analysis & Visualization": data_analysis.render,
    "🏀 Player Insights": player_insights.render,
    "🔍 Top 20 Players Deep Dive": deep_dive.render,
    "⚡ Feature Engineering": feature_engineering.render,
    "🎓 Model Development": model_development.render,
    "📉 Model Results & Analysis": model_analysis.render,
    "🎯 Demo: The Coach App": coach_app.render,
    "🔧 Trajectory data": trajectory_data.render,
    "💡 Conclusions & Recommendations": conclusions.render,
}

def get_selected_page():

    st.sidebar.markdown("# 🏀 NBA Shot Analysis")
    st.sidebar.markdown("### Top 20 Players Study")
    st.sidebar.markdown("---")

    return st.sidebar.radio(
        "Navigate Project Sections:",
        list(PAGE_MAP.keys())
    )