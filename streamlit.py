import streamlit as st

from streamlit_includes.ui.styles import load_styles
from streamlit_includes.ui.sidebar import get_selected_page

from streamlit_includes.pages import (
    introduction,
    data_presentation,
    preprocessing,
    data_analysis,
    feature_engineering,
    model_development,
    model_analysis,
    player_insights,
    deep_dive,
    conclusions
)

st.set_page_config(
    page_title="NBA Shot Analysis - Top 20 Players",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_styles()


PAGE_MAP = {
    "🎯 Introduction & Problem": introduction.render,
    "📊 Data Presentation": data_presentation.render,
    "🔧 Data Preprocessing": preprocessing.render,
    "📈 Data Analysis & Visualization": data_analysis.render,
    "⚡ Feature Engineering": feature_engineering.render,
    "🎓 Model Development": model_development.render,
    "📉 Model Results & Analysis": model_analysis.render,
    "🏀 Player Insights": player_insights.render,
    "🔍 Top 20 Players Deep Dive": deep_dive.render,
    "💡 Conclusions & Recommendations": conclusions.render,
}


def main():
    selected_page = get_selected_page()
    PAGE_MAP[selected_page]()


if __name__ == "__main__":
    main()