import streamlit as st


def get_selected_page():

    st.sidebar.markdown("# 🏀 NBA Shot Analysis")
    st.sidebar.markdown("### Top 20 Players Study")
    st.sidebar.markdown("---")

    return st.sidebar.radio(
        "Navigate Project Sections:",
        [
            "🎯 Introduction & Problem",
            "📊 Data Presentation",
            "🔧 Data Preprocessing",
            "📈 Data Analysis & Visualization",
            "⚡ Feature Engineering",
            "🎓 Model Development",
            "📉 Model Results & Analysis",
            "🔧 Trajectory data",
            "🏀 Player Insights",
            "🔍 Top 20 Players Deep Dive",
            "🎯 Shot Probability Predictor",
            "💡 Conclusions & Recommendations"
        ]
    )