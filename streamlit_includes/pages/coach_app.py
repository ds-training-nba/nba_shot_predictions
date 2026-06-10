import streamlit as st

from app.streamlit import sl_player_app, sl_alternatives_app


def render():
    st.markdown(
        '<div class="section-title">🎯 Demo: The Coach App</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Find alternative Players or moves"
    )
    tab1, tab2 = st.tabs(["Player Alternatives", "Alternative Moves"])
    with tab1:
        sl_player_app()
    with tab2:
        sl_alternatives_app()