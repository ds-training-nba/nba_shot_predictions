import streamlit as st

from streamlit_includes.ui.styles import load_styles
from streamlit_includes.ui.sidebar import get_selected_page, PAGE_MAP



st.set_page_config(
    page_title="NBA Shot Analysis - Top 20 Players",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_styles()





def main():
    selected_page = get_selected_page()
    PAGE_MAP[selected_page]()


if __name__ == "__main__":
    main()