import streamlit as st


def load_styles():

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f4788;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 2rem;
            font-weight: bold;
            color: #2e75b6;
            border-bottom: 4px solid #e74c3c;
            padding-bottom: 15px;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        .subsection {
            font-size: 1.3rem;
            font-weight: bold;
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .player-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        .highlight {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        section[data-testid="stSidebar"] {
            min-width: 340px;
        }
        </style>
    """, unsafe_allow_html=True)