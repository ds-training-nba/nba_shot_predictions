import streamlit as st

@st.cache_data
def load_top_20_players():
    """Load data for top 20 NBA players"""
    top_20_players = [
        'LeBron James',
        'Kobe Bryant',
        'Stephen Curry',
        'Tim Duncan',
        "Kevin Garnett",
        "Nikola Jokic",
        "Dwyane Wade",
        "Kevin Durant",
        "Dirk Nowitzki",
        "Giannis Antetokounmpo",
        "James Harden",
        "Chris Paul",
        "Kawhi Leonard",
        "Manu Ginobili",
        "Anthony Davis",
        "Tony Parker",
        "Draymond Green",
        "Russell Westbrook",
        "Pau Gasol",
        "Luka Doncic"
    ]
    return top_20_players
