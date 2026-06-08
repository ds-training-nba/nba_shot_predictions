import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots


def render():
    """Top 20 Players Deep Dive"""
    st.markdown('<div class="section-title">🔍 Top 20 Players - Deep Analysis</div>',
                unsafe_allow_html=True)

    df = get_top_20_shots()
    players = load_top_20_players()

    st.markdown("### Select Player for Detailed Analysis")
    selected_player = st.selectbox("Choose player:", players, key='player_deep')

    player_data = df[df['PLAYER_NAME'] == selected_player]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shots", len(player_data))
    with col2:
        st.metric("FG%", f"{player_data['SHOT_MADE_FLAG'].mean():.1%}")
    with col3:
        st.metric("3PT Attempts", len(player_data[player_data['SHOT_TYPE'] == '3PT Field Goal']))
    with col4:
        st.metric("Avg Distance", f"{player_data['SHOT_DISTANCE'].mean():.1f} ft")

    # Player profile
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Shot Type Distribution")
        shot_dist = player_data['SHOT_TYPE'].value_counts()
        fig = px.pie(values=shot_dist.values, names=shot_dist.index)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Success by Distance")
        dist_stats = player_data.copy()

        df_distance = dist_stats.groupby("SHOT_DISTANCE")["SHOT_MADE_FLAG"].mean() * 100
        df_distance = df_distance.sort_index()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_distance.index,
            y=df_distance.values,
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=4)
        ))

        # Free throw line
        fig.add_vline(
            x=15,
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text="FT line"
        )

        # 3PT line
        fig.add_vline(
            x=23.75,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text="3PT line"
        )

        fig.update_layout(
            title="Success Rate by Shot Distance",
            xaxis_title="Distance (feet)",
            yaxis_title="Success Rate (%)",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

