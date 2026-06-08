import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots




def render():
    """Player Insights Page"""
    st.markdown('<div class="section-title">🏀 Player Insights & Shooting Patterns</div>',
                unsafe_allow_html=True)

    df = get_top_20_shots()
    players = load_top_20_players()

    tab1, tab2, tab3 = st.tabs(["Overall Rankings", "Shot Type Efficiency", "Game Situation Analysis"])

    with tab1:
        st.markdown("### Top 20 Players - Overall Shooting Efficiency")

        player_stats = []
        for player in players:
            player_data = df[df['PLAYER_NAME'] == player]
            fg_pct = player_data['SHOT_MADE_FLAG'].mean() * 100
            total_shots = len(player_data)

            player_stats.append({
                'Player': player,
                'FG%': round(fg_pct, 1),
                'Total Shots': total_shots,
                'Made': int(player_data['SHOT_MADE_FLAG'].sum()),
                'Missed': int((1 - player_data['SHOT_MADE_FLAG']).sum())
            })

        df_stats = pd.DataFrame(player_stats).sort_values('FG%', ascending=False)

        fig = px.bar(df_stats, x='FG%', y='Player',
                     title='Field Goal Percentage by Player',
                     orientation='h',
                     color='FG%',
                     color_continuous_scale='Viridis')

        fig.update_layout(
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Efficiency by Shot Type")

        selected_player = st.selectbox("Select a player:", players)
        player_data = df[df['PLAYER_NAME'] == selected_player]

        shot_efficiency = []
        for shot_type in ['1PT Free Throw', '2PT Field Goal', '3PT Field Goal']:
            type_data = player_data[player_data['SHOT_TYPE'] == shot_type]
            if len(type_data) > 0:
                fg_pct = type_data['SHOT_MADE_FLAG'].mean() * 100
                attempts = len(type_data)
                made = int(type_data['SHOT_MADE_FLAG'].sum())

                shot_efficiency.append({
                    'Shot Type': shot_type,
                    'FG%': round(fg_pct, 1),
                    'Attempts': attempts,
                    'Made': made
                })

        col1, col2 = st.columns([1.5, 1])

        with col1:
            df_efficiency = pd.DataFrame(shot_efficiency)
            fig = px.bar(df_efficiency, x='Shot Type', y='FG%',
                         title=f'{selected_player} - Efficiency by Shot Type',
                         color='FG%',
                         color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(df_efficiency, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### Impact of Game Situations on Player Performance")

        selected_player = st.selectbox("Select player for situation analysis:", players, key='player2')
        player_data = df[df['PLAYER_NAME'] == selected_player]

        col1, col2 = st.columns(2)

        with col1:
            # Home vs Away
            home_away = player_data.groupby('IS_HOME')['SHOT_MADE_FLAG'].mean() * 100

            fig = px.bar(x=home_away.index, y=home_away.values,
                         title=f'{selected_player} - Home vs Away',
                         labels={'x': 'Situation', 'y': 'FG%'},
                         color=home_away.values,
                         color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Clutch vs Normal
            clutch_away = player_data.groupby('IsClutchTime')['SHOT_MADE_FLAG'].mean() * 100

            fig = px.bar(x=['Normal', 'Clutch'], y=clutch_away.values,
                         title=f'{selected_player} - Clutch Performance',
                         labels={'x': 'Time', 'y': 'FG%'},
                         color=clutch_away.values,
                         color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
