import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots

def render():
    """Data Analysis & Visualization"""
    st.markdown('<div class="section-title">📈 Data Analysis & Visualization</div>', unsafe_allow_html=True)

    df = get_top_20_shots()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Shot Distribution", "Player Comparison", "Game Situations", "Positional Analysis"])

    with tab1:
        st.markdown("### Shot Success by Type & Distance")

        col1, col2 = st.columns(2)

        with col1:
            # Shot type success rate
            shot_success = df.groupby('SHOT_TYPE')['SHOT_MADE_FLAG'].agg(['mean', 'count']).reset_index()
            shot_success['mean'] = shot_success['mean'] * 100

            fig = px.bar(shot_success, x='SHOT_TYPE', y='mean',
                         title="Success Rate by Shot Type",
                         labels={'mean': 'Success Rate (%)', 'SHOT_TYPE': 'Shot Type'},
                         color='mean',
                         color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_distance = df.groupby("SHOT_DISTANCE")["SHOT_MADE_FLAG"].mean() * 100
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
    with tab2:
        st.markdown("### Player Shooting Efficiency (All Players)")

        # --- aggregate ---
        player_stats = df.groupby("PLAYER_NAME").agg(
            total_shots=("SHOT_MADE_FLAG", "count"),
            overall_fg=("SHOT_MADE_FLAG", "mean"),
        ).reset_index()

        # filter noise
        player_stats = player_stats[player_stats["total_shots"] >= 100]

        # split by shot type
        one_pt = df[df["SHOT_TYPE"] == "1PT Free Throw"].groupby("PLAYER_NAME")["SHOT_MADE_FLAG"].mean()
        two_pt = df[df["SHOT_TYPE"] == "2PT Field Goal"].groupby("PLAYER_NAME")["SHOT_MADE_FLAG"].mean()
        three_pt = df[df["SHOT_TYPE"] == "3PT Field Goal"].groupby("PLAYER_NAME")["SHOT_MADE_FLAG"].mean()

        player_stats["1PT_FG"] = player_stats["PLAYER_NAME"].map(one_pt)
        player_stats["2PT_FG"] = player_stats["PLAYER_NAME"].map(two_pt)
        player_stats["3PT_FG"] = player_stats["PLAYER_NAME"].map(three_pt)

        # % conversion
        player_stats["Overall FG%"] = player_stats["overall_fg"] * 100
        player_stats["1PT FG%"] = player_stats["1PT_FG"] * 100
        player_stats["2PT FG%"] = player_stats["2PT_FG"] * 100
        player_stats["3PT FG%"] = player_stats["3PT_FG"] * 100

        player_stats = player_stats.sort_values("Overall FG%", ascending=False)

        # --- plot ---
        fig = go.Figure()

        # --- Bars (grouped) ---
        fig.add_trace(go.Bar(
            x=player_stats["PLAYER_NAME"],
            y=player_stats["1PT FG%"],
            name="1PT FG%"
        ))

        fig.add_trace(go.Bar(
            x=player_stats["PLAYER_NAME"],
            y=player_stats["2PT FG%"],
            name="2PT FG%"
        ))

        fig.add_trace(go.Bar(
            x=player_stats["PLAYER_NAME"],
            y=player_stats["3PT FG%"],
            name="3PT FG%"
        ))

        # --- OVERALL (overlay as line, not bar) ---
        fig.add_trace(go.Scatter(
            x=player_stats["PLAYER_NAME"],
            y=player_stats["Overall FG%"],
            mode="lines+markers",
            name="Overall FG%",
            line=dict(color="black", width=3),
            marker=dict(size=7)
        ))

        fig.update_layout(
            barmode="group",  # 🔥 важно: НЕ overlay
            title="Player Shooting Efficiency",
            xaxis_title="Player",
            yaxis_title="FG%",
            height=550,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Impact of Game Situations")

        col1, col2 = st.columns(2)

        with col1:
            # Home vs Away
            home_away = df.groupby('IS_HOME')['SHOT_MADE_FLAG'].mean() * 100

            fig = px.bar(x=home_away.index, y=home_away.values,
                         title="Success Rate: Home vs Away",
                         labels={'x': 'Game Situation', 'y': 'Success Rate (%)'},
                         color=home_away.values,
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Clutch vs Normal
            clutch_stats = df.groupby('IsClutchTime')['SHOT_MADE_FLAG'].mean() * 100

            fig = px.pie(
                values=clutch_stats.values,
                names=['Normal', 'Clutch'],
                title="Success Rate: Clutch vs Normal Time",
                color_discrete_map={'Normal': '#3498db', 'Clutch': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Player Shot Density Maps")

        st.image(
            "streamlit_includes/data/shot_density.png",
            use_container_width=True
        )