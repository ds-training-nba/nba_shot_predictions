import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots


def build_action_type(df):
    df = df.copy()

    is_ft = (
        (df["MAIN_ACTION_TYPE"] == "Other") &
        (df["ACTION_TYPE"] == "Free Throw")
    )

    df["ACTION_TYPE_CLEAN"] = df["MAIN_ACTION_TYPE"]

    # Free Throw separate
    df.loc[is_ft, "ACTION_TYPE_CLEAN"] = "Free Throw"

    # Remove Free Throw from Other
    df.loc[
        (df["MAIN_ACTION_TYPE"] == "Other") & (~is_ft),
        "ACTION_TYPE_CLEAN"
    ] = "Other"

    return df


def render():
    st.markdown(
        '<div class="section-title">🏀 Player Insights & Shooting Patterns</div>',
        unsafe_allow_html=True
    )

    df = get_top_20_shots()
    players = load_top_20_players()
    colors = px.colors.qualitative.Set2

    green = "#2ecc71"
    red = "#e74c3c"

    # =========================================================
    # TAB 1 — OVERVIEW
    # =========================================================
    tab1, tab2 = st.tabs([
        "Overview",
        "Player Analysis"
    ])

    with tab1:
        st.markdown("### Top Players Overview")

        player_stats = df.groupby("PLAYER_NAME").agg(
            FG_pct=("SHOT_MADE_FLAG", lambda x: x.mean() * 100),
            total_shots=("SHOT_MADE_FLAG", "count"),
            points=("points", "sum"),
        ).reset_index()

        player_stats = player_stats[player_stats["PLAYER_NAME"].isin(players)]
        player_stats = player_stats.sort_values("FG_pct", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                player_stats,
                x="FG_pct",
                y="PLAYER_NAME",
                orientation="h",
                color="FG_pct",
                color_continuous_scale="Viridis",
                title="FG% by Player"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                player_stats,
                x="total_shots",
                y="FG_pct",
                size="points",
                hover_name="PLAYER_NAME",
                title="Efficiency vs Volume"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(player_stats, use_container_width=True)

    # =========================================================
    # TAB 2 — PLAYER ANALYSIS
    # =========================================================
    with tab2:
        st.markdown("### Player Analysis")

        player = st.selectbox("Select player", players)

        d = df[df["PLAYER_NAME"] == player].copy()
        d = build_action_type(d)

        col1, col2, col3, col4 = st.columns(4)

        # =====================================================
        # SHOT PROFILE
        # =====================================================
        with col1:
            data = d.groupby("SHOT_TYPE").agg(
                fg_pct=("SHOT_MADE_FLAG", "mean"),
                volume=("SHOT_MADE_FLAG", "count")
            ).reset_index()

            data["fg_pct"] *= 100

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=data["SHOT_TYPE"],
                y=data["fg_pct"],
                name="FG%",
                marker_color=colors[:len(data)]
            ))

            fig.add_trace(go.Scatter(
                x=data["SHOT_TYPE"],
                y=data["volume"],
                name="Volume",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=red)
            ))

            fig.update_layout(
                title="Shot Type FG% + Volume",
                yaxis=dict(title="FG%"),
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        with col2:
            data = d.groupby("ACTION_TYPE_CLEAN").agg(
                fg_pct=("SHOT_MADE_FLAG", "mean"),
                volume=("SHOT_MADE_FLAG", "count")
            ).reset_index()

            data["fg_pct"] *= 100

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=data["ACTION_TYPE_CLEAN"],
                y=data["fg_pct"],
                name="FG%",
                marker_color=colors[:len(data)]
            ))

            fig.add_trace(go.Scatter(
                x=data["ACTION_TYPE_CLEAN"],
                y=data["volume"],
                name="Volume",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=red)
            ))

            fig.update_layout(
                title="Action Type FG% + Volume",
                yaxis=dict(title="FG%"),
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        with col3:
            data = d.groupby("SHOT_ZONE_BASIC").agg(
                fg_pct=("SHOT_MADE_FLAG", "mean"),
                volume=("SHOT_MADE_FLAG", "count")
            ).reset_index()

            data["fg_pct"] *= 100

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=data["SHOT_ZONE_BASIC"],
                y=data["fg_pct"],
                name="FG%",
                marker_color=colors[:len(data)]
            ))

            fig.add_trace(go.Scatter(
                x=data["SHOT_ZONE_BASIC"],
                y=data["volume"],
                name="Volume",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=red)
            ))

            fig.update_layout(
                title="Shot Zone FG% + Volume",
                yaxis=dict(title="FG%"),
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        with col4:
            d["DIST_RANGE_CLEAN"] = d["SHOT_ZONE_RANGE"]

            is_ft = (
                    (d["MAIN_ACTION_TYPE"] == "Other") &
                    (d["ACTION_TYPE"] == "Free Throw")
            )

            d.loc[is_ft, "DIST_RANGE_CLEAN"] = "Free Throw"

            data = d.groupby("DIST_RANGE_CLEAN").agg(
                fg_pct=("SHOT_MADE_FLAG", "mean"),
                volume=("SHOT_MADE_FLAG", "count")
            ).reset_index()

            data["fg_pct"] *= 100

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=data["DIST_RANGE_CLEAN"],
                y=data["fg_pct"],
                name="FG%",
                marker_color=colors[:len(data)]
            ))

            fig.add_trace(go.Scatter(
                x=data["DIST_RANGE_CLEAN"],
                y=data["volume"],
                name="Volume",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=red)
            ))

            fig.update_layout(
                title="Distance Range FG% + Volume",
                yaxis=dict(title="FG%"),
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # GAME CONTEXT
        # =====================================================
        st.markdown("### Game Context Impact")

        col1, col2, col3 = st.columns(3)

        with col1:
            home_away = d.groupby("IS_HOME")["SHOT_MADE_FLAG"].mean() * 100

            fig = go.Figure()

            fig.add_bar(
                x=["Home"],
                y=[home_away.loc[1]],
                marker_color=green
            )

            fig.add_bar(
                x=["Away"],
                y=[home_away.loc[0]],
                marker_color=red
            )

            fig.update_layout(title="Home vs Away FG%", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            clutch = d.groupby("IsClutchTime")["SHOT_MADE_FLAG"].mean() * 100

            fig = go.Figure()

            fig.add_bar(
                x=["Normal"],
                y=[clutch.loc[0]],
                marker_color=green
            )

            fig.add_bar(
                x=["Clutch"],
                y=[clutch.loc[1]],
                marker_color=red
            )

            fig.update_layout(title="Clutch vs Normal FG%", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            ot = d.groupby("IsOvertime")["SHOT_MADE_FLAG"].mean() * 100

            fig = go.Figure()

            fig.add_bar(
                x=["Regular"],
                y=[ot.loc[0]],
                marker_color=green
            )

            fig.add_bar(
                x=["Overtime"],
                y=[ot.loc[1]],
                marker_color=red
            )

            fig.update_layout(title="Regular vs Overtime FG%", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # SPATIAL ANALYSIS
        # =====================================================
        st.markdown("### Spatial Shot Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                d,
                x="LOC_X",
                y="LOC_Y",
                color="SHOT_MADE_FLAG",
                color_discrete_map={0: red, 1: green},
                opacity=0.5,
                title="Shot Chart"
            )

            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            d = d.copy()

            d["ANGLE_SECTOR_BIN"] = pd.cut(
                d["ANGLE"],
                bins=[0, 45, 95, 135, 180],
                labels=["0°–±45°", "±45°–±95°", "±95°–±135°", ">±135°"]
            )

            angle = d.groupby("ANGLE_SECTOR_BIN")["SHOT_MADE_FLAG"].mean() * 100

            fig = px.bar(
                x=angle.index,
                y=angle.values,
                title="FG% by Angle Sector",
                color=[green if i < 2 else red for i in range(len(angle))]
            )

            fig.update_layout(showlegend=False)

            st.plotly_chart(fig, use_container_width=True)