import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots


def render():
    """Data Analysis & Visualization"""
    st.markdown('<div class="section-title">📈 Data Analysis & Visualization</div>',
                unsafe_allow_html=True)

    df_main = get_top_20_shots()

    tab1, tab2, tab3 = st.tabs(["Shot Distribution", "Player Comparison", "Statistical Analysis"])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Shot Distribution
    # ══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Shots by Action Type & Distance")

        col1, col2 = st.columns(2)

        with col1:
            shot_counts = df_main.groupby("MAIN_ACTION_TYPE").size().reset_index(name="count")
            fig = px.bar(
                shot_counts, x="MAIN_ACTION_TYPE", y="count",
                title="Number of Shots by Shot Type",
                labels={"count": "Number of Shots", "MAIN_ACTION_TYPE": "Shot Type"},
                color="count", color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            dist_counts = (
                df_main[df_main["SHOT_DISTANCE"] <= 50]
                .groupby("SHOT_DISTANCE").size().sort_index()
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dist_counts.index, y=dist_counts.values,
                mode="lines+markers", name="Number of Shots",
                line=dict(color="#e74c3c", width=3), marker=dict(size=4),
            ))
            fig.add_vline(x=15, line_width=2, line_dash="dash", line_color="blue",
                          annotation_text="FT line")
            fig.add_vline(x=22, line_width=2, line_dash="dash", line_color="green",
                          annotation_text="3PT line")
            fig.update_layout(
                title="Number of Shots by Distance",
                xaxis_title="Distance (feet)", yaxis_title="Number of Shots",
                hovermode="x unified", height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            min_counts = df_main.groupby("MINUTES_REMAINING").size().reset_index(name="count")
            fig = px.bar(
                min_counts, x="MINUTES_REMAINING", y="count",
                title="Minutes Remaining",
                labels={"count": "Number of Shots", "MINUTES_REMAINING": "Minutes remaining"},
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            player_shots = df_main.groupby("PLAYER_NAME").size().reset_index(name="count")
            fig = px.bar(
                player_shots, x="count", y="PLAYER_NAME",
                title="Number of Shots by Player",
                labels={"PLAYER_NAME": "Player", "count": "Number of Shots"},
                orientation="h", color="count", color_continuous_scale="RdYlGn",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Player Comparison
    # ══════════════════════════════════════════════════════════════
    with tab2:
        COLS, ROWS = 5, 4  # 20 players in a 4 × 5 grid

        player_stats = df_main.groupby("PLAYER_NAME").agg(
            total_shots=("SHOT_MADE_FLAG", "count"),
            overall_fg=("SHOT_MADE_FLAG", "mean"),
        ).reset_index()
        players = (
            player_stats.sort_values("total_shots", ascending=False)["PLAYER_NAME"].tolist()
        )

        # ── Shot-type distribution grid ──────────────────────────
        st.markdown("### Shot Type Distribution per Player")
        st.caption(
            "Each bar shows the share of shots by action type (normalized to 100%). "
            "Free Throws are separated from 'Other'."
        )

        # only copy because we mutate a new column
        df_typed = df_main.copy()
        df_typed["SHOT_TYPE_CLEAN"] = df_typed["MAIN_ACTION_TYPE"]
        df_typed.loc[
            (df_typed["MAIN_ACTION_TYPE"] == "Other") &
            (df_typed["ACTION_TYPE"] == "Free Throw"),
            "SHOT_TYPE_CLEAN",
        ] = "Free Throw"

        type_order = ["Jump", "Layup", "Dunk", "Hook", "Free Throw", "Other"]
        type_colors = {
            "Jump": "#2196F3", "Layup": "#4CAF50", "Dunk": "#FF5722",
            "Hook": "#9C27B0", "Free Throw": "#FFC107", "Other": "#9E9E9E",
        }

        # simpler normalization
        raw_counts = df_typed.groupby(["PLAYER_NAME", "SHOT_TYPE_CLEAN"]).size().unstack(fill_value=0)
        dist_norm = raw_counts.div(raw_counts.sum(axis=1), axis=0) * 100

        fig_types = make_subplots(
            rows=ROWS, cols=COLS, subplot_titles=players,
            vertical_spacing=0.08, horizontal_spacing=0.06,
        )
        for idx, player in enumerate(players):
            row, col = divmod(idx, COLS)
            row, col = row + 1, col + 1
            series = dist_norm.loc[player] if player in dist_norm.index else pd.Series(dtype=float)
            for t in type_order:
                fig_types.add_trace(
                    go.Bar(
                        x=[t], y=[series.get(t, 0.0)],
                        marker_color=type_colors[t], name=t,
                        legendgroup=t, showlegend=(idx == 0),
                    ),
                    row=row, col=col,
                )
        fig_types.update_layout(
            height=1600, barmode="group",
            legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=140),
        )
        fig_types.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(fig_types, use_container_width=True)

        # ── Shot density map ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### Player Shot Density Maps")
        st.image("streamlit_includes/data/shot_density.png", use_container_width=True)

        # ── Shot volume by exact distance ────────────────────────
        st.markdown("---")
        st.markdown("### Shot Volume by Distance — Per Player")
        st.caption("Attempts at each exact distance (free throws excluded).")

        # copy because we filter rows
        df_fg = df_main[df_main["SHOT_TYPE"] != "1PT Free Throw"]

        dist_vol = (
            df_fg.groupby(["PLAYER_NAME", "SHOT_DISTANCE"]).size()
            .reset_index(name="count")
        )
        dist_vol = dist_vol[dist_vol["count"] >= 20]

        fig_dist = make_subplots(
            rows=ROWS, cols=COLS, subplot_titles=players,
            vertical_spacing=0.08, horizontal_spacing=0.06,
        )
        for idx, player in enumerate(players):
            row, col = divmod(idx, COLS)
            row, col = row + 1, col + 1
            pdata = dist_vol[dist_vol["PLAYER_NAME"] == player].sort_values("SHOT_DISTANCE")
            if not pdata.empty:
                fig_dist.add_trace(
                    go.Bar(
                        x=pdata["SHOT_DISTANCE"], y=pdata["count"],
                        marker_color="#1565C0", name=player, showlegend=False,
                    ),
                    row=row, col=col,
                )
        fig_dist.update_layout(
            height=1600, barmode="group", margin=dict(t=120),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── Hit rate vs exact distance ───────────────────────────
        st.markdown("---")
        st.markdown("### Hit Rate by Shot Distance — All Players")
        st.caption("FG% by exact shot distance. Distances with < 10 attempts per player are excluded.")

        hr_stats = (
            df_main.groupby(["PLAYER_NAME", "SHOT_DISTANCE"])["SHOT_MADE_FLAG"]
            .agg(["mean", "count"]).reset_index()
        )
        hr_stats = hr_stats[hr_stats["count"] >= 10]
        hr_stats["fg_pct"] = hr_stats["mean"] * 100

        fig_hr = go.Figure()
        for player in players:
            pdata = hr_stats[hr_stats["PLAYER_NAME"] == player].sort_values("SHOT_DISTANCE")
            if not pdata.empty:
                fig_hr.add_trace(go.Scatter(
                    x=pdata["SHOT_DISTANCE"], y=pdata["fg_pct"],
                    mode="lines+markers", name=player,
                    marker=dict(size=5), line=dict(width=2), connectgaps=False,
                ))
        fig_hr.update_layout(
            title="FG% by Shot Distance — All Players",
            xaxis_title="Shot Distance (ft)", yaxis_title="FG%",
            yaxis=dict(ticksuffix="%", range=[0, 100]),
            height=600, hovermode="x unified",
            legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
            margin=dict(r=180),
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — Statistical Analysis
    # ══════════════════════════════════════════════════════════════
    with tab3:

        # ── 3. Correlation matrix ────────────────────────────────
        st.markdown("### Correlation Matrix")
        st.caption(
            "Point-biserial / Pearson correlations between selected variables. "
            "Categorical variables are label-encoded. **SHOT_MADE_FLAG** (target) "
            "is highlighted with a border."
        )

        CORR_COLS = [
            "SHOT_DISTANCE", "SHOT_ZONE_BASIC", "ACTION_TYPE",
            "PERIOD_x", "is_playoffs", "IS_HOME",
            "scoreMarginBeforeShot", "TimeRemainingInPeriod",
            "OPPONENT_INTERFERED", "ANGLE_SECTOR", "SHOT_MADE_FLAG",
        ]

        available = [c for c in CORR_COLS if c in df_main.columns]
        df_corr = df_main[available].copy()

        # label-encode categoricals
        for col in df_corr.select_dtypes(include=["object", "category"]).columns:
            df_corr[col] = pd.factorize(df_corr[col])[0].astype(float)

        # ensure numeric & drop rows with NaN
        df_corr = df_corr.apply(pd.to_numeric, errors="coerce").dropna()

        corr_matrix = df_corr.corr()

        # build heatmap with SHOT_MADE_FLAG highlighted
        target = "SHOT_MADE_FLAG"
        cols_order = [c for c in available if c != target] + [target]
        corr_matrix = corr_matrix.loc[cols_order, cols_order]

        z = corr_matrix.values
        labels = corr_matrix.columns.tolist()
        n = len(labels)
        target_idx = labels.index(target)

        text_matrix = [[f"{v:.2f}" for v in row] for row in z]

        fig_corr = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels, text=text_matrix,
            texttemplate="%{text}",
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="r"),
        ))

        # draw a rectangle around the target column & row
        for i in range(n):
            # target column bar
            fig_corr.add_shape(
                type="rect",
                x0=target_idx - 0.5, x1=target_idx + 0.5,
                y0=i - 0.5, y1=i + 0.5,
                line=dict(color="gold", width=2),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
            # target row bar
            fig_corr.add_shape(
                type="rect",
                x0=i - 0.5, x1=i + 0.5,
                y0=target_idx - 0.5, y1=target_idx + 0.5,
                line=dict(color="gold", width=2),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

        fig_corr.update_layout(
            title="Correlation Matrix (target = SHOT_MADE_FLAG, highlighted in gold)",
            height=620, width=800,
            xaxis=dict(tickangle=-40),
            margin=dict(l=100, b=120),
        )
        st.plotly_chart(fig_corr, use_container_width=False)
        st.caption(
            "Categorical variables (SHOT_ZONE_BASIC, ACTION_TYPE) are label-encoded "
            "before computing Pearson correlation — treat their coefficients as indicative only."
        )

        # ── 1. Shot type × made/missed ───────────────────────────
        st.markdown("---")
        st.markdown("### Shot Type — Made vs Missed")
        st.caption(
            "Each SHOT_TYPE group contains two bars (Made / Missed) "
            "that together sum to 100% of that type's attempts."
        )

        type_counts = (
            df_main.groupby(["SHOT_TYPE", "SHOT_MADE_FLAG"])
            .size()
            .reset_index(name="count")
        )

        total = type_counts["count"].sum()
        type_counts["pct"] = type_counts["count"] / total * 100

        type_counts["Outcome"] = type_counts["SHOT_MADE_FLAG"].map({0: "Missed", 1: "Made"})

        type_label_order = ["1PT Free Throw", "2PT Field Goal", "3PT Field Goal"]
        type_counts["SHOT_TYPE"] = pd.Categorical(type_counts["SHOT_TYPE"],
                                                   categories=type_label_order, ordered=True)
        type_counts = type_counts.sort_values("SHOT_TYPE")

        fig_st = px.bar(
            type_counts,
            x="SHOT_TYPE", y="pct", color="Outcome", barmode="group",
            color_discrete_map={"Made": "#2ecc71", "Missed": "#e74c3c"},
            text=type_counts["pct"].map(lambda v: f"{v:.1f}%"),
            labels={"pct": "Share (%)", "SHOT_TYPE": "Shot Type"},
            title="Made vs Missed by Shot Type (% within each type)",
            category_orders={"SHOT_TYPE": type_label_order, "Outcome": ["Made", "Missed"]},
            height=450, width=600,
        )
        fig_st.update_traces(textposition="outside")
        fig_st.update_layout(yaxis=dict(ticksuffix="%", range=[0, 90]))
        st.plotly_chart(fig_st, use_container_width=False)


        # ── 2. Angle vs Distance ─────────────────────────────────
        st.markdown("---")
        st.markdown("### Angle vs Distance — FG% by Angle Sector and Shot Zone")
        st.caption(
            "Two angle-sector definitions compared. Each subplot = one distance zone. "
            "Y-axis = FG%, X-axis = angle sector."
        )

        ZONES = ["Less Than 8 ft.", "8-16 ft.", "16-24 ft.", "24+ ft.", "Back Court Shot"]
        ZONES_SHORT = ["< 8 ft", "8–16 ft", "16–24 ft", "24+ ft", "Back Court"]

        # only copy because we add two temporary columns
        df_angle = df_main[df_main["ABS_ANGLE"].notna()].copy()

        # sector definitions
        def make_sectors(df, breaks, labels):
            bins  = [0] + breaks + [180]
            return pd.cut(df["ABS_ANGLE"], bins=bins, labels=labels, right=True,
                          include_lowest=True)

        sector_defs = {
            "45/90°": {
                "breaks": [45, 90, 135],
                "labels": ["0–45°", "45–90°", "90–135°", ">135°"],
            },
            "55/100°": {
                "breaks": [55, 100, 145],
                "labels": ["0–55°", "55–100°", "100–145°", ">145°"],
            },
        }

        sector_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

        for def_name, params in sector_defs.items():
            st.markdown(f"**Sector definition: {def_name}**")

            df_angle["_sector"] = make_sectors(
                df_angle, params["breaks"], params["labels"]
            )

            fig_ang = make_subplots(
                rows=1, cols=len(ZONES),
                subplot_titles=ZONES_SHORT,
                shared_yaxes=True,
                horizontal_spacing=0.04,
            )

            for zi, zone in enumerate(ZONES):
                zone_data = df_angle[df_angle["SHOT_ZONE_RANGE"] == zone]
                hr = (
                    zone_data.groupby("_sector", observed=True)["SHOT_MADE_FLAG"]
                    .agg(["mean", "count"]).reset_index()
                )
                # suppress tiny samples
                hr.loc[hr["count"] < 20, "mean"] = np.nan
                hr["fg_pct"] = hr["mean"] * 100

                for si, sector_label in enumerate(params["labels"]):
                    row_data = hr[hr["_sector"] == sector_label]
                    val = row_data["fg_pct"].values[0] if not row_data.empty else np.nan
                    fig_ang.add_trace(
                        go.Bar(
                            x=[sector_label], y=[val],
                            marker_color=sector_colors[si],
                            name=sector_label,
                            legendgroup=sector_label,
                            showlegend=(zi == 0),
                        ),
                        row=1, col=zi + 1,
                    )

            fig_ang.update_layout(
                height=380, barmode="group",
                legend=dict(orientation="h", y=1.18, x=0),
                margin=dict(t=80, b=40),
            )
            fig_ang.update_yaxes(ticksuffix="%", range=[0, 100])
            fig_ang.update_xaxes(showticklabels=False)
            st.plotly_chart(fig_ang, use_container_width=True)

