import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional

from streamlit_includes.data.load_models import load_pipelines
from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots


# ── helpers ──────────────────────────────────────────────────────────────────

def _gauge(prob: float) -> go.Figure:
    color = "#e74c3c" if prob < 0.4 else "#f39c12" if prob < 0.55 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "Shot Probability", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0, 40],  "color": "#fdecea"},
                {"range": [40, 55], "color": "#fef9e7"},
                {"range": [55, 100],"color": "#eafaf1"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": prob * 100,
            },
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _historical_avg(df: pd.DataFrame, player: str,
                    shot_zone_basic: str, shot_type: str) -> Optional[float]:
    mask = (
        (df["PLAYER_NAME"] == player) &
        (df["SHOT_ZONE_BASIC"] == shot_zone_basic) &
        (df["SHOT_TYPE"] == shot_type)
    )
    sub = df[mask]
    return sub["SHOT_MADE_FLAG"].mean() if len(sub) >= 20 else None


# ── page ─────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        '<div class="section-title">🎯 Shot Probability Predictor</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Configure a shot scenario and the **XGBoost model** will estimate "
        "the probability of it going in."
    )

    pipelines = load_pipelines()
    pipe      = pipelines["XGBoost"]
    players   = load_top_20_players()
    df        = get_top_20_shots()

    # ── sidebar-style input panel ────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown("### ⚙️ Shot Parameters")

        player = st.selectbox("Player", players)

        shot_type = st.selectbox(
            "Shot Type",
            ["2PT Field Goal", "3PT Field Goal", "1PT Free Throw"]
        )

        shot_zone_basic = st.selectbox(
            "Zone (Basic)",
            ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
             "Left Corner 3", "Right Corner 3", "Above the Break 3", "Backcourt"]
        )

        shot_zone_area = st.selectbox(
            "Zone (Area)",
            ["Center(C)", "Left Side(L)", "Right Side(R)",
             "Left Side Center(LC)", "Right Side Center(RC)", "Back Court(BC)"]
        )

        shot_zone_range = st.selectbox(
            "Zone (Range)",
            ["Less Than 8 ft.", "8-16 ft.", "16-24 ft.", "24+ ft.", "Back Court Shot"]
        )

        action_type = st.selectbox(
            "Action Type",
            ["Jump Shot", "Layup Shot", "Driving Layup Shot", "Pullup Jump shot",
             "Step Back Jump shot", "Turnaround Jump Shot", "Hook Shot",
             "Floating Jump shot", "Dunk Shot", "Running Layup Shot"]
        )

        main_action_type = st.selectbox(
            "Main Action Type",
            ["Jump Shot", "Layup", "Dunk", "Hook Shot", "Tip Shot"]
        )

        shot_distance = st.slider("Shot Distance (ft)", 0, 40, 15)

        st.markdown("---")
        st.markdown("#### 📍 Court Location")
        loc_x = st.slider("LOC_X (horizontal, ft × 10)", -250, 250, 0)
        loc_y = st.slider("LOC_Y (vertical, ft × 10)", -50, 900, 150)

        # derive geometry
        angle      = float(np.arctan2(loc_x, max(loc_y, 1)))
        angle_sin  = float(np.sin(angle))
        angle_cos  = float(np.cos(angle))
        abs_angle  = float(abs(angle))
        angle_sect = int(np.digitize(angle, bins=[-1.0, -0.3, 0.3, 1.0]))

        st.markdown("---")
        st.markdown("#### 🕐 Game Context")

        period             = st.selectbox("Period", [1, 2, 3, 4])
        time_in_period     = st.slider("Time Remaining in Period (s)", 0, 720, 360)
        time_in_game       = st.slider("Time Remaining in Game (s)", 0, 2880, 1440)
        score_margin       = st.slider("Score Margin (home − away)", -30, 30, 0)
        score_home         = st.number_input("Home Score Before Shot", 0, 200, 50)
        score_away         = st.number_input("Away Score Before Shot", 0, 200, 50)
        total_played       = st.slider("Total Played Time (s)", 0, 2880, 1440)
        overtime_number    = st.number_input("Overtime Number", 0, 5, 0)

        st.markdown("---")
        st.markdown("#### 🏟️ Situation")

        is_home            = st.checkbox("Home Game", value=True)
        is_playoffs        = st.checkbox("Playoffs", value=False)
        is_overtime        = overtime_number > 0
        is_clutch          = (time_in_game <= 300) and (abs(score_margin) <= 5)
        opponent_interfered = st.checkbox("Shot Contested (opponent interfered)", value=False)

        # internal IDs — use median from dataset for selected player
        player_id = int(df[df["PLAYER_NAME"] == player]["PLAYER_ID"].median()) \
            if "PLAYER_ID" in df.columns else 0
        team_id   = int(df[df["PLAYER_NAME"] == player]["TEAM_ID"].median()) \
            if "TEAM_ID" in df.columns else 0

        predict_btn = st.button("🏀 Predict", use_container_width=True, type="primary")

    # ── prediction output ────────────────────────────────────────────────────
    with col_right:
        st.markdown("### 📊 Prediction Result")

        if not predict_btn:
            st.info("Configure the shot parameters on the left and click **Predict**.")
            st.stop()

        # build input row — column order must match training
        input_dict = {
            # numerical
            "SHOT_DISTANCE":        shot_distance,
            "LOC_X":                loc_x,
            "LOC_Y":                loc_y,
            "ANGLE_SIN":            angle_sin,
            "ANGLE_COS":            angle_cos,
            "ANGLE":                angle,
            "ANGLE_SECTOR":         angle_sect,
            "ABS_ANGLE":            abs_angle,
            "TimeRemainingInPeriod": time_in_period,
            "TimeRemainingInGame":  time_in_game,
            "scoreMarginBeforeShot": score_margin,
            "scoreHomeBeforeShot":  score_home,
            "scoreAwayBeforeShot":  score_away,
            "PERIOD_x":             period,
            "OvertimeNumber":       overtime_number,
            "TEAM_ID":              team_id,
            "PLAYER_ID":            player_id,
            "TotalPlayedTime":      total_played,
            # categorical
            "SHOT_TYPE":            shot_type,
            "SHOT_ZONE_RANGE":      shot_zone_range,
            "SHOT_ZONE_BASIC":      shot_zone_basic,
            "SHOT_ZONE_AREA":       shot_zone_area,
            "ACTION_TYPE":          action_type,
            "MAIN_ACTION_TYPE":     main_action_type,
            "PLAYER_NAME":          player,
            # boolean
            "is_playoffs":          is_playoffs,
            "IS_HOME":              is_home,
            "IsOvertime":           is_overtime,
            "IsClutchTime":         is_clutch,
            "OPPONENT_INTERFERED":  opponent_interfered,
        }

        X_input = pd.DataFrame([input_dict])

        try:
            prob = pipe.predict_proba(X_input)[0][1]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # gauge
        st.plotly_chart(_gauge(prob), use_container_width=True)

        # verdict
        verdict = "🟢 High probability — good look!" if prob >= 0.55 else \
                  "🟡 Moderate — could go either way" if prob >= 0.40 else \
                  "🔴 Low probability — tough shot"
        st.markdown(f"### {verdict}")

        st.markdown("---")

        # comparison with historical average
        hist = _historical_avg(df, player, shot_zone_basic, shot_type)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model Prediction", f"{prob:.1%}")
        if hist is not None:
            delta = prob - hist
            c2.metric("Historical Avg (this zone)", f"{hist:.1%}",
                      delta=f"{delta:+.1%}")
            c3.metric("Difference", f"{abs(delta):.1%}",
                      delta="above avg" if delta >= 0 else "below avg")
        else:
            c2.metric("Historical Avg", "n/a", delta="< 20 samples")
            c3.metric("Difference", "—")

        # clutch callout
        if is_clutch:
            st.warning(
                "⚡ **Clutch situation detected** — last 5 min, margin ≤ 5 pts. "
                "Historical clutch FG% tends to drop by 3–5 percentage points on average."
            )

        # context breakdown
        with st.expander("Show input summary"):
            st.json(input_dict)
