import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
from processing.compute_columns import add_shot_main_action_type_column
from processing.filtering import filter_for_players

from streamlit_includes.data.helpers_trajectory_demo import recompute_additional_features

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

UNKNOWN_PLAYER_IDX = 0
UNKNOWN_ACTION_IDX = 0

COURT_X_MIN = 47.0
COURT_X_MAX = 94.0
COURT_Y_MIN = 0.0
COURT_Y_MAX = 50.0

CANVAS_W = 700
CANVAS_H = 700

PLAYER_SELECTION = (
    ["shooter"]
    + [f"defender{i}" for i in range(1, 6)]
    + [f"attacker{i}" for i in range(1, 5)]
)

ARTIFACT_DIR = "streamlit_includes/data/streamlit_trajectory_artifacts"


# --------------------------------------------------------------------
# COORDINATE CONVERSION
# --------------------------------------------------------------------

def court_to_canvas(x, y):
    """
    Convert court coordinates to canvas pixels.
    Canvas:
    px left -> right corresponds to court y
    py top -> bottom corresponds to court x reversed
    """
    px = (y - COURT_Y_MIN) / (COURT_Y_MAX - COURT_Y_MIN) * CANVAS_W
    py = (COURT_X_MAX - x) / (COURT_X_MAX - COURT_X_MIN) * CANVAS_H
    return px, py


def canvas_to_court(px, py):
    y = COURT_Y_MIN + px / CANVAS_W * (COURT_Y_MAX - COURT_Y_MIN)
    x = COURT_X_MAX - py / CANVAS_H * (COURT_X_MAX - COURT_X_MIN)
    return x, y


# --------------------------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------------------------

@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, "deepNN.keras"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "deepNN_scaler.pkl"))
    player_to_idx = joblib.load(os.path.join(ARTIFACT_DIR, "deepNN_player_to_idx.pkl"))
    action_to_idx = joblib.load(os.path.join(ARTIFACT_DIR, "deepNN_action_to_idx.pkl"))
    continuous_features = joblib.load(os.path.join(ARTIFACT_DIR, "deepNN_continuous_features.pkl"))

    return model, scaler, player_to_idx, action_to_idx, continuous_features


@st.cache_data
def load_examples():
    df = pd.read_parquet(os.path.join(ARTIFACT_DIR, "example_shots.parquet") )
    df = add_shot_main_action_type_column(df)
    df = filter_for_players(df)
    df = df.loc[df['MAIN_ACTION_TYPE'] != 'Dunk']

    return df


# ============================================================
# COURT BACKGROUND
# ============================================================

@st.cache_data
def make_court_background():
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(img)

    def draw_arc_court(center_x, center_y, width, height, theta1, theta2,
                    line_width=2, n_points=120):

        a = width / 2.0
        b = height / 2.0

        thetas = np.linspace(np.deg2rad(theta1), np.deg2rad(theta2), n_points)

        points = []
        for t in thetas:
            x = center_x + b * np.sin(t)
            y = center_y + a * np.cos(t)
            points.append((x, y))

        line_court(points, width=line_width)


    def rect_court(x0, y0, x1, y1, width=2):
        px0, py0 = court_to_canvas(x0, y0)
        px1, py1 = court_to_canvas(x1, y1)

        left = min(px0, px1)
        right = max(px0, px1)
        top = min(py0, py1)
        bottom = max(py0, py1)

        draw.rectangle(
            [left, top, right, bottom],
            outline="black",
            width=width
        )

    def circle_court(cx, cy, r, width=2):
        px, py = court_to_canvas(cx, cy)

        rx = r / (COURT_Y_MAX - COURT_Y_MIN) * CANVAS_W
        ry = r / (COURT_X_MAX - COURT_X_MIN) * CANVAS_H

        draw.ellipse(
            [px - rx, py - ry, px + rx, py + ry],
            outline="black",
            width=width
        )

    def line_court(points, width=2):
        canvas_points = [court_to_canvas(x, y)for x, y in points]
        draw.line(canvas_points,fill="black",width=width)

    # Half court
    rect_court(47, 0, 94, 50)

    # Paint/key
    rect_court(75, 17, 94, 33)

    # Inner box
    rect_court(75, 19, 94, 31)

    # Rim
    circle_court(89, 25, 0.75)

    # Free throw circle
    circle_court(75, 25, 6)

    # Backboard
    line_court([(90, 22), (90, 28)], width=2)

    # Corner three left
    line_court([(80, 3), (94, 3)], width=2)

    # Corner three right
    line_court([(80, 47), (94, 47)], width=2)

    # Three-point arc
    draw_arc_court(
        center_x=89.25,
        center_y=25,
        width=47.5,
        height=47.5,
        theta1=202,
        theta2=337.5,
        line_width=2,
        n_points=160
    )

    return img.resize((CANVAS_W, CANVAS_H)).convert("RGBA")


# --------------------------------------------------
# POSITION HELPERS
# --------------------------------------------------

def get_all_player_positions_from_row(row):
    pos_dict = {}

    for frame in range(6):
        pos_dict[f"shooter_x_t{frame}"] = row[f"shooter_x_t{frame}"]
        pos_dict[f"shooter_y_t{frame}"] = row[f"shooter_y_t{frame}"]

        for player in PLAYER_SELECTION:
            if player == "shooter":
                continue

            pos_dict[f"{player}_x_t{frame}"] = (pos_dict[f"shooter_x_t{frame}"] + row[f"{player}_dx_t{frame}"])
            pos_dict[f"{player}_y_t{frame}"] = (pos_dict[f"shooter_y_t{frame}"] + row[f"{player}_dy_t{frame}"])

    return pd.DataFrame([pos_dict])


def make_trajectory_circle_object(frame, x, y, color, radius=8):
    px, py = court_to_canvas(x, y)

    return {
        "type": "circle",
        "version": "4.4.0",
        "originX": "center",
        "originY": "center",
        "left": px,
        "top": py,
        "fill": color,
        "stroke": "black",
        "strokeWidth": 2,
        "radius": radius,
        "scaleX": 1,
        "scaleY": 1,
        "opacity": max(0.20, 1.0 - frame * 0.2),
        "selectable": True,
        "hasControls": False,
        "lockScalingX": True,
        "lockScalingY": True,
        "lockRotation": True,
    }


def make_trajectory_drawing(df_pos, only_shot=False):
    objects = []

    for player in PLAYER_SELECTION:
        if player == "shooter":
            color = "green"
        elif player.startswith("defender"):
            color = "red"
        else:
            color = "blue"

        num_frames = 1 if only_shot else 6

        for frame in range(num_frames):
            objects.append(
                make_trajectory_circle_object(
                    frame=frame,
                    x=float(df_pos[f"{player}_x_t{frame}"].values[0]),
                    y=float(df_pos[f"{player}_y_t{frame}"].values[0]),
                    color=color,
                )
            )

    return {
        "version": "4.4.0",
        "objects": objects
    }


def extract_all_trajectory_positions_from_canvas(json_data, num_frames):
    """
    Extract all circle positions from canvas. Based on order (id does not seem to be possible)
    Output:
    {
        "shooter": {0: (x, y), 1: (x, y), ...},
        "defender1": {...},
        ...
    }
    """
    positions = {player: {} for player in PLAYER_SELECTION}

    if json_data is None:
        return positions

    circle_objects = [
        obj for obj in json_data.get("objects", [])
        if obj.get("type") == "circle"
    ]

    for idx, obj in enumerate(circle_objects):
        player_idx = idx // num_frames
        frame = idx % num_frames

        if player_idx >= len(PLAYER_SELECTION):
            continue

        player = PLAYER_SELECTION[player_idx]

        px = float(obj["left"])
        py = float(obj["top"])

        x, y = canvas_to_court(px, py)

        positions[player][frame] = (x, y)

    return positions


def update_row_from_player_trajectory(row, selected_player, positions_by_frame):
    row = row.copy()

    for frame, (x, y) in positions_by_frame.items():
        if selected_player == "shooter":

            row[f"shooter_x_t{frame}"] = x
            row[f"shooter_y_t{frame}"] = y

        else:
            sx = float(row[f"shooter_x_t{frame}"])
            sy = float(row[f"shooter_y_t{frame}"])

            dx = x - sx
            dy = y - sy

            row[f"{selected_player}_dx_t{frame}"] = dx
            row[f"{selected_player}_dy_t{frame}"] = dy
            row[f"{selected_player}_dist_t{frame}"] = np.sqrt(dx ** 2 + dy ** 2)

    return row


def update_row_from_all_positions(row, all_positions):
    modified_row = row.copy()

    for player, positions_by_frame in all_positions.items():
        if positions_by_frame:
            modified_row = update_row_from_player_trajectory(modified_row, player, positions_by_frame)

    modified_row = recompute_additional_features(modified_row)

    return modified_row


# ----------------------------------------------
# PREDICTION
# ----------------------------------------------

def predict_probability(row, model, scaler, player_to_idx, action_to_idx, continuous_features):

    X_cont = row[continuous_features].to_frame().T
    X_cont = X_cont.astype(float).values
    X_cont_scaled = scaler.transform(X_cont)

    player_raw = row["PLAYER_ID"]
    player_encoded = player_to_idx.get(player_raw, UNKNOWN_PLAYER_IDX)
    X_player = np.array([[player_encoded]], dtype=np.int32)

    X_period = np.array([[int(row["PERIOD"])]], dtype=np.int32)

    action_raw = row["MAIN_ACTION_TYPE"]
    action_encoded = action_to_idx.get(action_raw, UNKNOWN_ACTION_IDX)
    X_action = np.array([[action_encoded]], dtype=np.int32)

    prob = model.predict([X_cont_scaled, X_player, X_period, X_action])[0, 0]

    return float(prob)



# ----------------------------------------------
# SESSION STATE
# ----------------------------------------------

@st.cache_data
def get_player_options(shots_df):
    player_df = (
        shots_df[["PLAYER_ID", "PLAYER_NAME"]]
        .drop_duplicates()
        .sort_values("PLAYER_NAME")
        .reset_index(drop=True)
    )
    return player_df

@st.cache_data
def get_action_options(shots_df):
    main_actions = shots_df["MAIN_ACTION_TYPE"].sort_values().unique().tolist()
    main_actions.append('Dunk')
    return main_actions

def update_row_metadata(row, player_id=None, player_name=None, action_type=None):
    row = row.copy()

    if player_id is not None:
        row["PLAYER_ID"] = player_id

    if player_name is not None:
        row["PLAYER_NAME"] = player_name

    if action_type is not None:
        row["MAIN_ACTION_TYPE"] = action_type

    return row

def _state_key(prefix, key):
    return f"{prefix}_{key}"


def initialize_demo_state(prefix, shot_idx, base_row, base_proba):
    st.session_state[_state_key(prefix, "shot_idx")] = shot_idx
    st.session_state[_state_key(prefix, "base_row")] = base_row.copy()
    st.session_state[_state_key(prefix, "current_row")] = base_row.copy()
    st.session_state[_state_key(prefix, "canvas_version")] = 0
    st.session_state[_state_key(prefix, "base_proba")] = base_proba
    st.session_state[_state_key(prefix, "proba")] = base_proba

    st.session_state[_state_key(prefix, "selected_player_id")] = base_row["PLAYER_ID"]
    st.session_state[_state_key(prefix, "selected_player_name")] = base_row["PLAYER_NAME"]
    st.session_state[_state_key(prefix, "selected_action_type")] = base_row["MAIN_ACTION_TYPE"]


# ----------------------------------------------
# RENDER
# ----------------------------------------------

def render(prefix="trajectory_demo"):
    """
    Render interactive trajectory demo. Prefix for use with other pages
    """

    model, scaler, player_to_idx, action_to_idx, continuous_features = (load_model_and_artifacts())

    shots_df = load_examples()

    st.markdown("### 🎮 Interactive Shot Probability Demo")
    st.caption(
        "Select an existing shot, move the players, and compare the model's "
        "predicted probability before and after the change."
    )

    # ------------------------------------------------------------
    # Sidebar / controls
    # ------------------------------------------------------------

    shot_idx = st.selectbox(
        "Select example shot",
        options=shots_df.index,
        format_func=lambda idx: (
            f"{shots_df.loc[idx, 'PLAYER_NAME']} | "
            f"{shots_df.loc[idx, 'MAIN_ACTION_TYPE']} | "
            f"Made: {shots_df.loc[idx, 'SHOT_MADE_FLAG']}"
        ),
        key=_state_key(prefix, "shot_selectbox"),
    )

    base_row = shots_df.loc[shot_idx].copy()

    # ------------------------------------------------------------
    # Init / reset state
    # ------------------------------------------------------------

    need_init = (
        _state_key(prefix, "current_row") not in st.session_state
        or _state_key(prefix, "base_row") not in st.session_state
        or _state_key(prefix, "shot_idx") not in st.session_state
        or st.session_state[_state_key(prefix, "shot_idx")] != shot_idx
        or _state_key(prefix, "canvas_version") not in st.session_state
        or _state_key(prefix, "proba") not in st.session_state
    )

    if need_init:
        base_proba = predict_probability(base_row, model, scaler, player_to_idx, action_to_idx, continuous_features)
        initialize_demo_state(prefix, shot_idx, base_row, base_proba)
        st.rerun() 

    if st.button("Reset positions", key=_state_key(prefix, "reset_button")):
        base_row_state = st.session_state[_state_key(prefix, "base_row")].copy()
        base_proba = st.session_state[_state_key(prefix, "base_proba")]

        st.session_state[_state_key(prefix, "current_row")] = base_row_state
        st.session_state[_state_key(prefix, "proba")] = base_proba
        st.session_state[_state_key(prefix, "canvas_version")] += 1

        st.session_state[_state_key(prefix, "selected_player_id")] = base_row_state["PLAYER_ID"]
        st.session_state[_state_key(prefix, "selected_player_name")] = base_row_state["PLAYER_NAME"]
        st.session_state[_state_key(prefix, "selected_action_type")] = base_row_state["MAIN_ACTION_TYPE"]

        if _state_key(prefix, "player_dropdown") in st.session_state:
            st.session_state[_state_key(prefix, "player_dropdown")] = base_row_state["PLAYER_NAME"]
        if _state_key(prefix, "action_dropdown") in st.session_state:
            st.session_state[_state_key(prefix, "action_dropdown")] = base_row_state["MAIN_ACTION_TYPE"] 

        st.rerun()

    current_row = st.session_state[_state_key(prefix, "current_row")].copy()

    # ------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------

    df_pos = get_all_player_positions_from_row(current_row)

    initial_drawing = make_trajectory_drawing(df_pos, only_shot=False)

    background = make_court_background()

    canvas_key = (
        f"{prefix}_canvas_{st.session_state[_state_key(prefix, 'shot_idx')]}_{st.session_state[_state_key(prefix, 'canvas_version')]}"
    )

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Editing full trajectory")
        st.write(shot_idx)
        st.caption("Drag any circle for t0–t5. Darker circles are closer to the shot frame.")
        num_frames = 6

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            background_image=background,
            update_streamlit=True,
            height=CANVAS_H,
            width=CANVAS_W,
            drawing_mode="transform",
            initial_drawing=initial_drawing,
            key=canvas_key,
        )


    # ------------------------------------------------------------
    # Apply canvas changes
    # ------------------------------------------------------------

    if st.button("Set current position"):
        json_data = canvas_result.json_data if canvas_result is not None else None


        if json_data is not None:
            all_positions = extract_all_trajectory_positions_from_canvas(
                json_data,
                num_frames=num_frames,
            )

            modified_row = st.session_state[_state_key(prefix, "current_row")]

            modified_row = update_row_from_all_positions(
                modified_row,
                all_positions,
            )

            st.session_state[_state_key(prefix, "current_row")] = modified_row.copy()

            modified_row = update_row_metadata(
                    modified_row,
                    player_id=st.session_state[_state_key(prefix, "selected_player_id")],
                    player_name=st.session_state[_state_key(prefix, "selected_player_name")],
                    action_type=st.session_state[_state_key(prefix, "selected_action_type")],
            )

            st.session_state[_state_key(prefix, "proba")] = predict_probability(
                modified_row,
                model,
                scaler,
                player_to_idx,
                action_to_idx,
                continuous_features,
            )

            st.session_state[_state_key(prefix, "canvas_version")] += 1
            st.rerun()

    # ------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------

    with right_col:
        base_proba = st.session_state[_state_key(prefix, "base_proba")]
        current_proba = st.session_state[_state_key(prefix, "proba")]

        st.subheader("Prediction")

        st.metric(
            "Original probability",
            f"{base_proba:.1%}",
        )

        st.metric(
            "Current probability",
            f"{current_proba:.1%}",
            delta=f"{current_proba - base_proba:+.1%}",
        )

        st.markdown("---")

        st.subheader("Editable inputs")

        player_options_df = get_player_options(shots_df)
        action_options = get_action_options(shots_df)

        current_row = st.session_state[_state_key(prefix, "current_row")]

        current_player_id = current_row["PLAYER_ID"]
        current_action_type = current_row["MAIN_ACTION_TYPE"]

        # Find current player index
        player_ids = player_options_df["PLAYER_ID"].tolist()
        player_index = player_ids.index(current_player_id)

        selected_player_label = st.selectbox(
            "Shooter",
            options=player_options_df['PLAYER_NAME'].values,
            index=player_index,
            #format_func=lambda idx: (
            #    f"{player_options_df.loc[idx, 'PLAYER_NAME']} "
            #    f"({player_options_df.loc[idx, 'PLAYER_ID']})"
            #),
            key=_state_key(prefix, "player_dropdown"),
        )

        selected_player_id = player_options_df.loc[player_options_df["PLAYER_NAME"] == selected_player_label, "PLAYER_ID"].values[0]
        selected_player_name = player_options_df.loc[player_options_df["PLAYER_NAME"] == selected_player_label, "PLAYER_NAME"].values[0]

        # Find current action index
        action_index = action_options.index(current_action_type)

        selected_action_type = st.selectbox(
            "Main action type",
            options=action_options,
            index=action_index,
            key=_state_key(prefix, "action_dropdown"),
        )

        # If metadata changed -> update session state
        if selected_player_id != current_row["PLAYER_ID"] or selected_action_type != current_row["MAIN_ACTION_TYPE"]:
            st.session_state[_state_key(prefix, "selected_player_id")] = selected_player_id
            st.session_state[_state_key(prefix, "selected_player_name")] = selected_player_name
            st.session_state[_state_key(prefix, "selected_action_type")] = selected_action_type

        # -----------------------------
        # Info Dataframe
        # -----------------------------
        st.subheader("Shot information")

        info_cols = [
            "SHOT_TYPE",
            "SHOT_DISTANCE",
            "SHOT_ZONE_BASIC",
            "PERIOD",
            "MINUTES_REMAINING",
            "SECONDS_REMAINING",
            "SHOT_MADE_FLAG",
        ]

        existing_info_cols = [
            col for col in info_cols
            if col in current_row.index
        ]

        if existing_info_cols:
            st.dataframe(
                current_row[existing_info_cols].to_frame("value"),
                use_container_width=True,
            )