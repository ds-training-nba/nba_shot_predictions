from math import atan2

import numpy as np
import pandas as pd


def add_is_home_column(df: pd.DataFrame):
    def is_home(row):
        player_team = row['PLAYER1_TEAM_ABBREVIATION']
        home_team = row['HTM']
        return 1 if home_team == player_team else 0

    df['IS_HOME'] = df.apply(is_home, axis=1)
    return df

def fill_team_scores_and_margin(df):
    df = df.sort_values(["GAME_ID_x", "GAME_EVENT_ID"])

    df["points"] = df["SHOT_TYPE"].map({
        "1PT Free Throw": 1,
        "2PT Field Goal": 2,
        "3PT Field Goal": 3
    })

    df["pointsHome"] = df["points"].where(df["SHOT_MADE_FLAG"] == 1 & (df["IS_HOME"] == 1), 0)
    df["pointsAway"] = df["points"].where(df["SHOT_MADE_FLAG"] == 1 & (df["IS_HOME"] == 0), 0)

    df["scoreHome"] = df.groupby("GAME_ID_x")["pointsHome"].cumsum()
    df["scoreAway"] = df.groupby("GAME_ID_x")["pointsAway"].cumsum()

    df["scoreHomeBeforeShot"] = df.groupby("GAME_ID_x")["scoreHome"].shift(1).fillna(0)
    df["scoreAwayBeforeShot"] = df.groupby("GAME_ID_x")["scoreAway"].shift(1).fillna(0)

    df["scoreMargin"] = np.where(
        df["IS_HOME"],
        df["scoreHome"] - df["scoreAway"],
        df["scoreAway"] - df["scoreHome"]
    )

    df["scoreMarginBeforeShot"] = np.where(
        df["IS_HOME"],
        df["scoreHomeBeforeShot"] - df["scoreAwayBeforeShot"],
        df["scoreAwayBeforeShot"] - df["scoreHomeBeforeShot"]
    )

    return df

COMPUTED_FEATURES_FUNCTIONS = [
    add_is_home_column,
    fill_team_scores_and_margin
]


def add_computed_feature_columns(df):
    for func in COMPUTED_FEATURES_FUNCTIONS:
        df = func(df)
    return df


def add_opponent_interfered_column(df: pd.DataFrame):
    def opponent_interfered(row):
        return row['PLAYER1_TEAM_ABBREVIATION'] != row['PLAYER2_TEAM_ABBREVIATION'] and isinstance(row['PLAYER2_TEAM_ABBREVIATION'], str) and (len(row['PLAYER2_TEAM_ABBREVIATION']) > 0)

    df['OPPONENT_INTERFERED'] = df.apply(opponent_interfered, axis=1)
    return df


def add_angle_column(df: pd.DataFrame):
    def angle(row):
        x = row['LOC_X']
        y = row['LOC_Y']
        return 180 * atan2(x,y)/np.pi
    def angle_sector(angle_in_deg):
        # front
        if abs(angle_in_deg) <= 45:
            return 0
        # side
        if abs(angle_in_deg) > 45 and abs(angle_in_deg) <= 90:
            return 1
        # extreme side (far behind the basket line)
        if abs(angle_in_deg) > 90 and abs(angle_in_deg) < 135:
            return 2
        # directly behind the basket
        return 3
    df['ANGLE'] = df.apply(angle,axis=1)
    df['ANGLE_SECTOR'] = df['ANGLE'].apply(angle_sector)
    df['ABS_ANGLE'] = df['ANGLE'].apply(lambda val: abs(val))
    return df


def add_shot_main_action_type_column(df: pd.DataFrame):
    def main_category(val):
        other_str = 'Other'
        if not isinstance(val, str):
            return other_str
        keywords = ['Dunk', 'Layup', 'Hook', 'Jump']
        for keyword in keywords:
            if keyword in val:
                return keyword
        return other_str
    df['MAIN_ACTION_TYPE'] = df['ACTION_TYPE'].apply(main_category)
    return df

def add_shifted_score_columns(df: pd.DataFrame):
    df['SCORE_HOME_BT'] = 0
    df['SCORE_AWAY_BT'] = 0
    def shift_score(val):
        other_str = 'Other'
        if not isinstance(val, str):
            return other_str
        keywords = ['Dunk', 'Layup', 'Hook', 'Jump']
        for keyword in keywords:
            if keyword in val:
                return keyword
        return other_str
    df['MAIN_ACTION_TYPE'] = df['ACTION_TYPE'].apply(main_category)
    return df

