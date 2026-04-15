from math import atan2

import numpy as np
import pandas as pd
from app.config import TARGET_VARIABLE
def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    grouped_df = df.groupby(fields).agg({TARGET_VARIABLE: "mean"})
    return grouped_df

def add_shot_main_action_type_column(df: pd.DataFrame):
    def main_category(val):
        keywords = ['Dunk', 'Layup', 'Hook', 'Jump']
        for keyword in keywords:
            if keyword in val:
                return keyword
        return 'Other'
    df['MAIN_ACTION_TYPE'] = df['ACTION_TYPE'].apply(main_category)
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

def add_is_home_column(df: pd.DataFrame):
    def is_home(row):
        player_team = row['PLAYER1_TEAM_ABBREVIATION']
        home_team = row['HTM']
        return 1 if home_team == player_team else 0

    df['IS_HOME'] = df.apply(is_home, axis=1)
    return df