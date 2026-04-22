import pandas as pd
from app.config import TARGET_VARIABLE

def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    grouped_df = df.groupby(fields).agg({TARGET_VARIABLE: "mean"})
    return grouped_df


def preprocess_fields(df):
    
    # Remove rows with inconsistencies in the targets
    shot_result_conv = df['shotResult'].replace({'Made': 1, 'Missed': 0})
    df = df[~(shot_result_conv.notna() & (df['SHOT_MADE_FLAG'] != shot_result_conv))]

    # Drop subset of nan rows
    df = df.dropna(subset=['SHOT_DISTANCE', 'SHOT_TYPE', 'SHOT_ZONE_RANGE', 'SHOT_ZONE_BASIC', 'LOC_X', 'LOC_Y'])

    # Fill missing values based on valid values with same game id
    df['HTM'] = df.groupby('GAME_ID_x')['HTM'].transform('first')
    df['VTM'] = df.groupby('GAME_ID_x')['VTM'].transform('first')
    df['GAME_DATE'] = df.groupby('GAME_ID_x')['GAME_DATE'].transform('first')

    # We can fill PLAYER1_TEAM_ABBREVIATION, by checking other rows with same player and team id
    df['PLAYER1_TEAM_ABBREVIATION'] = df.groupby(['PLAYER_ID', 'TEAM_ID'])['PLAYER1_TEAM_ABBREVIATION'].transform('first')
    df = df.dropna(subset=['PLAYER1_TEAM_ABBREVIATION']) # drop remaining nans

    # Fix minutes and seconds remaining
    extracted = df['clock'].str.extract(r'PT(\d+)M(\d+)\.')
    df['MINUTES_REMAINING'] = df['MINUTES_REMAINING'].fillna(extracted[0].astype(float))
    df['SECONDS_REMAINING'] = df['SECONDS_REMAINING'].fillna(extracted[1].astype(float))

    # Drop duplicated rows, introduced by blocks
    df = df[~((df.duplicated(subset=['GAME_ID_x', 'GAME_EVENT_ID'], keep=False)) & df['shotResult'].isna())]
    df = df.drop_duplicates().reset_index(drop=True)

    return df

