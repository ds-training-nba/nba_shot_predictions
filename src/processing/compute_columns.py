from math import atan2
import numpy as np
import pandas as pd

from app.config import TARGET_VARIABLE
from processing.helpers import shot_accuracy_by_fields


def add_is_home_column(df: pd.DataFrame):
    """
    adds a column "IS_HOME" to the df, which indicates if the shooting player is playing at home.
    :param df:
    :return:
    """
    def is_home(row):
        player_team = row['PLAYER1_TEAM_ABBREVIATION']
        home_team = row['HTM']
        return 1 if home_team == player_team else 0

    df['IS_HOME'] = df.apply(is_home, axis=1)
    return df

def fill_team_scores_and_margin(df):
    """
    Adds several columns to the dataframe, to have every shot row with and current score, and to have a difference
    of the score according to the perspective of the player.
    :param df:
    :return:
    """
    df = df.sort_values(["GAME_ID_x", "GAME_EVENT_ID"])

    df["points"] = df["SHOT_TYPE"].map({
        "1PT Free Throw": 1,
        "2PT Field Goal": 2,
        "3PT Field Goal": 3
    })

    df["pointsHome"] = df["points"].where((df["SHOT_MADE_FLAG"] == 1) & (df["IS_HOME"] == 1), 0)
    df["pointsAway"] = df["points"].where((df["SHOT_MADE_FLAG"] == 1) & (df["IS_HOME"] == 0), 0)

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


def fill_time_features(df):
    """
    Adding calculated time features. Readable shot clock, IsOvertime, IsClutchTime (intense time shortly before and when
    no easy win can be expected for the own team)
    :param df:
    :return:
    """
    def played_time_seconds(row):
        period = row['PERIOD_x']
        period_time = 12 * 60
        OT_time = 5 * 60
        if period <= 4:
            return (period - 1) * period_time + (period_time - row['TimeRemainingInPeriod'])
        else:
            return 4 * period_time + (period - 5) * OT_time + (OT_time - row['TimeRemainingInPeriod'])

    def time_remaining_in_game(row):
        period = row['PERIOD_x']
        period_time = 12 * 60
        if period <= 4:
            return (4 - period) * period_time + row['TimeRemainingInPeriod']
        else:
            return row['TimeRemainingInPeriod']

    def parse_time(x):
        if pd.isnull(x) or ':' not in x:
            return None
        parts = x.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return None

    time_from_string = df["PCTIMESTRING"].apply(parse_time)
    time_from_parts  = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

    df["TimeRemainingInPeriod"] = time_from_string.fillna(time_from_parts).astype("int16")
    df['TotalPlayedTime']       = df.apply(played_time_seconds, axis=1).astype(int)
    df['TimeRemainingInGame']   = df.apply(time_remaining_in_game, axis=1).astype(int)
    df['IsOvertime']            = (df['PERIOD_x'] > 4).astype(int)
    df['OvertimeNumber']        = (df['PERIOD_x'] - 4).clip(lower=0).astype(int)
    df['IsClutchTime']          = (
                                    (df['TimeRemainingInGame'] <= 300) &(df['scoreMarginBeforeShot'].abs() <= 5)
                                ).astype('int8')
    return df


def add_player_data(df: pd.DataFrame):
    """
    Adding personal player information that is generally available and might be handy to make deductions (e.g. high age)
    :param df:
    :return:
    """
    df_players = pd.read_csv('data/orig/player_data.csv')
    df = df.merge(df_players, left_on="PLAYER_NAME", right_on="name")

    df['GAME_DATE'] = pd.to_datetime(
        df['GAME_DATE'].astype(str).apply(lambda val: val[:-2]),
        format='%Y%m%d',
        errors='coerce'
    )
    df['year'] = df['GAME_DATE'].dt.year
    df['years_experience'] = df['GAME_DATE'].dt.year - df['year_start']
    df['player_age'] = df['GAME_DATE'].dt.year - df['birth_date'].apply(lambda val: val[-4:]).astype(float)
    return df
def determine_best_age_per_player(df: pd.DataFrame):
    """
    Calculate some context: Player's age with the most precise shots
    :param df:
    :return: dict player_name: best_age
    """
    df_accuracy_by_age = shot_accuracy_by_fields(df, ['PLAYER_NAME', 'player_age']).reset_index()
    best_age_per_player = {}
    for name in df_accuracy_by_age['PLAYER_NAME'].unique():
        best_age_per_player[name] = df_accuracy_by_age.loc[
            df_accuracy_by_age[df_accuracy_by_age['PLAYER_NAME'] == name][TARGET_VARIABLE].argmax()]['player_age']
    return best_age_per_player

def add_last_match_precisions_for_prediction(df_train: pd.DataFrame, df_prediction: pd.DataFrame):
    """
    Context about the last match of the player.
    (The idea was that we could use the last game in the training set for the test set,
    but that was based on the idea the games in the test set were after the training.
    Using these results otherwise in the test set smell too much of target leak.)
    :param df_train:
    :param df_prediction:
    :return:
    """
    last_match_precisions_per_player = {}
    for name in df_train['PLAYER_NAME'].unique():
        df_sorted = df_train[df_train['PLAYER_NAME'] == name].sort_values(ascending=False, by="GAME_DATE")
        last_match_precisions_per_player[name] = df_sorted.iloc[0]['last_match_precision']
    df_prediction['last_match_precision'] = df_prediction.apply(lambda row: last_match_precisions_per_player[row['PLAYER_NAME']], axis=1)
    return df_prediction

def add_last_match_precision(df: pd.DataFrame):
    """
    Adds the last match precision column: An indicator of the players current form
    :param df:
    :return:
    """
    df_accuracy_per_player = shot_accuracy_by_fields(df, ['PLAYER_NAME'])
    df_accuracy_per_player_and_match = shot_accuracy_by_fields(df, ['PLAYER_NAME', 'GAME_ID_x'])
    precisions ={}
    for name in df['PLAYER_NAME'].unique():
        player_average = df_accuracy_per_player.loc[name, TARGET_VARIABLE]
        # Sort players matches from first to last
        df_sorted = df[df['PLAYER_NAME'] == name].sort_values(ascending=True,by="GAME_DATE")
        last_precision = player_average # initial value, begin with average
        precisions[name] = {}
        for match_id in df_sorted['GAME_ID_x'].unique():
            last_relative_precision = last_precision/player_average
            precisions[name][match_id] = last_relative_precision
            # precision
            new_precision = df_accuracy_per_player_and_match.loc[(name,match_id)][TARGET_VARIABLE]
            if new_precision > 0:
                last_precision = new_precision
            else:
                # print('Match Precision is 0', name, match_id)
                # Player seems to not have scored in the match. Maybe due to short appearance in the match and
                # injury/pause. Don't set last match precision to 0, to not confuse the algorithm, but
                last_precision *= 0.7

    df['last_match_precision'] = df.apply(lambda row: precisions[row['PLAYER_NAME']][row['GAME_ID_x']], axis=1)
    return df

def add_best_age_data(df: pd.DataFrame, best_ages):
    """
    Apply calculated best ages to dataFrame/Player Column
    :param df:
    :param best_ages:
    :return:
    """
    def best_age(name):
        return best_ages[name]
    df['best_age'] = df['PLAYER_NAME'].apply(best_age)
    return df

def add_opponent_interfered_column(df: pd.DataFrame):
    """
    When there is a player two mentioned in the data and it is an opponent, we calculate that an opponent has interfered
    with the shot. But this has a very strong POST labeling target leak. Every time the opponent is mentioned, the shot WAS MADE!
    So we do not use this column
    :param df:
    :return:
    """
    def opponent_interfered(row):
        return row['PLAYER1_TEAM_ABBREVIATION'] != row['PLAYER2_TEAM_ABBREVIATION'] and isinstance(row['PLAYER2_TEAM_ABBREVIATION'], str) and (len(row['PLAYER2_TEAM_ABBREVIATION']) > 0)

    df['OPPONENT_INTERFERED'] = df.apply(opponent_interfered, axis=1)
    return df


def add_angle_column(df: pd.DataFrame):
    """
    From locations calculate the angle
    :param df:
    :return:
    """
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
    df['ANGLE_SIN'] = df['ANGLE'].apply(lambda val: np.sin(val))
    df['ANGLE_COS'] = df['ANGLE'].apply(lambda val: np.cos(val))
    return df


def add_shot_main_action_type_column(df: pd.DataFrame):
    """
    Grouping the ACTION_TYPE into main groups. Seems to add information to the model and not add to overfitting
    although the columns highly correlate.
    Also, MAIN_ACTION_TYPE can be interpreted better.
    :param df:
    :return:
    """
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



# definition of added columns before uploading clean split version to huggingface
COMPUTED_FEATURES_FUNCTIONS = [
    add_is_home_column,
    fill_team_scores_and_margin,
    fill_time_features,
    add_opponent_interfered_column,
    add_angle_column,
    add_shot_main_action_type_column
]

def add_computed_feature_columns(df):
    for func in COMPUTED_FEATURES_FUNCTIONS:
        df = func(df)
    return df