from pandas import DataFrame
import numpy as np

def fix_score_column(df: DataFrame, game_id: int):
    df_for_game = df[df['GAME_ID_x'] == game_id].sort_values('GAME_EVENT_ID')
    score_home = 0
    score_away = 0
    score_home_shifted = 0
    score_away_shifted = 0
    def fix(row):
        nonlocal score_away, score_home, score_home_shifted, score_away_shifted
        if row['scoreHome'] != 0 or row['scoreAway'] != 0:
            score_home = score_home if np.isnan(row['scoreHome']) else row['scoreHome']
            score_away = score_away if np.isnan(row['scoreAway']) else row['scoreAway']
        row['scoreHome'] = score_home
        row['scoreAway'] = score_away
        row['SCORE_HOME_BT'] = score_home_shifted
        row['SCORE_AWAY_BT'] = score_away_shifted
        score_away_shifted = score_away
        score_home_shifted = score_home
        return row
        #print(score_home, score_away)

    df_for_game = df_for_game.apply(fix, axis=1)
    df_for_game = df_for_game.sort_index()
    df.loc[df['GAME_ID_x'] == game_id, 'scoreHome'] = df_for_game['scoreHome']
    df.loc[df['GAME_ID_x'] == game_id, 'scoreAway'] = df_for_game['scoreAway']
    df.loc[df['GAME_ID_x'] == game_id, 'SCORE_HOME_BT'] = df_for_game['SCORE_HOME_BT']
    df.loc[df['GAME_ID_x'] == game_id, 'SCORE_AWAY_BT'] = df_for_game['SCORE_AWAY_BT']
    return df

def fix_score_columns_for_all_games(df: DataFrame):
    df['SCORE_HOME_BT'] = 0
    df['SCORE_AWAY_BT'] = 0
    game_ids = df['GAME_ID_x'].unique()
    for game_id in game_ids:
        df = fix_score_column(df,game_id)
    return df

