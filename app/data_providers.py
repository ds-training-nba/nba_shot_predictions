from app.config import PLAYER_CHOICE
import pandas as pd

from processing.preprocessing import add_shot_main_action_type_column


def get_shots_dataframe(use_small = False):

    file = 'data/small/shots.csv' if use_small else 'data/orig/shots.csv'
    df = pd.read_csv(file)
    return df

def filtered_shots_dataframe(use_small = False):
    df = get_shots_dataframe(use_small)


    chosen_player_matrix = None

    for name in PLAYER_CHOICE:
        current_matrix = df['Player Name'] == name
        if chosen_player_matrix is None:
            chosen_player_matrix = current_matrix
        else:
            chosen_player_matrix = chosen_player_matrix | current_matrix

    return df[chosen_player_matrix]

def main_dataframe(use_small = False):
    df = filtered_shots_dataframe(use_small=use_small)
    df = add_shot_main_action_type_column(df)
    return df