from app.config import PLAYER_CHOICE
import pandas as pd


def get_shots_dataframe():

    file = 'data/orig/shots.csv'
    df = pd.read_csv(file)
    return df

def filtered_shots_dataframe():
    df = get_shots_dataframe()

    chosen_player_matrix = None

    for name in PLAYER_CHOICE:
        current_matrix = df['Player Name'] == name
        if chosen_player_matrix is None:
            chosen_player_matrix = current_matrix
        else:
            chosen_player_matrix = chosen_player_matrix | current_matrix

    return df[chosen_player_matrix]