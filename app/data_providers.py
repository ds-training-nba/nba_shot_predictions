from datasets import load_dataset
import pandas as pd

from app.config import PLAYER_CHOICE
from processing.preprocessing import add_shot_main_action_type_column, add_angle_column

def get_shots_dataframe(use_small = False):
    if use_small:
        return pd.read_csv('data/small/shots.csv')

    ds = load_dataset(
        "ds-training-nba/nba_shot_data",
        data_files={"train": "raw_merged/merged_dataset.parquet"}
    )
    return ds['train'].to_pandas()

def filtered_shots_dataframe(use_small = False):
    df = get_shots_dataframe(use_small)


    chosen_player_matrix = None

    for name in PLAYER_CHOICE:
        current_matrix = df['PLAYER_NAME'] == name
        if chosen_player_matrix is None:
            chosen_player_matrix = current_matrix
        else:
            chosen_player_matrix = chosen_player_matrix | current_matrix

    return df[chosen_player_matrix]

def main_dataframe(use_small = False):
    df = filtered_shots_dataframe(use_small=use_small)
    df = add_shot_main_action_type_column(df)
    df = add_angle_column(df)
    return df