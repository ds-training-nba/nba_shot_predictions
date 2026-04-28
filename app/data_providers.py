import dataclasses

from datasets import load_dataset
import pandas as pd

from processing.compute_columns import add_computed_feature_columns, add_is_home_column, add_opponent_interfered_column, \
    add_angle_column, add_shot_main_action_type_column
from processing.encoding import encode_for_model
from processing.filtering import filter_clean_source_columns, filter_pre_encoding_columns, filter_for_players
from processing.preprocessing import preprocess_fields

def get_shots_dataframe(use_small = False):
    """
    Returns the raw dataframe (default: from huggingface, use_small: local small csv)
    :param use_small:  use the small version for better performance when testing complicated calculations
    :return: pd.DataFrame
    """
    if use_small:
        return pd.read_csv('data/small/shots.csv')

    ds = load_dataset(
        "ds-training-nba/nba_shot_data",
        data_files={"train": "raw_merged/merged_dataset.parquet"}
    )
    return ds['train'].to_pandas()


def main_dataframe(use_small = False):
    """
        Returns the main dataframe to work with. Containing added columns and so on.
        :param use_small: use the small version for better performance when testing complicated calculations
        :return: pd.DataFrame
    """
    df = filtered_shots_dataframe(use_small=use_small)
    df = add_shot_main_action_type_column(df)
    df = add_angle_column(df)
    df = add_is_home_column(df)
    df = add_opponent_interfered_column(df)
    return df

def clean_source_dataframe(use_small = False):
    """
        Returns only the source columns to use. The goal is that this dataframe does not contain any missing values.
        :param use_small: use the small version for better performance when testing complicated calculations
        :return: pd.DataFrame
    """
    main_df = get_shots_dataframe(use_small)
    return filter_clean_source_columns(main_df)

@dataclasses.dataclass
class DataFrameRequest:
    model_to_encode_for: str = ""
    use_small: bool = False
    apply_preprocessing: bool = True
    filter_clean: bool = True
    add_computed: bool = True
    filter_pre_encoding_columns: bool = True
    encode_for_model: bool = True
    filter_top_players: bool = True


def provide_dataframe(request: DataFrameRequest):
    """
        Returns all the data for the model
        :param request: define data source size and what processing is to be done
        :return: pd.DataFrame
    """
    # base raw dataframe
    df = get_shots_dataframe(request.use_small)

    if request.filter_clean:
        # only use clean source columns
        df = filter_clean_source_columns(df)
    if request.apply_preprocessing:
        # Remove nans and duplicates
        df = preprocess_fields(df)
    if request.filter_top_players:
        # only use clean source columns
        df = filter_for_players(df)
    if request.add_computed:
        # computed/engineered features
        df = add_computed_feature_columns(df)
    if request.filter_pre_encoding_columns:
        # cleanup columns before encoding
        df = filter_pre_encoding_columns(df)
    if request.encode_for_model:
        # encode
        df = encode_for_model(df, request.model_to_encode_for)
    return df


def filtered_shots_dataframe(use_small = False):
    """
    Returns the dataframe filtered for our selected 20 players
    :param use_small: use the small version for better performance when testing complicated calculations
    :return: pd.DataFrame
    """
    df = get_shots_dataframe(use_small)
    return filter_for_players(df)

def test_train_dataset():
    ds = load_dataset(
        "ds-training-nba/nba_shot_data",
        data_files={
            "train": "processed/processed_20_players_train.parquet",
            "test": "processed/processed_20_players_test.parquet"
        }
    )
    return {
        "train": ds['train'].to_pandas(),
        "test": ds['test'].to_pandas(),
    }