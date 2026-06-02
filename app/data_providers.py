# imports of 3rd party packages
import dataclasses
from datasets import load_dataset
import pandas as pd

# imports of own packages
from app.conf.run import RunConfig
from app.config import TARGET_VARIABLE
from processing.compute_columns import add_computed_feature_columns, add_is_home_column, add_opponent_interfered_column, \
    add_angle_column, add_shot_main_action_type_column, add_player_data, determine_best_age_per_player, \
    add_best_age_data, add_last_match_precision, add_last_match_precisions_for_prediction
from processing.encoding import encode_for_model
from processing.filtering import filter_clean_source_columns, filter_pre_encoding_columns, filter_for_players
from processing.fixes import fix_action_type_target_leak
from processing.preprocessing import preprocess_fields



def get_shots_dataframe(use_small = False):
    """
    Returns the RAW dataframe (default: from huggingface, use_small: local small csv)
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
        Still quite raw and not directly used for training. (There is "caching" step in between. Training data is
        fetched from another hugging face data source)
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
        Not used for training: A more refined and test-train-split datasource is available on our hugging face repo)
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
        Returns all the data for the model.
        Also operates on the RAW datasource. only used to generate the test-train upload to huggingface
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
    if request.add_computed:
        # computed/engineered features
        df = add_computed_feature_columns(df)
    if request.filter_top_players:
        # only use clean source columns
        df = filter_for_players(df)
    if request.filter_pre_encoding_columns:
        # cleanup columns before encoding
        df = filter_pre_encoding_columns(df)
    # if request.encode_for_model:
    #     # encode
    #     df = encode_for_model(df, request.model_to_encode_for)
    return df


def filtered_shots_dataframe(use_small = False):
    """
    Returns the dataframe filtered for our selected 20 players
    :param use_small: use the small version for better performance when testing complicated calculations
    :return: pd.DataFrame
    """
    df = get_shots_dataframe(use_small)
    return filter_for_players(df)

def test_train_dataset(add_player_info=False):
    """
    Returns the actual filtered, cleaned and split version from huggingface.
    :param add_player_info:
    :return:
    """
    ds = load_dataset(
        "ds-training-nba/nba_shot_data",
        data_files={
            "train": "processed/processed_20_players_train.parquet",
            "test": "processed/processed_20_players_test.parquet"
        }
    )
    if add_player_info:
        # enrich with well known facts about the players from another database
        train = add_player_data(ds['train'].to_pandas())
        test = add_player_data(ds['test'].to_pandas())
        # determine best age from train and add column to test and train
        best_ages = determine_best_age_per_player(train)
        train = add_best_age_data(train, best_ages)
        test = add_best_age_data(test, best_ages)
        # consecutively add the precision of the last match to the data of the next match.
        train = add_last_match_precision(train)
        # get precision data only from train
        test = add_last_match_precisions_for_prediction(train, test)
    else:
        train = ds['train'].to_pandas()
        test = ds['test'].to_pandas()

    return {
        "train": train,
        "test": test,
    }
def full_dataset():
    """
    Convenience function to get merged test/train DF
    :return:
    """
    ds = test_train_dataset(True)
    return pd.concat([ds['train'],ds['test']], axis=0)

def ready_split_dataset(config: RunConfig):
    """
    Returns the processed (encoded) and split dataframes according to the RunConfig
    :param config:
    :return: X_train_enc, y_train, X_test_enc, y_test, X_train, X_test
    """
    dataset = test_train_dataset(True)
    df_train = dataset['train']
    df_test = dataset['test']

    if config.use_only_field_goals:
        df_train = df_train[df_train['points'] != 1]
        df_test = df_test[df_test['points'] != 1]
    if config.use_action_type_fix:
        df_train = fix_action_type_target_leak(df_train)
        df_test = fix_action_type_target_leak(df_test)
    X_test, y_test = split_x_y(df_test)
    X_train, y_train = split_x_y(df_train)
    X_train_enc, X_test_enc = encode_for_model(X_train, y_train, config.model_config.model_id, config.encoding_config, X_test)
    return X_train_enc, y_train, X_test_enc, y_test, X_train, X_test


def split_x_y(df):
    """
    Covenience function to avoid duplicate code
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    return X,y
