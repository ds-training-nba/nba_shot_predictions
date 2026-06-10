import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from datetime import date

# for profiling
from contextlib import contextmanager
import time

from alternatives import AlternativesCalculatorPipeline, ThreePointAlternativesCalculator, AlternativesProcessorPipeline
from app.conf.run import build_best_run_config, MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_LIGHT_GBM
from app.data_providers import full_dataset, prepare_dataset_for_prediction, enrich_with_player_info, \
    test_train_dataset
from processing.compute_columns import player_data_df
from app.experiments import experiment_current_path, load_runs_to_dataframe
from app.modeling import load_persisted_model
from evaluation.insights import get_false_predictions
from processing.helpers import get_player_id_by_name


def sl_show_experiment_results(experiment_id: str):
    """
    builds a streamlit page with the results table of an experiment
    :param experiment_id:
    :return:
    """
    df = load_runs_to_dataframe(experiment_current_path(experiment_id))
    st.title("Experiment Dashboard for " + experiment_id)
    default_fields = ['model', 'context_name', 'metric_accuracy', 'macro_avg_precision',
                      'macro_avg_recall', 'macro_avg_f1-score']
    if 'metric_brier_score' in df.columns:
        default_fields.insert(2,'metric_brier_score')
    if 'brier_decomposition_10_resolution' in df.columns:
        default_fields.insert(2,'brier_decomposition_10_resolution')
    if 'brier_decomposition_10_reliability' in df.columns:
        default_fields.insert(2, 'brier_decomposition_10_reliability')
    if 'brier_decomposition_20_resolution' in df.columns:
        default_fields.insert(2,'brier_decomposition_20_resolution')
    if 'brier_decomposition_20_reliability' in df.columns:
        default_fields.insert(2, 'brier_decomposition_20_reliability')
    st.dataframe(df[default_fields])

def sl_show_false_predictions():
    """
    builds a streamlit page to analyze false predictions

    :return:
    """
    config = build_best_run_config()
    page_size = 20
    false_positives, false_negatives = get_false_predictions(config)
    total_pages = (len(false_positives) - 1) // page_size + 1
    page_p = st.number_input("Seite", min_value=1, max_value=total_pages, step=1)
    st.title("False Positives")
    st.dataframe(paginate_dataframe(false_positives, page_size, page_p))

    plt.figure(figsize=(10, 8))
    sns.countplot(false_positives, x="MAIN_ACTION_TYPE",stat="percent")

    # Show the plot in Streamlit
    st.pyplot(plt)

    total_pages = (len(false_negatives) - 1) // page_size + 1
    page_n = st.number_input("Seite", min_value=1, max_value=total_pages, step=1)
    st.title("False negatives")
    st.dataframe(paginate_dataframe(false_negatives, page_size, page_n))

def paginate_dataframe(dataframe, page_size, page_num):

    page_size = page_size

    if page_size is None:

        return None

    offset = page_size*(page_num-1)

    return dataframe[offset:offset + page_size]

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.2f}s")

@st.cache_data
def sl_app_dataset():
    print("Cache miss Full DS")
    return full_dataset()

@st.cache_data
def sl_app_split_data():
    print("Cache miss test/train")
    return test_train_dataset()

def sl_team_options(df):
    options =  list(df['PLAYER1_TEAM_ABBREVIATION'].unique())
    options.insert(0, "")
    return options

def sl_player_options(df):
    return df['PLAYER_NAME'].unique()

def sl_filter_by_team(df, team):
    return df[df['PLAYER1_TEAM_ABBREVIATION'] == team]

def sl_filter_by_player(df, name):
    return df[df['PLAYER_NAME'] == name]

def sl_date_options(df):
    return df['GAME_DATE'].unique()

def sl_filter_by_date(df, date):
    return df[df['GAME_DATE'] == date]

def sl_prepare_dataset_for_prediction(df, config, simulate_year = None):
    # get cache training data
    ds = sl_app_split_data()

    df_train, df = enrich_with_player_info(ds['train'],df, simulate_year)

    return prepare_dataset_for_prediction(df, config, df_train)

def sl_enrich_with_player_info(df, simulate_year = None):
    # get cached split dataset
    split_data = sl_app_split_data()
    _, df = enrich_with_player_info(split_data['train'], df, simulate_year)
    return df

def sl_switch_player(df: pd.DataFrame, alternative_player_name:str, df_orig: pd.DataFrame):
    """
    provide an alternative version of inputs by changing player name
    :param df:
    :param alternative_player_name:
    :param df_orig:
    :return:
    """
    df['PLAYER_NAME'] = alternative_player_name

    df['PLAYER_ID'] = get_player_id_by_name(df_orig, alternative_player_name)
    return df

def sl_get_target_year_for_same_age(player, alternative_player, current_year):
    df = player_data_df()

    player_birth_year = df[df['name'] == player]['birth_date'].iloc[0][-4:]
    alternative_player_year = df[df['name'] == alternative_player]['birth_date'].iloc[0][-4:]
    age_diff = int(alternative_player_year) - int(player_birth_year)
    return current_year + age_diff

def sl_player_app():
    app_id = "player_app"

    col_left, col_right = st.columns([1, 1])

    with col_left:
        model_id = st.selectbox(
            "Choose the model",
            [MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_LIGHT_GBM],
            key=app_id + "_model"
        )
        df_orig = sl_app_dataset()
        with timer("team options"):
            team = st.selectbox(
                "Filter by team (optional)",
                sl_team_options(df_orig), key=app_id + "_team"
            )
        with timer("filter by team"):
            if len(team) > 0:
                df = sl_filter_by_team(df_orig,team)
            else:
                df = df_orig.copy()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.header('Original Player')
        with timer("player options"):
            player = st.selectbox(
                "Choose your player",
                sl_player_options(df),key=app_id + "_player"
            )
        with timer("filter by player"):
            df = sl_filter_by_player(df, player)

        chosen_date = st.selectbox(
            "Choose your game date",
            sl_date_options(df), key=app_id + "_date"
        )
        with timer("filter by date "):
            df = sl_filter_by_date(df, chosen_date)

        config = build_best_run_config()
        config.model_config.model_id = model_id
        with timer("load persisted model "):
            model = load_persisted_model(model_id)


        df['pointsMade'] = df.apply(lambda row: row['SHOT_MADE_FLAG'] * row['points'], axis=1 )
        columns = ['PLAYER_NAME', 'ACTION_TYPE', 'SHOT_DISTANCE', 'points','pointsPredicted', 'pointsMade']


        X_enc = sl_prepare_dataset_for_prediction(df, config)
        with timer("prediction 1"):
            y_proba = model.predict_proba(X_enc)

        df['probability'] = y_proba[:,1]
        df['pointsPredicted'] = df.apply(lambda row: row['probability'] * row['points'], axis=1)
        with timer("enrich 1"):
            df = sl_enrich_with_player_info(df)
        st.dataframe(df[columns])
        st.text('Player age {}'.format(df['player_age'].iloc[0]))
        st.text('Original points made: {}'.format(df['pointsMade'].sum()))
        st.text('Points predicted by model {}: {:.2f}'.format(model_id,df['pointsPredicted'].sum()))

    with col_right:
        ##################### Alternative Player ###################################
        st.header('Alternative Player')
        alternative_player = st.selectbox(
            "Choose your alternative player",
            sl_player_options(df_orig),
            key=app_id + "_alt_player"
        )
        age_mode = st.selectbox(
            "Choose the mode of age adjustment",
            ["none", "same age", "today"],
            key=app_id + "_age_mode"
        )
        with timer("filter alternative"):
            # start with a fresh version of df for original player
            df_alternative = sl_filter_by_date(sl_filter_by_player(df_orig, player), chosen_date)
        with timer("switch player on alternative"):
            df_alternative = sl_switch_player(df_alternative, alternative_player, df_orig)
        target_year = None
        if age_mode == "same age":
            target_year = sl_get_target_year_for_same_age(player, alternative_player, df['year'].iloc[0])
        if age_mode == "today":
            # Get the current date
            current_date = date.today()
            # Access the year attribute to get the current year
            current_year = current_date.year
            target_year = current_year
        # print(df_alternative.head(20))
        X_enc = sl_prepare_dataset_for_prediction(df_alternative, config, target_year)
        with timer("predict 2"):
            y_proba = model.predict_proba(X_enc)

        # actual points made (for comparison)
        df_alternative['pointsMade'] = df_alternative.apply(lambda row: row['SHOT_MADE_FLAG'] * row['points'], axis=1)
        df_alternative['SHOT_MADE_FLAG'] = y_proba[:, 1]
        df_alternative['pointsPredicted'] = df_alternative.apply(lambda row: row['SHOT_MADE_FLAG'] * row['points'], axis=1)


        with timer("enrich 2"):
            df_alternative = sl_enrich_with_player_info(df_alternative, target_year)
        st.dataframe(df_alternative[columns])
        st.text('Alternative player age, calculated in age mode "{}": {}'.format(age_mode, df_alternative['player_age'].iloc[0]))
        st.text('Points predicted by model {}: {:.2f}'.format(model_id, df_alternative['pointsPredicted'].sum()))

def sl_alternatives():
    return AlternativesCalculatorPipeline(
        [ThreePointAlternativesCalculator()]
    )

def sl_alternatives_app():
    app_id = "alternatives_app"
    model_id = st.selectbox(
        "Choose the model",
        [MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_LIGHT_GBM],
        key=app_id + "_model"
    )
    df_train_test = sl_app_split_data()
    df_orig = df_train_test['test']
    with timer("team options"):
        team = st.selectbox(
            "Filter by team (optional)",
            sl_team_options(df_orig),
            key=app_id + "_team"
        )
    with timer("filter by team"):
        if len(team) > 0:
            df = sl_filter_by_team(df_orig, team)
        else:
            df = df_orig.copy()

    with timer("player options"):
        player = st.selectbox(
            "Choose your player",
            sl_player_options(df),
            key=app_id + "_player"
        )
    with timer("filter by player"):
        df = sl_filter_by_player(df, player)



    config = build_best_run_config()
    config.model_config.model_id = model_id
    with timer("load persisted model "):
        model = load_persisted_model(model_id)


    df['pointsMade'] = df.apply(lambda row: row['SHOT_MADE_FLAG'] * row['points'], axis=1)

    X_enc = sl_prepare_dataset_for_prediction(df, config)
    y_proba = model.predict_proba(X_enc)
    df['probability'] = y_proba[:, 1]
    df['pointsPredicted'] = df.apply(lambda row: row['probability'] * row['points'], axis=1)


    st.text('Original points made: {}'.format(df['pointsMade'].sum()))
    st.text('Points predicted by model {}: {:.2f}'.format(model_id, df['pointsPredicted'].sum()))


    ########### Calculating alternatives #############
    alternatives_calculator = sl_alternatives()
    alternatives = alternatives_calculator.calculate_alternatives(df)
    processing_pipeline = AlternativesProcessorPipeline(alternatives)
    df_old = df.copy()
    df = processing_pipeline.process(df)

    X_enc = sl_prepare_dataset_for_prediction(df, config)
    y_proba = model.predict_proba(X_enc)
    df['probability_alternative'] = y_proba[:, 1]
    df['pointsPredicted_alternative'] = df.apply(lambda row: row['probability_alternative'] * row['points'], axis=1)

    st.text('Alternative Points predicted by model {}: {:.2f}'.format(model_id, df['pointsPredicted_alternative'].sum()))

    df = df.join(df_old,rsuffix="_old")
    columns = ['ACTION_TYPE', 'SHOT_DISTANCE', 'SHOT_DISTANCE_old','pointsPredicted', 'pointsPredicted_alternative']
    for alternative in alternatives:
        st.subheader(alternative.explanation)
        df_filtered = df[df["alternatives_explanation"] == alternative.explanation]
        diff = df_filtered['pointsPredicted_alternative'].sum() - df_filtered['pointsPredicted'].sum()
        st.dataframe(df_filtered[columns])
        st.text('"{}" can earn a difference of {:.2f} points in {} matches'.format(alternative.explanation, diff, len(df_filtered), df['GAME_DATE'].nunique()))