from app.config import PLAYER_CHOICE, CLEAN_SOURCE_COLUMNS, EXPLANATORY_VARIABLES_PRE_ENCODING, TARGET_VARIABLE


def filter_for_players(df):
    chosen_player_matrix = None

    for name in PLAYER_CHOICE:
        current_matrix = df['PLAYER_NAME'] == name
        if chosen_player_matrix is None:
            chosen_player_matrix = current_matrix
        else:
            chosen_player_matrix = chosen_player_matrix | current_matrix

    return df[chosen_player_matrix]


def filter_clean_source_columns(df):
    return df[CLEAN_SOURCE_COLUMNS]


def filter_pre_encoding_columns(df):
    columns = EXPLANATORY_VARIABLES_PRE_ENCODING + [TARGET_VARIABLE]
    return df[columns]


