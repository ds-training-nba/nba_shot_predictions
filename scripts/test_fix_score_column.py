from app.data_providers import provide_dataframe, DataFrameRequest
from processing.fixes import fix_score_column, fix_score_columns_for_all_games

request = DataFrameRequest(
    add_computed = False,
    filter_pre_encoding_columns = False,
    encode_for_model = False,
    filter_top_players=False,
    filter_clean=False
)
df = provide_dataframe(request)
# game_id = 20300354
# df = fix_score_column(df, 20300354)
# print(df[df['GAME_ID_x'] == game_id][['scoreHome', 'scoreAway', 'SCOREMARGIN']])

fix_score_columns_for_all_games(df)
