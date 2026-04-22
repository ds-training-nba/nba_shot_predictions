from app.data_providers import provide_dataframe, DataFrameRequest
from processing.fixes import fix_score_column, fix_score_columns_for_all_games
import pandas as pd
import resource
import sys

def limit_memory(max_bytes):
    # Set soft and hard limits
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

# Limit to 10 GB (1024 * 1024 * 1024 bytes)
limit_memory(22*1024 * 1024 * 1024)
# configuring pipeline
request = DataFrameRequest(
    add_computed = True,
    apply_preprocessing = True,
    filter_pre_encoding_columns = False,
    encode_for_model = False,
    filter_top_players=False,
    filter_clean= True
)
# pipeline execution
df = provide_dataframe(request)
game_id = 20300354
# df['SCORE_HOME_BT'] = 0
# df['SCORE_AWAY_BT'] = 0
# df = fix_score_column(df, 20300354)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df[df['GAME_ID_x'] == game_id][['SHOT_MADE_FLAG', 'scoreHome', 'scoreAway', 'IS_HOME', 'points', "pointsHome", "pointsAway", 'scoreMarginBeforeShot']])

# fix_score_columns_for_all_games(df)
