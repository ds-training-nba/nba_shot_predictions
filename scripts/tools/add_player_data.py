from app.data_providers import test_train_dataset, full_dataset
import pandas as pd

from processing.compute_columns import add_player_data


ds = test_train_dataset()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(ds['train'][ds['train']['PLAYER_NAME'] == 'Kevin Garnett'][['GAME_DATE', 'GAME_ID_x', 'PLAYER_NAME', 'best_age','last_match_precision']])
    #print(ds['test'][ds['test']['PLAYER_NAME'] == 'Kevin Garnett'][['year', 'PLAYER_NAME', 'best_age','last_match_precision']])
    print(ds['train'][(ds['train']['PLAYER_NAME'] == 'Kevin Garnett') & (ds['train']['GAME_ID_x'] == 21300692)])



