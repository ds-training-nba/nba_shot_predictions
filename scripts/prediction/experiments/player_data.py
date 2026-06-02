import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_LIGHT_GBM,
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path

# comparing the use of different player related data
experiment_id = "player_data"
# default: no added column
config1 = build_default_run_config()
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
config1.context_name = "Default for comparison"
config2 = copy.deepcopy(config1)
config2.context_name = "player age"
# adding player age columne
config2.encoding_config.passthrough_cols.append('player_age')
config3 = copy.deepcopy(config1)
config3.context_name = "experience"
#
config3.encoding_config.passthrough_cols.append('years_experience')
config4 = copy.deepcopy(config1)
config4.context_name = "year"
config4.encoding_config.passthrough_cols.append('year')
config5 = copy.deepcopy(config1)
config5.context_name = "age and best age"
config5.encoding_config.passthrough_cols.append('player_age')
config5.encoding_config.passthrough_cols.append('best_age')
config6 = copy.deepcopy(config5)
config6.context_name = "age and best age and year"
config6.encoding_config.passthrough_cols.append('year')
config7 = copy.deepcopy(config1)
config7.context_name = "last_match_precision"
config7.encoding_config.passthrough_cols.append('last_match_precision')
config8 = copy.deepcopy(config6)
config8.context_name = "age cols, year, last_match_precision"
config8.encoding_config.passthrough_cols.append('last_match_precision')

run_experiment([config1,config2, config3, config4,config5, config6, config7, config8], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

