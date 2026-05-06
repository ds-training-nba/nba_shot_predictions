import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_SVM, MODEL_ID_LIGHT_GBM
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "time_left"
# default: RandomForest + Player Encoding OneHot
config1 = build_default_run_config()
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
config1.context_name = "Time Left not present"
config2 = copy.deepcopy(config1)
config2.context_name = "Using Time Left std scaled"
config2.encoding_config.std_scale_cols.append('TimeRemainingInGame')
config4 = copy.deepcopy(config1)
config4.context_name = "Using Time Left passedthrough"
config4.encoding_config.passthrough_cols.append('TimeRemainingInGame')
config3 = copy.deepcopy(config1)
config3.context_name = "Using IsClutchTime"
# Target + RandomForest
config3.encoding_config.passthrough_cols.append('IsClutchTime')

config5 = copy.deepcopy(config3)
config5.context_name = "Using IsClutchTime and Time left passed through"
# Target + RandomForest
config5.encoding_config.passthrough_cols.append('TimeRemainingInGame')

run_experiment([config1,config2,config4,config3, config5], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

