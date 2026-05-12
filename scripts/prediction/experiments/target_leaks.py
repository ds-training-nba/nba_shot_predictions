import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "target_leakage"

config1 = build_default_run_config()
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
config1.context_name = "No suspicious column"

config2 = copy.deepcopy(config1)
config2.context_name = "Action Type with leakage"
config2.encoding_config.one_hot_cols.append('ACTION_TYPE')


config3 = copy.deepcopy(config2)
config3.context_name = "Action Type AND Opponent Interfered"
config3.encoding_config.passthrough_cols.append('OPPONENT_INTERFERED')







run_experiment([config1,config2, config3], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

