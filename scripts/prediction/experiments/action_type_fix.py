import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "action_type_fix"
# default: RandomForest
config1 = build_default_run_config()
config1.context_name = "Action Type with leakage"
config1.encoding_config.one_hot_cols.append('ACTION_TYPE')
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
# Model = Logistic Regression
config2 = copy.deepcopy(config1)
config2.context_name = "Action Type fixed"
config2.use_action_type_fix = True



run_experiment([config1,config2], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

