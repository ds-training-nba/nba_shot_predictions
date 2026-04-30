import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_SVM, MODEL_ID_LOGISTIC_REGRESSION
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "field_goals_only"
# default: RandomForest + Player Encoding OneHot
config1 = build_default_run_config()
config1.context_name = 'Field Goals Only'
config1.use_only_field_goals = True
config1.model_config.model_id = MODEL_ID_LOGISTIC_REGRESSION

run_experiment([config1], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

