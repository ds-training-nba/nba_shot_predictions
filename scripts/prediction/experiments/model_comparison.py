import pandas as pd
import copy

from app.conf.run import  MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM, build_best_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_DECISION_TREE
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "model_comparison"

config1 = build_best_run_config()
config1.context_name = "Random Forest"
config1.model_config.model_id = MODEL_ID_RANDOM_FOREST

config2 = copy.deepcopy(config1)
config2.context_name = "Logistic Regression"
config2.model_config.model_id = MODEL_ID_LOGISTIC_REGRESSION

config3 = copy.deepcopy(config1)
config3.context_name = "Light GBM"
config3.model_config.model_id = MODEL_ID_LIGHT_GBM

config4 = build_best_run_config()
config4.context_name = "Simple Lookup"
config4.model_config.model_id = MODEL_ID_SIMPLE_LOOKUP

config5 = build_best_run_config()
config5.context_name = "Decision Tree"
config5.model_config.model_id = MODEL_ID_DECISION_TREE


run_experiment([config1,config2,config3,config4, config5], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head(10))

