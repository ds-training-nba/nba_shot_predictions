import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST,  MODEL_ID_LIGHT_GBM, \
    build_best_run_config
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "additional_main_action_type"
# default: no added column
config1 = build_best_run_config()
config1.context_name = "Best (including MAIN_ACTION_TYPE) "
config2 = build_best_run_config()
config2.context_name = "Best without MAIN_ACTION_TYPE"
config2.encoding_config.str_cat_cols.remove("MAIN_ACTION_TYPE")
config3 = copy.deepcopy(config1)
config3.use_only_field_goals = True
config3.context_name = "Only Field Goals: Best including MAIN_ACTION_TYPE"
config4 = copy.deepcopy(config3)
config4.context_name = "Only Field Goals: Best WITHOUT MAIN_ACTION_TYPE"
config4.encoding_config.str_cat_cols.remove("MAIN_ACTION_TYPE")


run_experiment([config1,config2, config3,config4], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

