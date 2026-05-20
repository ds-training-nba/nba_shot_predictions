import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_LIGHT_GBM, \
    build_best_run_config, get_active_columns, EncodingConfig
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "non_encoding_for_tree_based_models"
# default: no added column
config1 = build_best_run_config()
config1.context_name = "Best with encoding"
config2 = build_best_run_config()
config2.context_name = "Best with encoding to passthrough"
config2.encoding_config.str_cat_cols = config2.encoding_config.one_hot_cols
config2.encoding_config.one_hot_cols = []




run_experiment([config1,config2], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

