import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST,  MODEL_ID_LIGHT_GBM, \
    build_best_run_config
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "best_vs_default"
# default: no added column
config1 = build_default_run_config()
config1.context_name = "Default"

config2 = build_best_run_config()
config2.context_name = "Best"


run_experiment([config1,config2], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

