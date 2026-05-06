import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_SVM, MODEL_ID_LIGHT_GBM
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "score_margin"
# default: RandomForest + Player Encoding OneHot
config1 = build_default_run_config()
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
config1.context_name = "Score Margin not present"
config2 = copy.deepcopy(config1)
config2.context_name = "Score Margin std scaled"
# Target + RandomForest
config2.encoding_config.std_scale_cols.append('scoreMarginBeforeShot')

run_experiment([config1,config2], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

