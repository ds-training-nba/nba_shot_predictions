import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "calibration"
# Automatically comparing the use of calibratedclassifiercv wrap to the bare model
config1 = build_default_run_config()
config1.context_name = "Raw Model"
config1.model_config.model_id = MODEL_ID_LIGHT_GBM

config2 = copy.deepcopy(config1)
config2.context_name = "Model Wrapped in CalibratedClassifierCV"
config2.model_config.wrap_calibrated = True



run_experiment([config1,config2], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

