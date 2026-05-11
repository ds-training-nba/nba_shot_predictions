import pandas as pd
import copy

from app.conf.run import build_default_run_config, MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path


experiment_id = "manual_feature_comparison"

config1 = build_default_run_config()
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
config1.context_name = "No suspicious column"

config2 = copy.deepcopy(config1)
config2.context_name = "Action Type with leakage"
config2.encoding_config.one_hot_cols.append('ACTION_TYPE')


config3 = copy.deepcopy(config2)
config3.context_name = "For Context: Action Type AND Opponent Interfered"
config3.encoding_config.passthrough_cols.append('OPPONENT_INTERFERED')

config4 = copy.deepcopy(config2)
config4.context_name = "Using Action Type WITH leakage FIX"
config4.use_action_type_fix = True

config5 = copy.deepcopy(config4)
config5.context_name = "Action Type WITH leakage FIX + other columns"
config5.encoding_config.std_scale_cols.append('ABS_ANGLE')
config5.encoding_config.std_scale_cols.append('TimeRemainingInGame')
config5.encoding_config.passthrough_cols.append('IsOvertime')
config5.encoding_config.one_hot_cols.append('SHOT_ZONE_RANGE')
config5.encoding_config.one_hot_cols.append('OvertimeNumber')
config5.encoding_config.one_hot_cols.append('PERIOD_x')
config5.encoding_config.one_hot_cols.append('SHOT_ZONE_BASIC')
config5.encoding_config.one_hot_cols.append('SHOT_ZONE_AREA')





run_experiment([config1,config2, config3, config4, config5], experiment_id)

df = load_runs_to_dataframe(experiment_current_path(experiment_id))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

