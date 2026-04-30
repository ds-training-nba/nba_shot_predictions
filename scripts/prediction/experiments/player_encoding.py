from app.conf.run import build_default_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_SVM, MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM
import copy
from app.experiments import run_experiment, load_runs_to_dataframe, experiment_current_path



#default: RandomForest + Player Encoding OneHot
config1 = build_default_run_config()
config1.model_config.model_id = MODEL_ID_LIGHT_GBM
config1.context_name = "Player Enc: One Hot"
config2 = copy.deepcopy(config1)
config2.context_name = "Player Enc: Target Enc"
# Target + RandomForest
config2.encoding_config.target_enc_cols.append('PLAYER_ID')
config2.encoding_config.one_hot_cols.remove('PLAYER_ID')

run_experiment([config1,config2], 'player_encoding')

df = load_runs_to_dataframe(experiment_current_path('player_encoding'))

print(df.head())

