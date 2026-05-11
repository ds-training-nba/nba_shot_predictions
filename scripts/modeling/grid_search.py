from app.conf.run import build_default_run_config, MODEL_ID_DECISION_TREE, MODEL_ID_LIGHT_GBM, \
    MODEL_ID_LOGISTIC_REGRESSION
import copy

from app.experiments import run_grid_search_experiment

experiment_id = "model_comparison_grid_search"
# default: RandomForest
config1 = build_default_run_config()
config1.context_name = "Random Forest"
# Adding to all Configs
config1.encoding_config.std_scale_cols.append('scoreMarginBeforeShot')
config1.encoding_config.passthrough_cols.append('IsClutchTime')
# Model = Logistic Regression
config2 = copy.deepcopy(config1)
config2.context_name = "Logistic Regression"
config2.model_config.model_id = MODEL_ID_LOGISTIC_REGRESSION
config3 = copy.deepcopy(config1)
config3.context_name = "Light GBM"
config3.model_config.model_id = MODEL_ID_LIGHT_GBM
config4 = copy.deepcopy(config1)
config4.context_name = "Decision Tree"
config4.model_config.model_id = MODEL_ID_DECISION_TREE


run_grid_search_experiment(
    [
        #config2,
        config1,
        #config3,
    #config4
    ], experiment_id
)