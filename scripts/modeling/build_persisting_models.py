from app.conf.run import build_best_run_config, MODEL_ID_RANDOM_FOREST, MODEL_ID_LOGISTIC_REGRESSION, \
    MODEL_ID_LIGHT_GBM, MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_DECISION_TREE
from app.modeling import build_persisted_model


build_persisted_model(MODEL_ID_RANDOM_FOREST)

build_persisted_model(MODEL_ID_LOGISTIC_REGRESSION)

build_persisted_model(MODEL_ID_LIGHT_GBM)

build_persisted_model(MODEL_ID_SIMPLE_LOOKUP)

build_persisted_model(MODEL_ID_DECISION_TREE)