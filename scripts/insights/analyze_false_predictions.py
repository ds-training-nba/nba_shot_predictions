from app.conf.run import build_default_run_config, MODEL_ID_LIGHT_GBM
from evaluation.insights import get_false_predictions


config = build_default_run_config()
config.return_probabilities = True
config.model_config.model_id = MODEL_ID_LIGHT_GBM
false_negatives, false_positives = get_false_predictions(config)


