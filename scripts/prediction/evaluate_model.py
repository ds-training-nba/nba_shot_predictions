from app.conf.run import build_default_run_config, MODEL_ID_SIMPLE_LOOKUP
from app.config import RESULTS_PATH
from app.modeling import model_prediction
from app.output import save_classification_run
config = build_default_run_config()
config.model_config.model_id = MODEL_ID_SIMPLE_LOOKUP
config.return_probabilities = True
y_pred, y_test, y_pred_train, y_train = model_prediction(config)
save_classification_run(y_test, y_pred[:,1] > config.decision_boundary, config,  RESULTS_PATH,y_proba=y_pred)