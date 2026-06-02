from app.conf.run import MODEL_ID_DEEP_LEARNING, build_best_run_config
from app.config import RESULTS_PATH
from app.modeling import model_prediction
from app.output import save_classification_run

# simple run of training and evaluation
config = build_best_run_config()
config.model_config.model_id = MODEL_ID_DEEP_LEARNING
config.return_probabilities = True
y_pred, y_test, y_pred_train, y_train = model_prediction(config)
save_classification_run(y_test, y_pred[:,1] > config.decision_boundary, config,  RESULTS_PATH,y_proba=y_pred)