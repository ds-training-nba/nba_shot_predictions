from app.conf.run import build_default_run_config
from app.config import RESULTS_PATH
from app.modeling import model_prediction
from app.output import save_classification_run
config = build_default_run_config()
y_pred, y_test = model_prediction(config)
save_classification_run(y_test, y_pred, config,  RESULTS_PATH)