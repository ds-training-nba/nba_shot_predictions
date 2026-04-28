from app.config import RESULTS_PATH
from app.modeling import model_prediction, MODEL_ID_RANDOM_FOREST
from app.output import save_classification_run
model = MODEL_ID_RANDOM_FOREST
y_pred, y_test = model_prediction(model)
save_classification_run(y_test, y_pred, model, {}, RESULTS_PATH)