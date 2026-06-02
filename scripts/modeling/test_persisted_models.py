from app.conf.run import build_best_run_config, MODEL_ID_LOGISTIC_REGRESSION, MODEL_ID_RANDOM_FOREST, \
    MODEL_ID_DECISION_TREE, MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_LIGHT_GBM
from app.data_providers import ready_split_dataset
from app.modeling import load_persisted_model, evaluate_predictions


config = build_best_run_config()

model_ids = [
    MODEL_ID_LOGISTIC_REGRESSION,
    MODEL_ID_RANDOM_FOREST,
    MODEL_ID_DECISION_TREE,
    MODEL_ID_SIMPLE_LOOKUP,
    MODEL_ID_LIGHT_GBM
]
# Load all persisted models in a loop
for model_id in model_ids:
    print("Loading model", model_id, "\n")
    config.model_config.model_id = model_id
    X_train_enc, y_train, X_test_enc, y_test, X_train, X_test = ready_split_dataset(config)
    model = load_persisted_model(config.model_config.model_id)
    y_pred= model.predict(X_test_enc)
    cm, cr = evaluate_predictions(y_test, y_pred)
    print(cm)
    print(cr)
    print("\n\n")
