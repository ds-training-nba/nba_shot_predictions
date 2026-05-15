
import matplotlib.pyplot as plt

from app.conf.run import build_best_run_config, MODEL_ID_LOGISTIC_REGRESSION, MODEL_ID_DECISION_TREE, \
    MODEL_ID_LIGHT_GBM, MODEL_ID_RANDOM_FOREST
from app.data_providers import ready_split_dataset
from app.modeling import build_model
from plot.interpretability import feature_importance_bar_plot

config = build_best_run_config()
config.model_config.model_id = MODEL_ID_LOGISTIC_REGRESSION
model = build_model(config.model_config)


X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
model.fit(X_train,y_train)
models = {
    MODEL_ID_LOGISTIC_REGRESSION: model
}
feature_importance_bar_plot(X_train.columns, model.coef_[0],"Logistic Regression Feature Importance", "coefficient")

tree_models = [MODEL_ID_DECISION_TREE, MODEL_ID_LIGHT_GBM, MODEL_ID_RANDOM_FOREST]
for model_id in tree_models:
    config.model_config.model_id = model_id
    model = build_model(config.model_config)
    model.fit(X_train,y_train)
    models[model_id] = model
    feature_importance_bar_plot(X_train.columns, model.feature_importances_, model_id + " Feature Importance")


from sklearn.inspection import permutation_importance
for model_id in models:
    model = models[model_id]

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42
    )
    feature_importance_bar_plot(X_train.columns, result.importances_mean, model_id + " Permutation Importance")

plt.show()