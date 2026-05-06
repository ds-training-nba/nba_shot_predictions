import pandas as pd

from app.conf.run import RunConfig, get_active_columns
from app.data_providers import ready_split_dataset
from app.modeling import build_model, predict_probabilities
from processing.helpers import combine_actual_and_prediction_dataframe, combine_result_and_x_orig_dataframe


def worst_prediction_failures(y_proba, y_test):
    df_results = combine_actual_and_prediction_dataframe(y_test,y_proba)
    worst_false_positives = df_results[(df_results['Actual'] == 0) & (df_results['Predicted'] > 0.5)].sort_values(by='Predicted', ascending=False)
    worst_false_negatives = df_results[(df_results['Actual'] == 1) & (df_results['Predicted'] < 0.5)].sort_values(by='Predicted', ascending=True)
    return worst_false_negatives, worst_false_positives

def get_false_predictions(config: RunConfig):
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    X_test_orig = X_test_orig.reset_index(drop=True)
    model = build_model(config.model_config)
    model.fit(X_train, y_train)

    y_proba = predict_probabilities(model, X_test)

    false_negatives, false_positives = worst_prediction_failures(y_proba, y_test)
    false_negatives = combine_result_and_x_orig_dataframe(X_test_orig, false_negatives)
    false_positives = combine_result_and_x_orig_dataframe(X_test_orig, false_positives)

    cols = get_active_columns(config.encoding_config) + ['Actual', 'Predicted']
    false_positives = false_positives[cols]
    false_negatives = false_negatives[cols]
    return false_negatives, false_positives