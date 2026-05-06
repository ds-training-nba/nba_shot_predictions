from app.conf.run import build_default_run_config, MODEL_ID_LIGHT_GBM
from app.data_providers import ready_split_dataset
from app.modeling import model_prediction, build_model, predict_probabilities
from evaluation.insights import get_false_predictions
from processing.helpers import combine_actual_and_prediction_dataframe, combine_result_and_x_orig_dataframe
import pandas as pd
config = build_default_run_config()
config.return_probabilities = True
config.model_config.model_id = MODEL_ID_LIGHT_GBM
X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
X_test_orig = X_test_orig.reset_index(drop=True)
model = build_model(config.model_config)
model.fit(X_train, y_train)

y_proba = predict_probabilities(model, X_test)

df_result = combine_actual_and_prediction_dataframe(y_test,y_proba)
df_result = combine_result_and_x_orig_dataframe(X_test_orig,df_result)
print(df_result['Actual'].value_counts(normalize=1))
df_result = df_result[df_result['OPPONENT_INTERFERED'] == 1]
print(df_result['Actual'].value_counts(normalize=1))




