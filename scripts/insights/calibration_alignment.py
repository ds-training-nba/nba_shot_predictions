import numpy as np

from app.conf.run import build_default_run_config, MODEL_ID_LIGHT_GBM
from app.data_providers import ready_split_dataset
from app.modeling import build_model, predict_probabilities
from processing.helpers import combine_actual_and_prediction_dataframe, combine_result_and_x_orig_dataframe

# manually analyzing predicted probabilities and real probabilities in bin

# training pipeline
config = build_default_run_config()
config.return_probabilities = True
config.model_config.model_id = MODEL_ID_LIGHT_GBM
model = build_model(config.model_config)
X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
model.fit(X_train,y_train)

y_proba = predict_probabilities(model, X_test)

df_result = combine_actual_and_prediction_dataframe(y_test,y_proba)
df_result = combine_result_and_x_orig_dataframe(X_test_orig,df_result)

bins = np.linspace(0.1,0.9,9)
for bin in bins:
    prob = df_result[(df_result['Predicted'] > bin - 0.05) & (df_result['Predicted'] < bin + 0.05)]
    print(bin,':')
    print(prob['Actual'].count())
    print(prob['Actual'].value_counts(normalize=True), "\n")


free_throw = df_result[(df_result['ACTION_TYPE'] == 'Free Throw')]
print('free_throw:')
print(free_throw['Actual'].value_counts(normalize=True), "\n")

print(free_throw[['Predicted', 'Actual', 'PLAYER_ID', 'ACTION_TYPE', 'IS_HOME']].head(20))

#
# fig = plt.figure()
#
# CalibrationDisplay.from_estimator(
#     model,
#     X_test,
#     y_test,
#     n_bins=20
# )
#
# Trying to improve via calibrated classifier
# calibration_model = CalibratedClassifierCV(
#     model,
#     method="isotonic",
#     cv=3
# )
# calibration_model.fit(X_train,y_train)
# fig = plt.figure()
# CalibrationDisplay.from_estimator(
#     calibration_model,
#     X_test,
#     y_test,
#     n_bins=20
# )
# plt.show()
