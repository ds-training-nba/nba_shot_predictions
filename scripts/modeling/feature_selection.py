from sklearn.metrics import classification_report

from app.conf.run import build_default_run_config, MODEL_ID_LOGISTIC_REGRESSION


from app.modeling import run_feature_selection

config = build_default_run_config()
config.model_config.model_id = MODEL_ID_LOGISTIC_REGRESSION
config.encoding_config.std_scale_cols.append('ABS_ANGLE')
config.encoding_config.one_hot_cols.append('SHOT_ZONE_RANGE')
config.encoding_config.one_hot_cols.append('SHOT_ZONE_BASIC')
config.encoding_config.one_hot_cols.append('SHOT_ZONE_AREA')
config.encoding_config.one_hot_cols.append('ACTION_TYPE')
config.encoding_config.passthrough_cols.append('OPPONENT_INTERFERED')
sel, y_pred, y_test = run_feature_selection(config)
print(sel)
print(classification_report(y_test,y_pred))