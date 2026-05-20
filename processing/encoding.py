from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler

from app.conf.run import EncodingConfig, can_handle_categories, MODEL_ID_SIMPLE_LOOKUP, MODEL_ID_LOGISTIC_REGRESSION


def encode_for_model(X_train, y_train, model_to_encode_for: str, encoding_config: EncodingConfig, X_test = None):

    if model_to_encode_for == MODEL_ID_SIMPLE_LOOKUP:
        return X_train, X_test
    one_hot_cols = []
    passthrough_cols = encoding_config.passthrough_cols
    std_scale_cols = encoding_config.std_scale_cols
    # decide what to do with str_cat_cols according to model capacities
    if can_handle_categories(model_to_encode_for):
        passthrough_cols += encoding_config.str_cat_cols
    else:
        one_hot_cols = encoding_config.str_cat_cols

    if model_to_encode_for == MODEL_ID_LOGISTIC_REGRESSION:
        std_scale_cols += encoding_config.passthrough_cols
        passthrough_cols = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_cols),
            ("target", TargetEncoder(), encoding_config.target_enc_cols),
            ("std_scale", StandardScaler(), encoding_config.std_scale_cols),
            ("pass", "passthrough", passthrough_cols)
        ],
        remainder="drop"  # drops all other columns, so that w have a clean definition of wanted cols from encoding config
    )
    preprocessor.set_output(transform="pandas")
    X_train = preprocessor.fit_transform(X_train, y_train)

    if X_test is not None:
        X_test = preprocessor.transform(X_test)
    if can_handle_categories(model_to_encode_for):
        # XGBoost Models can handle string categories, as long as they are of type "category"
        for col in encoding_config.str_cat_cols:
            X_train['pass__' + col] = X_train['pass__' + col].astype("category")
            if X_test is not None:
                X_test['pass__' + col] = X_test['pass__' + col].astype("category")

    return X_train, X_test
