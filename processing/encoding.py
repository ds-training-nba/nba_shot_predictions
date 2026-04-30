from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler

from app.conf.run import EncodingConfig


def encode_for_model(X_train, y_train, model_to_encode_for: str, encoding_config: EncodingConfig, X_test = None):


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), encoding_config.one_hot_cols),
            ("target", TargetEncoder(), encoding_config.target_enc_cols),
            ("std_scale", StandardScaler(), encoding_config.std_scale_cols),
            ("num", "passthrough", encoding_config.passthrough_cols)
        ],
        remainder="drop"  # lässt die anderen Spalten unverändert
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)

    return X_train, X_test
