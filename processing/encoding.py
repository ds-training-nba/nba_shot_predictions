from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from app.conf.run import EncodingConfig


def encode_for_model(X_train, model_to_encode_for: str, encoding_config: EncodingConfig, X_test = None):


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), encoding_config.one_hot_cols),
            ("num", "passthrough", encoding_config.passthrough_cols)
        ],
        remainder="drop"  # lässt die anderen Spalten unverändert
    )

    X_train = preprocessor.fit_transform(X_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)

    return X_train, X_test
