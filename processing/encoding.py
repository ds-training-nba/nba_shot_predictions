from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder





def encode_for_model(X_train, model_to_encode_for: str, X_test = None):
    categorical_cols = ["MAIN_ACTION_TYPE"]
    numeric_cols = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # lässt die anderen Spalten unverändert
    )

    X_train = preprocessor.fit_transform(X_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)

    return X_train, X_test
