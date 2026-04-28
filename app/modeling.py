from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

from app.config import TARGET_VARIABLE
from app.data_providers import test_train_dataset
from processing.encoding import encode_for_model
from processing.filtering import filter_pre_encoding_columns


MODEL_ID_RANDOM_FOREST = "RandomForest"


def evaluate_model(model_id: str):
    """
    whole processing pipeline, yet to be made testable and configurable
    :return: None
    """
    dataset = test_train_dataset()
    print(dataset['test'].head())
    print(dataset['train'].head())
    print(dataset['train'].columns)

    X_test, y_test = split_x_y(filter_pre_encoding_columns(dataset['test']))
    X_train, y_train = split_x_y(filter_pre_encoding_columns(dataset['train']))
    X_train, X_test = encode_for_model(X_train, model_id, X_test)
    model = build_model(model_id)
    model.fit(X_train, y_train)
    y_pred = predict(model, X_test)
    cm, cr = evaluate_predictions(y_test, y_pred)
    return cm,cr


def evaluate_predictions(y_test, y_pred):
    cm = pd.crosstab(y_test, y_pred, rownames=['Real Class'], colnames=['Predicted Class'])
    cr = classification_report(y_test, y_pred)
    return cm, cr

def predict(model, X):
    """
    Abstraction for models that do not always have sklearn interface
    :param model:
    :param X:
    :return:
    """
    return model.predict(X)

def split_x_y(df):
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    return X,y

def build_model(model_id: str):
    global MODEL_ID_RANDOM_FOREST
    match model_id:
        case MODEL_ID_RANDOM_FOREST:
            return RandomForestClassifier(n_jobs=-1, random_state=321)
