from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

import app.conf.run
from app.conf.run import RunConfig
from app.config import TARGET_VARIABLE
from app.data_providers import test_train_dataset
from processing.encoding import encode_for_model
from processing.filtering import filter_pre_encoding_columns


def model_prediction(config: RunConfig):

    """
    whole processing pipeline, yet to be made testable and configurable
    :return: None
    """
    dataset = test_train_dataset()
    X_test, y_test = split_x_y(dataset['test'])
    X_train, y_train = split_x_y(dataset['train'])
    X_train, X_test = encode_for_model(X_train, config.model_config.model_id,config.encoding_config, X_test)
    model = build_model(config.model_config.model_id)
    model.fit(X_train, y_train)
    y_pred = predict(model, X_test)
    return y_pred, y_test


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
    match model_id:
        case app.conf.run.MODEL_ID_RANDOM_FOREST:
            return RandomForestClassifier(n_jobs=-1, random_state=321)
