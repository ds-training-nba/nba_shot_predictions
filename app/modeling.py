from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import pandas as pd

import app.conf.run
from app.conf.run import RunConfig, ModelConfig
from app.data_providers import ready_split_dataset


def model_prediction(config: RunConfig):

    """
    whole processing pipeline, yet to be made testable and configurable
    :return: None
    """
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    model = build_model(config.model_config)
    model.fit(X_train, y_train)

    y_pred = predict(model, X_test) if not config.return_probabilities else predict_probabilities(model, X_test)
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

def predict_probabilities(model, X):
    """
    Abstraction for models that do not always have sklearn interface
    :param model:
    :param X:
    :return:
    """
    return model.predict_proba(X)


def build_model(model_config: ModelConfig):
    match model_config.model_id:
        case app.conf.run.MODEL_ID_RANDOM_FOREST:
            return RandomForestClassifier(n_jobs=-1, random_state=321)
        case app.conf.run.MODEL_ID_SVM:
            return SVC()
        case app.conf.run.MODEL_ID_LOGISTIC_REGRESSION:
            return LogisticRegression()
        case app.conf.run.MODEL_ID_LIGHT_GBM:
            return LGBMClassifier(
                        n_estimators=1000,
                        learning_rate=0.1,
                        max_depth=10,
                        n_jobs=-1
                    )
