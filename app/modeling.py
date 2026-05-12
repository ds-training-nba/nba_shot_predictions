from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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
            return RandomForestClassifier()
        case app.conf.run.MODEL_ID_LOGISTIC_REGRESSION:
            # params according to RandomSearchCV
            return LogisticRegression(solver="liblinear", l1_ratio=0, max_iter=1000, class_weight=None, C=0.01)
        case app.conf.run.MODEL_ID_DECISION_TREE:
            return DecisionTreeClassifier()
        case app.conf.run.MODEL_ID_LIGHT_GBM:
            return LGBMClassifier()
def build_param_grid(model_config: ModelConfig):
    match model_config.model_id:
        case app.conf.run.MODEL_ID_RANDOM_FOREST:
            return {
                "n_estimators": [100, 300],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
                "bootstrap": [True]
            }

        case app.conf.run.MODEL_ID_LOGISTIC_REGRESSION:
            return {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"],
                "max_iter": [1000]
            }
        case app.conf.run.MODEL_ID_LIGHT_GBM:
            return {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 5, 10],
                "min_child_samples": [10, 20, 50],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        case app.conf.run.MODEL_ID_DECISION_TREE:
            return {
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"]
            }
def run_grid_search(config: RunConfig, cv):
    model = build_model(config.model_config)
    param_grid = build_param_grid(config.model_config)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=cv,
        scoring=config.metric_string,
        n_jobs=-1,
        n_iter=10
    )
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    search.fit(X_train, y_train)

    y_pred = predict(search, X_test) if not config.return_probabilities else predict_probabilities(search, X_test)
    return y_pred, y_test, {"best_params": search.best_params_, "best_score": search.best_score_ }

def run_feature_selection(config: RunConfig):
    model = build_model(config.model_config)
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2
    )

    X_train_sel = rfecv.fit_transform(X_train, y_train)
    ranking = pd.Series(rfecv.ranking_, index=X_train.columns)
    y_pred = rfecv.predict(X_test)


    ranking = ranking.sort_values()
    selected_features = X_train.columns[rfecv.support_]
    rejected_features = X_train.columns[~rfecv.support_]
    return  ranking, y_pred, y_test
