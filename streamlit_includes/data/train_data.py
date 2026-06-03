import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@st.cache_resource
def train_models(df):

    target = "SHOT_MADE_FLAG"
    numerical_fatures = ['SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'ANGLE_SIN', 'ANGLE_COS', 'ANGLE', 'ANGLE_SECTOR',
                         'ABS_ANGLE', 'TimeRemainingInPeriod', 'TimeRemainingInGame', 'scoreMarginBeforeShot',
                         'scoreHomeBeforeShot', 'scoreAwayBeforeShot', 'PERIOD_x', 'OvertimeNumber', 'TEAM_ID',
                         'PLAYER_ID', 'TotalPlayedTime']
    categorical_features = ['SHOT_TYPE', 'SHOT_ZONE_RANGE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'ACTION_TYPE',
                            'MAIN_ACTION_TYPE', 'PLAYER_NAME']
    boolean_features = ['is_playoffs', 'IS_HOME', 'IsOvertime', 'IsClutchTime', 'OPPONENT_INTERFERED']
    all_features = numerical_fatures + categorical_features + boolean_features

    df_model = df[all_features + [target]].dropna().sample(50000, random_state=42)

    X = df_model[all_features]
    y = df_model[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    num_cols = X_train.select_dtypes(include=['int64','float64','bool']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
    }

    results = []
    roc_data = {}
    cm_data = {}

    fitted_models = {}

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # metrics
        results.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Log Loss": log_loss(y_test, y_proba),
            "Brier": brier_score_loss(y_test, y_proba),
            "Accuracy": (y_pred == y_test).mean()
        })

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr, "tpr": tpr}

        # CM
        cm_data[name] = confusion_matrix(y_test, y_pred)

        fitted_models[name] = pipe

    return (
        fitted_models,
        pd.DataFrame(results),
        roc_data,
        cm_data,
        X_train, X_test, y_train, y_test
    )