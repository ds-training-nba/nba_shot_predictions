"""
Run this script ONCE before launching the Streamlit app:
    python streamlit_includes/data/train_and_save.py

It trains all 4 models on the Top-20 NBA shot dataset and saves:
    streamlit_includes/data/models/
        xgboost_pipeline.joblib
        random_forest_pipeline.joblib
        logistic_regression_pipeline.joblib
        knn_pipeline.joblib
        metrics.pkl          ← results_df, roc_data, cm_data
        splits.pkl           ← X_test, y_test (for evaluation display)
        feature_importance.pkl ← sorted feature names + importances (XGBoost)
"""

import os

import pickle
import numpy as np
import pandas as pd
import joblib
import time


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    roc_curve, confusion_matrix, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from streamlit_includes.data.top_20_dataset import get_top_20_shots


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("Loading dataset...")
    df = get_top_20_shots()
    print(df.info())

    target = "SHOT_MADE_FLAG"
    numerical_features = [
        'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'ANGLE_SIN', 'ANGLE_COS',
        'ANGLE', 'ANGLE_SECTOR', 'ABS_ANGLE',
        'TimeRemainingInPeriod', 'TimeRemainingInGame',
        'scoreMarginBeforeShot', 'scoreHomeBeforeShot', 'scoreAwayBeforeShot',
        'PERIOD_x', 'OvertimeNumber', 'TEAM_ID', 'PLAYER_ID', 'TotalPlayedTime'
    ]
    categorical_features = [
        'SHOT_TYPE', 'SHOT_ZONE_RANGE', 'SHOT_ZONE_BASIC',
        'SHOT_ZONE_AREA', 'ACTION_TYPE', 'MAIN_ACTION_TYPE', 'PLAYER_NAME'
    ]
    boolean_features = [
        'is_playoffs', 'IS_HOME', 'IsOvertime', 'IsClutchTime', 'OPPONENT_INTERFERED'
    ]
    all_features = numerical_features + categorical_features + boolean_features

    df_model = df[all_features + [target]].dropna()
    X = df_model[all_features]
    y = df_model[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    models_config = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "KNN":                 KNeighborsClassifier(n_neighbors=15),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    }

    results = []
    roc_data = {}
    cm_data = {}

    for name, model in models_config.items():
        print(f"Training {name}...")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        start_time = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model":    name,
            "ROC-AUC":  round(roc_auc_score(y_test, y_proba), 4),
            "Log Loss": round(log_loss(y_test, y_proba), 4),
            "Brier":    round(brier_score_loss(y_test, y_proba), 4),
            "Accuracy": round((y_pred == y_test).mean(), 4),
            "F1":       round(f1_score(y_test, y_pred), 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        cm_data[name]  = confusion_matrix(y_test, y_pred)

        # save individual pipeline
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{safe_name}_pipeline.joblib"))
        print(f"  ✓ Saved {safe_name}_pipeline.joblib")
        print(f"  ✓ Training time {train_time} seconds")

    # feature importance from XGBoost
    xgb_pipe   = joblib.load(os.path.join(MODELS_DIR, "xgboost_pipeline.joblib"))
    ohe        = xgb_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_names  = ohe.get_feature_names_out(cat_cols).tolist()
    feat_names = num_cols + ohe_names
    importances = xgb_pipe.named_steps["model"].feature_importances_

    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # save metrics bundle
    with open(os.path.join(MODELS_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump({
            "results_df": pd.DataFrame(results),
            "roc_data":   roc_data,
            "cm_data":    cm_data,
        }, f)

    # save test split for evaluation display
    with open(os.path.join(MODELS_DIR, "splits.pkl"), "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)

    # save feature importance
    with open(os.path.join(MODELS_DIR, "feature_importance.pkl"), "wb") as f:
        pickle.dump(fi_df, f)

    print("\n✅ All models saved to", MODELS_DIR)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
