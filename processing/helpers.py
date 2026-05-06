import pandas as pd

from app.config import TARGET_VARIABLE


def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    grouped_df = df.groupby(fields).agg({TARGET_VARIABLE: "mean"})
    return grouped_df

def combine_actual_and_prediction_dataframe(y_test, y_proba):
    return pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_proba[:, 1]
    })

def combine_result_and_x_orig_dataframe(df_x_orig, df_result):
    df_x_orig = df_x_orig.reset_index(drop=True)
    return df_result.join(df_x_orig)