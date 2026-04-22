import pandas as pd
from app.config import TARGET_VARIABLE
def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    grouped_df = df.groupby(fields).agg({TARGET_VARIABLE: "mean"})
    return grouped_df


