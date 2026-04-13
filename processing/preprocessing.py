import pandas as pd

def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    grouped_df = df.groupby(fields)
    accuracy_column