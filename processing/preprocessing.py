import pandas as pd
from app.config import TARGET_VARIABLE
def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    grouped_df = df.groupby(fields).agg({TARGET_VARIABLE: "mean"})
    return grouped_df

def add_shot_main_action_type_column(df: pd.DataFrame):
    def main_category(val):
        keywords = ['Dunk', 'Layup', 'Hook', 'Jump']
        for keyword in keywords:
            if keyword in val:
                return keyword
        return 'Other'
        # if 'Dunk'
    df['Main Action Type'] = df['Action Type'].apply(main_category)
    return df