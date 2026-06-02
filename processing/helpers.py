import pandas as pd

from app.config import TARGET_VARIABLE


def shot_accuracy_by_fields(df: pd.DataFrame, fields):
    """
    Calculates the mean Target for a groupby of given fields
    An abstraction to provide a consistent
    :param df: DataFrame with both target and explanatory variables
    :param fields: list of strings indicating the fields to group by for calculating the mean target
    :return: A dataFrame with a (combined) index consisting of the given fields and a mean result for the target
    """
    grouped_df = df.groupby(fields).agg({TARGET_VARIABLE: "mean"})
    return grouped_df

def combine_actual_and_prediction_dataframe(y_test, y_proba):
    """
    Function to provide a consistent way of combining actual target and predicted probabilities
    :param y_test: Actual target series
    :param y_proba: Output of predict_proba in sklearn compatible format
    :return: a merged DataFrame
    """
    return pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_proba[:, 1]
    }, index=y_test.index)

def combine_result_and_x_orig_dataframe(df_x_orig, df_result: pd.DataFrame):
    """
    Function to provide a level of abstraction for the join in case more than just a join was necessary.
    Used everywhere I want to combine a result DataFrame (see combine_actual_and_prediction_dataframe)
    with original explanatory data
    :param df_x_orig: The original explanatory data
    :param df_result: The result data frame (e.g. obtained by combine_actual_and_prediction_dataframe)
    :return: A joint DataFrame
    """
    return df_result.join(df_x_orig)

def filter_false_negatives(df_combined):
    """
    Convenience function to filter only the false negatives of a given df
    :param df_combined: DataFrame containing the result DataFrame made with combine_actual_and_prediction_dataframe
    :return: filtered DataFrame
    """
    negative_predictions_mask = df_combined['Predicted'] < 0.5
    positive_outcome_mask = df_combined['Actual'] == 1
    return df_combined[ negative_predictions_mask & positive_outcome_mask ]

def filter_false_positives(df_combined):
    """
    Convenience function to filter only the false positives of a given df
    :param df_combined: DataFrame containing the result DataFrame made with combine_actual_and_prediction_dataframe
    :return: filtered DataFrame
    """
    positive_predictions_mask = df_combined['Predicted'] > 0.5
    negative_outcome_mask = df_combined['Actual'] == 0
    return df_combined[ positive_predictions_mask & negative_outcome_mask ]