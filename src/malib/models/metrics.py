# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss,
    mean_absolute_percentage_error


)
DEFAULT_USER_COL = ""
DEFAULT_ITEM_COL = ""
DEFAULT_RATING_COL = ""
DEFAULT_PREDICTION_COL = ""


def general_evaluation(
    rating_true,
    rating_pred,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Compare all evaluation metrics and return a DataFrame with the results.

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        pandas.DataFrame: DataFrame with comparison of evaluation metrics
    """
    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(rating_true[col_rating], rating_pred[col_prediction]))
    mae = mean_absolute_error(rating_true[col_rating], rating_pred[col_prediction])

    # Calculate the mean of the true values for normalization
    mean_true = rating_true[col_rating].median()

    # Calculate normalized RMSE and MAE
    normalized_rmse = rmse / mean_true
    normalized_mae = mae / mean_true

    # Calculate other metrics
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mean_absolute_percentage_error(rating_true[col_rating], rating_pred[col_prediction]),
        'sMAPE': 100 * np.mean(2 * np.abs(rating_pred[col_prediction].mean() - rating_true[col_rating].mean()) / 
                               (np.abs(rating_true[col_rating].mean()) + np.abs(rating_pred[col_prediction].mean()))),
        'R-squared': r2_score(rating_true[col_rating], rating_pred[col_prediction]),
        'Explained Variance': explained_variance_score(rating_true[col_rating], rating_pred[col_prediction]),
        'Accuracy': 1 - (np.abs((rating_true[col_rating] - rating_pred[col_prediction]) / rating_true[col_rating])).mean(),
        'Normalized RMSE': normalized_rmse,
        'Normalized MAE': normalized_mae
        # 'AUC': roc_auc_score(rating_true[col_rating], rating_pred[col_prediction]),
        # 'Log Loss': log_loss(rating_true[col_rating], rating_pred[col_prediction])
    }

    return pd.DataFrame(metrics, index=[0])


def daily_evaluation(
    df_true,
    df_pred,
    date_column,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Compare evaluation metrics for each day and return a DataFrame with the results.

    Args:
        df_true (pandas.DataFrame): DataFrame with true data
        df_pred (pandas.DataFrame): DataFrame with predicted data
        date_column (str): column name for date to join the DataFrames
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        pandas.DataFrame: DataFrame with comparison of evaluation metrics for each day
    """
    # Merge the true and predicted DataFrames on the specified date column
    merged_df = pd.merge(df_true, df_pred, on=date_column)

    # Initialize an empty list to store evaluation metrics for each day
    evaluation_results = []

    # Iterate over unique dates and calculate evaluation metrics for each day
    for date in merged_df[date_column].unique():
        # Filter data for the current date
        daily_data = merged_df[merged_df[date_column] == date]
        
        # Calculate evaluation metrics for the current date
        rmse = np.sqrt(mean_squared_error(daily_data[col_rating], daily_data[col_prediction]))
        mae = mean_absolute_error(daily_data[col_rating], daily_data[col_prediction])
        mape = mean_absolute_percentage_error(daily_data[col_rating], daily_data[col_prediction])

        # Calculate normalized RMSE and MAE
        true_value = daily_data[col_rating].to_list()[0]
        normalized_rmse = rmse / true_value if true_value != 0 else np.nan
        normalized_mae = mae / true_value if true_value != 0 else np.nan

        # Calculate accuracy for the current date
        predict = daily_data[col_prediction].to_list()[0]
        accuracy = 1 - abs((daily_data[col_rating].sum() - daily_data[col_prediction].sum()) / daily_data[col_rating].sum())

        # Append evaluation metrics for the current date to the results list
        evaluation_results.append({
            'Date': date,
            'true': true_value,
            'predict': predict,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Normalized RMSE': normalized_rmse,
            'Normalized MAE': normalized_mae,
            'Accuracy': accuracy
        })

    # Create a DataFrame from the list of evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)

    return evaluation_df





def merge_rating_true_pred(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Join truth and prediction data frames on userID and itemID and return the true
    and predicted rated with the correct index.

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        numpy.ndarray: Array with the true ratings
        numpy.ndarray: Array with the predicted ratings

    """

    # pd.merge will apply suffixes to columns which have the same name across both dataframes
    suffixes = ["_true", "_pred"]
    rating_true_pred = pd.merge(
        rating_true, rating_pred, on=[col_user], suffixes=suffixes
    )
    if col_rating in rating_pred.columns:
        col_rating = col_rating + suffixes[0]
    if col_prediction in rating_true.columns:
        col_prediction = col_prediction + suffixes[1]
    return rating_true_pred
    # return rating_true_pred[col_rating], rating_true_pred[col_prediction]

def rmse(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate Root Mean Squared Error

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: Root mean squared error
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate Mean Absolute Error.

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: Mean Absolute Error.
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return mean_absolute_error(y_true, y_pred)



def mape(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return mean_absolute_percentage_error(y_true, y_pred)

def smape(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """
    Calculate symmetric Mean Absolute Percentage Error (sMAPE) between the true and predicted ratings.

    Parameters:
    ----------
    rating_true : pd.DataFrame
        DataFrame containing the true ratings.
    rating_pred : pd.DataFrame
        DataFrame containing the predicted ratings.
    col_user : str, optional
        Column name for user IDs, by default DEFAULT_USER_COL.
    col_item : str, optional
        Column name for item IDs, by default DEFAULT_ITEM_COL.
    col_rating : str, optional
        Column name for true ratings, by default DEFAULT_RATING_COL.
    col_prediction : str, optional
        Column name for predicted ratings, by default DEFAULT_PREDICTION_COL.

    Returns:
    -------
    float
        The symmetric Mean Absolute Percentage Error (sMAPE) between the true and predicted ratings.
    """
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    
    # Calculate sMAPE
    smape_value = (100 / len(y_true)) * sum(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred)))
    return smape_value

def rsquared(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate R squared

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: R squared (min=0, max=1).
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return r2_score(y_true, y_pred)


def exp_var(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate explained variance.

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: Explained variance (min=0, max=1).
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return explained_variance_score(y_true, y_pred)


def auc(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate the Area-Under-Curve metric for implicit feedback typed
    recommender, where rating is binary and prediction is float number ranging
    from 0 to 1.

    https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        The evaluation does not require a leave-one-out scenario.
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: auc_score (min=0, max=1)
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return roc_auc_score(y_true, y_pred)


def logloss(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate the logloss metric for implicit feedback typed
    recommender, where rating is binary and prediction is float number ranging
    from 0 to 1.

    https://en.wikipedia.org/wiki/Loss_functions_for_classification#Cross_entropy_loss_(Log_Loss)

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: log_loss_score (min=-inf, max=inf)
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return log_loss(y_true, y_pred)
