import numpy as np
import pandas as pd
import math

def get_train_test_data(data, ratios, seed=42, shuffle=False,num=2):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from `this source <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.

    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.

    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if num not in [2, 3]:
        raise ValueError("num must be either 2 or 3")

    if math.fsum(ratios) != 1.0:
        raise ValueError("The ratios have to sum to 1")

    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # Add split index (this makes splitting by group more efficient).
    # for i in range(len(ratios)):
    #     splits[i]["split_index"] = i
    if num == 2:
        return splits[0],splits[1]
    if num == 3:
        return splits[0],splits[1],splits[2]

def get_train_test_by_date_ratio(data, ratios, date_column, seed=42, shuffle=False, num=2):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from `this source <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.

    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        date_column (str): Name of the column containing dates.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.

    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if num not in [2, 3]:
        raise ValueError("num must be either 2 or 3")

    if math.isclose(sum(ratios), 1.0) == False:
        raise ValueError("The ratios have to sum to 1")

    # Sort DataFrame by the date column
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)

    # Calculate split indices
    split_indices = [round(r * len(data)) for r in np.cumsum(ratios[:-1])]

    # Split the data based on the calculated indices
    splits = np.split(data, split_indices)

    if num == 2:
        return splits[0], splits[1]
    if num == 3:
        return splits[0], splits[1], splits[2]


def get_train_test_by_date(data, date_column, split_date):
    """Split pandas DataFrame into two based on a date condition.

    Args:
        data (pandas.DataFrame): DataFrame to be split.
        date_column (str): Name of the column containing dates.
        split_date (str): Date string in format 'YYYY-MM-DD'. 
                          Data with dates less than or equal to this date will be in the first DataFrame.

    Returns:
        tuple: Two pandas DataFrames split by the given date.
    """
    # Convert string date to datetime
    split_date = pd.to_datetime(split_date)

    # Split the DataFrame based on the date condition
    lower_dates_df = data[data[date_column] <= split_date]
    greater_dates_df = data[data[date_column] > split_date]

    return lower_dates_df, greater_dates_df

def get_train_test_by_days(data, date_column, num_days):
    """Split pandas DataFrame into two based on a fixed number of days for the test set.

    Args:
        data (pandas.DataFrame): DataFrame to be split.
        date_column (str): Name of the column containing dates.
        num_days (int): Number of days for the test set starting from the end.

    Returns:
        tuple: Two pandas DataFrames split by the given number of days.
    """
    # Convert date column to datetime
    data[date_column] = pd.to_datetime(data[date_column])

    # Sort DataFrame by the date column
    data = data.sort_values(by=date_column)

    # Determine the cutoff date for the test set
    cutoff_date = data[date_column].max() - pd.Timedelta(days=num_days)

    # Split the DataFrame based on the cutoff date
    train_df = data[data[date_column] <= cutoff_date]
    test_df = data[data[date_column] > cutoff_date]

    return train_df, test_df

def get_df_n_days(df_input: pd.DataFrame) -> int:
    """
    Helper function to get Test's dataframe numbers of date

    Parameters
    ----------
    df_input : pd.DataFrame
        Input test dataframe 


    Returns
    -------
    int
        Number of days in Test Dataframe
    """
    df = df_input.copy()
    n_days = df.shape[0]
    return n_days