from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

def clean_df(df: pd.DataFrame,target:str, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Cleans the input dataframe according to cleaning dict specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    target: str
        target column
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = _remove_rows(df, target,cleaning)
    df = _log_transform(df, target,cleaning)
    return df


def _remove_rows(df: pd.DataFrame, target:str, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Removes some rows of the input dataframe according to cleaning dict specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    target: str
        target column
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df_clean = df.copy()  # To avoid CachedObjectMutationWarning
    df_clean["__to_remove"] = 0
    if cleaning["del_negative"]:
        df_clean["__to_remove"] = np.where(df_clean[target] < 0, 1, df_clean["__to_remove"])
    if cleaning["del_days"] is not None:
        df_clean["__to_remove"] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning["del_days"]), 1, df_clean["__to_remove"]
        )
    if cleaning["del_zeros"]:
        df_clean["__to_remove"] = np.where(df_clean[target] == 0, 1, df_clean["__to_remove"])
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean



def _log_transform(df: pd.DataFrame,target:str, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Applies a log transform to the y column of input dataframe, if possible.
    Raises an error in streamlit dashboard if not possible.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    target: str
        target column 
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df_clean = df.copy()  # To avoid CachedObjectMutationWarning
    if cleaning["log_transform"]:
        if df_clean.y.min() <= 0:
            raise ValueError("The target has values <= 0. Please remove negative and 0 values when applying log transform.")
        else:
            df_clean["y"] = np.log(df_clean["y"])
    return df_clean

