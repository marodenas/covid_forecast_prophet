from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd 

def format_date(
    df_input: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Formats date and target columns of input dataframe.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe whose columns will be formatted.
    date_col : str
        Name of date column in input dataframe.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns formatted.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _format_date(df, date_col)
    return df

def format_input_and_target(
    df_input: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Formats date and target columns of input dataframe.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe whose columns will be formatted.
    date_col : str
        Name of date column in input dataframe.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns formatted.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    # df = _format_date(df, date_col)
    df = _format_target(df, target_col)
    df = _rename_cols(df, date_col, target_col)
    return df



def _rename_cols(df_input: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Renames date and target columns of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be renamed.
    date_col : str
        Name of date column in input dataframe.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns renamed.
    """
    df = df_input.copy() 
    if (target_col != "y") and ("y" in df.columns):
        df = df.rename(columns={"y": "y_2"})
    if (date_col != "ds") and ("ds" in df.columns):
        df = df.rename(columns={"ds": "ds_2"})
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    return df

def _format_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Formats date column of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be formatted.
    date_col : str
        Name of date column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with date column formatted.
    """
    try:
        # Convert the date column to datetime
        df[date_col] = pd.to_datetime(df[date_col],format='%Y-%m-%d')
        return df
    except ValueError:
        print("Error: The date column could not be converted. Make sure it contains valid dates.")


def _format_target(df_input: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Formats target column of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be formatted.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with date column formatted.
    """
    try:
        df = df_input.copy() 
        # Convertir la columna de destino a tipo de dato float
        df[target_col] = df[target_col].astype("float")
        
        # Verificar si el número de valores únicos en la columna de destino es menor que el umbral especificado
        if df[target_col].nunique() < 2:
            raise ValueError("Please select the correct target column (should be numerical, not categorical).")
        
        # Retornar el DataFrame modificado
        return df
    except ValueError as e:
        # Manejar el error si no se puede realizar la conversión o no se cumple la condición de cardinalidad mínima
        raise ValueError("Error al formatear la columna de destino:", e)


def group_by_columns_and_sum(df_input, group_cols, sum_cols):
    """
    Group by specified columns and sum the rest of the columns.

    Args:
    df_input (pd.DataFrame): The DataFrame to be grouped and summed.
    group_cols (list of str): The column names to group by.
    sum_cols (list of str): The column names to be summed.

    Returns:
    pd.DataFrame: The resulting DataFrame after grouping and summing.
    """
    df = df_input.copy() 
    # Validate column names
    invalid_cols = set(group_cols + sum_cols) - set(df.columns)
    if invalid_cols:
        raise ValueError(f"Invalid column names: {invalid_cols}")

    # Group by specified columns and sum the rest
    grouped = df.groupby(group_cols)[sum_cols].sum().reset_index()

    return grouped