import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_time_series(
    df_input: pd.DataFrame,
    time_column: str,
    value_columns: list,
    as_date: bool = True,
    title: str = 'Time Series Plot',
    x_label: str = 'Date',
    y_label: str = 'Value',
    figsize: tuple = (14, 7)
) -> None:
    """Plots a time series line plot for one or more value columns in a DataFrame using seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    time_column : str
        Name of the column containing the time series data.
    value_columns : list
        List of column names to plot on the Y axis.
    as_date : bool, optional
        Whether to treat the time_column as dates or numerical values, by default True.
    title : str, optional
        Title of the plot, by default 'Time Series Plot'.
    x_label : str, optional
        Label for the X axis, by default 'Date'.
    y_label : str, optional
        Label for the Y axis, by default 'Value'.
    figsize : tuple, optional
        Size of the figure, by default (14, 7).

    Returns
    -------
    None
        This function does not return anything. It plots the time series.
    """
    dfp = df_input.copy() 
    # Optionally convert the time column to datetime if needed
    if as_date:
        dfp[time_column] = pd.to_datetime(dfp[time_column])
    
    df_grouped = dfp.groupby(time_column).sum().reset_index()

    # Set the figure size
    plt.figure(figsize=figsize)
    
    # Plot each column using seaborn
    for col in value_columns:
        plt.plot(df_grouped[time_column], df_grouped[col], label=col)
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(x_label if as_date else 'Time (Numerical)')
    plt.ylabel(y_label)
    
    # Show legend
    plt.legend()
    
    # Show the plot
    plt.show()

# Example usage
# Assuming 'df' is your DataFrame and it has columns 'Date', 'Confirmed', 'Deaths', 'Recovered'
# plot_time_series(df, 'Date', ['Confirmed', 'Deaths', 'Recovered'])


def plot_numeric_correlation_heatmap(
    df_input: pd.DataFrame,
    title: str = 'Correlation Heatmap',
    figsize: tuple = (10, 8)
) -> None:
    """Calculates the correlation matrix for numeric columns and plots it as a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    title : str, optional
        Title of the heatmap, by default 'Correlation Heatmap'.
    figsize : tuple, optional
        Size of the figure, by default (10, 8).

    Returns
    -------
    None
        This function does not return anything. It plots the correlation heatmap.
    """


    dfp = df_input.copy() 
    # Filter out the numeric columns
    numeric_df = dfp.select_dtypes(include=[float, int])
    
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Set the figure size
    plt.figure(figsize=figsize)
    
    # Plot the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f')
    
    # Set the title
    plt.title(title)
    
    # Show the plot
    plt.show()

# Example usage
# Assuming 'df' is your DataFrame
# plot_numeric_correlation_heatmap(df)



def plot_decompose_time_series(
    df_input: pd.DataFrame,
    time_column: str,
    value_column: str,
    model: str = 'additive',
    freq: int = 30,
    title: str = 'Decomposition of Time Series',
    figsize: tuple = (16, 7)
) -> None:
    """Decomposes a time series into trend, seasonal, and residual components.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    time_column : str
        Name of the column containing the time series data.
    value_column : str
        Name of the column to decompose.
    model : str, optional
        Type of seasonal component ('additive' or 'multiplicative'), by default 'additive'.
    freq : int, optional
        Frequency of the time series, by default 30.
    title : str, optional
        Title of the decomposition plot, by default 'Decomposition of Time Series'.

    Returns
    -------
    None
        This function does not return anything. It plots the decomposed components.
    """

    dfp = df_input.copy() 
    # Ensure the dataframe is sorted by time column
    dfp.sort_values(by=time_column, inplace=True)
    
    # Aggregate data by the specified time column and value column
    daily_cases = dfp.groupby(time_column)[value_column].sum()

    # Decompose the time series
    result = seasonal_decompose(daily_cases, model=model, period=freq)
    
    # Plot the decomposition
    plt.figure(figsize=figsize)
    result.plot()
    plt.suptitle(title, y=0.95)  # Adjust the y position to avoid overlap
    plt.show()


# Example usage
# Assuming 'df' is your DataFrame and has columns 'Date' and 'Confirmed'
# decompose_time_series(df, 'Date', 'Confirmed', model='additive', freq=30)



def plot_autocorrelation(
    df_input: pd.DataFrame,
    time_column: str,
    value_column: str,
    title: str = 'Autocorrelation Plot',
    figsize: tuple = (14, 7)
) -> None:
    """Plots the autocorrelation of a time series column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    time_column : str
        Name of the column containing the time series data.
    value_column : str
        Name of the column to plot for autocorrelation.
    title : str, optional
        Title of the plot, by default 'Autocorrelation Plot'.
    figsize : tuple, optional
        Size of the figure, by default (14, 7).

    Returns
    -------
    None
        This function does not return anything. It plots the autocorrelation.
    """
    df_input[time_column] = pd.to_datetime(df_input[time_column])
    
    # Set the figure size
    plt.figure(figsize=figsize)
    
    # Plot the autocorrelation
    autocorrelation_plot(df_input.set_index(time_column)[value_column])
    
    # Set the title
    plt.title(title)
    
    # Show the plot
    plt.show()

# Example usage
# plot_autocorrelation(df, 'Date', 'Confirmed')


def plot_moving_std(
    df_input: pd.DataFrame,
    time_column: str,
    value_column: str,
    window: int = 12,
    title: str = 'Moving Standard Deviation Plot',
    x_label: str = 'Date',
    y_label: str = 'Standard Deviation',
    figsize: tuple = (14, 7)
) -> None:
    """Plots the moving standard deviation of a time series column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    time_column : str
        Name of the column containing the time series data.
    value_column : str
        Name of the column to plot for moving standard deviation.
    window : int, optional
        Window size for moving standard deviation, by default 12.
    title : str, optional
        Title of the plot, by default 'Moving Standard Deviation Plot'.
    x_label : str, optional
        Label for the X axis, by default 'Date'.
    y_label : str, optional
        Label for the Y axis, by default 'Standard Deviation'.
    figsize : tuple, optional
        Size of the figure, by default (14, 7).

    Returns
    -------
    None
        This function does not return anything. It plots the moving standard deviation.
    """
    dfp = df_input.copy() 
    dfp.set_index(time_column, inplace=True)
    
    # Calculate moving standard deviation
    moving_std = dfp[value_column].rolling(window=window).std()
    
    # Set the figure size
    plt.figure(figsize=figsize)
    
    # Plot the moving standard deviation
    plt.plot(moving_std)
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Show the plot
    plt.show()

# Example usage
# plot_moving_std(df, 'Date', 'Confirmed', window=12)

def plot_boxplot(df, category_col, value_col, title='Box Plot', x_label=None, y_label=None, figsize=(14, 7)):
    """
    Plots a box plot for a categorical column against a value column in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - category_col: string name of the categorical column
    - value_col: string name of the column with values to plot
    - title: string, the title of the plot
    - x_label: string, the label for the X axis (optional)
    - y_label: string, the label for the Y axis (optional)
    - figsize: tuple, the size of the figure
    """
    
    # Set default labels if not provided
    if x_label is None:
        x_label = category_col
    if y_label is None:
        y_label = value_col
    
    # Set the figure size
    plt.figure(figsize=figsize)
    
    # Create the box plot
    sns.boxplot(data=df, x=category_col, y=value_col)
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Show the plot
    plt.show()