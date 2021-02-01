# Given a document term matrix, we want to work out how to pick up the terms with higher
# frequency for long term and for recent increases in frequency

# %% Package Imports

import pandas as pd


# %% Pick out the top N columns for overall frequency

def get_top_mentions_all_time(document_term_matrix:pd.DataFrame, N:int, date_col_name:str):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): dataframe with vocab frequency as cols
        N (int): [description]
        date_col_name (str): [description]

    Returns:
        [type]: [description]
    """

    document_term_matrix.set_index(date_col_name, inplace=True)
    vocab_cols = document_term_matrix.iloc[:,document_term_matrix.columns.get_loc('documents')+1:]

    # Get a sum for each col and sort for top N
    top_mentions = vocab_cols.sum(axis=0).sort_values(ascending=False)[:N]

    top_mention_cols = list(top_mentions.index)

    info_cols = list(document_term_matrix.iloc[:,:document_term_matrix.columns.get_loc('documents')+1].columns)

    document_term_matrix_top_mentions = document_term_matrix[info_cols+top_mention_cols]

    return document_term_matrix_top_mentions


# %% Pick out the top N columns for highest frequency over last 30 days

# //// I should actually be setting the datetime as index outside of this function if using inplace=True
# Depends entirely on the date used, pass as arg

def get_top_mentions_last_T_days(document_term_matrix:pd.DataFrame, N:int, T:int, date_col_name:str):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): dataframe with vocab frequency as cols
        N (int): [description]
        T (int): [description]
        date_col_name (str): [description]

    Returns:
        [type]: [description]
    """

    # Set Date as index and filter
    document_term_matrix.set_index(date_col_name, inplace=True)
    latest = document_term_matrix[document_term_matrix.last_valid_index()-pd.DateOffset(T, 'D'):]

    # We make an assumption that the date is read in as datetime

    vocab_cols = latest.iloc[:,latest.columns.get_loc('documents')+1:]

    top_mentions_latest = vocab_cols.sum(axis=0).sort_values(ascending=False)[:N]
    top_mention_cols = list(top_mentions_latest.index)
    info_cols = list(document_term_matrix.iloc[:,:document_term_matrix.columns.get_loc('documents')+1].columns)

    document_term_matrix_top_mentions = document_term_matrix[info_cols+top_mention_cols]

    return document_term_matrix_top_mentions


# %% Pick out the top N columns for increase in frequency?

# This is a little trickier
# Lets try looking at the average mentions before and after the last 3 months and take a ratio
# of which we'l ltake the top N. We should probably impose a min absolute value and obviously avoid div0

def get_increased_mentions(document_term_matrix:pd.DataFrame, N:int, T:int, date_col_name:str):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): dataframe with vocab frequency as cols
        N (int): [description]
        T (int): [description]
        date_col_name (str): [description]

    Returns:
        [type]: [description]
    """

    # Set datetime index
    document_term_matrix.set_index(date_col_name, inplace=True)
    latest = document_term_matrix[document_term_matrix.last_valid_index()-pd.DateOffset(T, 'M'):]
    historical = document_term_matrix[:document_term_matrix.last_valid_index()-pd.DateOffset(T, 'M')]

    info_cols = list(document_term_matrix.iloc[:,:document_term_matrix.columns.get_loc('documents')+1].columns)
    vocab_cols = list(document_term_matrix.iloc[:,document_term_matrix.columns.get_loc('documents')+1:].columns)

    # Average monthly mentions in the historical?
    hist_monthly = historical[vocab_cols].resample('M').sum().mean()
    latest_monthly = latest[vocab_cols].resample('M').sum().mean()

    hist_monthly.rename('historical', inplace=True)
    latest_monthly.rename('latest', inplace=True)

    comparison = pd.DataFrame(hist_monthly).merge(latest_monthly, left_index=True, right_index=True)

    # Filter here for latest and historical

    comparison['relative_increase'] = comparison['latest'] / comparison['historial']

    increased_mentions = list(comparison['relative_increase'].sort_values(ascending=False)[:N].index)

    return document_term_matrix[info_cols+increased_mentions]


# %% We'll likely want to look at a list of reasonably frequent occurrences, especially if seasonal
# so we'll need to maintain a list of terms somewhere to keep as standard

def get_standard_terms(document_term_matrix:pd.DataFrame, date_col_name:str, standard_cols:list):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): dataframe with vocab frequency as cols
        date_col_name (str): [description]
        standard_cols (list): [description]

    Returns:
        [type]: [description]
    """

    # Set datetime index
    document_term_matrix.set_index(date_col_name, inplace=True)
    info_cols = list(document_term_matrix.iloc[:,:document_term_matrix.columns.get_loc('documents')+1].columns)

    return document_term_matrix[info_cols+standard_cols]

