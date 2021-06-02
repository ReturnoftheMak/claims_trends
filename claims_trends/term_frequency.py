# Given a document term matrix, we want to work out how to pick up the terms with higher
# frequency for long term and for recent increases in frequency

# %% Package Imports

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer


# %% Pick out the top N columns for overall frequency

def get_top_mentions_all_time(document_term_matrix:pd.DataFrame, N:int, date_col_name:str, additional_groups:list=[]):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): vectorised dataframe after preprocessing
        N (int): Number of terms to return
        date_col_name (str): name of the date column to use
        additional_groups (list, optional): Additional columns to use for group by aggregation. Defaults to [].

    Returns:
        tuple: 2 x pd.DataFrame, one for time series, the other for cross filtering.
    """

    document_term_matrix_top_mentions = document_term_matrix
    vocab_cols = document_term_matrix_top_mentions.iloc[:,document_term_matrix_top_mentions.columns.get_loc('documents')+1:]

    # Get a sum for each col and sort for top N
    top_mentions = pd.to_numeric(vocab_cols.sum(axis=0), errors='coerce').fillna(0).sort_values(ascending=False)[:N]

    top_mention_cols = list(top_mentions.index)

    info_cols = list(document_term_matrix_top_mentions.iloc[:,:document_term_matrix_top_mentions.columns.get_loc('documents')+1].columns)

    reduced = document_term_matrix_top_mentions[info_cols+top_mention_cols]

    flattened = pd.melt(reduced, id_vars=info_cols, var_name='term', value_name='mentions')
    flattened['month'] = flattened['date'].dt.to_period('M')

    group = ['month', 'term', 'class'] + additional_groups

    grouped = flattened.groupby(['month', 'class', 'term'], as_index=False).sum()
    grouped_add = flattened.groupby(group, as_index=False).sum()

    # turn the period back to date format
    grouped['date'] = grouped.month.dt.to_timestamp('d')
    grouped_add['date'] = grouped_add.month.dt.to_timestamp('d')

    return grouped, grouped_add


# %% Pick out the top N columns for highest frequency over last 30 days

# //// I should actually be setting the datetime as index outside of this function if using inplace=True
# Depends entirely on the date used, pass as arg

def get_top_mentions_last_T_days(document_term_matrix:pd.DataFrame, N:int, T:int, date_col_name:str, additional_groups:list=[]):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): vectorised dataframe after preprocessing
        N (int): Number of terms to return
        T (int): Number of days to calculate over
        date_col_name (str): name of the date column to use
        additional_groups (list, optional): Additional columns to use for group by aggregation. Defaults to [].

    Returns:
        tuple: 2 x pd.DataFrame, one for time series, the other for cross filtering.
    """

    # Set Date as index and filter
    document_term_matrix_top_mentions = document_term_matrix.set_index(date_col_name)
    latest = document_term_matrix_top_mentions[document_term_matrix_top_mentions.last_valid_index()-pd.DateOffset(T, 'D'):]

    # We make an assumption that the date is read in as datetime

    vocab_cols = latest.iloc[:,latest.columns.get_loc('documents')+1:]

    top_mentions_latest = vocab_cols.sum(axis=0).sort_values(ascending=False)[:N]
    top_mention_cols = list(top_mentions_latest.index)
    info_cols = list(document_term_matrix_top_mentions.iloc[:,:document_term_matrix_top_mentions.columns.get_loc('documents')+1].columns)

    reduced = document_term_matrix_top_mentions[info_cols+top_mention_cols]

    flattened = pd.melt(reduced, id_vars=info_cols, var_name='term', value_name='mentions')
    flattened['month'] = flattened['date'].dt.to_period('M')

    group = ['month', 'term'] + additional_groups

    grouped = flattened.groupby(['month', 'term'], as_index=False).sum()
    grouped_add = flattened.groupby(group, as_index=False).sum()

    # turn the period back to date format
    grouped['date'] = grouped.month.dt.to_timestamp('d')
    grouped_add['date'] = grouped_add.month.dt.to_timestamp('d')

    return grouped, grouped_add


# %% Pick out the top N columns for increase in frequency?

# This is a little trickier
# Lets try looking at the average mentions before and after the last 3 months and take a ratio
# of which we'l ltake the top N. We should probably impose a min absolute value and obviously avoid div0

def get_increased_mentions(document_term_matrix:pd.DataFrame, N:int, T:int, date_col_name:str, additional_groups:list=[]):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): vectorised dataframe after preprocessing
        N (int): Number of terms to return
        T (int): Number of days to calculate over
        date_col_name (str): name of the date column to use
        additional_groups (list, optional): Additional columns to use for group by aggregation. Defaults to [].

    Returns:
        tuple: 2 x pd.DataFrame, one for time series, the other for cross filtering.
    """

    # Set datetime index
    document_term_matrix_im = document_term_matrix.set_index(date_col_name)
    latest = document_term_matrix_im[document_term_matrix_im.last_valid_index()-pd.DateOffset(T, 'M'):]
    historical = document_term_matrix_im[:document_term_matrix_im.last_valid_index()-pd.DateOffset(T, 'M')]

    info_cols = list(document_term_matrix_im.iloc[:,:document_term_matrix_im.columns.get_loc('documents')+1].columns)
    vocab_cols = list(document_term_matrix_im.iloc[:,document_term_matrix_im.columns.get_loc('documents')+1:].columns)

    # Average monthly mentions in the historical?
    hist_monthly = historical[vocab_cols].resample('M').sum().mean()
    latest_monthly = latest[vocab_cols].resample('M').sum().mean()

    hist_monthly.rename('historical', inplace=True)
    latest_monthly.rename('latest', inplace=True)

    comparison = pd.DataFrame(hist_monthly).merge(latest_monthly, left_index=True, right_index=True)

    # Filter here for latest and historical

    comparison['relative_increase'] = comparison['latest'] / comparison['historial']

    increased_mentions = list(comparison['relative_increase'].sort_values(ascending=False)[:N].index)

    reduced = document_term_matrix_im[info_cols+increased_mentions]

    flattened = pd.melt(reduced, id_vars=info_cols, var_name='term', value_name='mentions')
    flattened['month'] = flattened['date'].dt.to_period('M')

    group = ['month', 'term'] + additional_groups

    grouped = flattened.groupby(['month', 'term'], as_index=False).sum()
    grouped_add = flattened.groupby(group, as_index=False).sum()

    # turn the period back to date format
    grouped['date'] = grouped.month.dt.to_timestamp('d')
    grouped_add['date'] = grouped_add.month.dt.to_timestamp('d')

    return grouped, grouped_add


# %% We'll likely want to look at a list of reasonably frequent occurrences, especially if seasonal
# so we'll need to maintain a list of terms somewhere to keep as standard

def get_standard_terms(document_term_matrix:pd.DataFrame, date_col_name:str, standard_cols:list, additional_groups:list=[]):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): vectorised dataframe after preprocessing
        date_col_name (str): name of the date column to use
        standard_cols (list): input list for terms to use
        additional_groups (list, optional): Additional columns to use for group by aggregation. Defaults to [].

    Returns:
        tuple: 2 x pd.DataFrame, one for time series, the other for cross filtering.
    """

    # Set datetime index
    document_term_matrix_sm = document_term_matrix
    info_cols = list(document_term_matrix_sm.iloc[:,:document_term_matrix_sm.columns.get_loc('documents')+1].columns)

    # We may want to standardise the entered list first?
    # Use WordNetLemmatizer
    stemmer = PorterStemmer()
    stemmed_cols = set([stemmer.stem(word) for word in standard_cols])

    cols_to_use = [col for col in stemmed_cols if col in document_term_matrix_sm.columns]

    reduced = document_term_matrix_sm[info_cols+cols_to_use]

    flattened = pd.melt(reduced, id_vars=info_cols, var_name='term', value_name='mentions')
    flattened['month'] = flattened['date'].dt.to_period('M')

    group = ['month', 'term'] + additional_groups

    grouped = flattened.groupby(['month', 'term', 'claim_id'], as_index=False).sum()
    grouped_add = flattened.groupby(group, as_index=False).sum()

    # turn the period back to date format
    grouped['date'] = grouped.month.dt.to_timestamp('d')
    grouped_add['date'] = grouped_add.month.dt.to_timestamp('d')

    return grouped, grouped_add

