# Given a document term matrix, we want to work out how to pick up the terms with higher
# frequency for long term and for recent increases in frequency

# %% Package Imports

import pandas as pd


# %% Pick out the top N columns for overall frequency

def get_top_mentions_all_time(document_term_matrix:pd.DataFrame, N:int, date_col_name:str):
    """[summary]

    Args:
        document_term_matrix (pd.DataFrame): [description]
        N (int): [description]
        date_col_name (str): [description]

    Returns:
        [type]: [description]
    """

    vocab_cols = document_term_matrix.iloc[:,document_term_matrix.columns.get_loc('documents')+1:]

    # Get a sum for each col and sort for top N
    top_mentions = vocab_cols.sum(axis=0).sort_values(ascending=False)[:N]

    top_mention_cols = list(top_mentions.index)

    info_cols = list(document_term_matrix.iloc[:,:document_term_matrix.columns.get_loc('documents')+1].columns)

    document_term_matrix_top_mentions = document_term_matrix[info_cols+top_mention_cols]

    return document_term_matrix_top_mentions



# %% Pick out the top N columns for highest frequency over last 30 days

def get_top_mentions_last_T_days(document_term_matrix:pd.DataFrame, N:int, T:int):

    # Set Date as index and filter
    document_term_matrix.set_index('date', inplace=True)
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
