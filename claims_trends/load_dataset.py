# %% Package Imports

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import scikit_vectorizer, LemmaTokenizer, document_term_matrix, get_stopwords
from term_frequency import get_top_mentions_all_time

# %% Get recombined df, including all classes

def load_data():
    """Gets the claims data with required fields

    Returns:
        pd.DataFrame: [description]
    """

    df_scm = pd.read_csv(r'C:\Users\makhan.gill\SQL_DATA\LMM_SCM_all.csv',
                        usecols=['ClaimDetailID', 'LossDescription', 'ExtendedLossDetails', 'LossLocation', 'OutstandingLoss', 'TotalLossPaid'])
    df_cgen = pd.read_csv(r'C:\Users\makhan.gill\SQL_DATA\claim_data_general_all.csv',
                        usecols=['ClaimDetailID', 'ClaimAdvisedDate', 'HandlingClass'])
    df = df_scm.merge(df_cgen, how="left", on='ClaimDetailID')

    document_list = ['LossDescription', 'ExtendedLossDetails']
    df['documents'] = df[document_list].apply(lambda x: ' '.join(x.dropna().astype(str)).lower(), axis=1)
    df['incurred_loss'] = df['OutstandingLoss'] + df['TotalLossPaid']

    df = df[['ClaimAdvisedDate', 'LossLocation', 'ClaimDetailID', 'HandlingClass', 'documents', 'incurred_loss']]
    df.columns = ['date', 'loss_location', 'claim_id', 'class', 'documents', 'incurred_loss']
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

    # Turn this into a groupby
    df = df.groupby(['date', 'loss_location', 'claim_id', 'class', 'documents'], as_index=False).sum()

    return df


def get_cob_vectorised_df(class_df:pd.DataFrame):
    """[summary]

    Args:
        class_df (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    stop_words_combined = get_stopwords()
    vectorizer = scikit_vectorizer(stop_words_combined, LemmaTokenizer, CountVectorizer)

    doc_tm = document_term_matrix(class_df, vectorizer)

    return doc_tm


def get_class_mentions(loaded_df:pd.DataFrame, N:int, date_col_name:str, additional_groups:list=[], cobs:list=[]):
    """[summary]

    Args:
        loaded_df (pd.DataFrame): [description]
        N (int): [description]
        date_col_name (str): [description]
        additional_groups (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """

    class_mentions = {}

    if len(cobs) < 1:
        cobs = loaded_df['class'].unique()

    for cob in cobs:
        # // print(cob)
        vectorised_df = get_cob_vectorised_df(loaded_df[loaded_df['class'] == cob])
        class_mentions[cob] = get_top_mentions_all_time(vectorised_df, N, date_col_name, additional_groups)

    # // df_combined_0 = pd.concat([class_mentions[key][0] for key in class_mentions.keys()])
    df_combined_1 = pd.concat([class_mentions[key][1] for key in class_mentions.keys()])

    # // df_dict_combined = {'base': df_combined_0, 'add_cols':df_combined_1}

    return df_combined_1


# ? df_combined =  get_class_mentions(load_data(), 30, 'date', additional_groups=['loss_location'],
# ? cobs=['Accident & Health', 'Offshore Energy', 'Aviation'])
# %%
