# %% Package Imports

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import scikit_vectorizer, LemmaTokenizer, document_term_matrix, get_stopwords
from term_frequency import get_standard_terms


# %% Params

key_words = ['concussion', 'cranial', 'brain',
             'head', 'headache', 'migraine', 'dizzy',
             'dizziness', 'fatigue', 'insomnia']


# %%

df_scm = pd.read_csv(r'C:\Users\makhan.gill\SQL_DATA\LMM_SCM_all.csv',
                     usecols=['ClaimDetailID', 'LossDescription', 'LossLocation'])
df_cgen = pd.read_csv(r'C:\Users\makhan.gill\SQL_DATA\claim_data_general_all.csv',
                      usecols=['ClaimDetailID', 'ClaimAdvisedDate', 'HandlingClass'])
df = df_scm.merge(df_cgen, how="left", on='ClaimDetailID')

df.columns = ['claim_id', 'documents', 'loss_location', 'class', 'date']
df = df[['date', 'loss_location', 'claim_id', 'class', 'documents']]
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

stop_words_combined = get_stopwords()
vectorizer = scikit_vectorizer(stop_words_combined, LemmaTokenizer, CountVectorizer)

# Liability only?

df_liab = df[df['class'].isin(['Non Marine Liability', 'Marine Energy Liability', 'iBott General Liability'])]

# Chunk it?

grouped_list = []

n = 10000

list_df = [df_liab[i:i+n] for i in range(0,len(df_liab),n)]

for df_chunk in list_df:
    doc_tm = document_term_matrix(df_chunk, vectorizer)
    g1, g2 =get_standard_terms(doc_tm, 'date', key_words)
    grouped_list.append(g1)


concussion_terms = pd.concat(grouped_list)

