# %% Import Packages

import re
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords


# %% Need a SQL connection

def sql_connection(ServerName, DBName):
    """Returns a SQLAlchemy engine, given the server and database name.

    Params:
        ServerName (str): - Server name
        DBName (str): - Database name

    Returns:
        Object of type (sqlalchemy.engine.base.Engine) for use in pandas pd.to_sql functions
    """

    from sqlalchemy import create_engine

    sqlcon = create_engine('mssql+pyodbc://@' +
                           ServerName +
                           '/' +
                           DBName +
                           '?driver=ODBC+Driver+13+for+SQL+Server',
                           fast_executemany=True)

    return sqlcon


# %% Getting data from the SCM and USM

# I think we'll want to keep a look at the date the claim was made as our time measure,
# although we could just use the time of the message.
# Use the ClaimReference for claim level info, and ClaimDetailID for lowest level

# We'll need to collect a non-repeating set of texts for each claim, then apply some nlp
# and return key terms

# Essentially we want to end up with a sparse matrix with every word and how often it appears per day?
#  We can then track increases over time


# %% Pull in the SCM data

def get_scm_data(sql_con, table_name='T_LMM_SCM'):
    """[summary]

    Args:
        sql_con ([type]): [description]
        table_name (str, optional): [description]. Defaults to 'T_LMM_SCM'.

    Returns:
        [type]: [description]
    """

    df = pd.read_sql_table(sql_con, table_name)

    # Should return the whole table, can filter down once we know what we want

    # Text and ID cols

    return df


def get_usm_data(sql_con, table_name = 'T_LMM_SCM'):
    """[summary]

    Args:
        sql_con ([type]): [description]
        table_name (str, optional): [description]. Defaults to 'T_LMM_SCM'.

    Returns:
        [type]: [description]
    """

    df = pd.read_sql_table(sql_con, table_name)

    # Filter for all of the Lloyd's narrative fields

    return df

# %% Remove the stopwords from a string

# Run both these functions for getting list of lemmatised words from one string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download stopwords list
# //// nltk.download('punkt')
stop_words_en = set(stopwords.words('english'))

def remove_stopwords(text:str, stop_words):
    """
    """

    filtered_words = [word for word in word_tokenize(text) if word not in stop_words]

    return filtered_words


def get_word_stems(word_list:list):
    """
    """

    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

    return lemmatized_words


# It may also be useful to add a different function here which specifically looks at insurance language
# and how these words can be grouped. This could be abbreviations or the like.

def preprocessing(text:str):
    return get_word_stems(remove_stopwords(text, stop_words_en))


# %% Could also use the sklearn tfidf vectoriser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '%', '&', '(', ')', '-', '/', '+', '\\']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens and not bool(re.match(r'\d+', t)) and not bool(re.match(r'usd+', t))]

class PorterTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '%', '&', '(', ')', '-', '/', '+', '\\']
    def __init__(self):
        self.prt = PorterStemmer()
    def __call__(self, doc):
        return [self.prt.stem(t) for t in word_tokenize(doc) if t not in self.ignore_tokens and not bool(re.match(r'\d+', t)) and not bool(re.match(r'usd+', t))]


def scikit_vectorizer(stop_words, LemmaTokenizer, CountVectorizer):
    """[summary]

    Args:
        stop_words ([type]): [description]
        PorterTokenizer ([type]): [description]
        CountVectorizer ([type]): [description]

    Returns:
        [type]: [description]
    """

    tokenizer = LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words))

    vectorizer = CountVectorizer(stop_words=token_stop, tokenizer=tokenizer)

    return vectorizer


vectorizer = scikit_vectorizer(stop_words_en, LemmaTokenizer, CountVectorizer)

# //// corpus = ['first doc', 'the second text']
# Corpus here is the entire set of documents in a list, on string per doc
# //// vectorizer.fit_transform(corpus)


# %% How do we account for multiple docs per record? We could just join all these together?
# This is also a different problem in that we need to get docs for all of a particular date claimed
# plus have the claim ID in there as well

# Thus I think we'll need to come up with our own functions to do this, not rely on scikit

# We still want a sparse document term matrix, but with additional fields for dates and claims ID
# The question may be what fields will actually make up our document, can try just a couple to start,
# then all on a later run?

# %% Clean the document // unused

def clean_document_field(data:pd.DataFrame):
    """[summary]

    Args:
        data (pd.DataFrame): [description]
    """

    data['tokens'] = data['document'].apply(preprocessing)

    #


# %% Return a single document from many provided a list of col headers

def get_document_list(data:pd.DataFrame, document_list:list, key_fields:list):
    """[summary]

    Args:
        data (pd.DataFrame): [description]
        document_list (list): [description]
        key_fields (list): Key fields here are a list of things like date and claims ID

    Returns:
        [type]: [description]
    """

    data['document'] = data[document_list].apply(lambda x: ' '.join(x.dropna().astype(str)).lower(), axis=1)

    return data[key_fields+['document']]


# If we now create a sparse matrix using the document array, we should be able to keep the documents
# matched up to the other fields.

# %% Get a set of cleaned tokens from the document field for a vocabulary

def get_vocabulary(data:pd.DataFrame):
    """[summary]

    Args:
        data (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """

    documents = data['document']

    # Add all text together
    text = documents.apply(lambda x: ' '.join(x.dropna().astype(str)).lower(), axis=0)

    tokens = preprocessing(text)

    vocab = set(tokens)

    return vocab


# %% Test data to check if the CountVectorizer will work for ordered data separated from its key

docs = ['fires destroyed house',
        'flooding in block of flats',
        'flood damage',
        'theft of sum £300']

dates = ['1/1/2021','2/1/2021','3/1/2021','4/1/2021']

claim_ids = [101,102,103,104]

data = {'claim_id':claim_ids,
        'date':dates,
        'documents':docs}

df = pd.DataFrame(data)


# %% Transposing and returning results, again testing works here

X = vectorizer.fit_transform(data['documents'])

df_docterm = pd.DataFrame(X.toarray(),
                          index=data['claim_id'],
                          columns=vectorizer.get_feature_names())


# %% Data manipulation to get to the CountVectorizer result plus additional fields

# Definitely need to look at the fix for merging, shouldn't be duplicating here

def document_term_matrix(data:pd.DataFrame, vectorizer:CountVectorizer):
    """[summary]

    Args:
        data (pd.DataFrame): [description]
        vectorizer (CountVectorizer): [description]

    Returns:
        pd.DataFrame: [description]
    """

    X = vectorizer.fit_transform(data['documents'])

    info_cols = list(data.iloc[:, :data.columns.get_loc('documents')+1].columns)

    df_docterm = pd.DataFrame(X.toarray(), index=data['claim_id'], columns=vectorizer.get_feature_names())

    drops = [col for col in info_cols if col in df_docterm.columns]

    if len(drops) > 0:
        df_docterm.drop(drops, axis=1, inplace=True)

    data = data.merge(df_docterm, how="inner", left_on='claim_id', right_index=True)

    data.drop_duplicates(subset=['claim_id', 'documents'], inplace=True)

    return data


# %% Check what dates are in the data, format if needed



# %% Get stopwords

def get_stopwords(filename='insurance_stopwords.json'):
    """[summary]

    Args:
        filename (str, optional): [description]. Defaults to 'insurance_stopwords.json'.

    Returns:
        [type]: [description]
    """

    stemmer = PorterStemmer()
    stop_words_en = set([stemmer.stem(word) for word in stopwords.words('english')])
    with open(filename, 'r') as json_file:
        stops_ins = json.loads(json_file.read())
    stop_words_ins = set(stemmer.stem(word) for word in stops_ins['insurance_stopwords'])
    stop_words_combined = stop_words_en.union(stop_words_ins)

    return stop_words_combined
