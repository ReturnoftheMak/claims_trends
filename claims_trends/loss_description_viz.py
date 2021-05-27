# %% Package Imports

import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Output, Input
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from preprocessing import scikit_vectorizer, LemmaTokenizer, document_term_matrix, get_stopwords
from term_frequency import get_top_mentions_all_time


# %% Data Loads

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

doc_tm = document_term_matrix(df.head(5000), vectorizer)


def get_mentions_per_class(vectorised_df:pd.DataFrame, N:int, date_col_name:str, additional_groups:list=[]):
    """[summary]

    Args:
        vectorised_df (pd.DataFrame): [description]
        N (int): [description]
        date_col_name (str): [description]
        additional_groups (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """

    class_mentions = {}

    cobs = vectorised_df['class'].unique()

    for cob in cobs:
        class_mentions[cob] = get_top_mentions_all_time(vectorised_df[vectorised_df['class'] == cob], N, date_col_name, additional_groups)

    df_combined_0 = pd.concat([class_mentions[key][0] for key in class_mentions.keys()])
    df_combined_1 = pd.concat([class_mentions[key][1] for key in class_mentions.keys()])

    df_dict_combined = {'base': df_combined_0, 'add_cols':df_combined_1}

    return class_mentions, df_dict_combined

df_dict, df_dict_combined =  get_mentions_per_class(doc_tm, 30, 'date', additional_groups=['loss_location'])


# %%

# /// data, data_b = get_top_mentions_all_time(doc_tm, 30, 'date', additional_groups=['loss_location'])

# Might need to standardise the input here to make this code a bit more extensible
# In this case the columns from the get_mentions should be used as color/filter
# /// color_var = data.term
# /// class_var = data['class']

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Claims Keywords"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Claims Keywords", className="header-title"
                ),
                html.P(
                    children="SCM Data - Property Claims",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Terms", className="menu-title"),
                        dcc.Dropdown(
                            id="term-filter",
                            value=['earthquake', 'wind', 'fire'],
                            multi=True,
                            className="dropdown",
                            persistence_type="local",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Class", className="menu-title"),
                        dcc.Dropdown(
                            id="class-filter",
                            options=[{"label": class_, "value": class_} for class_ in list(df_dict.keys())],
                            value=list(df_dict.keys())[0],
                            multi=False,
                            className="dropdown",
                            persistence_type="local",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range",
                            className="menu-title"
                            ),
                        dcc.DatePickerRange(
                            id="date-range",
                            persistence_type="local",
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
    dcc.Graph(id="line-chart"),
    dcc.Graph(id="bar-chart"),
    ]
)


@app.callback(
    dash.dependencies.Output('term-filter', 'options'),
    [dash.dependencies.Input('class-filter', 'value')]
)
def update_term_dropdown(name):
    return [{'label': i, 'value': i} for i in df_dict[name][0].term.unique()]


@app.callback(
    [
        dash.dependencies.Output('date-range', 'min_date_allowed'),
        dash.dependencies.Output('date-range', 'max_date_allowed'),
        dash.dependencies.Output('date-range', 'start_date'),
        dash.dependencies.Output('date-range', 'end_date'),
    ],
    [dash.dependencies.Input('class-filter', 'value')]
)
def update_date_dropdown(name):
    return df_dict[name][0].date.min().date(), df_dict[name][0].date.max().date(), df_dict[name][0].date.min().date(), df_dict[name][0].date.max().date()


@app.callback(
    Output("line-chart", "figure"),
    [
        Input("term-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("class-filter", "value"),
    ],
)
def update_line_chart(terms, start_date, end_date, class_):
    if len(terms) == 1:
        terms = list(terms)
    mask = ((df_dict[class_][0].term.isin(terms))
            & (df_dict[class_][0].date >= start_date)
            & (df_dict[class_][0].date <= end_date)
    )
    fig = px.line(df_dict[class_][0][mask],
                x="date",
                y="mentions",
                color='term',
                labels={'date':'Date Claim Advised',
                        'mentions':'Number of Occurences'
                        }
                )
    return fig


@app.callback(
    Output("bar-chart", "figure"),
    [
    Input("line-chart", 'hoverData'),
    Input("term-filter", "value"),
    Input("class-filter", "value"),
    ]
)
def bar_chart(hoverData, terms, class_):
    if len(terms) == 1:
        terms = list(terms)
    date_hover = hoverData['points'][0]['x']
    mask = (
        (df_dict[class_][1].term.isin(terms))
        & (df_dict[class_][1].date == date_hover)
        & (df_dict[class_][1].mentions >= 1)
    )
    fig = px.bar(df_dict[class_][1][mask],
                y='loss_location',
                x='mentions',
                color='term',
                labels={'loss_location':'Location of Loss',
                        'mentions':'Number of Occurences'
                        }
                )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

# %%
