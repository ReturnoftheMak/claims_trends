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

df_scm = pd.read_csv(r'C:\Users\makhan.gill\SQL_DATA\LMM_SCM_all.csv', usecols=['ClaimDetailID', 'LossDescription', 'LossLocation'])
df_cgen = pd.read_csv(r'C:\Users\makhan.gill\SQL_DATA\claim_data_general_all.csv', usecols=['ClaimDetailID', 'ClaimAdvisedDate', 'HandlingClass'])
df = df_scm.merge(df_cgen, how="left", on='ClaimDetailID')

df.columns = ['claim_id', 'documents', 'loss_location', 'class', 'date']
df = df[['date', 'loss_location', 'claim_id', 'class', 'documents']]
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

stop_words_combined = get_stopwords()
vectorizer = scikit_vectorizer(stop_words_combined, LemmaTokenizer, CountVectorizer)

doc_tm = document_term_matrix(df.head(5000), vectorizer)

data, data_b = get_top_mentions_all_time(doc_tm, 30, 'date', additional_groups=['loss_location'])

# Need to standardise the input here to make this code a bit more extensible
# In this case the columns from the get_mentions should be used as color/filter
color_var = data.term
class_var = data['class']

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
                    children="SCM Data - Energy Claims",
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
                            options=[{"label": country, "value": country} for country in np.sort(color_var.unique())],
                            value=['fire', 'water', 'hurricane'],
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
                            options=[{"label": class_, "value": class_} for class_ in np.sort(class_var.unique())],
                            value=['Aviation'],
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
                            min_date_allowed=data.date.min().date(),
                            max_date_allowed=data.date.max().date(),
                            start_date=data.date.min().date(),
                            end_date=data.date.max().date(),
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
    mask = ((color_var.isin(terms))
            & (data.date >= start_date)
            & (data.date <= end_date)
            & (data['class'] == class_)
    )
    fig = px.line(data[mask],
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
    ]
)
def pie_chart(hoverData, terms):
    if len(terms) == 1:
        terms = list(terms)
    date_hover = hoverData['points'][0]['x']
    mask = (
        (data_b.term.isin(terms))
        & (data_b.date == date_hover)
        & (data_b.mentions >= 1)
    )
    fig = px.bar(data_b[mask],
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
