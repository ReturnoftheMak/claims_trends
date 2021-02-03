# %% Package Imports

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

df = pd.read_csv(r'C:\Users\makhan.gill\Documents\GitHub\claims_trends\claims_trends\claim_ids.csv')
df.columns = ['claim_id', 'date', 'documents']
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

stop_words_combined = get_stopwords()
vectorizer = scikit_vectorizer(stop_words_combined, LemmaTokenizer, CountVectorizer)

doc_tm = document_term_matrix(df.head(5000), vectorizer)

data = get_top_mentions_all_time(doc_tm, 20, 'date')

# Need to standardise the input here to make this code a bit more extensible
# In this case the columns from the get_mentions should be used as color/filter
color_var = data.term

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Covid Vaccinations"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Claims Keywords", className="header-title"
                ),
                html.P(
                    children="SCM Data",
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
                            value=['fire', 'water'],
                            multi=True,
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
    ]
)

@app.callback(
    Output("line-chart", "figure"),
    [
        Input("term-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_line_chart(var, start_date, end_date):
    if len(var) == 1:
        var = list(var)
    mask = ((color_var.isin(var))
            & (data.date >= start_date)
            & (data.date <= end_date)
    )
    fig = px.line(data[mask],
                x="date",
                y="mentions",
                color='term'
                )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
# %%
