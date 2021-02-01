import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Output, Input

data = pd.read_csv('country_vaccinations.csv')
# //// data = data.query("iso_code == 'GBR'")
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
# //// data.sort_values("date", inplace=True)

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
                    children="Covid Vaccinations", className="header-title"
                ),
                html.P(
                    children="Data from the Our World in Data GitHub",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Country", className="menu-title"),
                        dcc.Dropdown(
                            id="country-filter",
                            options=[{"label": country, "value": country} for country in np.sort(data.country.unique())],
                            value=['United Kingdom', 'United States'],
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
        Input("country-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_line_chart(countries, start_date, end_date):
    if len(countries) == 1:
        countries = list(countries)
    mask = ((data.country.isin(countries))
            & (data.date >= start_date)
            & (data.date <= end_date)
    )
    fig = px.line(data[mask],
        x="date", y="people_vaccinated_per_hundred", color='country')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)