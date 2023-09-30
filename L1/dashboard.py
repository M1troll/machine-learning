import dash
from dash import html, dcc
import requests
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output


API_TEMPLATE = "http://asterank.com/api/kepler?query={query}&limit={limit}"

NAMING_MAPPING = {
    "RPLANET" : "Planetary radius (Earth radii)",
    "TPLANET" : "Equilibrium temperature of planet, per Borucki et al. (k)",
    "A" : "Semi-major axis (AU)",
}


def get_data_from_api(query: dict = {}, limit: int = 2000) -> pd.DataFrame:
    """Send request to api and return data as `DataFrame`."""
    response = requests.get(API_TEMPLATE.format(query=query, limit=limit))
    response.raise_for_status()

    data = pd.json_normalize(response.json())
    return data


def create_dasboard(data: pd.DataFrame) -> dash.Dash:
    """."""
    # Add star size categories to data
    bins = [0, 1, 2, 30]
    star_size_names = ["small", "similar", "bigger"]
    data["StarSize"] = pd.cut(data["RSTAR"], bins, labels=star_size_names)

    app = dash.Dash(__name__)

    min_range, max_range = min(data["RPLANET"]), max(data["RPLANET"])
    rplanet_selector = dcc.RangeSlider(
        id="range_slider",
        min=min_range,
        max=max_range,
        marks={i: str(i) for i in range(5, int(max_range), 5)},
        step=1,
        value=[min_range, max_range],
    )

    star_size_selector = dcc.Dropdown(
        id = "star_size_selector",
        options=[{"label": name, "value": name} for name in star_size_names],
        value=["similar"],
        multi=True,
    )

    app.layout = html.Div([
        html.H1("Dashboard by Oboleninov Anton... Good luck..."),
        # Selectors
        html.Div([
            html.Div([
                html.H2("Planet range selector"),
                html.Div(rplanet_selector, style={"width": "30%"}),
            ]),
            html.Div([
                html.H2("Star size filter"),
                html.Div(star_size_selector, style={"width": "30%"}),
            ]),
        ]),

        html.Br(),

        # Graphs
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="radius-temp-figure"),
                    style={"float": "left", "width": "50%"},
                ),
                html.Div(
                    dcc.Graph(id="temp-a-figure"),
                    style={"margin-left": "50%", "width": "50%"},
                ),
            ],
            style={"width": "100%"},
        ),
    ])

    return app


def add_callbacks(app: dash.Dash, data: pd.DataFrame):
    """."""

    @app.callback(
        [
            Output(component_id="radius-temp-figure", component_property="figure"),
            Output(component_id="temp-a-figure", component_property="figure"),
        ],
        [
            Input(component_id="range_slider", component_property="value"),
            Input(component_id="star_size_selector", component_property="value"),
        ],
    )
    def update(rplanet, sizes):
        """."""
        new_data = data[
            (data["RPLANET"] >= rplanet[0])
            & (data["RPLANET"] <= rplanet[1])
            & (data["StarSize"].isin(sizes))
        ]

        # Range & temperature
        radius_figure = px.scatter(
            new_data,
            x="RPLANET",
            y="TPLANET",
            labels=NAMING_MAPPING,
            title="Planet range - planet temperature",
            color="StarSize",
        )

        # Temperature & distance to sun
        temperature_figure = px.scatter(
            new_data,
            x="TPLANET",
            y="A",
            labels=NAMING_MAPPING,
            title="Planet temperature - distance from the star",
            color="StarSize",
        )

        return radius_figure, temperature_figure


def main():
    """Get data from api and run dasboard's server."""
    data = get_data_from_api()
    app = create_dasboard(data)

    add_callbacks(app, data)

    app.run_server(debug=True)


if __name__ == "__main__":
    main()
