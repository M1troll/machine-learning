import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import requests
from dash import dcc, html
from dash.dependencies import Input, Output

API_TEMPLATE = "http://asterank.com/api/kepler?query={query}&limit={limit}"

NAMING_MAPPING = {
    "RPLANET" : "Planetary radius (Earth radii)",
    "TPLANET" : "Equilibrium temperature of planet, per Borucki et al. (k)",
    "A" : "Semi-major axis (AU)",
}

STAR_SIZES = ["small", "similar", "bigger"]



def get_data_from_api(query: dict = {}, limit: int = 2000) -> pd.DataFrame:
    """Send request to api and return data as `DataFrame`."""
    response = requests.get(API_TEMPLATE.format(query=query, limit=limit))
    response.raise_for_status()
    data = pd.json_normalize(response.json())

    # Add star size categories to data
    bins = [0, 1, 2, 20]
    data["StarSize"] = pd.cut(data["RSTAR"], bins, labels=STAR_SIZES)

    return data


def create_dasboard(data: pd.DataFrame) -> dash.Dash:
    """."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Best Dashboard",
    )

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
        options=[{"label": name, "value": name} for name in STAR_SIZES],
        value=STAR_SIZES,
        multi=True,
    )

    app.layout = html.Div(
        html.Div(
            [
                # Header row
                dbc.Row(
                    html.H1("Dashboard by Oboleninov Anton... Good luck..."),
                ),
                html.Br(),

                # Selectors row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Planet range selector"),
                                html.Div(rplanet_selector),
                            ],
                            width={"size": 4},
                        ),
                        dbc.Col(
                            [
                                html.H3("Star size filter"),
                                html.Div(star_size_selector),
                            ],
                            width={"size": 4},
                        ),
                    ],
                    style={"bottom-margin": "2%"},
                ),

                # Graphics row
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="radius-temp-figure"),
                            style={"margin": "2% 2% 2% 2%"},
                        ),
                        dbc.Col(
                            dcc.Graph(id="temp-a-figure"),
                            style={"margin": "2% 2% 2% 2%"},
                        ),
                    ],
                    style={"bottom-margin": "2%"},
                ),
            ],
        ),
        style={
            "background-image": "url('static/background-1.jpg')",
            "background-repeat": "no-repeat",
            "background-size": "100% auto",
            "height": "100vh",
            "width": "100vw",
            "margin": "0",
        },
    )

    return app


def add_callbacks(app: dash.Dash, data: pd.DataFrame):
    """."""

    @app.callback(
        [
            Output("radius-temp-figure", "figure"),
            Output("temp-a-figure", "figure"),
        ],
        [
            Input("range_slider", "value"),
            Input("star_size_selector", "value"),
        ],
    )
    def update(rplanet, sizes):
        """."""
        new_data = data[
            (data["RPLANET"] >= rplanet[0])
            & (data["RPLANET"] <= rplanet[1])
        ]

        if sizes:
            new_data = new_data[(data["StarSize"].isin(sizes))]

        # Range & temperature
        radius_figure = px.scatter(
            new_data,
            x="RPLANET",
            y="TPLANET",
            labels=NAMING_MAPPING,
            title="Planet range - planet temperature",
            color="StarSize",
            template="plotly_dark",
        )

        # Temperature & distance to sun
        temperature_figure = px.scatter(
            new_data,
            x="TPLANET",
            y="A",
            labels=NAMING_MAPPING,
            title="Planet temperature - distance from the star",
            color="StarSize",
            template="plotly_dark",
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
