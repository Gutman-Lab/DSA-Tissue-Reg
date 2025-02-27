import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


from components.registrationControls import (
    osdViewer_layout,
    data_stores,
    caseSelect_controls,
    thumbnail_grid,
    regPoint_layout,
)


from settings import background_callback_manager

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=background_callback_manager,
)

# This is the line that was missing - expose the server variable
server = app.server


registrationApp_layout = dbc.Container(
    [
        caseSelect_controls,
        thumbnail_grid,
        osdViewer_layout,
        data_stores,
        regPoint_layout,
    ]
)


# Create tabs using dcc.Tabs with supported styling options
app.layout = dbc.Container(
    [
        html.Div(id="caseSetViewer"),
        html.H3(
            "DSA Tissue Registration",
            className="mb-2",
            style={"align-self": "center", "text-align": "center"},
        ),
        registrationApp_layout,
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True)
