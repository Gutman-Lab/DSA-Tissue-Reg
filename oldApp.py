import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# from components.caseViewer import caseViewer_layout
from components.registrationControls import registrationControls_layout
from settings import background_callback_manager

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=background_callback_manager,
)

# This is the line that was missing - expose the server variable
server = app.server

# Create tabs using dcc.Tabs with supported styling options
app.layout = dbc.Container(
    [
        html.H3(
            "DSA Tissue Registration",
            className="mb-2",
            style={"align-self": "center", "text-align": "center"},
        ),
        registrationControls_layout,
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True)
