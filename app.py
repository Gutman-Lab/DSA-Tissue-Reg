import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from components.caseViewer import caseViewer_layout
from components.registrationControls import registrationControls_layout

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# This is the line that was missing - expose the server variable
server = app.server

# Create tabs using dcc.Tabs instead of dbc.Tabs
app.layout = dbc.Container(
    [
        html.H1("DSA Tissue Registration", className="mb-4"),
        dcc.Tabs(
            [
                dcc.Tab(caseViewer_layout, label="Case Viewer", value="caseViewer"),
                dcc.Tab(
                    registrationControls_layout,
                    label="Registration Controls",
                    value="registrationControls",
                ),
            ],
            value="registrationControls",
        ),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)
