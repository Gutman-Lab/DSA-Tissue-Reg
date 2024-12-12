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

# Create tabs using dcc.Tabs with supported styling options
app.layout = dbc.Container(
    [
        html.H3("DSA Tissue Registration", className="mb-2"),
        dcc.Tabs(
            [
                dcc.Tab(
                    caseViewer_layout,
                    label="Case Viewer",
                    value="caseViewer",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                ),
                dcc.Tab(
                    registrationControls_layout,
                    label="Registration Controls",
                    value="registrationControls",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                ),
            ],
            value="registrationControls",
            className="custom-tabs",
            colors={"border": "white", "primary": "#007bff", "background": "#f8f9fa"},
            style={"height": "40px", "padding": "0px 8px"},
        ),
    ],
    fluid=True,
)

# Add custom CSS to the app's assets folder
# Create a file: assets/custom.css with the following content:
"""
.custom-tab {
    padding: 6px 16px;
    border-radius: 4px;
    margin-right: 4px;
    border: 1px solid #dee2e6;
    background-color: #f8f9fa;
}

.custom-tab--selected {
    padding: 6px 16px;
    border-radius: 4px;
    margin-right: 4px;
    border: 1px solid #007bff;
    background-color: #007bff !important;
    color: white;
}

.custom-tabs {
    border-bottom: none !important;
}
"""

if __name__ == "__main__":
    app.run_server(debug=True)
