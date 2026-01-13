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


from components.showStoredReg import showReg_layout

from settings import background_callback_manager

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=background_callback_manager,
)

# This is the line that was missing - expose the server variable
server = app.server


# Define TabUno (your current layout)
tab_uno_content = dbc.Container(
    [
        data_stores,
        dbc.Row(
            [
                dbc.Col(caseSelect_controls, width=12),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(thumbnail_grid, width=12),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(regPoint_layout, width=12),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(osdViewer_layout, width=12),
            ],
        ),
    ],
    fluid=True,
)

# Define the tabs layout
app.layout = dbc.Container(
    [
        html.H1("Image Registration Tool", className="text-center mb-4"),
        dcc.Tabs(
            [
                dcc.Tab(showReg_layout, label="Stored Registrations"),
                dcc.Tab(tab_uno_content, label="Registration"),
            ],
            id="tabs-content",
        ),
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True)
