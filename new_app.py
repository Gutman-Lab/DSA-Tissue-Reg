import dash
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from components.caseSelectionView import caseSelection_layout
from dash import Input, Output, callback

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# This is the line that was missing - expose the server variable
server = app.server

# Define the app layout with fixed height and no scrolling
app.layout = html.Div(
    style={
        "height": "100vh",  # Full viewport height
        "overflow": "hidden",  # Prevent scrolling
        "display": "flex",
        "flexDirection": "column",
    },
    children=[
        html.H1(
            "DSA Tissue Registration",
            className="text-center py-2",  # Reduced padding
            style={"flexShrink": 0},  # Prevent header from shrinking
        ),
        # Add Bootstrap pills for navigation
        dbc.Nav(
            [
                dbc.NavItem(
                    dbc.NavLink(
                        "Case Selection",
                        id="case-selection-tab",
                        n_clicks=0,
                        className="active",
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        "Registration",
                        id="registration-tab",
                        n_clicks=0,
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        "LightGlue",
                        id="lightglue-tab",
                        n_clicks=0,
                    )
                ),
            ],
            pills=True,
            className="mb-3",
            style={"flexShrink": 0},  # Prevent nav from shrinking
        ),
        # Content area
        html.Div(
            id="tab-content",
            style={
                "flex": 1,
                "overflow": "hidden",
                "minHeight": 0,  # Important for nested flex scrolling
            },
            children=caseSelection_layout,  # Default to case selection view
        ),
    ],
)


# Callback to handle tab switching
@callback(
    Output("tab-content", "children"),
    [
        Input("case-selection-tab", "n_clicks"),
        Input("registration-tab", "n_clicks"),
        Input("lightglue-tab", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def switch_tab(case_clicks, reg_clicks, lightglue_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return caseSelection_layout

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "case-selection-tab":
        return caseSelection_layout
    elif button_id == "registration-tab":
        # TODO: Import and return registration layout
        return html.Div("Registration Tab Content")
    elif button_id == "lightglue-tab":
        # TODO: Import and return lightglue layout
        return html.Div("LightGlue Tab Content")

    return caseSelection_layout


# Callback to update active tab
@callback(
    [
        Output("case-selection-tab", "active"),
        Output("registration-tab", "active"),
        Output("lightglue-tab", "active"),
    ],
    [
        Input("case-selection-tab", "n_clicks"),
        Input("registration-tab", "n_clicks"),
        Input("lightglue-tab", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def update_active_tab(case_clicks, reg_clicks, lightglue_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, False

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "case-selection-tab":
        return True, False, False
    elif button_id == "registration-tab":
        return False, True, False
    elif button_id == "lightglue-tab":
        return False, False, True

    return True, False, False


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
