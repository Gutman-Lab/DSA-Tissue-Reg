import dash
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from components.caseSelectionView import caseSelection_layout, get_cases
from components.lightGlueView import lightGlue_layout
from dash import Input, Output, callback, State, ctx

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,  # Add this to suppress callback exceptions
)

# This is the line that was missing - expose the server variable
server = app.server

# Get initial case list
initial_cases = get_cases()
initial_case_id = initial_cases[0]["value"] if initial_cases else None

# Define the app layout
app.layout = html.Div(
    [
        # Stores - accessible across all tabs
        dcc.Store(id="case_images_store", data=[]),
        dcc.Store(id="selected_case_id", data=initial_case_id),
        # Header
        html.H1("DSA Tissue Registration", className="p-3 bg-light border-bottom"),
        # Tabs
        dbc.Tabs(
            [
                dbc.Tab(
                    caseSelection_layout,
                    label="Case Selection",
                    tab_id="case-selection",
                ),
                dbc.Tab(
                    html.Div("Registration Tab Content"),
                    label="Registration",
                    tab_id="registration",
                ),
                dbc.Tab(lightGlue_layout, label="LightGlue", tab_id="lightglue"),
            ],
            id="tabs",
            active_tab="lightglue",
        ),
    ],
    style={
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
    },
)


# Callback to update selected_case_id when case is selected
@callback(
    Output("selected_case_id", "data"),
    Input("caseSelect", "value"),
)
def update_selected_case(case_id):
    return case_id


# Single callback to handle case_images_store updates
@callback(
    Output("case_images_store", "data", allow_duplicate=True),
    Input("selected_case_id", "data"),
    prevent_initial_call=True,
)
def update_case_images(case_id):
    if not case_id:
        return []

    from settings import gc

    print(f"Fetching images for case: {case_id}")
    try:
        # Get all items in the case folder
        items = list(gc.get(f"resource/{case_id}/items?type=folder&limit=0"))
        print(f"Found {len(items)} items in case folder")
        return items
    except Exception as e:
        print(f"Error fetching items: {e}")
        return []


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
