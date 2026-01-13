from dash import html, Input, Output, State, callback, dcc, no_update
from settings import gc, memory, DSA_BASE_URL, token_info
import dash_bootstrap_components as dbc
import dash_ag_grid
from pprint import pprint
import shutil
import os

# Root folder ID for the application
ROOT_FOLDER_ID = "67cf332a65fd0aa585997f2a"


# Function to clear the cache directory
def clear_cache():
    cache_dir = os.path.expanduser("~/.cache/npCacheDir")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache directory: {cache_dir}")


# Function to get cases from DSA
@memory.cache
def get_cases():
    try:
        # Get the list of folders directly under the root folder
        folders = list(gc.listFolder(ROOT_FOLDER_ID))
        print(f"Found {len(folders)} folders")

        # Simply format all folders for the dropdown
        cases = [
            {
                "label": folder["name"],  # Use the folder name as the label
                "value": folder["_id"],  # Use the folder ID as the value
            }
            for folder in folders
        ]

        print("Available cases:")
        for case in cases:
            print(f"  - {case['label']} ({case['value']})")

        return cases
    except Exception as e:
        print(f"Error fetching cases: {e}")
        return []


# Get initial case list
caseList = get_cases()

# Create the case selection controls with cache clear button and block ID filter
filter_controls = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    html.Label(
                        "Select a case ",
                        className="me-2",
                        style={"marginTop": "2px"},
                    ),
                    dbc.Select(
                        id="caseSelect",
                        options=caseList,
                        value=caseList[0]["value"],
                        style={"width": "200px"},
                    ),
                ],
                className="d-flex align-items-center",
            ),
            width="auto",
        ),
        dbc.Col(
            html.Div(
                [
                    html.Label(
                        "Block ID:",
                        className="me-2",
                        style={"marginTop": "2px"},
                    ),
                    dbc.Select(
                        id="blockID_filter_select",
                        options=[],
                        style={"width": "200px"},
                    ),
                ],
                className="d-flex align-items-center",
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.Button(
                "Clear Cache",
                id="clear_cache_button",
                color="secondary",
                size="sm",
                className="ms-2",
            ),
            width="auto",
        ),
    ],
    className="g-2 mb-2",  # Reduced gap between columns and bottom margin
    align="center",
)

# Create the case folder table
adrcCaseTable = dash_ag_grid.AgGrid(
    id="currentCaseTable",
    columnDefs=[
        {"headerName": "Name", "field": "name"},
        # {"headerName": "Created", "field": "created"},
        {"headerName": "Region Name", "field": "meta.npSchema.regionName"},
        {"headerName": "Stain", "field": "meta.npSchema.stainID"},
    ],
    defaultColDef={
        "flex": 1,
        "minWidth": 80,
        "filterParams": {"debounceMs": 2500},
        "floatingFilter": True,
        "sortable": True,
        "resizable": True,
    },
    dashGridOptions={
        "rowSelection": "single",
        "domLayout": "autoHeight",  # Makes the table height fit its content
    },
    className="ag-theme-alpine",
    style={"width": "100%", "marginBottom": "1rem"},
)

# Create the case folder table and metadata panel row
table_and_metadata = dbc.Row(
    [
        dbc.Col(
            adrcCaseTable,
            width=8,
        ),
        dbc.Col(
            [
                html.H6("Metadata", className="mb-2"),
                html.Pre(
                    id="metadata-display",
                    className="p-2 bg-light border rounded",
                    style={
                        "maxHeight": "300px",
                        "overflowY": "auto",
                        "fontSize": "0.8rem",
                        "fontFamily": "monospace",
                        "whiteSpace": "pre-wrap",
                        "wordWrap": "break-word",
                    },
                ),
            ],
            width=4,
            style={"borderLeft": "1px solid #dee2e6"},
        ),
    ],
    className="mb-2",
)

# Create the thumbnail container
thumbnail_grid = html.Div(
    id="thumbnail-grid",
    style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "0.25rem",  # Reduced gap between thumbnails
        "padding": "0.25rem",  # Reduced padding around the grid
        "overflowY": "auto",
        "overflowX": "hidden",
    },
)

# Create the layout for the case selection view
caseSelection_layout = html.Div(
    style={
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",
        "padding": "0.25rem",
    },
    children=[
        filter_controls,
        html.Div(
            style={
                "flex": "0 0 auto",
                "marginBottom": "0.25rem",
            },
            children=[table_and_metadata],
        ),
        html.Div(
            id="thumbnail_container",
            style={
                "flex": 1,
                "overflowY": "auto",
                "overflowX": "hidden",
                "minHeight": 0,
                "marginTop": "0.25rem",
            },
            children=[thumbnail_grid],
        ),
    ],
)


# Callback for cache clear button
@callback(
    Output("caseSelect", "options"),
    Output("caseSelect", "value"),
    Input("clear_cache_button", "n_clicks"),
    prevent_initial_call=True,
)
def clear_cache_and_refresh(n_clicks):
    if n_clicks:
        clear_cache()
        # Refetch cases after clearing cache
        new_cases = get_cases()
        return new_cases, new_cases[0]["value"] if new_cases else None
    return no_update, no_update


# Callback to update the case images store when a case is selected
@callback(Output("case_images_store", "data"), Input("caseSelect", "value"))
# @memory.cache
def update_case_images(caseFolderId):
    if not caseFolderId:
        return []

    print(f"Fetching images for case folder: {caseFolderId}")
    try:
        # Get all items in the case folder, specifying type=folder
        items = list(gc.get(f"resource/{caseFolderId}/items?type=folder&limit=0"))
        print(f"Found {len(items)} items in case folder")
        return items
    except Exception as e:
        print(f"Error fetching items: {e}")
        return []


# Callback to update block ID filter options
@callback(
    [
        Output("blockID_filter_select", "options"),
        Output("blockID_filter_select", "value"),
    ],
    Input("case_images_store", "data"),
)
def update_blockID_filter_options(slideList):
    if not slideList:
        return [], None

    # Extract unique blockIDs from the data
    unique_blockIDs = list(
        set(
            slide.get("meta", {}).get("npSchema", {}).get("blockID", "")
            for slide in slideList
        )
    )
    unique_blockIDs = [bid for bid in unique_blockIDs if bid]  # Remove empty values

    if not unique_blockIDs:
        return [], None

    # Create options list and select first block by default
    options = [{"label": bid, "value": bid} for bid in sorted(unique_blockIDs)]
    return options, options[0]["value"]


# Callback to update the table when the case images store changes or block filter changes
@callback(
    [
        Output("currentCaseTable", "rowData"),
        Output("currentCaseTable", "selectedRows"),
    ],
    [Input("case_images_store", "data"), Input("blockID_filter_select", "value")],
)
def update_table_from_store(case_images, selected_block):
    if not case_images:
        return [], []

    # Filter by block ID if a specific block is selected
    if selected_block and selected_block != "all":
        filtered_images = [
            img
            for img in case_images
            if img.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
        ]
    else:
        filtered_images = case_images

    # Return the filtered images data and select the first row
    return filtered_images, [filtered_images[0]] if filtered_images else []


# Callback to update thumbnails when table data changes
@callback(Output("thumbnail-grid", "children"), [Input("currentCaseTable", "rowData")])
def update_thumbnails(table_data):
    if not table_data:
        return []

    # Create thumbnail cards for each image
    thumbnail_cards = [create_thumbnail_card(item) for item in table_data]
    return thumbnail_cards


# Function to create a thumbnail card for an image
def create_thumbnail_card(item):
    return dbc.Card(
        [
            dbc.CardHeader(
                item.get("meta", {})
                .get("npSchema", {})
                .get("stainID", "Unknown Stain"),
                className="text-center p-1",  # Minimal padding in header
                style={
                    "font-size": "0.9rem",
                    "border-bottom": "none",  # Remove border between header and image
                },
            ),
            dbc.CardImg(
                src=f"{DSA_BASE_URL}/item/{item['_id']}/tiles/thumbnail?token={token_info['_id']}&width=256",
                top=True,
                style={
                    "height": "200px",  # Reduced height
                    "objectFit": "contain",
                    "padding": "0",  # Remove padding around image
                    "backgroundColor": "#f8f9fa",  # Light gray background
                },
            ),
        ],
        className="m-0",  # No margin
        style={
            "width": "280px",
            "backgroundColor": "#f8f9fa",  # Match image background
            "border": "1px solid #dee2e6",  # Subtle border
        },
    )


# Add callback to update metadata display when a row is selected
@callback(
    Output("metadata-display", "children"),
    [Input("currentCaseTable", "selectedRows")],
)
def update_metadata_display(selected_rows):
    if not selected_rows or len(selected_rows) == 0:
        return "Select a row to view metadata"

    import json

    selected_item = selected_rows[0]

    # Format the JSON with indentation for better readability
    formatted_json = json.dumps(selected_item, indent=2)
    return formatted_json
