## This will be a component that shows the stored registrations for a given case based on what's in the DSA case Folder

from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from settings import gc, memory, DSA_BASE_URL, token_info
import dash_ag_grid
from components.carlos_reg_utils import (
    dice_coefficient,
    apply_affine_transform,
    register_fixed_moving,
)
import cv2
import base64
import numpy as np

## Good exaples to start with..
caseList = [
    {"label": "E20-106", "value": "6731978f900c0c0559aef4ee"},
    {"label": "E20-11", "value": "641bfd45867536bb7a236ae1"},
    {"label": "E20-121", "value": "6734d5e8d4bd86dddb18e1cd"},
]

# Define the grid columns
grid_columns = [
    {"headerName": "Name", "field": "name", "sortable": True, "filter": True},
    {"headerName": "ID", "field": "_id", "sortable": True, "filter": True},
    {
        "headerName": "PreRotate",
        "field": "preRotate",
        "cellEditor": "agSelectCellEditor",
        "cellEditorParams": {"values": ["-90", "FlipXY", "None"]},
        "editable": True,
        "sortable": True,
    },
    {"headerName": "Size", "field": "size", "sortable": True},
    {
        "headerName": "Reg Image Size",
        "field": "regImageSize",
        "sortable": True,
    },
    {"headerName": "Rotation", "field": "rotation", "sortable": True},
    {"headerName": "Scale", "field": "scale", "sortable": True},
    {"headerName": "Source Image", "field": "srcImage", "sortable": True},
    {"headerName": "X Offset", "field": "xOffset", "sortable": True},
    {"headerName": "Y Offset", "field": "yOffset", "sortable": True},
]


@callback(
    Output("folder-contents-grid", "rowData"), Input("stored-reg-case-select", "value")
)
def update_folder_contents(selected_case_id):
    if not selected_case_id:
        return []

    try:
        # Get the contents of the selected folder from Girder
        folder_contents = list(gc.listItem(selected_case_id))
        # print(folder_contents, "folder contents")
        # Format the data for the grid
        row_data = []
        for item in folder_contents:
            # Get registration metadata if it exists
            reg_data = item.get("meta", {}).get("npReg", {})

            row_data.append(
                {
                    "name": item.get("name", ""),
                    "_id": item.get("_id", ""),
                    "preRotate": "None",  # Default value
                    "size": item.get("size", 0),
                    # Add registration metadata fields
                    "regImageSize": reg_data.get("regImageSize", ""),
                    "rotation": (
                        round(reg_data.get("rotation", 0), 3)
                        if reg_data.get("rotation") is not None
                        else ""
                    ),
                    "scale": (
                        round(reg_data.get("scale", 0), 3)
                        if reg_data.get("scale") is not None
                        else ""
                    ),
                    "srcImage": reg_data.get("srcImage", ""),
                    "xOffset": (
                        round(reg_data.get("xOffset", 0), 3)
                        if reg_data.get("xOffset") is not None
                        else ""
                    ),
                    "yOffset": (
                        round(reg_data.get("yOffset", 0), 3)
                        if reg_data.get("yOffset") is not None
                        else ""
                    ),
                }
            )

        return row_data
    except Exception as e:
        print(f"Error fetching folder contents: {e}")
        return []


@callback(
    Output("registered-image-preview", "src"),
    Input("folder-contents-grid", "selectedRows"),
)
def update_registered_image(selected_rows):
    if not selected_rows or len(selected_rows) == 0:
        return ""

    selected_item = selected_rows[0]
    src_image_id = selected_item.get("srcImage", "")
    target_id = selected_item.get("_id", "")
    print(selected_item, "was selected")
    # Get the registered image
    reg_matrix, reg_image = register_fixed_moving(src_image_id, target_id)
    resampled_image = apply_affine_transform(reg_image, reg_matrix)

    # Convert numpy array to base64 image
    # Convert to uint8 if not already
    if resampled_image.dtype != np.uint8:
        resampled_image = (resampled_image * 255).astype(np.uint8)

    # Convert grayscale to RGB if needed
    if len(resampled_image.shape) == 2:
        resampled_image = cv2.cvtColor(resampled_image, cv2.COLOR_GRAY2RGB)

    print("--- Now Resampling the image ---")

    # Encode the image
    success, buffer = cv2.imencode(".png", resampled_image)
    if success:
        registered_thumb = "data:image/png;base64," + base64.b64encode(buffer).decode(
            "utf-8"
        )
        return registered_thumb
    else:
        print("Error encoding image")
        registered_thumb = ""


@callback(
    [
        Output("selected-source-thumbnail", "src"),
        Output("selected-target-thumbnail", "src"),
    ],
    Input("folder-contents-grid", "selectedRows"),
)
def update_thumbnails(selected_rows):
    if not selected_rows or len(selected_rows) == 0:
        return "", "", ""

    selected_item = selected_rows[0]
    src_image_id = selected_item.get("srcImage", "")

    # Only proceed if srcImage exists and is not empty
    if not src_image_id:
        return "", "", ""

    target_id = selected_item.get("_id", "")

    source_thumb = (
        f"{DSA_BASE_URL}/item/{src_image_id}/tiles/thumbnail?token={token_info['_id']}"
    )
    target_thumb = (
        f"{DSA_BASE_URL}/item/{target_id}/tiles/thumbnail?token={token_info['_id']}"
    )

    return source_thumb, target_thumb


showReg_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select Case:", className="mb-2"),
                        dcc.Dropdown(
                            id="stored-reg-case-select",
                            options=caseList,
                            value=caseList[0]["value"],  # Default to first case
                            className="mb-4",
                        ),
                    ],
                    width={"size": 6, "offset": 3},  # Center the dropdown
                ),
            ],
            className="mb-4",
        ),
        html.H1("Stored Registrations"),
        dbc.Row(
            [
                dbc.Col(
                    dash_ag_grid.AgGrid(
                        id="folder-contents-grid",
                        columnDefs=grid_columns,
                        rowData=[],
                        columnSize="sizeToFit",
                        defaultColDef={
                            "resizable": True,
                            "sortable": True,
                            "filter": True,
                        },
                        dashGridOptions={
                            "pagination": True,
                            "rowSelection": "single",  # Enable single row selection
                            "editType": "fullRow",
                        },
                    ),
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        # Thumbnails row with three columns
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Source Image", className="text-center mb-2"),
                        html.Img(
                            id="selected-source-thumbnail",
                            style={
                                "maxWidth": "100%",
                                "height": "auto",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                                "padding": "5px",
                            },
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.H4("Target Image", className="text-center mb-2"),
                        html.Img(
                            id="selected-target-thumbnail",
                            style={
                                "maxWidth": "100%",
                                "height": "auto",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                                "padding": "5px",
                            },
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.H4("Registered Image", className="text-center mb-2"),
                        html.Img(
                            id="registered-image-preview",
                            style={
                                "maxWidth": "100%",
                                "height": "auto",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                                "padding": "5px",
                            },
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)


def find_relative_rotation(padded_mask1, padded_mask2) -> tuple[int, float]:
    """Function to find the relative rotation. Rotate counterclockwise"""
    max_dice = 0
    best_k = 0
    for k in range(4):  # 0°, 90°, 180°, 270°
        rotated_mask2 = np.rot90(padded_mask2, k)
        dice = dice_coefficient(padded_mask1, rotated_mask2)
        if k == 0:
            first_dice = dice
        if dice > max_dice:

            if dice > (first_dice + 0.005):

                best_k = k
            # else:
            #     max_dice = dice
            max_dice = dice

    return best_k * 90, max_dice  # Rotation angle and similarity score
