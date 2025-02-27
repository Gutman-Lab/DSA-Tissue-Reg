import numpy as np
from dash import (
    html,
    Input,
    Output,
    State,
    callback,
    dcc,
    clientside_callback,
    ctx,
    no_update,
    ALL,
)


import json
from pprint import pprint

import dash

import dash_bootstrap_components as dbc
from settings import gc, memory, DSA_BASE_URL, token_info
import dash_paperdragon
import cv2
import requests
from io import BytesIO
import base64
from utils.registration_utils import (
    get_thumbnail_image,
    normalize_image_sizes,
    create_geojson_features,
    scale_points_to_full_size,
    get_slides_for_registration,
    create_registration_points,
    generate_distinct_colors,
)
import time
from settings import background_callback_manager
import dash_ag_grid
from components.osdRegViewers import (
    fixed_image_viewer,
    moving_image_viewer,
    merged_image_viewer,
    osdViewer_layout,
)

## Good exaples to start with..
caseList = [
    {"label": "E20-11", "value": "641bfd45867536bb7a236ae1"},
    {"label": "E20-106", "value": "641bfdd9867536bb7a236c3d"},
    {"label": "E20-121", "value": "6734d5e8d4bd86dddb18e1cd"},
]


import dash_core_components as dcc

# Add these controls to your layout
parameter_controls = dbc.Row(
    [
        dbc.Col(
            [
                html.Label("Rotation (°):"),
                dcc.Input(
                    id="rotation-input",
                    type="number",
                    value=0,
                    step=0.01,
                    readOnly=True,
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Scale:"),
                dcc.Input(
                    id="scale-input", type="number", value=1, step=0.01, readOnly=True
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Offset X:"),
                dcc.Input(
                    id="offset-x-input", type="number", value=0, step=1, readOnly=True
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Offset Y:"),
                dcc.Input(
                    id="offset-y-input", type="number", value=0, step=1, readOnly=True
                ),
            ],
            width=4,
        ),
    ]
)


### Populate caseSlideSet based on the currently slided caseID
@callback(
    Output("registration_caseSlideSet_store", "data"),
    [Input("registration_caseSelect", "value")],
)
@memory.cache  ## Remove this when you start doing any updates..
def populate_caseSlideSet(caseFolderId):
    ## May want to add schema validation here in the future
    slideList = list(gc.listItem(caseFolderId))

    annotationCountStr = (
        f"annotation/counts?items={','.join([x['_id'] for x in slideList])}"
    )
    annotationCounts = gc.get(annotationCountStr)
    for sl in slideList:
        print(sl)
        sl["annotationCount"] = annotationCounts[sl["_id"]]
    return slideList


### This callback is used to populate the blockID filter options
@callback(
    Output("registration_blockID_filter_select", "options"),
    Output("registration_blockID_filter_select", "value"),
    [
        Input("registration_caseSlideSet_store", "data"),
        Input("registration_show_annotated_only", "value"),
    ],
)
def update_blockID_filter_options(slideList, show_annotated_only):
    # First filter slides by annotation count if needed
    if show_annotated_only:
        slideList = [
            slide for slide in slideList if slide.get("annotationCount", 0) > 0
        ]

    # Extract unique blockIDs from the filtered data
    unique_blockIDs = list(
        set(
            slide.get("meta", {}).get("npSchema", {}).get("blockID", "")
            for slide in slideList
        )
    )
    unique_blockIDs = [bid for bid in unique_blockIDs if bid]  # Remove empty values

    options = [{"label": bid, "value": bid} for bid in sorted(unique_blockIDs)]

    return options, options[0]["value"]


@callback(
    Output("selectedRegionData", "data"),
    Input("registration_blockID_filter_select", "value"),
    Input("registration_caseSlideSet_store", "data"),
    Input("registration_show_annotated_only", "value"),
)
def updateSelectedRegionData(blockID, slideList, show_annotated_only):
    if blockID:
        slideList = [
            slide
            for slide in slideList
            if slide.get("meta", {}).get("npSchema", {}).get("blockID", "") == blockID
        ]

    ## This will show only slides that have annotations... may want to change this
    ## to be based on whether the FIXED image has annotations instead of any slide.. TBD
    if show_annotated_only:
        slideList = [
            slide for slide in slideList if slide.get("annotationCount", 0) > 0
        ]

    return slideList


# ##https://www.sciencedirect.com/science/article/pii/S0010482522000932

# ### LAYOUTS-- Moving to separate variables to make this more readable

data_stores = dbc.Container(
    [
        dcc.Store(id="optimal-transform-store", data=None),
        dcc.Store(id="registration_caseId", data="641bfd45867536bb7a236ae1"),
        dcc.Store(id="registration_blockId", data="5"),
        dcc.Store(id="selectedRegionData", data=None),
        dcc.Store(id="fixed_image_id", data=None),
        dcc.Store(id="moving_image_id", data=None),
        dcc.Store(id="moving-image-metadata", data={}),
        dcc.Store(id="registration_caseRootFolderId_store", data=caseList[0]["value"]),
        dcc.Store(id="registration_caseSlideSet_store", data=[]),
        dcc.Store(id="selected-moving-slide", data=None),
        dcc.Store(id="fixedImage_fiducial_points", data=None),
        dcc.Store(id="movingImage_fiducial_points", data=None),
    ]
)


## Populate the grids when I update or push points to it
def convert_points_to_row_data(points):
    """Convert a list of points into a list of dictionaries for rowData."""
    row_data = []
    for index, (x, y) in enumerate(points):
        row_data.append(
            {"globalX": x, "globalY": y, "index": index}  # Add the index position
        )
    return row_data


@callback(
    Output("regPoint_fixed_image_grid", "rowData"),
    Output("regPoint_moving_image_grid", "rowData"),
    Input("fixedImage_fiducial_points", "data"),
    Input("movingImage_fiducial_points", "data"),
)
def populate_regPoint_grids(fixed_points, moving_points):
    fixed_row_data = convert_points_to_row_data(fixed_points) if fixed_points else []
    moving_row_data = convert_points_to_row_data(moving_points) if moving_points else []

    return fixed_row_data, moving_row_data


caseSelect_controls = dbc.Row(
    [
        dbc.Col(
            [
                html.Label(
                    "Select Case:",
                    className="mb-1 small",
                ),
                dbc.Select(
                    id="registration_caseSelect",
                    options=caseList,
                    value=caseList[0]["value"],
                    size="sm",
                ),
            ],
            width="auto",
        ),
        dbc.Col(
            [
                html.Label("Block ID:", className="mb-1 small"),
                dbc.Select(
                    id="registration_blockID_filter_select",
                    options=[{}],
                    size="sm",
                ),
            ],
            width="auto",
        ),
        dbc.Col(
            dbc.Checkbox(
                id="registration_show_annotated_only",
                label="Only annotated",
                value=False,
                className="ml-3",
            ),
            width="auto",
        ),
        dbc.Col(
            [
                html.Label("Image ROI Size:", className="mb-1 small"),
                dbc.Select(
                    id="regImage_size_select",
                    options=[256, 384, 512, 1024],
                    value=256,
                    style={"width": "200px", "marginLeft": "15px"},
                ),
            ],
            width="auto",
        ),
        dbc.Col(
            [
                html.Label("Registration Method:", className="mb-1 small"),
                dbc.Select(
                    id="registration_method_select",
                    options=["SIFT", "ORB"],
                    value="SIFT",
                    style={"width": "200px", "marginLeft": "15px"},
                ),
            ],
            width="auto",
        ),
    ]
)


# Function to convert numpy array to base64 image
def array_to_data_url(img_array):
    """Convert numpy array to base64 encoded image for display"""
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Encode image
    success, encoded = cv2.imencode(".png", img_array)
    if success:
        return "data:image/png;base64," + base64.b64encode(encoded).decode()
    return ""


thumbnail_grid = dcc.Loading(
    id="registration-loading",
    type="circle",
    children=[
        html.Div(
            id="registration-thumbnail-grid",
            className="d-flex flex-nowrap gap-2",
            style={
                "overflowX": "auto",
                "whiteSpace": "nowrap",
                "paddingBottom": "10px",
                "maxHeight": "200px",
            },
        ),
    ],
)


def compute_affine_parameters(fixed_points, moving_points):
    """
    Compute the affine transformation parameters (rotation, scaling, and offset)
    to optimize the registration of the fixed image to the moving image.

    Parameters:
    - fixed_points: List of fiducial points from the fixed image.
    - moving_points: List of fiducial points from the moving image.

    Returns:
    - rotation: Rotation angle in degrees.
    - scale: Scaling factor.
    - offset: (x_offset, y_offset) translation.
    """
    # Convert points to numpy arrays
    fixed_np = np.array(fixed_points, dtype=np.float32)
    moving_np = np.array(moving_points, dtype=np.float32)

    # Estimate the affine transformation matrix
    transform_matrix = cv2.estimateAffinePartial2D(moving_np, fixed_np)[0]

    if transform_matrix is None:
        raise ValueError("Could not compute affine transformation.")

    # Extract rotation, scaling, and translation from the transformation matrix
    scale_x = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
    scale_y = np.sqrt(transform_matrix[1, 0] ** 2 + transform_matrix[1, 1] ** 2)
    rotation = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]) * (
        180.0 / np.pi
    )  # Convert to degrees
    x_offset = transform_matrix[0, 2]
    y_offset = transform_matrix[1, 2]

    return (
        rotation,
        (scale_x + scale_y) / 2,
        (x_offset, y_offset),
    )  # Return average scale


regPoint_grid_columns = [
    {"headerName": "Point ID", "field": "index", "width": 50},
    {"headerName": "globalX", "field": "globalX"},
    {"headerName": "globalY", "field": "globalY"},
]


regPoint_layout = dbc.Row(
    [
        dbc.Col(
            [
                html.Label("Fixed Image Fiducial Points:", className="mb-1 small"),
                html.Div(
                    id="regPoint_fixed_image_info",
                    className="image-info mb-2",
                ),
                html.Div(
                    [
                        dash_ag_grid.AgGrid(
                            id="regPoint_fixed_image_grid",
                            columnDefs=regPoint_grid_columns,
                            rowData=[],
                        ),
                    ],
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label(
                    "Moving Image Fiducial Points:",
                    className="mb-1 small",
                ),
                html.Div(
                    [
                        dash_ag_grid.AgGrid(
                            id="regPoint_moving_image_grid",
                            columnDefs=regPoint_grid_columns,
                            rowData=[],
                        ),
                    ],
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Offset Info", className="mb-1 small"),
                html.Div(id="manual_offset_data"),
                html.Div(
                    id="regPoint_offset_info",
                    className="image-info mb-2",
                ),
                parameter_controls,
            ],
            width=4,
        ),
    ]
)


@callback(
    Output("regPoint_offset_info", "children"),
    Input("fixedImage_fiducial_points", "data"),
    Input("movingImage_fiducial_points", "data"),
)
def update_regPoint_offset_info(fixed_points, moving_points):
    rotation, scale, offset = compute_affine_parameters(fixed_points, moving_points)
    return html.Div(
        f"Rotation: {rotation:.2f}°, Scale: {scale:.2f}, Offset: ({offset[0]:.2f}, {offset[1]:.2f})",
        style={"fontSize": "24px"},
    )


## This generates the thumbnail grid, and also the fixed and moving slide IDs
@callback(
    Output("registration-thumbnail-grid", "children"),
    Output("fixed_image_id", "data"),
    Output("moving_image_id", "data"),
    Input("selectedRegionData", "data"),
    # Input({"type": "thumbnail-card", "index": ALL}, "n_clicks"),
    Input(
        {"type": "thumbnail-card", "index": ALL, "stain": ALL}, "n_clicks"
    ),  ## Want both who was clicke,d and the id of the clicked.
    State({"type": "thumbnail-card", "index": ALL, "stain": ALL}, "id"),
    State("fixed_image_id", "data"),
    State("moving_image_id", "data"),
)
def generate_thumbnail_grid(
    slideList,
    thumbnail_clicks,
    thumbnail_ids,
    fixed_slide_id,
    moving_slide_id,
):
    triggered = ctx.triggered_id
    # print(triggered, "triggered...")

    if not ctx.triggered_id:
        return no_update, no_update, no_update

    ## The thumbnail card is clicked, so update the moving slide ID
    if (
        triggered
        and isinstance(triggered, dict)
        and triggered.get("type") == "thumbnail-card"
    ):
        # Get the clicked card's ID and stain
        moving_slide_id = ctx.triggered_id["index"]

    ## Initialize the fixed and moving slide IDs if the slide list is empty
    if not slideList:
        return [], None, None

    ## If the fixed slide ID is not set, or the trigger is from the selected region data, then update the fixed slide ID
    if not fixed_slide_id or triggered == "selectedRegionData":
        ### See if there is a HE slide in the list
        fixed_slide_id = next(
            (
                x["_id"]
                for x in slideList
                if x.get("meta", {}).get("npSchema", {}).get("stainID", "") == "HE"
            ),
            None,
        )
        ## If there is no HE slide, then use the first slide in the list

        if not fixed_slide_id:
            fixed_slide_id = slideList[0]["_id"]

    ## If the moving slide ID is not set, or the trigger is from the selected region data, then update the moving slide ID
    ## This is needed when the region changes

    if not moving_slide_id or triggered == "selectedRegionData":
        moving_slide_id = slideList[1]["_id"]

    # Create thumbnail cards for each slide
    thumbnail_cards = [
        create_thumbnail_card(slide, selected=(slide.get("_id") == moving_slide_id))
        for slide in slideList
    ]

    return thumbnail_cards, fixed_slide_id, moving_slide_id


@callback(
    Output("moving-image-info", "children"),
    Output("moving-image-viewer", "tileSources"),
    Input("moving_image_id", "data"),
)
def update_movingImageData(moving_image_id):
    ## This updates the metadata, and also the fixed image viewer tile source
    ## Trying to tie these callbacks to a single variable change to make it more manageable
    if moving_image_id:

        moving_tile_info = gc.get(f"item/{moving_image_id}/tiles")
        return (
            [
                html.Div(
                    f"Size: {moving_tile_info.get('sizeX', 'N/A')}×{moving_tile_info.get('sizeY', 'N/A')}  |  "
                    f"Resolution: {moving_tile_info.get('mm_x', 'N/A')}  |  "
                    f"Mag: {moving_tile_info.get('magnification', 'N/A')}"
                )
            ],
            [
                {
                    "tileSource": f"{DSA_BASE_URL}/item/{moving_image_id}/tiles/dzi.dzi?token={token_info['_id']}",
                    "width": moving_tile_info.get("width", 100000),
                }
            ],
        )
    else:
        return [], []


@callback(
    Output("fixed-image-info", "children"),
    Output("fixed-image-viewer", "tileSources"),
    Input("fixed_image_id", "data"),
)
def update_fixedImageData(fixed_image_id):
    ## This updates the metadata, and also the fixed image viewer tile source
    ## Trying to tie these callbacks to a single variable change to make it more manageable
    if fixed_image_id:

        fixed_tile_info = gc.get(f"item/{fixed_image_id}/tiles")
        return (
            [
                html.Div(
                    f"Size: {fixed_tile_info.get('sizeX', 'N/A')}×{fixed_tile_info.get('sizeY', 'N/A')}  |  "
                    f"Resolution: {fixed_tile_info.get('mm_x', 'N/A')}  |  "
                    f"Mag: {fixed_tile_info.get('magnification', 'N/A')}"
                )
            ],
            [
                {
                    "tileSource": f"{DSA_BASE_URL}/item/{fixed_image_id}/tiles/dzi.dzi?token={token_info['_id']}",
                    "width": fixed_tile_info.get("width", 100000),
                }
            ],
        )
    else:
        return [], []


### Try and register the slides based on the thumbnail images
## This will dump the fidicual points..


@callback(
    Output("fixedImage_fiducial_points", "data"),
    Output("movingImage_fiducial_points", "data"),
    Input("fixed_image_id", "data"),
    Input("moving_image_id", "data"),
    Input("regImage_size_select", "value"),
)
def register_slides(fixed_image_id, moving_image_id, regImage_size_select):

    ## Get the fiducial points from the fixed and moving images
    if fixed_image_id and moving_image_id:
        print(fixed_image_id, moving_image_id, "are being registered ")
        # Get image dimensions for both slides
        try:
            fixed_tiles_info = gc.get(f"item/{fixed_image_id}/tiles")
            moving_tiles_info = gc.get(f"item/{moving_image_id}/tiles")

            fixed_bounds = {
                "width": fixed_tiles_info.get("sizeX", 10000),
                "height": fixed_tiles_info.get("sizeY", 10000),
            }
            moving_bounds = {
                "width": moving_tiles_info.get("sizeX", 10000),
                "height": moving_tiles_info.get("sizeY", 10000),
            }

            thumbnail_width = int(regImage_size_select)
            fixed_thumb = get_thumbnail_image(fixed_image_id, width=thumbnail_width)
            moving_thumb = get_thumbnail_image(moving_image_id, width=thumbnail_width)

            if fixed_thumb is None or moving_thumb is None:
                print("Failed to retrieve thumbnails")
                return no_update, no_update

            norm_fixed, norm_moving, (fixed_scale, moving_scale) = (
                normalize_image_sizes(fixed_thumb, moving_thumb)
            )

            detection_method = "sift"
            num_points = 10

            print(
                f"Normalized shapes - fixed: {norm_fixed.shape}, moving: {norm_moving.shape}"
            )
            print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

            fixed_points, moving_points, debug_info = create_registration_points(
                norm_fixed,
                norm_moving,
                detection_method,
                num_points,
                fixed_bounds,
            )

            if fixed_points and moving_points:
                """Scale points from thumbnail coordinates to full image coordinates"""
                scale_to_fullRes = fixed_bounds["width"] / thumbnail_width
                ## The images are scaled based on the width only
                scaled_fixed_points = []
                for point in fixed_points:
                    scaled_fixed_points.append(
                        (point[0] * scale_to_fullRes, point[1] * scale_to_fullRes)
                    )

                scaled_moving_points = []
                for point in moving_points:
                    scaled_moving_points.append(
                        (point[0] * scale_to_fullRes, point[1] * scale_to_fullRes)
                    )

                # Scale fixed points back to original fixed image size
                fixed_points = [
                    (x * fixed_scale["x"], y * fixed_scale["y"])
                    for x, y in fixed_points
                ]
                moving_points = [
                    (x * moving_scale["x"], y * moving_scale["y"])
                    for x, y in moving_points
                ]

                # Scale to full image coordinates
                fixed_points = scale_points_to_full_size(
                    fixed_points, fixed_thumb.shape, fixed_bounds
                )
                moving_points = scale_points_to_full_size(
                    moving_points, moving_thumb.shape, moving_bounds
                )

                # print("After scaling to full size:")
                # print("Fixed points:", fixed_points)
                # print("Moving points:", moving_points)

                # Print original points
                # if fixed_scale != moving_scale:
                # print("Original fixed points:", fixed_points)
                # print("Original moving points:", moving_points)

                # Scale fixed points back to original fixed image size
                # fixed_points = [
                #     (x * fixed_scale["x"], y * fixed_scale["y"])
                #     for x, y in fixed_points
                # ]
                # moving_points = [
                #     (x * moving_scale["x"], y * moving_scale["y"])
                #     for x, y in moving_points
                # ]
                print(
                    fixed_points,
                    moving_points,
                    "fixed and moving points...",
                )
                return fixed_points, moving_points
            else:
                return no_update, no_update

        except Exception as e:
            print(f"Error getting tiles info: {str(e)}")
            fixed_bounds = {"width": 10000, "height": 10000}
            moving_bounds = fixed_bounds.copy()

        return no_update, no_update
    else:
        return no_update, no_update


@callback(
    Output("fixed-image-viewer", "inputToPaper"),
    Input("fixedImage_fiducial_points", "data"),
)
def update_fixed_fiducial_points(fixed_points):
    if fixed_points:
        # Generate colors and create GeoJSON features
        colors = generate_distinct_colors(len(fixed_points))
        fixed_items = create_geojson_features(fixed_points, colors, "fixed")
        return {"actions": [{"type": "drawItems", "itemList": fixed_items}]}
    else:
        return {"actions": []}


## image space.
@callback(
    Output("moving-image-viewer", "inputToPaper"),
    Input("movingImage_fiducial_points", "data"),
)
def update_moving_fiducial_points(moving_points):
    if moving_points:
        # Generate colors and create GeoJSON features
        colors = generate_distinct_colors(len(moving_points))
        moving_items = create_geojson_features(moving_points, colors, "moving")
        return {"actions": [{"type": "drawItems", "itemList": moving_items}]}
    else:
        return {"actions": []}


#   Output("fixedImage_fiducial_points", "data"),
#     Output("movingImage_fiducial_points", "data"),


@callback(
    [
        Output("merged-image-viewer", "tileSources"),
        Output("merged-image-viewer", "inputToPaper"),
        Output("merged-image-info", "children"),
    ],
    [
        Input("fixed-image-viewer", "tileSources"),
        Input("moving-image-viewer", "tileSources"),
        Input("fixed-image-viewer", "inputToPaper"),
        Input("moving-image-viewer", "inputToPaper"),
        # Input("moving-image-opacity", "value"),
    ],
)
def update_merged_viewer(fixed_tiles, moving_tiles, fixed_paper, moving_paper):
    """Combine fixed and moving images in the merged viewer"""
    if not fixed_tiles or not moving_tiles:
        return [], {}, ["No tile information available"]
    opacity = 0.5

    try:
        merged_sources = [
            fixed_tiles[0],  # Fixed image as base layer
            {
                **moving_tiles[0],  # Moving image with opacity
                "opacity": opacity,  # Use the slider value
                "compositeOperation": "source-over",  # This controls how images are blended
            },
        ]

        # Combine the paper inputs (registration points) from both viewers
        merged_paper = {"actions": []}
        if fixed_paper and "actions" in fixed_paper:
            merged_paper["actions"].extend(fixed_paper["actions"])
        if moving_paper and "actions" in moving_paper:
            merged_paper["actions"].extend(moving_paper["actions"])

        ### This should contain the merged points for both the fixed and moving images
        # print(merged_paper, "merged_paper items")

        # Get info for the merged viewer
        merged_info = [
            html.Div(
                "Merged View (Moving image opacity: 0.5)",
                style={
                    "whiteSpace": "nowrap",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
            )
        ]

        return merged_sources, merged_paper, merged_info

    except Exception as e:
        print(f"Error in merged viewer: {str(e)}")
        return [], {}, ["Error creating merged view"]


def create_thumbnail_card(slide, selected=False, fixed_slide=False, moving_slide=False):
    """Create a thumbnail card with colored headers for fixed/moving images"""
    stain_id = slide.get("meta", {}).get("npSchema", {}).get("stainID", "Unknown")
    slide_id = slide.get("_id", "")

    # Determine header color based on stain type and selection
    header_style = {
        "padding": "0.25rem 0.5rem",
        "marginBottom": "4px",
    }

    if stain_id.upper() == "HE":
        header_style["backgroundColor"] = "#e3f2fd"  # Light blue for fixed/HE
        header_style["cursor"] = "pointer"
    elif selected:
        header_style["backgroundColor"] = "#c8e6c9"  # Light green for selected
        header_style["cursor"] = "pointer"
    elif stain_id.upper() not in ["HE", "UNKNOWN"]:
        header_style["backgroundColor"] = "#fff3e0"  # Light orange for moving/IHC
        header_style["cursor"] = "pointer"

    return dbc.Card(
        [
            html.Div(
                dbc.CardHeader(
                    [
                        html.P(
                            f"Stain: {stain_id}",
                            className="small mb-0",
                        )
                    ],
                    className="p-1",
                    style=header_style,
                ),
                id={"type": "thumbnail-card", "index": slide_id, "stain": stain_id},
                n_clicks=0,
            ),
            dbc.CardImg(
                src=f"{DSA_BASE_URL}/item/{slide_id}/tiles/thumbnail?token={token_info['_id']}",
                top=True,
                style={
                    "objectFit": "contain",
                    "marginTop": "4px",
                },
            ),
        ],
        className="border-0",
        style={"backgroundColor": "transparent"},
    )


# @callback(
#     Output("caseSetViewer", "children"),
#     Input("selectedRegionData", "data"),
# )
# def update_SelectedRegionSetViewer(slideList):
#     # print(slideList)
#     if slideList:
#         return html.Div(children=[len(slideList), [x["name"] for x in slideList]])
#     else:
#         return html.Div(children=["No slides selected"])

#     # Get slides
#     fixed_slide_id, moving_slide = get_slides_for_registration(
#         slideList, selected_block
#     )
# #     print(he_slide, "-----HE-----", moving_slide, "HE and moving slide...")
#     if not selected_slide_id:
#         selected_slide_id = moving_slide["_id"]

#     available_stains = get_slide_stain_info(filtered_slides)

#     # Create thumbnail cards for each slide
#     thumbnail_cards = [
#         create_thumbnail_card(slide, selected=(slide.get("_id") == selected_slide_id))
#         for slide in filtered_slides
#     ]

#     if "HE" in available_stains:
#         fixed_slide_id = available_stains["HE"]

#     # he_tiles_info = gc.get(f"item/{fixed_slide_id}/tiles")
#     # # print(he_tiles_info)

#     # if selected_slide_id:
#     #     moving_tiles_info = gc.get(f"item/{selected_slide_id}/tiles")

#     return thumbnail_cards, fixed_slide_id, selected_slide_id


# ## The selected region contains all the cases from the same Block ID... this update should trigger when the blockID is changed


###


# dcc.Store(id="selectedRegionData", data=None),

# # Define merged image controls with the modal button
# merged_image_controls = dbc.Row(
#     [
#         dbc.Col(
#             [
#                 html.Label("Opacity:", className="mb-1 small"),
#                 dcc.Slider(
#                     id="moving-image-opacity",
#                     min=0,
#                     max=1,
#                     step=0.1,
#                     value=0.5,
#                     marks=None,
#                     tooltip={"placement": "bottom", "always_visible": True},
#                     className="mt-1 narrow-slider",
#                 ),
#             ],
#             width="auto",
#             style={"width": "150px"},
#         ),
#         dbc.Col(
#             [
#                 html.Label("Step:", className="mb-1 small"),
#                 dbc.Select(
#                     id="offset-step-size",
#                     options=[
#                         {"label": "±1", "value": 1},
#                         {"label": "±10", "value": 10},
#                         {"label": "±100", "value": 100},
#                     ],
#                     value=1,
#                     size="sm",
#                 ),
#             ],
#             width="auto",
#             style={"width": "80px"},
#         ),
#         dbc.Col(
#             [
#                 html.Label("X:", className="mb-1 small"),
#                 dcc.Input(
#                     id="moving-image-x-offset",
#                     type="number",
#                     value=0,
#                     step="any",
#                     className="form-control form-control-sm",
#                     style={"width": "80px"},
#                     inputMode="numeric",
#                     pattern="[0-9]*",
#                 ),
#             ],
#             width="auto",
#         ),
#         dbc.Col(
#             [
#                 html.Label("Y:", className="mb-1 small"),
#                 dcc.Input(
#                     id="moving-image-y-offset",
#                     type="number",
#                     value=0,
#                     step="any",
#                     className="form-control form-control-sm",
#                     style={"width": "80px"},
#                     inputMode="numeric",
#                     pattern="[0-9]*",
#                 ),
#             ],
#             width="auto",
#         ),
#         dbc.Col(
#             [
#                 html.Label("Rot:", className="mb-1 small"),
#                 dcc.Input(
#                     id="moving-image-rotation",
#                     type="number",
#                     value=0,
#                     className="form-control form-control-sm",
#                     style={"width": "80px"},
#                     inputMode="numeric",
#                     pattern="[0-9]*",
#                 ),
#             ],
#             width="auto",
#         ),
#         # Add the modal toggle button
#         dbc.Col(
#             [
#                 html.Label(
#                     "\u00A0", className="mb-1 small d-block"
#                 ),  # Non-breaking space for alignment
#                 dbc.Button(
#                     "Points",
#                     id="open-registration-points-modal",
#                     color="secondary",
#                     size="sm",
#                     className="mt-1",
#                 ),
#             ],
#             width="auto",
#         ),
#     ],
#     className="g-2 align-items-end",
# )


# thumbnail_size_select = [
#     html.Label("Width:", className="mb-1 small"),
#     dbc.Select(
#         id="thumbnail-width-selector",
#         options=[
#             {
#                 "label": "256px",
#                 "value": 256,
#             },
#             {
#                 "label": "512px",
#                 "value": 512,
#             },
#             {
#                 "label": "1024px",
#                 "value": 1024,
#             },
#             {
#                 "label": "2048px",
#                 "value": 2048,
#             },
#         ],
#         value=1024,
#         size="sm",
#     ),
# ]


# # Basic layout for registration controls
# registrationControls_layout = html.Div(
#     [
#         # Add the store component
#         data_stores,
#         # Your existing layout components
#         dbc.Container(
#             [
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             [
#                                 # Static controls section
#                                 dbc.Row(
#                                     [
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Select Case:",
#                                                     className="mb-1 small",
#                                                 ),
#                                                 dbc.Select(
#                                                     id="registration_caseSelect",
#                                                     options=caseList,
#                                                     value=caseList[0]["value"],
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                         ),
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Block ID:", className="mb-1 small"
#                                                 ),
#                                                 dbc.Select(
#                                                     id="registration_blockID_filter_select",
#                                                     options=[{}],
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                         ),
#                                         dbc.Col(
#                                             dbc.Checkbox(
#                                                 id="registration_show_annotated_only",
#                                                 label="Only annotated",
#                                                 value=False,
#                                                 className="ml-3",
#                                             ),
#                                             width="auto",
#                                         ),
#                                         # Feature Detection Method - made narrower
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Method:", className="mb-1 small"
#                                                 ),
#                                                 dbc.Select(
#                                                     id="feature-detection-method",
#                                                     options=[
#                                                         {
#                                                             "label": "ORB (Fast)",
#                                                             "value": "orb",
#                                                         },
#                                                         {
#                                                             "label": "SIFT (Accurate)",
#                                                             "value": "sift",
#                                                         },
#                                                         {
#                                                             "label": "AKAZE",
#                                                             "value": "akaze",
#                                                         },
#                                                         {
#                                                             "label": "Mutual Information",
#                                                             "value": "intensity",
#                                                         },
#                                                     ],
#                                                     value="akaze",
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                             style={
#                                                 "width": "160px"
#                                             },  # Specify exact width
#                                         ),
#                                         # Number of Points - made narrower
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Points:", className="mb-1 small"
#                                                 ),
#                                                 dbc.Select(
#                                                     id="num-points-selector",
#                                                     options=[
#                                                         {"label": "6 pts", "value": 6},
#                                                         {"label": "8 pts", "value": 8},
#                                                         {
#                                                             "label": "12 pts",
#                                                             "value": 12,
#                                                         },
#                                                         {
#                                                             "label": "16 pts",
#                                                             "value": 16,
#                                                         },
#                                                     ],
#                                                     value=8,
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                             style={
#                                                 "width": "90px"
#                                             },  # Specify exact width
#                                         ),
#                                         # Thumbnail Width
#                                         dbc.Col(
#                                             thumbnail_size_select,
#                                             width="auto",
#                                             style={
#                                                 "width": "110px"
#                                             },  # Specify exact width
#                                         ),
#                                     ],
#                                     className="mb-2 g-2 align-items-end",
#                                 ),
#                                 # Thumbnails and merged controls in same row
#                                 dbc.Row(
#                                     [
#                                         # Thumbnail grid with horizontal scroll
#                                         dbc.Col(
#                                             dcc.Loading(
#                                                 id="registration-loading",
#                                                 type="circle",
#                                                 children=[
#                                                     html.Div(
#                                                         id="registration-thumbnail-grid",
#                                                         className="d-flex flex-nowrap gap-2",
#                                                         style={
#                                                             "overflowX": "auto",
#                                                             "whiteSpace": "nowrap",
#                                                             "paddingBottom": "10px",
#                                                             "maxHeight": "200px",
#                                                         },
#                                                     ),
#                                                 ],
#                                             ),
#                                             width=7,
#                                             style={
#                                                 "minWidth": 0,
#                                             },
#                                         ),
#                                         # Merged image controls
#                                         dbc.Col(
#                                             merged_image_controls,
#                                             width=5,
#                                             className="align-self-center",
#                                         ),
#                                     ],
#                                     className="mb-3",
#                                 ),
#                                 # Viewers section
#                                 osdViewer_layout,
#                             ],
#                             className="mb-2",
#                         ),
#                     ],
#                 ),
#                 # Add the modal here
#                 registration_points_modal,  # This is the key addition
#             ],
#             fluid=True,
#             className="px-2",
#         ),
#     ]
# )


# @callback(
#     Output("fixed-image-info", "children"),
#     Output("fixed-image-viewer", "tileSources"),
#     Input("fixed_image_id", "data"),
# )
# def update_fixedImageData(fixed_image_id):
#     ## This updates the metadata, and also the fixed image viewer tile source
#     ## Trying to tie these callbacks to a single variable change to make it more manageable
#     if fixed_image_id:

#         fixed_tile_info = gc.get(f"item/{fixed_image_id}/tiles")
#         return (
#             [
#                 html.Div(
#                     f"Size: {fixed_tile_info.get('sizeX', 'N/A')}×{fixed_tile_info.get('sizeY', 'N/A')}  |  "
#                     f"Resolution: {fixed_tile_info.get('mm_x', 'N/A')}  |  "
#                     f"Magnification: {fixed_tile_info.get('magnification', 'N/A')}"
#                 )
#             ],
#             [
#                 {
#                     "tileSource": f"{DSA_BASE_URL}/item/{fixed_image_id}/tiles/dzi.dzi?token={token_info['_id']}",
#                     "width": fixed_tile_info.get("width", 100000),
#                 }
#             ],
#         )


# # ## This will actually update the tileSourceprops / location for the moving image
# # @callback(

# #         Output("moving-image-x-offset", "value"),
# #         Output("moving-image-y-offset", "value"),
# #         Output("moving-image-rotation", "value"),
# #         Input()


# # Callback to populate thumbnails using the shared caseSlideSet_store
# @callback(
#     Output("registration-thumbnail-grid", "children"),
#     Output("fixed_image_id", "data"),
#     Output("moving_image_id", "data"),
#     [
#         Input("registration_caseSlideSet_store", "data"),
#         Input("registration_blockID_filter_select", "value"),
#         Input("selected-moving-slide", "data"),
#     ],
# )
# def update_registration_thumbnails(slideList, selected_block, selected_slide_id):
#     """Single callback to handle all thumbnail grid updates"""
#     if not slideList or not selected_block:
#         return [], {}, {}, no_update

#     """Extract slide ID and stain ID from slides if available"""

#     def get_slide_stain_info(slides):
#         stain_slideId_dict = {}

#         for slide in slides:
#             stain_id = slide.get("meta", {}).get("npSchema", {}).get("stainID")
#             if stain_id:  # Only return if stainID exists
#                 stain_slideId_dict[stain_id] = slide["_id"]

#         return stain_slideId_dict

#     # Filter slides by block ID
#     filtered_slides = [
#         slide
#         for slide in slideList
#         if slide.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
#     ]

#     # from pprint import pprint

#     # ### Get the stain and ID for the selected set

#     # Get slides
#     fixed_slide_id, moving_slide = get_slides_for_registration(
#         slideList, selected_block
#     )
#     print(he_slide, "-----HE-----", moving_slide, "HE and moving slide...")
#     if not selected_slide_id:
#         selected_slide_id = moving_slide["_id"]

#     available_stains = get_slide_stain_info(filtered_slides)

#     # Create thumbnail cards for each slide
#     thumbnail_cards = [
#         create_thumbnail_card(slide, selected=(slide.get("_id") == selected_slide_id))
#         for slide in filtered_slides
#     ]

#     if "HE" in available_stains:
#         fixed_slide_id = available_stains["HE"]

#     # he_tiles_info = gc.get(f"item/{fixed_slide_id}/tiles")
#     # # print(he_tiles_info)

#     # if selected_slide_id:
#     #     moving_tiles_info = gc.get(f"item/{selected_slide_id}/tiles")

#     return thumbnail_cards, fixed_slide_id, selected_slide_id


# ## CREATE A CALLBACK THAT ACTUALLY MOVES THE MOVING IMAGE TILE SOURCE..


# # ## THIS IS A NIGHTMARE CALLBACK...
# # # Simplified callbacks that use the imported functions
# # @callback(
# #     [
# #         # Output("fixed-image-viewer", "tileSources"),
# #         # Output("moving-image-viewer", "tileSources"),
# #         Output("moving-image-x-offset", "value"),
# #         Output("moving-image-y-offset", "value"),
# #         Output("moving-image-rotation", "value"),
# #         Output("moving-image-info", "children"),
# #         Output("moving-image-metadata", "data"),
# #         Output("fixedImage_fiducial_points", "data"),
# #         Output("movingImage_fiducial_points", "data"),
# #     ],
# #     [
# #         Input("registration_caseSlideSet_store", "data"),
# #         Input("registration_blockID_filter_select", "value"),
# #         Input("feature-detection-method", "value"),
# #         Input("num-points-selector", "value"),
# #         Input("thumbnail-width-selector", "value"),
# #     ],
# #     background=True,
# #     background_callback_manager=background_callback_manager,
# #     allow_duplicates=True,
# # )
# # def setup_registration_images(
# #     slideList,
# #     selected_block,
# #     detection_method,
# #     num_points,
# #     thumbnail_width,
# # ):
# #     # print(f"selected_block: {selected_block}")

# #     # Get slides
# #     he_slide, moving_slide = get_slides_for_registration(slideList, selected_block)
# #     if not he_slide or not moving_slide:
# #         empty_return = 0, 0, 0, [], {}, [], []
# #         return empty_return

# #     # Get image dimensions for both slides
# #     try:
# #         he_tiles_info = gc.get(f"item/{he_slide['_id']}/tiles")
# #         moving_tiles_info = gc.get(f"item/{moving_slide['_id']}/tiles")

# #         fixed_bounds = {
# #             "width": he_tiles_info.get("sizeX", 10000),
# #             "height": he_tiles_info.get("sizeY", 10000),
# #         }
# #         moving_bounds = {
# #             "width": moving_tiles_info.get("sizeX", 10000),
# #             "height": moving_tiles_info.get("sizeY", 10000),
# #         }

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         try:
# #             # Get thumbnails with specified width
# #             fixed_thumb = get_thumbnail_image(he_slide["_id"], width=thumbnail_width)
# #             moving_thumb = get_thumbnail_image(
# #                 moving_slide["_id"], width=thumbnail_width
# #             )

# #             if fixed_thumb is None or moving_thumb is None:
# #                 print("Failed to retrieve thumbnails")
# #                 return (
# #                     # fixed_tile_source,
# #                     # moving_tile_source,
# #                     {"actions": []},
# #                     0,  # Default X offset
# #                     0,  # Default Y offset
# #                     0,  # Default rotation
# #                     [],
# #                     {},
# #                     [],
# #                     [],
# #                     None,
# #                     None,
# #                 )

# #             # Normalize image sizes for feature detection
# #             norm_fixed, norm_moving, (fixed_scale, moving_scale) = (
# #                 normalize_image_sizes(fixed_thumb, moving_thumb)
# #             )

# #             # print(
# #             #     f"Normalized shapes - fixed: {norm_fixed.shape}, moving: {norm_moving.shape}"
# #             # )
# #             # print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

# #             # Generate registration points on normalized images
# #             fixed_points, moving_points, debug_info = create_registration_points(
# #                 norm_fixed,
# #                 norm_moving,
# #                 detection_method,
# #                 num_points,
# #                 fixed_bounds,
# #             )

# #             if fixed_points and moving_points:
# #                 # Print original points
# #                 # if fixed_scale != moving_scale:
# #                 # print("Original fixed points:", fixed_points)
# #                 # print("Original moving points:", moving_points)

# #                 # Scale fixed points back to original fixed image size
# #                 fixed_points = [
# #                     (x * fixed_scale["x"], y * fixed_scale["y"])
# #                     for x, y in fixed_points
# #                 ]
# #                 moving_points = [
# #                     (x * moving_scale["x"], y * moving_scale["y"])
# #                     for x, y in moving_points
# #                 ]

# #                 # print("After scaling to original size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Scale to full image coordinates
# #                 fixed_points = scale_points_to_full_size(
# #                     fixed_points, fixed_thumb.shape, fixed_bounds
# #                 )
# #                 moving_points = scale_points_to_full_size(
# #                     moving_points, moving_thumb.shape, moving_bounds
# #                 )

# #                 # print("After scaling to full size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Convert points to numpy arrays for transformation calculation
# #                 fixed_np = np.float32(fixed_points)
# #                 moving_np = np.float32(moving_points)

# #                 # Calculate transformation matrix
# #                 transform_matrix = cv2.estimateAffinePartial2D(moving_np, fixed_np)[0]

# #                 # Extract transformation parameters
# #                 scale = np.sqrt(
# #                     transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2
# #                 )
# #                 rotation = np.degrees(
# #                     np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
# #                 )
# #                 x_offset = transform_matrix[0, 2]
# #                 y_offset = transform_matrix[1, 2]

# #                 # Generate colors and create GeoJSON features
# #                 colors = generate_distinct_colors(len(fixed_points))
# #                 fixed_items = create_geojson_features(fixed_points, colors, "fixed")
# #                 moving_items = create_geojson_features(
# #                     moving_points, colors, "moving", layer_idx=1
# #                 )

# #                 # print("=== Successful Feature Matching Return ===")
# #                 # print("Fixed Items:", fixed_items)
# #                 # print("Number of items:", len(fixed_items))
# #                 # print(
# #                 #     "First item example:", fixed_items[0] if fixed_items else "No items"
# #                 # )
# #                 # print(
# #                 #     "Paper Input Structure:",
# #                 #     {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                 # )
# #                 # print("=====================================")
# #                 # print(fixed_points, moving_points, "fixed_points, moving_points")
# #                 return (
# #                     # fixed_tile_source,
# #                     # {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                     # moving_tile_source,
# #                     x_offset,  # X offset value
# #                     y_offset,  # Y offset value
# #                     rotation,  # Rotation value
# #                     [
# #                         html.Div(
# #                             f"Size: {moving_tiles_info.get('sizeX', 'N/A')}×{moving_tiles_info.get('sizeY', 'N/A')}  |  "
# #                             f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                             f"Magnification: {moving_tiles_info.get('magnification', 'N/A')}"
# #                         )
# #                     ],
# #                     {
# #                         "sizeX": moving_tiles_info.get("sizeX", 1.0),
# #                         "sizeY": moving_tiles_info.get("sizeY", 1.0),
# #                         f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                         "magnification": moving_tiles_info.get("magnification", "N/A"),
# #                     },
# #                     fixed_points,
# #                     moving_points,
# #                 )

# #         except Exception as e:
# #             print(f"Error in feature matching: {str(e)}")
# #             import traceback

# #             traceback.print_exc()

# #         # Return defaults if registration fails
# #         return (
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #         )

# #     except Exception as e:
# #         print(f"Error getting tiles info: {str(e)}")
# #         fixed_bounds = {"width": 10000, "height": 10000}
# #         moving_bounds = fixed_bounds.copy()

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         # Return defaults if registration fails
# #         return (
# #             # fixed_tile_source,
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #             None,
# #             None,
# #         )


# ## The fixed and moving points are stored in separate stores and can be accessed independent
# ## of all the other operations.. this code simply draws the unadultered points on the original merged or fixed
# ## image space.
# ## MAY BE BETTER TO MERGE THESE T OMAKE SURE COLORS ARE ALWAYS CONSISTENT..


# @callback(
#     [
#         Output("merged-image-viewer", "tileSources"),
#         Output("merged-image-viewer", "inputToPaper"),
#         Output("merged-image-info", "children"),
#     ],
#     [
#         Input("fixed-image-viewer", "tileSources"),
#         Input("moving-image-viewer", "tileSources"),
#         Input("fixed-image-viewer", "inputToPaper"),
#         Input("moving-image-viewer", "inputToPaper"),
#         Input("moving-image-opacity", "value"),
#     ],
# )
# def update_merged_viewer(fixed_tiles, moving_tiles, fixed_paper, moving_paper, opacity):
#     """Combine fixed and moving images in the merged viewer"""
#     if not fixed_tiles or not moving_tiles:
#         return [], {}, ["No tile information available"]

#     try:
#         merged_sources = [
#             fixed_tiles[0],  # Fixed image as base layer
#             {
#                 **moving_tiles[0],  # Moving image with opacity
#                 "opacity": opacity,  # Use the slider value
#                 "compositeOperation": "source-over",  # This controls how images are blended
#             },
#         ]

#         # Combine the paper inputs (registration points) from both viewers
#         merged_paper = {"actions": []}
#         if fixed_paper and "actions" in fixed_paper:
#             merged_paper["actions"].extend(fixed_paper["actions"])
#         if moving_paper and "actions" in moving_paper:
#             merged_paper["actions"].extend(moving_paper["actions"])

#         ### This should contain the merged points for both the fixed and moving images
#         # print(merged_paper, "merged_paper items")

#         # Get info for the merged viewer
#         merged_info = [
#             html.Div(
#                 "Merged View (Moving image opacity: 0.5)",
#                 style={
#                     "whiteSpace": "nowrap",
#                     "overflow": "hidden",
#                     "textOverflow": "ellipsis",
#                 },
#             )
#         ]

#         return merged_sources, merged_paper, merged_info

#     except Exception as e:
#         print(f"Error in merged viewer: {str(e)}")
#         return [], {}, ["Error creating merged view"]


# @callback(
#     Output("merged-image-viewer", "tileSourceProps"),
#     [
#         Input("moving-image-opacity", "value"),
#         Input("moving-image-x-offset", "value"),
#         Input("moving-image-y-offset", "value"),
#         Input("moving-image-rotation", "value"),
#         Input("moving-image-metadata", "data"),
#     ],
# )
# def update_transform(opacity, x_offset, y_offset, rotation, metadata):
#     """Update the moving image transformation based on control values"""
#     try:
#         opacity = float(opacity) if opacity is not None else 1.0
#         x_offset = float(x_offset) if x_offset is not None else 0
#         y_offset = float(y_offset) if y_offset is not None else 0
#         rotation = float(rotation) if rotation is not None else 0

#         # Get scale factor from stored metadata
#         scale_factor = metadata.get("sizeX", 1.0)

#         props = [
#             {
#                 "opacity": opacity,
#                 "x": x_offset,
#                 "y": y_offset,
#                 "rotation": rotation,
#                 "flipped": False,
#                 "scaleFactor": scale_factor,
#                 "index": 0,
#             }
#         ]

#         # print(
#         #     f"Updating transform - Opacity: {opacity}, Scale: {scale_factor}, Rotation: {rotation}, X: {x_offset}, Y: {y_offset}"
#         # )
#         return props

#     except Exception as e:
#         print(f"Error in transform update: {str(e)}")
#         return [
#             {
#                 "opacity": 1.0,
#                 "x": 0,
#                 "y": 0,
#                 "rotation": 0,
#                 "flipped": False,
#                 "scaleFactor": 1.0,
#                 "index": 0,
#             }
#         ]


# clientside_callback(
#     """
#     function(stepSize) {
#         // Update the step attribute of x and y offset inputs
#         const xOffset = document.getElementById('moving-image-x-offset');
#         const yOffset = document.getElementById('moving-image-y-offset');
#         if (xOffset && yOffset) {
#             xOffset.step = stepSize;
#             yOffset.step = stepSize;
#         }
#         return [null, null  ];
#     }
#     """,
#     [Output("moving-image-x-offset", "step"), Output("moving-image-y-offset", "step")],
#     [Input("offset-step-size", "value")],
# )


# @callback(
#     [
#         Output("selected-moving-slide", "data"),
#         Output("registration-loading", "children", allow_duplicate=True),
#     ],
#     Input({"type": "thumbnail-card", "index": ALL, "stain": ALL}, "n_clicks"),
#     State({"type": "thumbnail-card", "index": ALL, "stain": ALL}, "id"),
#     State("selected-moving-slide", "data"),
#     prevent_initial_call=True,
# )
# def handle_thumbnail_click(n_clicks, ids, selected_moving_slide):
#     """Handle clicks on thumbnail card headers"""
#     ## This will also select the moving-slide if the value is not already set...

#     if not ctx.triggered_id:
#         return no_update, no_update

#     # Get the clicked card's ID and stain
#     clicked_id = ctx.triggered_id["index"]
#     clicked_stain = ctx.triggered_id["stain"]

#     def get_first_non_HE_slide(slides_list):
#         """Return the first slide index that isn't HE"""
#         for slide in slides_list:
#             if slide["stain"] != "HE":
#                 return slide["index"]
#         return None  # Return None if no non-HE slides found

#     if not selected_moving_slide:
#         selected_moving_slide = get_first_non_HE_slide(ids)

#     # Only update selection if it's not an HE slide
#     if clicked_stain.upper() != "HE":
#         print(f"Selected moving slide: {clicked_id}")
#         return clicked_id, no_update

#     return no_update, selected_moving_slide


# # # Add callbacks to control the modal and populate its content
# # @callback(
# #     Output("registration-points-modal", "is_open"),
# #     [
# #         Input("open-registration-points-modal", "n_clicks"),
# #         Input("close-registration-points-modal", "n_clicks"),
# #     ],
# #     State("registration-points-modal", "is_open"),
# # )
# # def toggle_modal(n1, n2, is_open):
# #     if n1 or n2:
# #         return not is_open
# #     return is_open


# def format_coord(value):
#     """Helper function to format coordinates"""
#     return f"{value:0.2f}" if isinstance(value, (int, float)) else "N/A"


# def compute_difference(moving_val, fixed_val):
#     """Compute difference between moving and fixed coordinates"""
#     try:
#         if moving_val is not None and fixed_val is not None:
#             diff = moving_val - fixed_val
#             return f"{diff:0.2f}"
#         return "N/A"
#     except:
#         return "N/A"

# # # Add callback to store optimal transform
# # @callback(
# #     Output("optimal-transform-store", "data"),
# #     [
# #         Input("fixed-image-viewer", "inputToPaper"),
# #         Input("moving-image-viewer", "inputToPaper"),
# #     ],
# # )
# # def update_optimal_transform(fixed_points, moving_points):
# #     if not fixed_points or not moving_points:
# #         return None

# #     try:
# #         # Get the points from the viewers
# #         fixed_items = fixed_points.get("actions", [{}])[0].get("itemList", [])
# #         moving_items = moving_points.get("actions", [{}])[0].get("itemList", [])

# #         # Calculate the transform
# #         calc_x, calc_y, calc_rotation = calculate_optimal_transform(
# #             fixed_items, moving_items
# #         )
# #         if calc_x is None:
# #             return None

# #         return {"x": calc_x, "y": calc_y, "rotation": calc_rotation}
# #     except Exception as e:
# #         print(f"Error updating optimal transform: {str(e)}")
# #         return None


# # @callback(
# #     [
# #         Output("moving-image-x-offset", "value", allow_duplicate=True),
# #         Output("moving-image-y-offset", "value", allow_duplicate=True),
# #         Output("moving-image-rotation", "value", allow_duplicate=True),
# #     ],
# #     Input("toggle-transform-button", "n_clicks"),
# #     [
# #         State("moving-image-x-offset", "value"),
# #         State("moving-image-y-offset", "value"),
# #         State("moving-image-rotation", "value"),
# #         State("optimal-transform-store", "data"),
# #     ],
# #     prevent_initial_call=True,
# # )
# # def toggle_transform(
# #     n_clicks, current_x, current_y, current_rotation, optimal_transform
# # ):
# #     if not n_clicks or optimal_transform is None:
# #         return dash.no_update, dash.no_update, dash.no_update

# #     # On odd clicks, switch to optimal transform
# #     if n_clicks % 2 == 1:
# #         print("Switching to optimal transform:", optimal_transform)
# #         return (
# #             optimal_transform["x"],
# #             optimal_transform["y"],
# #             optimal_transform["rotation"],
# #         )
# #     # On even clicks, switch back to previous manual transform
# #     else:
# #         print("Switching to manual transform:", current_x, current_y, current_rotation)
# #         return current_x, current_y, current_rotation

# #     # Get slides
# #     he_slide, moving_slide = get_slides_for_registration(slideList, selected_block)
# #     if not he_slide or not moving_slide:
# #         empty_return = 0, 0, 0, [], {}, [], []
# #         return empty_return


# #         fixed_bounds = {
# #             "width": he_tiles_info.get("sizeX", 10000),
# #             "height": he_tiles_info.get("sizeY", 10000),
# #         }
# #         moving_bounds = {
# #             "width": moving_tiles_info.get("sizeX", 10000),
# #             "height": moving_tiles_info.get("sizeY", 10000),
# #         }

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         try:
# #             # Get thumbnails with specified width
# #             fixed_thumb = get_thumbnail_image(he_slide["_id"], width=thumbnail_width)
# #             moving_thumb = get_thumbnail_image(
# #                 moving_slide["_id"], width=thumbnail_width
# #             )

# #             if fixed_thumb is None or moving_thumb is None:
# #                 print("Failed to retrieve thumbnails")
# #                 return (
# #                     # fixed_tile_source,
# #                     # moving_tile_source,
# #                     {"actions": []},
# #                     0,  # Default X offset
# #                     0,  # Default Y offset
# #                     0,  # Default rotation
# #                     [],
# #                     {},
# #                     [],
# #                     [],
# #                     None,
# #                     None,
# #                 )

# #             # Normalize image sizes for feature detection
# #             norm_fixed, norm_moving, (fixed_scale, moving_scale) = (
# #                 normalize_image_sizes(fixed_thumb, moving_thumb)
# #             )

# #             # print(
# #             #     f"Normalized shapes - fixed: {norm_fixed.shape}, moving: {norm_moving.shape}"
# #             # )
# #             # print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

# #             # Generate registration points on normalized images
# #             fixed_points, moving_points, debug_info = create_registration_points(
# #                 norm_fixed,
# #                 norm_moving,
# #                 detection_method,
# #                 num_points,
# #                 fixed_bounds,
# #             )

# #             if fixed_points and moving_points:
# #                 # Print original points
# #                 # if fixed_scale != moving_scale:
# #                 # print("Original fixed points:", fixed_points)
# #                 # print("Original moving points:", moving_points)

# #                 # Scale fixed points back to original fixed image size
# #                 fixed_points = [
# #                     (x * fixed_scale["x"], y * fixed_scale["y"])
# #                     for x, y in fixed_points
# #                 ]
# #                 moving_points = [
# #                     (x * moving_scale["x"], y * moving_scale["y"])
# #                     for x, y in moving_points
# #                 ]

# #                 # print("After scaling to original size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Scale to full image coordinates
# #                 fixed_points = scale_points_to_full_size(
# #                     fixed_points, fixed_thumb.shape, fixed_bounds
# #                 )
# #                 moving_points = scale_points_to_full_size(
# #                     moving_points, moving_thumb.shape, moving_bounds
# #                 )

# #                 # print("After scaling to full size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Convert points to numpy arrays for transformation calculation
# #                 fixed_np = np.float32(fixed_points)
# #                 moving_np = np.float32(moving_points)

# #                 # Calculate transformation matrix
# #                 transform_matrix = cv2.estimateAffinePartial2D(moving_np, fixed_np)[0]

# #                 # Extract transformation parameters
# #                 scale = np.sqrt(
# #                     transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2
# #                 )
# #                 rotation = np.degrees(
# #                     np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
# #                 )
# #                 x_offset = transform_matrix[0, 2]
# #                 y_offset = transform_matrix[1, 2]

# #                 # Generate colors and create GeoJSON features
# #                 colors = generate_distinct_colors(len(fixed_points))
# #                 fixed_items = create_geojson_features(fixed_points, colors, "fixed")
# #                 moving_items = create_geojson_features(
# #                     moving_points, colors, "moving", layer_idx=1
# #                 )

# #                 # print("=== Successful Feature Matching Return ===")
# #                 # print("Fixed Items:", fixed_items)
# #                 # print("Number of items:", len(fixed_items))
# #                 # print(
# #                 #     "First item example:", fixed_items[0] if fixed_items else "No items"
# #                 # )
# #                 # print(
# #                 #     "Paper Input Structure:",
# #                 #     {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                 # )
# #                 # print("=====================================")
# #                 # print(fixed_points, moving_points, "fixed_points, moving_points")
# #                 return (
# #                     # fixed_tile_source,
# #                     # {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                     # moving_tile_source,
# #                     x_offset,  # X offset value
# #                     y_offset,  # Y offset value
# #                     rotation,  # Rotation value
# #                     [
# #                         html.Div(
# #                             f"Size: {moving_tiles_info.get('sizeX', 'N/A')}×{moving_tiles_info.get('sizeY', 'N/A')}  |  "
# #                             f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                             f"Magnification: {moving_tiles_info.get('magnification', 'N/A')}"
# #                         )
# #                     ],
# #                     {
# #                         "sizeX": moving_tiles_info.get("sizeX", 1.0),
# #                         "sizeY": moving_tiles_info.get("sizeY", 1.0),
# #                         f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                         "magnification": moving_tiles_info.get("magnification", "N/A"),
# #                     },
# #                     fixed_points,
# #                     moving_points,
# #                 )

# #         except Exception as e:
# #             print(f"Error in feature matching: {str(e)}")
# #             import traceback

# #             traceback.print_exc()

# #         # Return defaults if registration fails
# #         return (
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #         )

# #     except Exception as e:
# #         print(f"Error getting tiles info: {str(e)}")
# #         fixed_bounds = {"width": 10000, "height": 10000}
# #         moving_bounds = fixed_bounds.copy()

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         # Return defaults if registration fails
# #         return (
# #             # fixed_tile_source,
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #             None,
# #             None,
# #         )


@callback(
    [
        Output("rotation-input", "value"),
        Output("scale-input", "value"),
        Output("offset-x-input", "value"),
        Output("offset-y-input", "value"),
    ],
    [
        Input("fixedImage_fiducial_points", "data"),
        Input("movingImage_fiducial_points", "data"),
    ],
)
def update_parameters(fixed_points, moving_points):
    if fixed_points and moving_points:
        rotation, scale, offset = compute_affine_parameters(fixed_points, moving_points)
        return rotation, scale, offset[0], offset[1]

    # Return default values if points are not available
    return 0, 1, 0, 0  # Default values for rotation, scale, offset X, offset Y


@callback(
    Output("manual_offset_data", "children"),
    [
        Input("rotation-input", "value"),
        Input("scale-input", "value"),
        Input("offset-x-input", "value"),
        Input("offset-y-input", "value"),
    ],
)
def update_manual_offset_data(rotation, scale, offset_x, offset_y):
    # Create a string representation of the parameters
    parameters = (
        f"Rotation: {rotation:.2f}°, "
        f"Scale: {scale:.2f}, "
        f"Offset: ({offset_x:.2f}, {offset_y:.2f})"
    )

    return parameters
