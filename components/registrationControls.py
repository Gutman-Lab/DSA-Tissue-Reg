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

# import dash_core_components as dcc

from pprint import pprint

import dash

import dash_bootstrap_components as dbc
from settings import gc, memory, DSA_BASE_URL, token_info
import cv2
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
                    min=-360,
                    max=360,
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Scale:"),
                dcc.Input(
                    id="scale-input",
                    type="number",
                    value=1,
                    step=0.01,
                    min=0,
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Offset X:"),
                dcc.Input(
                    id="offset-x-input",
                    type="number",
                    value=0,
                    step=1,
                ),
            ],
            width=4,
        ),
        dbc.Col(
            [
                html.Label("Offset Y:"),
                dcc.Input(
                    id="offset-y-input",
                    type="number",
                    value=0,
                    step=1,
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
        dcc.Store(id="moving-image-info_store", data={}),
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
    Output("moving-image-info_store", "data"),
    Input("moving_image_id", "data"),
)
def update_movingImageData(moving_image_id):
    ## This updates the metadata, and also the fixed image viewer tile source
    ## Trying to tie these callbacks to a single variable change to make it more manageable
    if moving_image_id:

        ## NEED TO MAKE SURE THIS IMAGE IS IN A STORE AS WELL

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
            moving_tile_info,
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


### This callback for the merged-image viewer allows me to rotate the moving image
## based on the control inputs from the registration controls.
@callback(
    Output("merged-image-viewer", "tileSourceProps"),
    Input("rotation-input", "value"),
    Input("scale-input", "value"),
    Input("offset-x-input", "value"),
    Input("offset-y-input", "value"),
    Input("moving-image-info_store", "data"),
)
def update_merged_viewer_tile_source_props(
    rotation, scale, offset_x, offset_y, moving_image_info
):

    ## IF ANY OF THE PROPERTIES ARE INVALID, JUST CATCH THEM HERE AND RETURN A NO UPDATE
    if rotation is None or scale is None or offset_x is None or offset_y is None:
        return no_update

    ## Make sure all the properties are numbers
    if rotation is not None:
        rotation = float(rotation)
    if scale is not None:
        scale = float(scale)
        if scale < 0:
            scale = 1

        scale = moving_image_info.get("width", 100000) * scale
    if offset_x is not None:
        offset_x = float(offset_x)
    if offset_y is not None:
        offset_y = float(offset_y)

    tile_source_props = {}
    try:
        if rotation is not None:
            tile_source_props["rotation"] = float(rotation)
        if scale is not None:
            tile_source_props["scaleFactor"] = float(scale)
        if offset_x is not None:
            tile_source_props["x"] = float(offset_x)
        if offset_y is not None:
            tile_source_props["y"] = float(offset_y)
    except Exception as e:
        print(f"Error in tile source props: {str(e)}")
        return no_update

    ## SO I AM ACTUALLY NOT SETTING THE SCAELFACTOR BUT THE ACTUAL WIDTH
    ## OF THE SECOND CONTAINER.. FOR NOW I WILL JUST MANAGE THE TRANSLATION IN THIS CODE

    print("generated tile source props", tile_source_props)
    return [{}, tile_source_props]


# REMEMBER I PROBABLY WANT TO PASS AN ARRAY, and have a null value for the first
## image since it is the fixed image.

#    if (curTileSource) {
#           props = tileSourceProps[i];

#           //Updating opacity, position and rotation...
#           //Future state could only update properties that have changed, but these operations seem
#           //fast enough that it may not be necessary
#           // console.log(props);
#           if (props.opacity !== undefined) curTileSource.setOpacity(props.opacity);
#           if (props.x !== undefined && props.y !== undefined) curTileSource.setPosition({ x: props.x, y: props.y })
#           if (props.rotation !== undefined) curTileSource.setRotation(props.rotation);
#           if (props.flipped !== undefined) curTileSource.setFlip(props.flipped);
#           if (props.compositeOperation !== undefined) curTileSource.setCompositeOperation(props.compositeOperation);
#           if (props.scaleFactor !== undefined) curTileSource.setWidth(props.scaleFactor);
#           // TO DO IS AGAIN DOUBLE CHECK THE SCALE FACTOR OR WIDTH IS CORRECT

#         }
#       }


## Tile sourcep roperties apply a shim without forcing the tile source to completely reload


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
    # Provide default values if any parameter is None
    rotation = rotation if rotation is not None else 0
    scale = scale if scale is not None else 1
    offset_x = offset_x if offset_x is not None else 0
    offset_y = offset_y if offset_y is not None else 0

    # Create a string representation of the parameters
    parameters = (
        f"Rotation: {rotation:.2f}°, "
        f"Scale: {scale:.2f}, "
        f"Offset: ({offset_x:.2f}, {offset_y:.2f})"
    )

    return parameters
