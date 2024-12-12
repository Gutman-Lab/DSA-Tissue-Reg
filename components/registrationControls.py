import numpy as np
from dash import html, Input, Output, State, callback, dcc
import dash_bootstrap_components as dbc
from settings import gc, memory, DSA_BASE_URL, token_info
import dash_paperdragon
import colorsys
import cv2
import requests
from io import BytesIO
from PIL import Image
import base64
from utils.registration_utils import (
    get_thumbnail_image,
    normalize_image_sizes,
    find_matching_points,
    filter_matches_geometric,
    create_geojson_features,
    scale_points_to_full_size,
    get_slides_for_registration,
    create_registration_points,
    generate_distinct_colors,
    generate_fiducial_points,
    create_thumbnail_card,
)
import time


osdConfig = config = {
    "eventBindings": [
        {"event": "keyDown", "key": "c", "action": "cycleProp", "property": "class"},
        {
            "event": "keyDown",
            "key": "x",
            "action": "cyclePropReverse",
            "property": "class",
        },
        {"event": "keyDown", "key": "d", "action": "deleteItem"},
        {"event": "keyDown", "key": "n", "action": "newItem", "tool": "rectangle"},
        {"event": "keyDown", "key": "e", "action": "editItem", "tool": "rectangle"},
        {"event": "keyDown", "key": "l", "action": "grabColor"},
        {"event": "mouseEnter", "action": "dashCallback", "callback": "mouseEnter"},
        {"event": "mouseLeave", "action": "dashCallback", "callback": "mouseLeave"},
    ],
    "callbacks": [
        {"eventName": "item-created", "callback": "createItem"},
        {"eventName": "property-changed", "callback": "propertyChanged"},
        {"eventName": "item-deleted", "callback": "itemDeleted"},
        {"eventName": "item-edited", "callback": "itemEdited"},
    ],
}

##https://www.sciencedirect.com/science/article/pii/S0010482522000932

# Add viewer components for fixed (HE) and moving images
fixed_image_viewer = dash_paperdragon.DashPaperdragon(
    id="fixed-image-viewer",
    viewerHeight=400,
    viewerWidth=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    config=osdConfig,
)

moving_image_viewer = dash_paperdragon.DashPaperdragon(
    id="moving-image-viewer",
    viewerHeight=400,
    viewerWidth=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    config=osdConfig,
)


merged_image_viewer = dash_paperdragon.DashPaperdragon(
    id="merged-image-viewer",
    viewerHeight=400,
    viewerWidth=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    config=osdConfig,
)


# Add thumbnail grid container
thumbnail_grid = html.Div(
    id="registration-thumbnail-grid", className="d-flex flex-wrap gap-3 mt-3"
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


# Add collapsible thumbnail sections to the layout
thumbnail_sections = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Button(
                    "Show Registration Thumbnails",
                    id="registration-thumbnails-collapse-button",
                    className="mb-3",
                    color="secondary",
                    size="sm",
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Registration Thumbnails", className="card-title"
                                ),
                                html.Div(id="registration-thumbnails-content"),
                            ]
                        )
                    ),
                    id="registration-thumbnails-collapse",
                    is_open=False,
                ),
            ],
            width=6,
        ),
        dbc.Col(
            [
                dbc.Button(
                    "Show Moving Image Thumbnails",
                    id="moving-thumbnails-collapse-button",
                    className="mb-3",
                    color="secondary",
                    size="sm",
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Moving Image Thumbnails", className="card-title"
                                ),
                                html.Div(id="moving-thumbnails-content"),
                            ]
                        )
                    ),
                    id="moving-thumbnails-collapse",
                    is_open=False,
                ),
            ],
            width=6,
        ),
    ],
    className="mb-3",
)

# Define merged image controls as a separate component
merged_image_controls = dbc.Row(
    [
        dbc.Col(
            [
                html.Label("Opacity:", className="mb-1 small"),
                dcc.Slider(
                    id="moving-image-opacity",
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mt-1 narrow-slider",
                ),
            ],
            width="auto",
            style={"width": "150px"},
        ),
        dbc.Col(
            [
                html.Label("X:", className="mb-1 small"),
                dcc.Input(
                    id="moving-image-x-offset",
                    type="number",
                    value=0,
                    className="form-control form-control-sm",
                    style={"width": "80px"},
                ),
            ],
            width="auto",
        ),
        dbc.Col(
            [
                html.Label("Y:", className="mb-1 small"),
                dcc.Input(
                    id="moving-image-y-offset",
                    type="number",
                    value=0,
                    className="form-control form-control-sm",
                    style={"width": "80px"},
                ),
            ],
            width="auto",
        ),
        dbc.Col(
            [
                html.Label("Rot:", className="mb-1 small"),
                dcc.Input(
                    id="moving-image-rotation",
                    type="number",
                    value=0,
                    className="form-control form-control-sm",
                    style={"width": "80px"},
                ),
            ],
            width="auto",
        ),
    ],
    className="g-2 align-items-end",
)

# Basic layout for registration controls
registrationControls_layout = dbc.Container(
    [
        dcc.Store(id="registration_caseId", data="641bfd45867536bb7a236ae1"),
        dcc.Store(id="registration_blockId", data="5"),
        dcc.Store(id="moving-image-metadata", data={}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # html.H3("Registration Controls", className="mb-2"),
                        dbc.Row(
                            [
                                # Feature Detection Method
                                dbc.Col(
                                    [
                                        html.Label("Method:", className="mb-1 small"),
                                        dbc.Select(
                                            id="feature-detection-method",
                                            options=[
                                                {"label": "ORB (Fast)", "value": "orb"},
                                                {
                                                    "label": "SIFT (Accurate)",
                                                    "value": "sift",
                                                },
                                                {"label": "AKAZE", "value": "akaze"},
                                                {"label": "BRISK", "value": "brisk"},
                                                {
                                                    "label": "Canny Edge",
                                                    "value": "canny",
                                                },
                                                {
                                                    "label": "Adaptive",
                                                    "value": "adaptive",
                                                },
                                                {
                                                    "label": "Watershed",
                                                    "value": "watershed",
                                                },
                                                {
                                                    "label": "Mutual Information",
                                                    "value": "intensity",
                                                },
                                            ],
                                            value="akaze",
                                            size="sm",
                                        ),
                                    ],
                                    width=2,
                                ),
                                # Number of Points
                                dbc.Col(
                                    [
                                        html.Label("Points:", className="mb-1 small"),
                                        dbc.Select(
                                            id="num-points-selector",
                                            options=[
                                                {"label": "6 pts", "value": 6},
                                                {"label": "8 pts", "value": 8},
                                                {"label": "12 pts", "value": 12},
                                                {"label": "16 pts", "value": 16},
                                            ],
                                            value=8,
                                            size="sm",
                                        ),
                                    ],
                                    width=1,
                                ),
                                # Thumbnail Width
                                dbc.Col(
                                    [
                                        html.Label("Width:", className="mb-1 small"),
                                        dbc.Select(
                                            id="thumbnail-width-selector",
                                            options=[
                                                {"label": "256px", "value": 256},
                                                {"label": "512px", "value": 512},
                                                {"label": "1024px", "value": 1024},
                                                {"label": "2048px", "value": 2048},
                                            ],
                                            value=1024,
                                            size="sm",
                                        ),
                                    ],
                                    width=1,
                                ),
                                # Registration Parameters
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Constraints:", className="mb-1 small"
                                        ),
                                        dbc.Checklist(
                                            id="registration-constraints",
                                            options=[
                                                {
                                                    "label": "Rotation ±10°",
                                                    "value": "limit_rotation",
                                                },
                                                {
                                                    "label": "Scale ±10%",
                                                    "value": "similar_scale",
                                                },
                                                {
                                                    "label": "Center",
                                                    "value": "center_aligned",
                                                },
                                                {
                                                    "label": "Shape",
                                                    "value": "preserve_shape",
                                                },
                                            ],
                                            value=["limit_rotation", "similar_scale"],
                                            inline=True,
                                            className="small",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ],
                            className="mb-2 g-2 align-items-end",
                        ),
                        dcc.Loading(
                            id="registration-loading",
                            type="circle",
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    id="registration-thumbnail-grid",
                                                    className="d-flex flex-wrap gap-2 mb-3",
                                                )
                                            ]
                                        ),
                                        dbc.Col([merged_image_controls], width=4),
                                    ]
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="viewers-loading",
                            type="circle",
                            children=[
                                dbc.Row(
                                    [
                                        # Fixed viewer column
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    id="fixed-image-info",
                                                    className="image-info mb-2",
                                                ),
                                                html.Div(
                                                    className="viewer-container",
                                                    children=[
                                                        html.Div(
                                                            "FIXED",
                                                            className="viewer-label",
                                                        ),
                                                        fixed_image_viewer,
                                                    ],
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        # Moving viewer column
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    id="moving-image-info",
                                                    className="image-info mb-2",
                                                ),
                                                html.Div(
                                                    className="viewer-container",
                                                    children=[
                                                        html.Div(
                                                            "MOVING",
                                                            className="viewer-label",
                                                        ),
                                                        moving_image_viewer,
                                                    ],
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        # Merged Image Viewer
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    id="merged-image-info",
                                                    className="image-info mb-2",
                                                ),
                                                html.Div(
                                                    className="viewer-container",
                                                    children=[
                                                        html.Div(
                                                            "MERGED",
                                                            className="viewer-label",
                                                        ),
                                                        merged_image_viewer,
                                                    ],
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                            ],
                        ),
                    ],
                    className="mb-2",
                )
            ]
        ),
        # thumbnail_debug_modal,
    ],
    fluid=True,
    className="px-2",
)


# Callback to populate thumbnails using the shared caseSlideSet_store
@callback(
    Output("registration-thumbnail-grid", "children"),
    [Input("caseSlideSet_store", "data"), Input("registration_blockId", "data")],
)
def update_registration_thumbnails(slideList, selected_block):
    if not slideList or not selected_block:
        return []

    # Filter slides by block ID
    filtered_slides = [
        slide
        for slide in slideList
        if slide.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
    ]

    # Create thumbnail cards for each slide
    thumbnail_cards = [create_thumbnail_card(slide) for slide in filtered_slides]

    return thumbnail_cards


# Simplified callbacks that use the imported functions
@callback(
    [
        Output("fixed-image-viewer", "tileSources"),
        Output("fixed-image-viewer", "inputToPaper"),
        Output("moving-image-viewer", "tileSources"),
        Output("moving-image-viewer", "inputToPaper"),
        # Add outputs for the control values
        Output("moving-image-x-offset", "value"),
        Output("moving-image-y-offset", "value"),
        Output("moving-image-rotation", "value"),
    ],
    [
        Input("caseSlideSet_store", "data"),
        Input("registration_blockId", "data"),
        Input("feature-detection-method", "value"),
        Input("num-points-selector", "value"),
        Input("registration-constraints", "value"),
        Input("thumbnail-width-selector", "value"),
    ],
)
def setup_registration_images(
    slideList,
    selected_block,
    detection_method,
    num_points,
    constraints,
    thumbnail_width,
):
    # Get slides
    he_slide, moving_slide = get_slides_for_registration(slideList, selected_block)
    if not he_slide or not moving_slide:
        return [], {}, [], {}, 0, 0, 0

    # Get image dimensions for both slides
    try:
        he_tiles_info = gc.get(f"item/{he_slide['_id']}/tiles")
        moving_tiles_info = gc.get(f"item/{moving_slide['_id']}/tiles")

        print(f"HE tiles info: {he_tiles_info}")
        print(f"Moving tiles info: {moving_tiles_info}")

        fixed_bounds = {
            "width": he_tiles_info.get("sizeX", 10000),
            "height": he_tiles_info.get("sizeY", 10000),
        }
        moving_bounds = {
            "width": moving_tiles_info.get("sizeX", 10000),
            "height": moving_tiles_info.get("sizeY", 10000),
        }

        print(f"Fixed bounds: {fixed_bounds}")
        print(f"Moving bounds: {moving_bounds}")

    except Exception as e:
        print(f"Error getting tiles info: {str(e)}")
        fixed_bounds = {"width": 10000, "height": 10000}
        moving_bounds = fixed_bounds.copy()

    # Create tile sources with correct dimensions
    fixed_tile_source = [
        {
            "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
            "width": fixed_bounds["width"],
            "height": fixed_bounds["height"],  # Add height to tile source
        }
    ]
    moving_tile_source = [
        {
            "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
            "width": moving_bounds["width"],
            "height": moving_bounds["height"],  # Add height to tile source
        }
    ]

    try:
        # Get thumbnails with specified width
        fixed_thumb = get_thumbnail_image(he_slide["_id"], width=thumbnail_width)
        moving_thumb = get_thumbnail_image(moving_slide["_id"], width=thumbnail_width)

        if fixed_thumb is None or moving_thumb is None:
            print("Failed to retrieve thumbnails")
            return (
                fixed_tile_source,
                {"actions": []},
                moving_tile_source,
                {"actions": []},
                0,  # Default X offset
                0,  # Default Y offset
                0,  # Default rotation
            )

        print(f"Using thumbnail width: {thumbnail_width}")
        print(
            f"Thumbnail shapes - fixed: {fixed_thumb.shape}, moving: {moving_thumb.shape}"
        )

        # Normalize image sizes for feature detection
        norm_fixed, norm_moving, (fixed_scale, moving_scale) = normalize_image_sizes(
            fixed_thumb, moving_thumb
        )

        print(
            f"Normalized shapes - fixed: {norm_fixed.shape}, moving: {norm_moving.shape}"
        )
        print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

        # Generate registration points on normalized images
        fixed_points, moving_points, debug_info = create_registration_points(
            norm_fixed,
            norm_moving,
            detection_method,
            num_points,
            constraints,
            fixed_bounds,
        )

        if fixed_points and moving_points:
            # Print original points
            print("Original fixed points:", fixed_points)
            print("Original moving points:", moving_points)

            # Scale fixed points back to original fixed image size
            fixed_points = [
                (x * fixed_scale["x"], y * fixed_scale["y"]) for x, y in fixed_points
            ]
            moving_points = [
                (x * moving_scale["x"], y * moving_scale["y"]) for x, y in moving_points
            ]

            print("After scaling to original size:")
            print("Fixed points:", fixed_points)
            print("Moving points:", moving_points)

            # Scale to full image coordinates
            fixed_points = scale_points_to_full_size(
                fixed_points, fixed_thumb.shape, fixed_bounds
            )
            moving_points = scale_points_to_full_size(
                moving_points, moving_thumb.shape, moving_bounds
            )

            print("After scaling to full size:")
            print("Fixed points:", fixed_points)
            print("Moving points:", moving_points)

            # Convert points to numpy arrays for transformation calculation
            fixed_np = np.float32(fixed_points)
            moving_np = np.float32(moving_points)

            # Calculate transformation matrix
            transform_matrix = cv2.estimateAffinePartial2D(moving_np, fixed_np)[0]

            # Extract transformation parameters
            scale = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
            rotation = np.degrees(
                np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            )
            x_offset = transform_matrix[0, 2]
            y_offset = transform_matrix[1, 2]

            # Generate colors and create GeoJSON features
            colors = generate_distinct_colors(len(fixed_points))
            fixed_items = create_geojson_features(fixed_points, colors, "fixed")
            moving_items = create_geojson_features(moving_points, colors, "moving")

            return (
                fixed_tile_source,
                {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
                moving_tile_source,
                {"actions": [{"type": "drawItems", "itemList": moving_items}]},
                x_offset,  # X offset value
                y_offset,  # Y offset value
                rotation,  # Rotation value
            )

    except Exception as e:
        print(f"Error in feature matching: {str(e)}")
        import traceback

        traceback.print_exc()

    # Return defaults if registration fails
    return (
        fixed_tile_source,
        {"actions": []},
        moving_tile_source,
        {"actions": []},
        0,  # Default X offset
        0,  # Default Y offset
        0,  # Default rotation
    )


# Add new callback for metadata
@callback(
    [
        Output("fixed-image-metadata", "children"),
        Output("moving-image-metadata", "children"),
    ],
    [Input("caseSlideSet_store", "data"), Input("registration_blockId", "data")],
)
def update_image_metadata(slideList, selected_block):
    if not slideList or not selected_block:
        return [], []

    filtered_slides = [
        slide
        for slide in slideList
        if slide.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
    ]

    he_slide = next(
        (
            slide
            for slide in filtered_slides
            if slide.get("meta", {}).get("npSchema", {}).get("stainID", "").upper()
            == "HE"
        ),
        None,
    )
    moving_slide = next(
        (
            slide
            for slide in filtered_slides
            if slide.get("meta", {}).get("npSchema", {}).get("stainID", "").upper()
            != "HE"
        ),
        None,
    )

    if not he_slide or not moving_slide:
        return [], []

    def get_metadata_component(slide, title):
        try:
            tiles_info = gc.get(f"item/{slide['_id']}/tiles")
            return html.Div(
                [
                    html.H5(title),
                    html.P(f"Magnification: {tiles_info.get('magnification', 'N/A')}"),
                    html.P(f"Size X: {tiles_info.get('sizeX', 'N/A')}"),
                    html.P(f"Size Y: {tiles_info.get('sizeY', 'N/A')}"),
                    html.P(f"mm_x: {tiles_info.get('mm_x', 'N/A')}"),
                ]
            )
        except Exception as e:
            return html.Div(
                [html.H5(title), html.P(f"Error fetching metadata: {str(e)}")]
            )

    fixed_metadata = get_metadata_component(he_slide, "Fixed Image Metadata")
    moving_metadata = get_metadata_component(moving_slide, "Moving Image Metadata")

    return fixed_metadata, moving_metadata


# Add new callback for thumbnail display
@callback(
    [
        Output("fixed-thumbnail-display", "src"),
        Output("moving-thumbnail-display", "src"),
    ],
    [
        Input("caseSlideSet_store", "data"),
        Input("registration_blockId", "data"),
    ],
)
def update_thumbnail_display(slideList, selected_block):
    if not slideList or not selected_block:
        return "", ""

    # Filter slides
    filtered_slides = [
        slide
        for slide in slideList
        if slide.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
    ]

    # Find HE and moving slides
    he_slide = next(
        (
            slide
            for slide in filtered_slides
            if slide.get("meta", {}).get("npSchema", {}).get("stainID", "").upper()
            == "HE"
        ),
        None,
    )
    moving_slide = next(
        (
            slide
            for slide in filtered_slides
            if slide.get("meta", {}).get("npSchema", {}).get("stainID", "").upper()
            != "HE"
        ),
        None,
    )

    if not he_slide or not moving_slide:
        return "", ""

    try:
        # Get thumbnails
        fixed_thumb = get_thumbnail_image(he_slide["_id"])
        moving_thumb = get_thumbnail_image(moving_slide["_id"])

        # Convert to data URLs
        fixed_url = array_to_data_url(fixed_thumb)
        moving_url = array_to_data_url(moving_thumb)

        return fixed_url, moving_url

    except Exception as e:
        print(f"Error displaying thumbnails: {str(e)}")
        return "", ""


# Add callbacks to toggle the collapses
@callback(
    Output("registration-thumbnails-collapse", "is_open"),
    [Input("registration-thumbnails-collapse-button", "n_clicks")],
    [State("registration-thumbnails-collapse", "is_open")],
)
def toggle_registration_thumbnails(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("moving-thumbnails-collapse", "is_open"),
    [Input("moving-thumbnails-collapse-button", "n_clicks")],
    [State("moving-thumbnails-collapse", "is_open")],
)
def toggle_moving_thumbnails(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Add callbacks to update the thumbnail content
@callback(
    Output("registration-thumbnails-content", "children"),
    [Input("fixed-image-viewer", "inputToPaper")],
)
def update_registration_thumbnails(fixed_paper):
    if not fixed_paper or "debug_info" not in fixed_paper:
        return "No thumbnail data available"

    debug_info = fixed_paper["debug_info"]
    return html.Div(
        [
            html.P(
                f"Total keypoints: {debug_info['stats'].get('total_keypoints', (0, 0))[0]}"
            ),
            html.P(f"Total matches: {debug_info['stats'].get('total_matches', 0)}"),
            html.P(f"RANSAC inliers: {debug_info['stats'].get('ransac_inliers', 0)}"),
        ]
    )


@callback(
    Output("moving-thumbnails-content", "children"),
    [Input("moving-image-viewer", "inputToPaper")],
)
def update_moving_thumbnails(moving_paper):
    if not moving_paper or "debug_info" not in moving_paper:
        return "No thumbnail data available"

    debug_info = moving_paper["debug_info"]
    return html.Div(
        [
            html.P(
                f"Total keypoints: {debug_info['stats'].get('total_keypoints', (0, 0))[1]}"
            ),
            html.P(f"Total matches: {debug_info['stats'].get('total_matches', 0)}"),
            html.P(f"RANSAC inliers: {debug_info['stats'].get('ransac_inliers', 0)}"),
        ]
    )


# # Add callback for modal control
# @callback(
#     Output("thumbnail-debug-modal", "is_open"),
#     [
#         Input("open-thumbnail-modal", "n_clicks"),
#         Input("close-thumbnail-modal", "n_clicks"),
#     ],
#     [State("thumbnail-debug-modal", "is_open")],
# )
# def toggle_modal(n1, n2, is_open):
#     if n1 or n2:
#         return not is_open
#     return is_open


# Add callbacks for modal content
@callback(
    Output("modal-fixed-debug-content", "children"),
    [Input("fixed-image-viewer", "inputToPaper")],
)
def update_fixed_debug_content(fixed_paper):
    if not fixed_paper or "debug_info" not in fixed_paper:
        return "No debug information available"

    debug_info = fixed_paper.get("debug_info", {})
    return dbc.Table(
        [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Total Keypoints:"),
                            html.Td(
                                f"{debug_info.get('stats', {}).get('total_keypoints', (0, 0))[0]}"
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Total Matches:"),
                            html.Td(
                                f"{debug_info.get('stats', {}).get('total_matches', 0)}"
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("RANSAC Inliers:"),
                            html.Td(
                                f"{debug_info.get('stats', {}).get('ransac_inliers', 0)}"
                            ),
                        ]
                    ),
                ]
            )
        ],
        bordered=True,
        hover=True,
        size="sm",
    )


@callback(
    Output("modal-moving-debug-content", "children"),
    [Input("moving-image-viewer", "inputToPaper")],
)
def update_moving_debug_content(moving_paper):
    if not moving_paper or "debug_info" not in moving_paper:
        return "No debug information available"

    debug_info = moving_paper.get("debug_info", {})
    return dbc.Table(
        [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Total Keypoints:"),
                            html.Td(
                                f"{debug_info.get('stats', {}).get('total_keypoints', (0, 0))[1]}"
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Total Matches:"),
                            html.Td(
                                f"{debug_info.get('stats', {}).get('total_matches', 0)}"
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("RANSAC Inliers:"),
                            html.Td(
                                f"{debug_info.get('stats', {}).get('ransac_inliers', 0)}"
                            ),
                        ]
                    ),
                ]
            )
        ],
        bordered=True,
        hover=True,
        size="sm",
    )


# Optional: Add a text indicator for more explicit status
@callback(
    Output("registration-loading", "children"),
    [Input("feature-detection-method", "value"), Input("num-points-selector", "value")],
    prevent_initial_call=True,
)
def update_loading_status(method, num_points):
    """Show loading status while registration is running"""
    # This will trigger the loading spinner
    time.sleep(0.1)  # Small delay to ensure spinner shows
    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id="registration-thumbnail-grid",
                            className="d-flex flex-wrap gap-2 mb-3",
                        )
                    ]
                )
            ]
        )
    ]


@callback(
    [
        Output("fixed-image-info", "children"),
        Output("moving-image-info", "children"),
        Output("moving-image-metadata", "data"),
    ],
    [
        Input("fixed-image-viewer", "tileSources"),
        Input("moving-image-viewer", "tileSources"),
    ],
)
def update_image_info(fixed_tiles, moving_tiles):
    """Update the image information display and store metadata"""
    if not fixed_tiles or not moving_tiles:
        return "", "", {}

    def get_girder_info(tiles):
        if isinstance(tiles, list) and len(tiles) > 0:
            tile_source = tiles[0].get("tileSource", "")
            item_id = tile_source.split("/item/")[1].split("/")[0]

            try:
                tile_info = gc.get(f"item/{item_id}/tiles")
                return tile_info
            except Exception as e:
                print(f"Error getting tile info: {e}")
                return {}
        return {}

    fixed_info = get_girder_info(fixed_tiles)
    moving_info = get_girder_info(moving_tiles)

    # Store moving image metadata
    moving_metadata = {
        "sizeX": moving_info.get("sizeX", 1.0),
        "sizeY": moving_info.get("sizeY", 1.0),
        "magnification": moving_info.get("magnification", "N/A"),
    }

    # Create display info
    fixed_display = [
        html.Div(
            f"Size: {fixed_info.get('sizeX', 'N/A')}×{fixed_info.get('sizeY', 'N/A')}  |  "
            f"Magnification: {fixed_info.get('magnification', 'N/A')}"
        )
    ]

    moving_display = [
        html.Div(
            f"Size: {moving_info.get('sizeX', 'N/A')}×{moving_info.get('sizeY', 'N/A')}  |  "
            f"Magnification: {moving_info.get('magnification', 'N/A')}"
        )
    ]

    return fixed_display, moving_display, moving_metadata


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
        Input("moving-image-opacity", "value"),
    ],
)
def update_merged_viewer(fixed_tiles, moving_tiles, fixed_paper, moving_paper, opacity):
    """Combine fixed and moving images in the merged viewer"""
    if not fixed_tiles or not moving_tiles:
        return [], {}, ["No tile information available"]

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


@callback(
    Output("merged-image-viewer", "tileSourceProps"),
    [
        Input("moving-image-opacity", "value"),
        Input("moving-image-x-offset", "value"),
        Input("moving-image-y-offset", "value"),
        Input("moving-image-rotation", "value"),
        Input("moving-image-metadata", "data"),
    ],
)
def update_transform(opacity, x_offset, y_offset, rotation, metadata):
    """Update the moving image transformation based on control values"""
    try:
        opacity = float(opacity) if opacity is not None else 1.0
        x_offset = float(x_offset) if x_offset is not None else 0
        y_offset = float(y_offset) if y_offset is not None else 0
        rotation = float(rotation) if rotation is not None else 0

        # Get scale factor from stored metadata
        scale_factor = metadata.get("sizeX", 1.0)

        props = [
            {
                "opacity": opacity,
                "x": x_offset,
                "y": y_offset,
                "rotation": rotation,
                "flipped": False,
                "scaleFactor": scale_factor,
                "index": 0,
            }
        ]

        print(
            f"Updating transform - Opacity: {opacity}, Scale: {scale_factor}, Rotation: {rotation}, X: {x_offset}, Y: {y_offset}"
        )
        return props

    except Exception as e:
        print(f"Error in transform update: {str(e)}")
        return [
            {
                "opacity": 1.0,
                "x": 0,
                "y": 0,
                "rotation": 0,
                "flipped": False,
                "scaleFactor": 1.0,
                "index": 0,
            }
        ]


# def create_tile_source_props(r, index=None):
#     """Create tile source properties for viewer updates"""
#     try:
#         defaults = get_default_tilesource_props()
#         sizeX = float(r.get("sizeX", 1)) or 1  # Use 1 if sizeX is 0 or None

#         imgTileSource = f"{r['apiUrl']}/item/{r['_id']}/tiles/dzi.dzi"
#         props = {
#             "tileSource": imgTileSource,
#             "opacity": (
#                 float(r.get("opacity", defaults["opacity"]))
#                 if r.get("isVisible", defaults["isVisible"])
#                 else 0
#             ),
#             "x": float(r.get("xOffset", defaults["x"])) / sizeX,
#             "y": float(r.get("yOffset", defaults["y"])) / sizeX,
#             "rotation": float(r.get("rotation", defaults["rotation"])),
#             "flipped": bool(r.get("flipped", defaults["flipped"])),
#             "scaleFactor": float(r.get("scaleFactor", defaults["scaleFactor"])),
#             "tileSource": imgTileSource,
#         }

#         if index is not None:
#             props["index"] = index

#         return props
