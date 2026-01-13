import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
from utils.registration_utils import calculate_optimal_transform, get_thumbnail_image
import numpy as np
from itertools import zip_longest  # Add this for the zip_longest function
import cv2
from PIL import Image
import io
import base64


from itertools import zip_longest


def format_coord(value):
    """Helper function to format coordinates"""
    return f"{value:0.2f}" if isinstance(value, (int, float)) else "N/A"


def compute_difference(moving_val, fixed_val):
    """Compute difference between moving and fixed coordinates"""
    try:
        if moving_val is not None and fixed_val is not None:
            diff = moving_val - fixed_val
            return f"{diff:0.2f}"
        return "N/A"
    except:
        return "N/A"


print("registrationModal.py loaded")


# Helper function to convert numpy array to base64 image
def array_to_base64(img_array):
    """Convert numpy array to base64 string"""
    try:
        # Convert to uint8 if not already
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(img_array)

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Convert to base64
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


# Store component
modal_stores = [
    dcc.Store(id="optimal-transform-store", data=None),
]

# Modal layout
registration_points_modal = dbc.Modal(
    [
        dbc.ModalHeader("Registration Points"),
        dbc.ModalBody(
            [
                # Image Preview Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6("Fixed Image", className="text-center mb-2"),
                                html.Img(
                                    id="modal-fixed-preview",
                                    className="img-fluid mb-3",
                                    style={"maxHeight": "200px"},
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.H6("Moving Image", className="text-center mb-2"),
                                html.Img(
                                    id="modal-moving-preview",
                                    className="img-fluid mb-3",
                                    style={"maxHeight": "200px"},
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.H6("Overlay", className="text-center mb-2"),
                                html.Img(
                                    id="modal-overlay-preview",
                                    className="img-fluid mb-3",
                                    style={"maxHeight": "200px"},
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                # Transform toggle button
                dbc.Button(
                    "Toggle Transform (Manual/Optimal)",
                    id="toggle-transform-button",
                    color="primary",
                    className="mb-4 w-100",
                ),
                # Transform Parameters Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6("Transform Parameters", className="mb-3"),
                                html.Div(
                                    id="registration-transform-info", className="mb-4"
                                ),
                            ]
                        )
                    ]
                ),
                # Matched Points Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6("Matched Points", className="mb-3"),
                                html.Div(
                                    id="registration-points-table", className="mb-4"
                                ),
                            ]
                        )
                    ]
                ),
                # Original Fiducial Points Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6("Original Fiducial Points", className="mb-3"),
                                html.Div(
                                    id="original-fiducial-points", className="mb-4"
                                ),
                            ]
                        )
                    ]
                ),
            ]
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close-registration-points-modal", className="ms-auto"
            )
        ),
    ],
    id="registration-points-modal",
    size="lg",
)


# Add callback to update image previews
@callback(
    [
        Output("modal-fixed-preview", "src"),
        Output("modal-moving-preview", "src"),
        Output("modal-overlay-preview", "src"),
    ],
    [
        Input("registration-points-modal", "is_open"),
        Input("optimal-transform-store", "data"),
        Input("fixed_image_id", "data"),
        Input("moving_image_id", "data"),
    ],
    [
        State("fixed-image-viewer", "inputToPaper"),
    ],
)
def update_preview_images(
    is_open, optimal_transform, fixed_id, moving_id, fixed_points
):
    print("update_preview_images called")
    if not is_open or not fixed_id or not moving_id:
        return no_update, no_update, no_update

    try:
        # Get fixed and moving images using the stored IDs
        fixed_img = get_thumbnail_image(fixed_id, width=512)
        moving_img = get_thumbnail_image(moving_id, width=512)

        print(f"Image shapes - Fixed: {fixed_img.shape}, Moving: {moving_img.shape}")

        # Convert to RGB if grayscale
        if len(fixed_img.shape) == 2:
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2RGB)
        if len(moving_img.shape) == 2:
            moving_img = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2RGB)

        # Apply histogram normalization to grayscale versions
        fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
        moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)

        # Normalize histograms
        fixed_gray = cv2.equalizeHist(fixed_gray)
        moving_gray = cv2.equalizeHist(moving_gray)

        # Create colored versions
        fixed_colored = np.zeros_like(fixed_img)
        moving_colored = np.zeros_like(moving_img)

        # Set red channel for fixed image (using normalized grayscale)
        fixed_colored[:, :, 2] = fixed_gray  # Red channel
        # Set blue channel for moving image (using normalized grayscale)
        moving_colored[:, :, 0] = moving_gray  # Blue channel

        # Convert regular images to base64 (showing original images in preview)
        fixed_b64 = array_to_base64(fixed_img)
        moving_b64 = array_to_base64(moving_img)

        # Create overlay if we have optimal transform
        if optimal_transform:
            print("Applying transform:", optimal_transform)
            # Create transformation matrix
            M = cv2.getRotationMatrix2D(
                (moving_img.shape[1] / 2, moving_img.shape[0] / 2),
                optimal_transform["rotation"],
                1.0,
            )
            M[0, 2] += optimal_transform["x"]
            M[1, 2] += optimal_transform["y"]

            # Apply transformation to the blue version
            transformed_moving = cv2.warpAffine(
                moving_colored,
                M,
                (moving_img.shape[1], moving_img.shape[0]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # Combine images
            overlay = fixed_colored.copy()
            overlay = cv2.add(overlay, transformed_moving)
        else:
            # If no transform, combine the colored versions
            overlay = fixed_colored.copy()
            overlay = cv2.add(overlay, moving_colored)

        # Ensure the overlay is properly visible
        overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX)

        print(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")
        print(
            f"Overlay min/max values per channel: R:{overlay[:,:,2].max()}, G:{overlay[:,:,1].max()}, B:{overlay[:,:,0].max()}"
        )

        overlay_b64 = array_to_base64(overlay)

        return fixed_b64, moving_b64, overlay_b64

    except Exception as e:
        print(f"Error updating preview images: {e}")
        import traceback

        print(traceback.format_exc())
        return no_update, no_update, no_update


@callback(
    Output("registration-points-modal", "is_open"),
    [
        Input("open-registration-points-modal", "n_clicks"),
        Input("close-registration-points-modal", "n_clicks"),
    ],
    State("registration-points-modal", "is_open"),
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    [
        Output("registration-transform-info", "children"),
        Output("registration-points-table", "children"),
        Output("original-fiducial-points", "children"),
    ],
    [
        Input("registration-points-modal", "is_open"),
        State("moving-image-x-offset", "value"),
        State("moving-image-y-offset", "value"),
        State("moving-image-rotation", "value"),
        State("fixed-image-viewer", "inputToPaper"),
        State("moving-image-viewer", "inputToPaper"),
    ],
)
def update_registration_info(
    is_open, x_offset, y_offset, rotation, fixed_points, moving_points
):
    if not is_open:
        return dash.no_update, dash.no_update, dash.no_update

    # Extract points first
    fixed_items = []
    moving_items = []
    if fixed_points and "actions" in fixed_points:
        for action in fixed_points["actions"]:
            if action.get("type") == "drawItems" and "itemList" in action:
                fixed_items = [
                    item
                    for item in action["itemList"]
                    if item.get("type") == "Feature"
                    and item.get("geometry", {}).get("type") == "Point"
                ]

    if moving_points and "actions" in moving_points:
        for action in moving_points["actions"]:
            if action.get("type") == "drawItems" and "itemList" in action:
                moving_items = [
                    item
                    for item in action["itemList"]
                    if item.get("type") == "Feature"
                    and item.get("geometry", {}).get("type") == "Point"
                ]

    # Calculate optimal transform
    calc_x, calc_y, calc_rotation = calculate_optimal_transform(
        fixed_items, moving_items
    )

    # Create transform info display
    transform_info = html.Div(
        [
            html.H6("Current Transform", className="small mb-2"),
            dbc.Table(
                [
                    html.Thead([html.Tr([html.Th("Parameter"), html.Th("Value")])]),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("X Offset"),
                                    html.Td(
                                        f"{x_offset:0.2f}"
                                        if x_offset is not None
                                        else "N/A"
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Y Offset"),
                                    html.Td(
                                        f"{y_offset:0.2f}"
                                        if y_offset is not None
                                        else "N/A"
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Rotation"),
                                    html.Td(
                                        f"{rotation:0.2f}°"
                                        if rotation is not None
                                        else "N/A"
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                size="sm",
                className="mb-4",
            ),
        ]
    )

    # Create points table
    points_table = create_points_table(fixed_items, moving_items)

    # Create original fiducial points display
    original_fiducial_table = create_fiducial_table(fixed_items, moving_items)

    return transform_info, points_table, original_fiducial_table


def create_points_table(fixed_items, moving_items):
    return dbc.Table(
        [
            html.Thead(
                [
                    html.Tr(
                        [
                            html.Th("Point"),
                            html.Th("Fixed X"),
                            html.Th("Fixed Y"),
                            html.Th("Moving X"),
                            html.Th("Moving Y"),
                            html.Th("X Diff", style={"backgroundColor": "#f8f9fa"}),
                            html.Th("Y Diff", style={"backgroundColor": "#f8f9fa"}),
                        ]
                    )
                ]
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(f"Point {i+1}"),
                            html.Td(
                                format_coord(
                                    fixed.get("geometry", {}).get(
                                        "coordinates", [None]
                                    )[0]
                                    if fixed
                                    else None
                                )
                            ),
                            html.Td(
                                format_coord(
                                    fixed.get("geometry", {}).get(
                                        "coordinates", [None, None]
                                    )[1]
                                    if fixed
                                    else None
                                )
                            ),
                            html.Td(
                                format_coord(
                                    moving.get("geometry", {}).get(
                                        "coordinates", [None]
                                    )[0]
                                    if moving
                                    else None
                                )
                            ),
                            html.Td(
                                format_coord(
                                    moving.get("geometry", {}).get(
                                        "coordinates", [None, None]
                                    )[1]
                                    if moving
                                    else None
                                )
                            ),
                            html.Td(
                                compute_difference(
                                    moving.get("geometry", {}).get(
                                        "coordinates", [None]
                                    )[0],
                                    fixed.get("geometry", {}).get(
                                        "coordinates", [None]
                                    )[0],
                                ),
                                style={"backgroundColor": "#f8f9fa"},
                            ),
                            html.Td(
                                compute_difference(
                                    moving.get("geometry", {}).get(
                                        "coordinates", [None, None]
                                    )[1],
                                    fixed.get("geometry", {}).get(
                                        "coordinates", [None, None]
                                    )[1],
                                ),
                                style={"backgroundColor": "#f8f9fa"},
                            ),
                        ]
                    )
                    for i, (fixed, moving) in enumerate(
                        zip_longest(fixed_items, moving_items, fillvalue={})
                    )
                ]
            ),
        ],
        bordered=True,
        size="sm",
        className="mt-3",
    )


def create_fiducial_table(fixed_items, moving_items):
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H6("Fixed Image Points", className="small mb-2"),
                    create_single_fiducial_table(fixed_items),
                ],
                width=6,
            ),
            dbc.Col(
                [
                    html.H6("Moving Image Points", className="small mb-2"),
                    create_single_fiducial_table(moving_items),
                ],
                width=6,
            ),
        ]
    )


def create_single_fiducial_table(items):
    return dbc.Table(
        [
            html.Thead(
                [
                    html.Tr(
                        [
                            html.Th("Point"),
                            html.Th("X"),
                            html.Th("Y"),
                            html.Th("Color"),
                        ]
                    )
                ]
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                f"{item.get('properties', {}).get('name', f'Point {i}')}"
                            ),
                            html.Td(
                                format_coord(
                                    item.get("geometry", {}).get("coordinates", [None])[
                                        0
                                    ]
                                )
                            ),
                            html.Td(
                                format_coord(
                                    item.get("geometry", {}).get(
                                        "coordinates", [None, None]
                                    )[1]
                                )
                            ),
                            html.Td(
                                html.Div(
                                    "",
                                    style={
                                        "backgroundColor": item.get(
                                            "properties", {}
                                        ).get("fillColor", "#000"),
                                        "width": "20px",
                                        "height": "20px",
                                        "borderRadius": "50%",
                                    },
                                )
                            ),
                        ]
                    )
                    for i, item in enumerate(items)
                ]
            ),
        ],
        bordered=True,
        size="sm",
    )


@callback(
    Output("optimal-transform-store", "data"),
    [
        Input("fixed-image-viewer", "inputToPaper"),
        Input("moving-image-viewer", "inputToPaper"),
    ],
)
def update_optimal_transform(fixed_points, moving_points):
    if not fixed_points or not moving_points:
        return None

    try:
        fixed_items = fixed_points.get("actions", [{}])[0].get("itemList", [])
        moving_items = moving_points.get("actions", [{}])[0].get("itemList", [])

        calc_x, calc_y, calc_rotation = calculate_optimal_transform(
            fixed_items, moving_items
        )
        print("Calculated optimal transform:", calc_x, calc_y, calc_rotation)
        return {"x": calc_x, "y": calc_y, "rotation": calc_rotation}
    except Exception as e:
        print(f"Error updating optimal transform: {str(e)}")
        return None


@callback(
    [
        Output("moving-image-x-offset", "value", allow_duplicate=True),
        Output("moving-image-y-offset", "value", allow_duplicate=True),
        Output("moving-image-rotation", "value", allow_duplicate=True),
    ],
    Input("toggle-transform-button", "n_clicks"),
    [
        State("moving-image-x-offset", "value"),
        State("moving-image-y-offset", "value"),
        State("moving-image-rotation", "value"),
        State("optimal-transform-store", "data"),
    ],
    prevent_initial_call=True,
)
def toggle_transform(
    n_clicks, current_x, current_y, current_rotation, optimal_transform
):
    if not n_clicks or optimal_transform is None:
        return no_update, no_update, no_update

    if n_clicks % 2 == 1:
        print("Switching to optimal transform:", optimal_transform)
        return (
            optimal_transform["x"],
            optimal_transform["y"],
            optimal_transform["rotation"],
        )
    else:
        print("Switching to manual transform:", current_x, current_y, current_rotation)
        return current_x, current_y, current_rotation


@callback(
    Output("toggle-transform-button", "children"),
    Input("toggle-transform-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_button_text(n_clicks):
    if not n_clicks:
        return "Switch to Optimal Transform"
    return (
        "Switch to Manual Transform"
        if n_clicks % 2 == 1
        else "Switch to Optimal Transform"
    )
