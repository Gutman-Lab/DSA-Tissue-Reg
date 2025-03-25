import dash
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import cv2
from PIL import Image
import io
import base64
import requests
from settings import gc, DSA_BASE_URL, token_info
import dash_ag_grid
import warnings
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# Suppress warnings about NNPACK
warnings.filterwarnings("ignore", message=".*nnpack.*")

# Initialize the feature extractor and matcher with explicit CPU device
device = torch.device("cpu")  # Force CPU usage
print(f"Using device: {device}")

try:
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    print("Successfully initialized LightGlue models")
except Exception as e:
    print(f"Error initializing LightGlue models: {str(e)}")
    extractor = None
    matcher = None


def get_dsa_image(item_id, width=1024):
    """Fetch image from DSA and convert to numpy array"""
    url = f"{DSA_BASE_URL}/item/{item_id}/tiles/thumbnail?token={token_info['_id']}&width={width}"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return np.array(img)


def array_to_data_url(img_array):
    """Convert numpy array to base64 data URL"""
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


def draw_matches(img0, img1, kpts0, kpts1, matches):
    """Draw matches between two images"""
    # Create a new image by concatenating the two images horizontally
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1

    # Create color version of the images if they're grayscale
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

    # Create the visualization image
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    viz[:h0, :w0] = img0
    viz[:h1, w0 : w0 + w1] = img1

    # Draw matches
    for idx in range(len(matches)):
        # Get the matching keypoints for each image
        kp1 = kpts0[matches[idx, 0]]
        kp2 = kpts1[matches[idx, 1]]

        # Convert to integer coordinates
        x1, y1 = int(kp1[0]), int(kp1[1])
        x2, y2 = int(kp2[0]) + w0, int(kp2[1])  # Add w0 to x2 to shift to second image

        # Draw a line between the matched keypoints
        color = np.random.randint(0, 255, 3).tolist()
        cv2.line(viz, (x1, y1), (x2, y2), color, 1)
        cv2.circle(viz, (x1, y1), 2, color, -1)
        cv2.circle(viz, (x2, y2), 2, color, -1)

    return viz


def match_images(img0, img1):
    """Match two images using LightGlue"""
    if extractor is None or matcher is None:
        raise RuntimeError("LightGlue models not properly initialized")

    try:
        # Convert to torch tensors and move to device
        img0_tensor = torch.from_numpy(img0)[None].to(device) / 255.0
        img1_tensor = torch.from_numpy(img1)[None].to(device) / 255.0

        # Extract features
        with torch.no_grad():  # Add no_grad context for inference
            feats0 = extractor.extract(img0_tensor)
            feats1 = extractor.extract(img1_tensor)

            # Match features
            matches01 = matcher({"image0": feats0, "image1": feats1})

        # Get matching points
        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()
        matches = matches01["matches"][0].cpu().numpy()

        # Draw matches using our custom function
        viz = draw_matches(img0, img1, kpts0, kpts1, matches)
        return viz

    except Exception as e:
        print(f"Error in match_images: {str(e)}")
        raise


# Layout for the LightGlue test component
lightGlue_layout = html.Div(
    [
        # html.H2("LightGlue Feature Matching", className="mb-4"),
        # Image selection table
        html.Div(
            [
                html.H4("Select Images for Registration", className="mb-3"),
                dash_ag_grid.AgGrid(
                    id="lightglue-image-table",
                    columnDefs=[
                        {
                            "headerName": "Select",
                            "checkboxSelection": True,
                            "headerCheckboxSelection": True,
                            "width": 75,
                        },
                        {"headerName": "File Name", "field": "name", "flex": 2},
                        {"headerName": "Stain", "field": "stain"},
                        {"headerName": "Block ID", "field": "block_id"},
                    ],
                    defaultColDef={
                        "flex": 1,
                        "minWidth": 100,
                        "filter": True,
                        "sortable": True,
                    },
                    dashGridOptions={
                        "rowSelection": "multiple",
                        "suppressRowClickSelection": True,
                        "rowMultiSelectWithClick": False,
                    },
                    style={"height": "300px"},
                ),
            ],
            className="mb-4",
        ),
        # Combined row for image previews and results
        html.Div(
            [
                # html.H4("Image Registration", className="mb-3"),
                dbc.Row(
                    [
                        # Fixed image preview
                        dbc.Col(
                            [
                                html.H5("Fixed Image", className="mb-2"),
                                html.Div(
                                    id="fixed-image-preview",
                                    style={
                                        "minHeight": "200px",
                                        "border": "1px solid #dee2e6",
                                        "borderRadius": "4px",
                                        "padding": "10px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                    },
                                ),
                            ],
                            width=3,
                        ),
                        # Moving image preview
                        dbc.Col(
                            [
                                html.H5("Moving Image", className="mb-2"),
                                html.Div(
                                    id="moving-image-preview",
                                    style={
                                        "minHeight": "200px",
                                        "border": "1px solid #dee2e6",
                                        "borderRadius": "4px",
                                        "padding": "10px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                    },
                                ),
                            ],
                            width=3,
                        ),
                        # Feature matching results
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H5("Feature Matches", className="mb-2"),
                                        dbc.Button(
                                            "Run Feature Matching",
                                            id="run-feature-matching",
                                            color="primary",
                                            className="mb-2",
                                        ),
                                        dbc.Spinner(
                                            html.Div(id="feature-matching-status")
                                        ),
                                        html.Div(id="feature-matching-results"),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
            ]
        ),
    ]
)


# Callback to update the image selection table
@callback(
    [
        Output("lightglue-image-table", "rowData"),
        Output("lightglue-image-table", "selectedRows"),
    ],
    Input("currentCaseTable", "rowData"),
)
def update_image_table(case_images):
    print("Trying to update image table")
    if not case_images:
        return [], []

    # Format data for the table
    row_data = []
    for img in case_images:
        row_data.append(
            {
                "id": img["_id"],
                "name": img["name"],
                "stain": img.get("meta", {})
                .get("npSchema", {})
                .get("stainID", "Unknown"),
                "block_id": img.get("meta", {})
                .get("npSchema", {})
                .get("blockID", "Unknown"),
            }
        )

    # Pre-select first two rows
    selected_rows = row_data[:2] if len(row_data) >= 2 else []

    print(f"Created row data with {len(row_data)} rows")
    return row_data, selected_rows


# Callback to get selected images for matching
@callback(
    Output("test-images-store", "data"),
    Input("lightglue-image-table", "selectedRows"),
)
def store_test_images(selected_rows):
    if not selected_rows or len(selected_rows) < 2:
        return None

    # Get the first two selected images
    return {
        "image0_id": selected_rows[0]["id"],
        "image1_id": selected_rows[1]["id"],
        "image0_name": selected_rows[0]["stain"],
        "image1_name": selected_rows[1]["stain"],
    }


# Callback to run feature matching
@callback(
    [
        Output("feature-matching-status", "children"),
        Output("feature-matching-results", "children"),
    ],
    Input("run-feature-matching", "n_clicks"),
    State("lightglue-image-table", "selectedRows"),
    prevent_initial_call=True,
)
def run_feature_matching(n_clicks, selected_rows):
    if not selected_rows or len(selected_rows) < 2:
        return (
            None,
            dbc.Alert(
                "Please select at least two images for matching",
                color="warning",
                dismissable=True,
            ),
        )

    try:
        # Load images
        img0 = get_dsa_image(selected_rows[0]["id"])
        img1 = get_dsa_image(selected_rows[1]["id"])

        # Convert to grayscale if needed
        if len(img0.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # Match images
        result_img = match_images(img0, img1)

        # Convert result to data URL
        img_url = array_to_data_url(result_img)

        return None, html.Div(
            [
                dbc.Alert(
                    f"Successfully matched features between {selected_rows[0]['stain']} and {selected_rows[1]['stain']}",
                    color="success",
                    className="mb-3",
                ),
                html.Img(
                    src=img_url,
                    style={
                        "width": "100%",
                        "maxWidth": "1200px",
                        "border": "1px solid #dee2e6",
                        "borderRadius": "4px",
                    },
                ),
            ]
        )

    except Exception as e:
        return None, dbc.Alert(
            f"Error during feature matching: {str(e)}",
            color="danger",
            dismissable=True,
        )


# Callback to update image previews
@callback(
    [
        Output("fixed-image-preview", "children"),
        Output("moving-image-preview", "children"),
    ],
    Input("lightglue-image-table", "selectedRows"),
)
def update_image_previews(selected_rows):
    if not selected_rows or len(selected_rows) < 2:
        return (
            html.P("Please select fixed image", className="text-muted"),
            html.P("Please select moving image", className="text-muted"),
        )

    # Create thumbnails for both images
    fixed_img = html.Img(
        src=f"{DSA_BASE_URL}/item/{selected_rows[0]['id']}/tiles/thumbnail?token={token_info['_id']}&width=512",
        style={
            "maxWidth": "100%",
            "maxHeight": "200px",
            "objectFit": "contain",
        },
    )
    moving_img = html.Img(
        src=f"{DSA_BASE_URL}/item/{selected_rows[1]['id']}/tiles/thumbnail?token={token_info['_id']}&width=512",
        style={
            "maxWidth": "100%",
            "maxHeight": "200px",
            "objectFit": "contain",
        },
    )

    # Add stain labels
    fixed_preview = html.Div(
        [
            fixed_img,
            html.P(f"Stain: {selected_rows[0]['stain']}", className="mt-2 mb-0"),
        ]
    )
    moving_preview = html.Div(
        [
            moving_img,
            html.P(f"Stain: {selected_rows[1]['stain']}", className="mt-2 mb-0"),
        ]
    )

    return fixed_preview, moving_preview
