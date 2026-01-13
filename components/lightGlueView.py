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
from settings import gc, DSA_BASE_URL, token_info, memory
import dash_ag_grid
import warnings
import torch
from sklearn.metrics import mutual_info_score
from scipy.interpolate import RBFInterpolator

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


def calculate_mutual_information(img1, img2, bins=20):
    """Calculate mutual information between two images"""
    # Ensure images are same size and grayscale
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Flatten the images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Calculate histogram
    hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)

    # Convert to probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x
    py = np.sum(pxy, axis=0)  # marginal for y

    # Calculate mutual information
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # Only include non-zero elements

    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


def filter_matches(kpts0, kpts1, matches, scores, min_score=0.5):
    """Filter matches by score and remove potential outliers"""
    # Filter by score
    mask = scores > min_score
    matches_filtered = matches[mask]
    scores_filtered = scores[mask]

    if len(matches_filtered) < 4:  # Need at least 4 points for a reasonable transform
        return matches, scores  # Return original if we don't have enough matches

    # Get the filtered point pairs
    pts0 = kpts0[matches_filtered[:, 0]]
    pts1 = kpts1[matches_filtered[:, 1]]

    # Use RANSAC to find a homography and identify inliers
    H, inliers = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

    if H is None:
        return matches_filtered, scores_filtered

    # Keep only inlier matches
    matches_filtered = matches_filtered[inliers.ravel().astype(bool)]
    scores_filtered = scores_filtered[inliers.ravel().astype(bool)]

    return matches_filtered, scores_filtered


def apply_tps_warp(src_img, src_points, dst_points, target_shape):
    """Apply Thin Plate Spline-like warping to the source image using RBF interpolation"""
    # Convert points to numpy arrays if they aren't already
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)

    # First, estimate a homography for global transform
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    if H is None:
        print(
            "Warning: Could not estimate homography, falling back to centroid-based scaling"
        )
        # Fall back to centroid-based scaling
        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)
        src_spread = np.mean(np.linalg.norm(src_points - src_centroid, axis=1))
        dst_spread = np.mean(np.linalg.norm(dst_points - dst_centroid, axis=1))
        global_scale = dst_spread / src_spread if src_spread > 0 else 1.0

        # Apply global transform
        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid
        src_scaled = src_centered * global_scale

        # Create coordinate grid
        h, w = target_shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        coords = np.vstack((x.ravel(), y.ravel())).T
        coords_centered = coords - src_centroid
        coords_scaled = coords_centered * global_scale

        # Create RBF interpolator for residual deformation
        rbf = RBFInterpolator(
            src_scaled, dst_centered, kernel="thin_plate_spline", smoothing=1.0
        )
        warped_coords = rbf(coords_scaled)
        final_coords = warped_coords + dst_centroid

    else:
        # Apply homography to get initial global transform
        h, w = target_shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        coords = np.vstack((x.ravel(), y.ravel())).T

        # Apply homography to all points
        coords_h = np.column_stack([coords, np.ones(len(coords))])
        warped_h = coords_h @ H.T
        warped_coords = warped_h[:, :2] / warped_h[:, 2:]

        # Calculate residual displacements
        src_warped = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H).reshape(
            -1, 2
        )

        # Create RBF interpolator for residual deformation
        rbf = RBFInterpolator(
            src_warped,
            dst_points - src_warped,
            kernel="thin_plate_spline",
            smoothing=1.0,
        )
        residual = rbf(warped_coords)
        final_coords = warped_coords + residual

    # Reshape for cv2.remap
    map_x = final_coords[:, 0].reshape(h, w)
    map_y = final_coords[:, 1].reshape(h, w)

    # Apply the warping
    warped = cv2.remap(
        src_img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return warped


# Update match_images function to handle color
@memory.cache
def match_images(img0_color, img1_color):
    """Match two images using LightGlue and calculate mutual information"""
    if extractor is None or matcher is None:
        raise RuntimeError("LightGlue models not properly initialized")

    try:
        # Convert to grayscale for feature matching while keeping color originals
        img0_gray = (
            cv2.cvtColor(img0_color, cv2.COLOR_RGB2GRAY)
            if len(img0_color.shape) == 3
            else img0_color
        )
        img1_gray = (
            cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY)
            if len(img1_color.shape) == 3
            else img1_color
        )

        # Convert to torch tensors and move to device
        img0_tensor = torch.from_numpy(img0_gray)[None].to(device) / 255.0
        img1_tensor = torch.from_numpy(img1_gray)[None].to(device) / 255.0

        # Extract features
        with torch.no_grad():
            feats0 = extractor.extract(img0_tensor)
            feats1 = extractor.extract(img1_tensor)

            # Match features
            matches01 = matcher({"image0": feats0, "image1": feats1})

        # Get matching points
        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()
        matches = matches01["matches"][0].cpu().numpy()
        scores = matches01["scores"][0].cpu().numpy()

        # Filter matches
        matches_filtered, scores_filtered = filter_matches(
            kpts0, kpts1, matches, scores, min_score=0.7
        )

        # Get matched point pairs
        src_points = kpts1[matches_filtered[:, 1]]  # Points from moving image
        dst_points = kpts0[matches_filtered[:, 0]]  # Points from fixed image

        # Apply warping
        warped_img1 = apply_tps_warp(
            img1_color, src_points, dst_points, img0_color.shape
        )

        # Calculate mutual information using grayscale versions
        mi_score = calculate_mutual_information(
            img0_gray,
            (
                cv2.cvtColor(warped_img1, cv2.COLOR_RGB2GRAY)
                if len(warped_img1.shape) == 3
                else warped_img1
            ),
        )

        # Create visualization of original matches using filtered matches
        viz = draw_matches(img0_color, img1_color, kpts0, kpts1, matches_filtered)

        # Create side-by-side comparison with labels
        h, w = img0_color.shape[:2]
        padding = np.zeros((h, 50, 3), dtype=np.uint8) + 255  # white padding
        comparison = np.hstack([img0_color, padding, warped_img1])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Fixed", (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(comparison, "Warped", (w + 60, 30), font, 1, (0, 0, 0), 2)

        # Create diagnostic overlay
        overlay = img0_color.copy()
        alpha = 0.5
        overlay_warped = cv2.addWeighted(overlay, alpha, warped_img1, 1 - alpha, 0)

        # Add overlay to comparison
        comparison = np.hstack([comparison, padding, overlay_warped])
        cv2.putText(comparison, "Overlay", (2 * w + 110, 30), font, 1, (0, 0, 0), 2)

        # Prepare match data for storage
        match_data = []
        for idx in range(len(matches_filtered)):
            match_data.append(
                {
                    "match_idx": idx,
                    "score": float(scores_filtered[idx]),
                    "img0_x": float(kpts0[matches_filtered[idx, 0]][0]),
                    "img0_y": float(kpts0[matches_filtered[idx, 0]][1]),
                    "img1_x": float(kpts1[matches_filtered[idx, 1]][0]),
                    "img1_y": float(kpts1[matches_filtered[idx, 1]][1]),
                }
            )

        # Add mutual information score and match statistics
        if match_data:
            match_data[0]["mutual_info"] = float(mi_score)
            match_data[0]["total_matches"] = len(matches)
            match_data[0]["filtered_matches"] = len(matches_filtered)

        return viz, comparison, match_data

    except Exception as e:
        print(f"Error in match_images: {str(e)}")
        raise


# Layout for the LightGlue test component
lightGlue_layout = html.Div(
    [
        dcc.Store(id="match-data-store", data=[]),
        # Image selection dropdowns and button row
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Fixed Image", className="mb-2"),
                        dcc.Dropdown(
                            id="fixed-image-dropdown",
                            placeholder="Select fixed image...",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Moving Image", className="mb-2"),
                        dcc.Dropdown(
                            id="moving-image-dropdown",
                            placeholder="Select moving image...",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label(
                            "\u00a0", className="mb-2"
                        ),  # Non-breaking space for alignment
                        dbc.Button(
                            "Run Feature Matching",
                            id="run-feature-matching",
                            color="primary",
                            className="w-100",  # Make button full width
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-4 align-items-end",  # align-items-end aligns everything to bottom
        ),
        # Match data table and status message row
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Match Details", className="mb-3"),
                        html.Div(
                            [
                                dash_ag_grid.AgGrid(
                                    id="match-data-table",
                                    columnDefs=[
                                        {
                                            "headerName": "Match #",
                                            "field": "match_idx",
                                            "width": 100,
                                        },
                                        {
                                            "headerName": "Score",
                                            "field": "score",
                                            "width": 100,
                                        },
                                        {
                                            "headerName": "MI Before",
                                            "field": "mutual_info_before",
                                            "width": 120,
                                        },
                                        {
                                            "headerName": "MI After",
                                            "field": "mutual_info_after",
                                            "width": 120,
                                        },
                                        {
                                            "headerName": "Image 1 X",
                                            "field": "img0_x",
                                            "width": 100,
                                        },
                                        {
                                            "headerName": "Image 1 Y",
                                            "field": "img0_y",
                                            "width": 100,
                                        },
                                        {
                                            "headerName": "Image 2 X",
                                            "field": "img1_x",
                                            "width": 100,
                                        },
                                        {
                                            "headerName": "Image 2 Y",
                                            "field": "img1_y",
                                            "width": 100,
                                        },
                                    ],
                                    defaultColDef={
                                        "sortable": True,
                                        "filter": True,
                                        "resizable": True,
                                    },
                                    dashGridOptions={
                                        "suppressHorizontalScroll": False,
                                        "suppressVerticalScroll": False,
                                    },
                                    className="ag-theme-alpine",
                                    style={
                                        "height": "300px",
                                        "width": "100%",
                                        "margin": 0,
                                    },
                                ),
                            ],
                            id="match-data-container",
                            style={
                                "display": "block",
                                "height": "300px",
                                "maxHeight": "300px",
                                "overflowY": "auto",
                                "overflowX": "auto",
                                "border": "1px solid #dee2e6",
                                "borderRadius": "4px",
                                "padding": 0,
                            },
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H4(
                            "\u00a0", className="mb-3"
                        ),  # Empty header for alignment
                        dbc.Spinner(html.Div(id="feature-matching-status")),
                    ],
                    width=2,
                ),
            ],
            className="mb-4",
        ),
        # Image previews and results
        html.Div(
            [
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
                                html.H5("Feature Matches", className="mb-2"),
                                html.Div(id="feature-matching-results"),
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


# Update the run_feature_matching callback
@callback(
    [
        Output("feature-matching-status", "children"),
        Output("feature-matching-results", "children"),
        Output("match-data-store", "data"),
        Output("match-data-container", "style"),
        Output("match-data-table", "rowData"),
    ],
    Input("run-feature-matching", "n_clicks"),
    [
        State("fixed-image-dropdown", "value"),
        State("moving-image-dropdown", "value"),
        State("currentCaseTable", "rowData"),
    ],
    prevent_initial_call=True,
)
def run_feature_matching(n_clicks, fixed_image, moving_image, case_images):
    if not fixed_image or not moving_image:
        return (
            dbc.Alert(
                "Please select both fixed and moving images",
                color="warning",
                dismissable=True,
            ),
            None,
            [],
            {"display": "block", "height": "300px"},
            [],
        )

    # Get stain types
    fixed_stain = next(
        (
            img.get("meta", {}).get("npSchema", {}).get("stainID", "Unknown")
            for img in case_images
            if img["_id"] == fixed_image
        ),
        "Unknown",
    )
    moving_stain = next(
        (
            img.get("meta", {}).get("npSchema", {}).get("stainID", "Unknown")
            for img in case_images
            if img["_id"] == moving_image
        ),
        "Unknown",
    )

    try:
        # Load images (keeping color)
        img0 = get_dsa_image(fixed_image)
        img1 = get_dsa_image(moving_image)

        # Match images and get warped comparison
        matches_viz, warped_comparison, match_data = match_images(img0, img1)

        # Convert results to data URLs
        matches_url = array_to_data_url(matches_viz)
        comparison_url = array_to_data_url(warped_comparison)

        return (
            dbc.Alert(
                f"Successfully matched and warped {moving_stain} to {fixed_stain}",
                color="success",
                className="mb-3",
            ),
            html.Div(
                [
                    html.H6("Feature Matches", className="mt-3 mb-2"),
                    html.Img(
                        src=matches_url,
                        style={
                            "width": "100%",
                            "maxWidth": "1200px",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "4px",
                            "marginBottom": "20px",
                        },
                    ),
                    html.H6(
                        "Fixed Image vs Warped Moving Image", className="mt-3 mb-2"
                    ),
                    html.Img(
                        src=comparison_url,
                        style={
                            "width": "100%",
                            "maxWidth": "1200px",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "4px",
                        },
                    ),
                ]
            ),
            match_data,
            {"display": "block", "height": "300px"},
            match_data,
        )

    except Exception as e:
        return (
            dbc.Alert(
                f"Error during feature matching: {str(e)}",
                color="danger",
                dismissable=True,
            ),
            None,
            [],
            {"display": "none"},
            [],
        )


# Update the image preview callback
@callback(
    [
        Output("fixed-image-preview", "children"),
        Output("moving-image-preview", "children"),
    ],
    [
        Input("fixed-image-dropdown", "value"),
        Input("moving-image-dropdown", "value"),
    ],
    State("currentCaseTable", "rowData"),
)
def update_image_previews(fixed_image, moving_image, case_images):
    if not fixed_image or not moving_image:
        return (
            html.P("Please select fixed image", className="text-muted"),
            html.P("Please select moving image", className="text-muted"),
        )

    # Get stain types from case_images
    fixed_stain = next(
        (
            img.get("meta", {}).get("npSchema", {}).get("stainID", "Unknown")
            for img in case_images
            if img["_id"] == fixed_image
        ),
        "Unknown",
    )
    moving_stain = next(
        (
            img.get("meta", {}).get("npSchema", {}).get("stainID", "Unknown")
            for img in case_images
            if img["_id"] == moving_image
        ),
        "Unknown",
    )

    # Create thumbnails
    fixed_preview = html.Div(
        [
            html.Img(
                src=f"{DSA_BASE_URL}/item/{fixed_image}/tiles/thumbnail?token={token_info['_id']}&width=512",
                style={
                    "maxWidth": "100%",
                    "maxHeight": "200px",
                    "objectFit": "contain",
                },
            ),
            html.P(f"Stain: {fixed_stain}", className="mt-2 mb-0"),
        ]
    )

    moving_preview = html.Div(
        [
            html.Img(
                src=f"{DSA_BASE_URL}/item/{moving_image}/tiles/thumbnail?token={token_info['_id']}&width=512",
                style={
                    "maxWidth": "100%",
                    "maxHeight": "200px",
                    "objectFit": "contain",
                },
            ),
            html.P(f"Stain: {moving_stain}", className="mt-2 mb-0"),
        ]
    )

    return fixed_preview, moving_preview


@callback(
    [
        Output("fixed-image-dropdown", "options"),
        Output("moving-image-dropdown", "options"),
    ],
    Input("currentCaseTable", "rowData"),
)
def update_image_options(case_images):
    print("\n=== Dropdown Update Debug ===")
    print(f"Received case images: {case_images}")

    if not case_images:
        print("No case images - returning empty options")
        return [], []

    # Create options list with stain type in the label
    options = []
    for img in case_images:
        stain = img.get("meta", {}).get("npSchema", {}).get("stainID", "Unknown")
        label = f"{stain} - {img['name']}"
        options.append({"label": label, "value": img["_id"]})

    print(f"\nCreated {len(options)} options:")
    for opt in options:
        print(f"- {opt['label']}: value = {opt['value']}")

    # Return the same options for both dropdowns
    return options, options


@callback(
    [
        Output("fixed-image-dropdown", "value"),
        Output("moving-image-dropdown", "value"),
    ],
    Input("currentCaseTable", "rowData"),
)
def set_default_images(case_images):
    if not case_images:
        return None, None

    # Find HE image for fixed image
    he_image = next(
        (
            img["_id"]
            for img in case_images
            if img.get("meta", {}).get("npSchema", {}).get("stainID") == "HE"
        ),
        None,
    )

    # If HE image found, use it as fixed and select first non-HE image as moving
    if he_image:
        other_image = next(
            (
                img["_id"]
                for img in case_images
                if img["_id"] != he_image
                and img.get("meta", {}).get("npSchema", {}).get("stainID") != "HE"
            ),
            None,
        )
        return he_image, other_image

    # If no HE image, just use first two images
    if len(case_images) >= 2:
        return case_images[0]["_id"], case_images[1]["_id"]
    elif len(case_images) == 1:
        return case_images[0]["_id"], None
    else:
        return None, None
