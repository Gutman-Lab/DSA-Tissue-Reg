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

##https://www.sciencedirect.com/science/article/pii/S0010482522000932

# Add viewer components for fixed (HE) and moving images
fixed_image_viewer = dash_paperdragon.DashPaperdragon(
    id="fixed-image-viewer",
    viewerHeight=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
)

moving_image_viewer = dash_paperdragon.DashPaperdragon(
    id="moving-image-viewer",
    viewerHeight=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
)


# Function to generate distinct colors
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        # Use high saturation and value for vibrant, visible colors
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        # Convert to hex color
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


# Function to generate fiducial points
def generate_fiducial_points(image_bounds, num_points=8):
    """
    Generate evenly distributed points within the image bounds
    """
    width = image_bounds["width"]
    height = image_bounds["height"]

    # Create a grid of points, avoiding edges
    margin = 0.1  # 10% margin from edges
    x_points = np.linspace(width * margin, width * (1 - margin), 3)
    y_points = np.linspace(height * margin, height * (1 - margin), 3)

    # Generate distinct colors for each point
    colors = generate_distinct_colors(num_points)

    points = []
    i = 0
    for x in x_points:
        for y in y_points:
            if i < num_points:  # Only add up to num_points
                points.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "radius": 5,
                        "color": colors[i],  # Assign unique color to each point
                        "fillColor": colors[i],
                        "strokeColor": colors[i],
                    }
                )
                i += 1

    return points[:num_points]


# Create thumbnail card component (same as in caseViewer)
def create_thumbnail_card(item):
    return dbc.Card(
        [
            dbc.CardHeader(
                item.get("meta", {})
                .get("npSchema", {})
                .get("stainID", "Unknown Stain"),
                className="text-center",
            ),
            dbc.CardImg(
                src=f"{DSA_BASE_URL}/item/{item['_id']}/tiles/thumbnail?token={token_info['_id']}",
                top=True,
                style={"height": "150px", "objectFit": "contain"},
            ),
        ],
        className="mb-3",
        style={"width": "200px"},
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

# Add modal for thumbnail debugging
thumbnail_debug_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Registration Debug Information")),
        dbc.ModalBody(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [html.Div(id="modal-fixed-debug-content")],
                            label="Fixed Image",
                        ),
                        dbc.Tab(
                            [html.Div(id="modal-moving-debug-content")],
                            label="Moving Image",
                        ),
                    ]
                )
            ]
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close-thumbnail-modal", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="thumbnail-debug-modal",
    size="lg",
)

# Basic layout for registration controls
registrationControls_layout = dbc.Container(
    [
        dcc.Store(id="registration_caseId", data="641bfd45867536bb7a236ae1"),
        dcc.Store(id="registration_blockId", data="5"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Registration Controls", className="mb-2"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Feature Detection Method:",
                                            className="mb-1",
                                        ),
                                        dbc.RadioItems(
                                            id="feature-detection-method",
                                            options=[
                                                {"label": "ORB", "value": "orb"},
                                                {"label": "SIFT", "value": "sift"},
                                                {"label": "AKAZE", "value": "akaze"},
                                                {"label": "BRISK", "value": "brisk"},
                                                {"label": "Canny", "value": "canny"},
                                                {
                                                    "label": "Adaptive",
                                                    "value": "adaptive",
                                                },
                                                {
                                                    "label": "Watershed",
                                                    "value": "watershed",
                                                },
                                            ],
                                            value="orb",
                                            inline=True,
                                            className="small",
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Registration Parameters:", className="mb-1"
                                        ),
                                        dbc.Checklist(
                                            id="registration-constraints",
                                            options=[
                                                {
                                                    "label": "Limit Rotation (±10°)",
                                                    "value": "limit_rotation",
                                                },
                                                {
                                                    "label": "Similar Scale (±10%)",
                                                    "value": "similar_scale",
                                                },
                                                {
                                                    "label": "Center Aligned",
                                                    "value": "center_aligned",
                                                },
                                                {
                                                    "label": "Preserve Tissue Shape",
                                                    "value": "preserve_shape",
                                                },
                                            ],
                                            value=["limit_rotation", "similar_scale"],
                                            inline=True,
                                            className="small",
                                        ),
                                    ],
                                    width=8,
                                ),
                            ],
                            className="mb-2 g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Number of Points:", className="mb-1"
                                        ),
                                        dcc.Slider(
                                            id="num-points-slider",
                                            min=6,
                                            max=16,
                                            step=2,
                                            value=8,
                                            marks={6: "6", 8: "8", 12: "12", 16: "16"},
                                            className="mb-1",
                                        ),
                                        html.Small(
                                            "Recommended: 8-12 points",
                                            className="text-muted",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Thumbnail Width:", className="mb-1"
                                        ),
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
                                    width=4,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Show Debug Info",
                                        id="open-thumbnail-modal",
                                        color="secondary",
                                        size="sm",
                                        className="mt-4",
                                    ),
                                    width=2,
                                ),
                            ],
                            className="mb-2 g-2",
                        ),
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
                        ),
                    ],
                    className="mb-2",
                )
            ]
        ),
        # Viewers Row
        dbc.Row(
            [
                dbc.Col([fixed_image_viewer], width=6),
                dbc.Col([moving_image_viewer], width=6),
            ],
            className="g-2",
        ),
        thumbnail_debug_modal,
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


def get_thumbnail_image(item_id, width=1024):
    """Fetch thumbnail image from DSA and resize to specified width"""
    url = f"{DSA_BASE_URL}/item/{item_id}/tiles/thumbnail?token={token_info['_id']}&width={width}"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    # Convert to grayscale if it's not already
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return img_np


def find_tissue_contours(img_array, method="canny"):
    """Find tissue boundaries using various edge/contour detection methods"""
    # Convert BGR to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Convert to HSV for better tissue segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    if method == "canny":
        # Use Canny edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    elif method == "threshold":
        # Use saturation channel for tissue detection
        _, thresh = cv2.threshold(hsv[:, :, 1], 50, 255, cv2.THRESH_BINARY)
        return thresh

    elif method == "adaptive":
        # Adaptive thresholding on grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Invert the image so tissue (darker regions) becomes white
        gray = cv2.bitwise_not(gray)
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh

    elif method == "watershed":
        # Watershed segmentation
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Invert for better tissue detection
        gray = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        return sure_fg.astype(np.uint8)


def find_matching_points(fixed_img, moving_img, method="orb", max_points=8):
    debug_info = {
        "all_keypoints_fixed": [],
        "all_keypoints_moving": [],
        "initial_matches": [],
        "ransac_inliers": [],
        "stats": {},
    }

    if method in ["orb", "sift", "akaze", "brisk"]:
        try:
            # Ensure images are properly preprocessed
            if len(fixed_img.shape) == 3:
                fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
                moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
            else:
                fixed_gray = fixed_img
                moving_gray = moving_img

            # Enhance contrast
            fixed_gray = cv2.equalizeHist(fixed_gray)
            moving_gray = cv2.equalizeHist(moving_gray)

            # Initialize detector with appropriate parameters
            if method == "orb":
                detector = cv2.ORB_create(nfeatures=500)
                norm_type = cv2.NORM_HAMMING
                crossCheck = True
            elif method == "sift":
                detector = cv2.SIFT_create(nfeatures=500)
                norm_type = cv2.NORM_L2
                crossCheck = False
            elif method == "akaze":
                detector = cv2.AKAZE_create(
                    descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                    descriptor_size=0,  # Full size
                    descriptor_channels=3,
                    threshold=0.001,  # Lower threshold to detect more features
                    nOctaves=4,
                    nOctaveLayers=4,
                )
                norm_type = cv2.NORM_HAMMING
                crossCheck = True
            elif method == "brisk":
                detector = cv2.BRISK_create()
                norm_type = cv2.NORM_HAMMING
                crossCheck = True

            # Find keypoints and descriptors
            kp1, des1 = detector.detectAndCompute(fixed_gray, None)
            kp2, des2 = detector.detectAndCompute(moving_gray, None)

            print(
                f"{method.upper()} detected keypoints - fixed: {len(kp1) if kp1 else 0}, moving: {len(kp2) if kp2 else 0}"
            )

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                print(f"Not enough keypoints found for {method}")
                return [], [], debug_info

            # Store all keypoints for visualization
            debug_info["all_keypoints_fixed"] = [
                (int(kp.pt[0]), int(kp.pt[1])) for kp in kp1
            ]
            debug_info["all_keypoints_moving"] = [
                (int(kp.pt[0]), int(kp.pt[1])) for kp in kp2
            ]
            debug_info["stats"]["total_keypoints"] = (len(kp1), len(kp2))

            # Match features based on method
            if method == "sift":
                bf = cv2.BFMatcher(norm_type)
                raw_matches = bf.knnMatch(des1, des2, k=2)
                matches = []
                for m, n in raw_matches:
                    if m.distance < 0.75 * n.distance:
                        matches.append(m)
            else:
                bf = cv2.BFMatcher(norm_type, crossCheck=crossCheck)
                matches = bf.match(des1, des2)

            if len(matches) < 4:
                print(f"Not enough matches found for {method}: {len(matches)}")
                return [], [], debug_info

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            print(f"Total matches found for {method}: {len(matches)}")

            # Take top matches (more than needed for filtering)
            num_matches = min(len(matches), max_points * 4)
            top_matches = matches[:num_matches]

            # Store initial matches for debug
            debug_info["initial_matches"] = [
                (kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in top_matches
            ]
            debug_info["stats"]["total_matches"] = len(matches)

            # Convert matches to point pairs
            fixed_points = []
            moving_points = []
            for match in top_matches:
                fixed_points.append(kp1[match.queryIdx].pt)
                moving_points.append(kp2[match.trainIdx].pt)

            return fixed_points, moving_points, debug_info

        except Exception as e:
            print(f"Error in {method} matching: {str(e)}")
            import traceback

            traceback.print_exc()
            return [], [], debug_info

    # For non-feature detection methods (tissue detection)
    else:
        try:
            # ... existing tissue detection code ...
            return fixed_points, moving_points, debug_info
        except Exception as e:
            print(f"Error in {method} detection: {str(e)}")
            return [], [], debug_info


def scale_points_to_full_size(points, thumbnail_size, full_size):
    """
    Scale points from thumbnail coordinates to full image coordinates
    Explicitly handle both width and height
    """
    # thumbnail_size is (height, width) from numpy shape
    # full_size has 'width' and 'height' keys

    print(
        f"Scaling points from thumbnail size {thumbnail_size} to full size {full_size}"
    )

    # Calculate scaling factors for both dimensions
    scale_x = (
        full_size["width"] / thumbnail_size[1]
    )  # width is second dimension in shape
    scale_y = (
        full_size["height"] / thumbnail_size[0]
    )  # height is first dimension in shape

    print(f"Scale factors - x: {scale_x}, y: {scale_y}")

    scaled_points = []
    for x, y in points:
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_points.append((scaled_x, scaled_y))
        print(f"Point scaled from ({x}, {y}) to ({scaled_x}, {scaled_y})")

    return scaled_points


# Add a function to filter matches based on geometric consistency
def filter_matches_geometric(fixed_points, moving_points, num_points, debug_info=None):
    """
    Filter matches using geometric constraints
    Args:
        fixed_points: list of points from fixed image
        moving_points: list of points from moving image
        num_points: number of points to return
        debug_info: dictionary for storing debug information
    Returns:
        filtered fixed points, filtered moving points, updated debug_info
    """
    if debug_info is None:
        debug_info = {}

    if len(fixed_points) < num_points or len(moving_points) < num_points:
        return fixed_points, moving_points, debug_info

    print(f"Starting geometric filtering with {len(fixed_points)} points")

    # Convert to numpy arrays
    fixed_arr = np.array(fixed_points)
    moving_arr = np.array(moving_points)

    # Find homography matrix
    H, mask = cv2.findHomography(fixed_arr, moving_arr, cv2.RANSAC, 5.0)

    if H is None:
        print("No homography found")
        return fixed_points[:num_points], moving_points[:num_points], debug_info

    # Get inliers
    inliers = mask.ravel().tolist()
    print(f"RANSAC found {sum(inliers)} inlier matches")

    # Filter points using mask
    fixed_filtered = []
    moving_filtered = []
    for i, inlier in enumerate(inliers):
        if inlier:
            fixed_filtered.append(fixed_points[i])
            moving_filtered.append(moving_points[i])

    # Take the best num_points
    if len(fixed_filtered) > num_points:
        fixed_filtered = fixed_filtered[:num_points]
        moving_filtered = moving_filtered[:num_points]

    print(f"Final number of points after geometric filtering: {len(fixed_filtered)}")
    return fixed_filtered, moving_filtered, debug_info


# Callback to set up images and generate fiducial points
@callback(
    [
        Output("fixed-image-viewer", "tileSources"),
        Output("fixed-image-viewer", "inputToPaper"),
        Output("moving-image-viewer", "tileSources"),
        Output("moving-image-viewer", "inputToPaper"),
    ],
    [
        Input("caseSlideSet_store", "data"),
        Input("registration_blockId", "data"),
        Input("feature-detection-method", "value"),
        Input("num-points-slider", "value"),
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
        return [], {}, [], {}

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

            # Generate colors and create GeoJSON features
            colors = generate_distinct_colors(len(fixed_points))
            fixed_items = create_geojson_features(fixed_points, colors, "fixed")
            moving_items = create_geojson_features(moving_points, colors, "moving")

            return (
                fixed_tile_source,
                {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
                moving_tile_source,
                {"actions": [{"type": "drawItems", "itemList": moving_items}]},
            )

    except Exception as e:
        print(f"Error in feature matching: {str(e)}")
        import traceback

        traceback.print_exc()

    return fixed_tile_source, {"actions": []}, moving_tile_source, {"actions": []}


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


def filter_points_with_constraints(
    fixed_points, moving_points, constraints, image_bounds
):
    """Apply pathology-specific constraints to point pairs"""
    if not constraints or not image_bounds:
        return fixed_points, moving_points

    try:
        valid_pairs = []
        for i in range(len(fixed_points)):
            fp = np.array(fixed_points[i])
            mp = np.array(moving_points[i])

            valid = True
            f_center = np.array([image_bounds["width"] / 2, image_bounds["height"] / 2])
            m_center = f_center  # Assuming same size images

            if "limit_rotation" in constraints:
                f_angle = np.arctan2(fp[1] - f_center[1], fp[0] - f_center[0])
                m_angle = np.arctan2(mp[1] - m_center[1], mp[0] - m_center[0])
                angle_diff = np.abs(np.degrees(f_angle - m_angle))
                if angle_diff > 10:  # 10 degree threshold
                    valid = False

            if "similar_scale" in constraints:
                f_dist = np.linalg.norm(fp - f_center)
                m_dist = np.linalg.norm(mp - m_center)
                scale_diff = abs(f_dist - m_dist) / f_dist
                if scale_diff > 0.1:  # 10% threshold
                    valid = False

            if "center_aligned" in constraints:
                center_threshold = (
                    min(image_bounds["width"], image_bounds["height"]) * 0.2
                )
                if np.linalg.norm((fp - f_center) - (mp - m_center)) > center_threshold:
                    valid = False

            if "preserve_shape" in constraints and len(valid_pairs) >= 2:
                prev_f = np.array(fixed_points[valid_pairs[-1]])
                prev_m = np.array(moving_points[valid_pairs[-1]])
                f_dist = np.linalg.norm(fp - prev_f)
                m_dist = np.linalg.norm(mp - prev_m)
                if abs(f_dist - m_dist) / f_dist > 0.1:  # 10% threshold
                    valid = False

            if valid:
                valid_pairs.append(i)

        filtered_fixed = [fixed_points[i] for i in valid_pairs]
        filtered_moving = [moving_points[i] for i in valid_pairs]

        print(f"Points after constraint filtering: {len(filtered_fixed)}")
        return filtered_fixed, filtered_moving

    except Exception as e:
        print(f"Error in constraint filtering: {str(e)}")
        return fixed_points, moving_points


def get_slides_for_registration(slideList, selected_block):
    """Get the fixed (HE) and moving slides for registration from a slide list"""
    if not slideList or not selected_block:
        return None, None

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

    return he_slide, moving_slide


def create_registration_points(
    fixed_img, moving_img, method, num_points, constraints=None, image_bounds=None
):
    """
    Main registration point generation function that handles different methods
    Returns: (fixed_points, moving_points)
    """
    try:
        # Get initial matches with double the requested points for better filtering
        fixed_points, moving_points, debug_info = find_matching_points(
            fixed_img, moving_img, method=method, max_points=num_points * 2
        )

        print(f"Initial points found: {len(fixed_points)}")

        # Filter points if we have enough
        if len(fixed_points) >= 4 and len(moving_points) >= 4:
            fixed_points, moving_points, debug_info = filter_matches_geometric(
                fixed_points, moving_points, num_points, debug_info
            )

            print(f"Points after geometric filtering: {len(fixed_points)}")

            # Apply additional constraints if specified
            if constraints and image_bounds:
                fixed_points, moving_points = filter_points_with_constraints(
                    fixed_points, moving_points, constraints, image_bounds
                )

        return fixed_points, moving_points, debug_info

    except Exception as e:
        print(f"Error in registration point creation: {str(e)}")
        import traceback

        traceback.print_exc()
        return [], [], {}


def create_geojson_features(points, colors, prefix="fiducial"):
    """Create GeoJSON features for a set of points with colors"""
    return [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {
                "radius": 5,
                "fillColor": color,
                "strokeColor": color,
                "name": f"{prefix}_{i}",
            },
        }
        for i, ((x, y), color) in enumerate(zip(points, colors))
    ]


def normalize_image_sizes(fixed_img, moving_img):
    """
    Resize images to have the same dimensions while preserving aspect ratio
    Returns: (normalized_fixed, normalized_moving, scale_factors)
    """
    # Get original dimensions
    fixed_h, fixed_w = fixed_img.shape[:2]
    moving_h, moving_w = moving_img.shape[:2]

    print(
        f"Original dimensions - fixed: {fixed_h}x{fixed_w}, moving: {moving_h}x{moving_w}"
    )

    # Calculate target size (use smaller dimensions to prevent upscaling)
    target_w = min(fixed_w, moving_w)
    target_h = min(fixed_h, moving_h)

    # Calculate scaling factors for later use
    fixed_scale = {"x": fixed_w / target_w, "y": fixed_h / target_h}
    moving_scale = {"x": moving_w / target_w, "y": moving_h / target_h}

    print(f"Target dimensions: {target_h}x{target_w}")
    print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

    # Resize images
    fixed_resized = cv2.resize(fixed_img, (target_w, target_h))
    moving_resized = cv2.resize(moving_img, (target_w, target_h))

    return fixed_resized, moving_resized, (fixed_scale, moving_scale)


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


# Add callback for modal control
@callback(
    Output("thumbnail-debug-modal", "is_open"),
    [
        Input("open-thumbnail-modal", "n_clicks"),
        Input("close-thumbnail-modal", "n_clicks"),
    ],
    [State("thumbnail-debug-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


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
