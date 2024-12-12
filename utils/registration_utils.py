import cv2
import numpy as np
import colorsys
import requests
from io import BytesIO
from PIL import Image
from settings import gc, DSA_BASE_URL, token_info


def get_thumbnail_image(item_id, width=1024):
    """Fetch thumbnail image from DSA and resize to specified width"""
    url = f"{DSA_BASE_URL}/item/{item_id}/tiles/thumbnail?token={token_info['_id']}&width={width}"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return img_np


def normalize_image_sizes(fixed_img, moving_img):
    """Resize images to have the same dimensions while preserving aspect ratio"""
    fixed_h, fixed_w = fixed_img.shape[:2]
    moving_h, moving_w = moving_img.shape[:2]

    print(
        f"Original dimensions - fixed: {fixed_h}x{fixed_w}, moving: {moving_h}x{moving_w}"
    )

    target_w = min(fixed_w, moving_w)
    target_h = min(fixed_h, moving_h)

    fixed_scale = {"x": fixed_w / target_w, "y": fixed_h / target_h}
    moving_scale = {"x": moving_w / target_w, "y": moving_h / target_h}

    print(f"Target dimensions: {target_h}x{target_w}")
    print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

    fixed_resized = cv2.resize(fixed_img, (target_w, target_h))
    moving_resized = cv2.resize(moving_img, (target_w, target_h))

    return fixed_resized, moving_resized, (fixed_scale, moving_scale)


def generate_distinct_colors(n):
    """Generate visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def calculate_mutual_information(hist):
    """Calculate mutual information from a joint histogram"""
    # Convert histogram to probability distribution
    pxy = hist / float(np.sum(hist))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]

    # Avoid log(0)
    nonzero = pxy > 0

    # Calculate mutual information
    nmi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    return nmi


def find_matching_points_intensity(fixed_img, moving_img, max_points=8):
    """Find matching points using mutual information with more visualization points"""
    h, w = fixed_img.shape

    # Create a denser grid than requested points for better visualization
    visualization_points = max(
        16, max_points * 2
    )  # At least 16 points, or double requested
    grid_size = int(np.sqrt(visualization_points))

    step_y = h // grid_size
    step_x = w // grid_size
    window_size = min(step_x, step_y) // 2
    search_radius = window_size // 2

    fixed_points = []
    moving_points = []
    nmi_scores = []  # Store scores for ranking

    # Create grid of points
    for y in range(step_y, h - step_y, step_y):
        for x in range(step_x, w - step_x, step_x):
            # Get window around point in fixed image
            f_window = fixed_img[
                y - window_size : y + window_size, x - window_size : x + window_size
            ]

            # Define search area in moving image
            search_y_min = max(window_size, y - search_radius)
            search_y_max = min(h - window_size, y + search_radius)
            search_x_min = max(window_size, x - search_radius)
            search_x_max = min(w - window_size, x + search_radius)

            best_nmi = -np.inf
            best_pos = (x, y)

            # Search for best matching position
            for sy in range(search_y_min, search_y_max, 2):
                for sx in range(search_x_min, search_x_max, 2):
                    m_window = moving_img[
                        sy - window_size : sy + window_size,
                        sx - window_size : sx + window_size,
                    ]

                    # Calculate normalized mutual information
                    hist, _, _ = np.histogram2d(
                        f_window.ravel(), m_window.ravel(), bins=32
                    )
                    nmi = calculate_mutual_information(hist)

                    if nmi > best_nmi:
                        best_nmi = nmi
                        best_pos = (sx, sy)

            # Store all points with their NMI scores
            if best_nmi > 0.2:  # Lower threshold to get more points
                fixed_points.append((x, y))
                moving_points.append(best_pos)
                nmi_scores.append(best_nmi)

    # Sort points by NMI score and keep the best ones
    if len(fixed_points) > max_points:
        # Sort by NMI score
        sorted_indices = np.argsort(nmi_scores)[::-1]  # Descending order

        # Select points with good spatial distribution
        selected_indices = []
        min_distance = min(step_x, step_y) / 2  # Minimum distance between points

        for idx in sorted_indices:
            point = fixed_points[idx]
            # Check if point is far enough from already selected points
            if not selected_indices or all(
                np.hypot(point[0] - fixed_points[i][0], point[1] - fixed_points[i][1])
                > min_distance
                for i in selected_indices
            ):
                selected_indices.append(idx)
                if len(selected_indices) >= max_points:
                    break

        # Update point lists with selected points
        fixed_points = [fixed_points[i] for i in selected_indices]
        moving_points = [moving_points[i] for i in selected_indices]
        nmi_scores = [nmi_scores[i] for i in selected_indices]

    print(f"Found {len(fixed_points)} points with NMI scores: {nmi_scores}")
    return fixed_points, moving_points


def find_matching_points(fixed_img, moving_img, method="orb", max_points=8):
    """Find matching points between two images using various methods"""
    debug_info = {
        "stats": {"total_keypoints": (0, 0), "total_matches": 0, "ransac_inliers": 0}
    }

    try:
        # Convert max_points to int if it's a string
        max_points = int(max_points) if isinstance(max_points, str) else max_points

        if method == "intensity":
            fixed_points, moving_points = find_matching_points_intensity(
                fixed_img, moving_img, max_points
            )
            debug_info["stats"]["total_matches"] = len(fixed_points)
            return fixed_points, moving_points, debug_info

        elif method == "orb":
            detector = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)
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
                threshold=0.0005,  # Lower threshold for more features
                nOctaves=6,  # More octaves for multi-scale detection
                nOctaveLayers=6,  # More layers per octave
                diffusivity=cv2.KAZE_DIFF_PM_G2,  # Better edge detection
            )
            norm_type = cv2.NORM_HAMMING
            crossCheck = True
            max_points = max_points * 2  # Allow more initial points for AKAZE

        # Continue with feature detection for non-intensity methods
        if method != "intensity":
            kp1, des1 = detector.detectAndCompute(fixed_img, None)
            kp2, des2 = detector.detectAndCompute(moving_img, None)

            debug_info["stats"]["total_keypoints"] = (len(kp1), len(kp2))
            print(
                f"{method.upper()} detected keypoints - fixed: {len(kp1)}, moving: {len(kp2)}"
            )

            if len(kp1) == 0 or len(kp2) == 0 or des1 is None or des2 is None:
                print(f"No keypoints or descriptors found for {method}")
                return [], [], debug_info

            # Match features using appropriate norm type and crossCheck
            bf = cv2.BFMatcher(norm_type, crossCheck=crossCheck)
            matches = bf.match(des1, des2)

            debug_info["stats"]["total_matches"] = len(matches)
            print(f"Total matches found for {method}: {len(matches)}")

            if len(matches) == 0:
                print(f"No matches found for {method}")
                return [], [], debug_info

            # Sort matches by distance and apply ratio test for AKAZE
            matches = sorted(matches, key=lambda x: x.distance)
            if method == "akaze":
                # Use distance ratio test
                good_matches = []
                min_dist = matches[0].distance
                max_dist = matches[-1].distance
                threshold = min_dist + 0.8 * (max_dist - min_dist)
                good_matches = [m for m in matches if m.distance < threshold]
                matches = good_matches[: max_points * 4]
            else:
                num_matches = min(len(matches), max_points * 4)
                matches = matches[:num_matches]

            # Extract matched keypoints
            fixed_points = [kp1[m.queryIdx].pt for m in matches]
            moving_points = [kp2[m.trainIdx].pt for m in matches]

        return fixed_points, moving_points, debug_info

    except Exception as e:
        print(f"Error in {method} matching: {str(e)}")
        import traceback

        traceback.print_exc()
        return [], [], debug_info


def filter_matches_geometric(fixed_points, moving_points, num_points, debug_info=None):
    """Filter matches using geometric constraints"""
    if debug_info is None:
        debug_info = {}

    try:
        if len(fixed_points) < 4 or len(moving_points) < 4:
            return fixed_points, moving_points, debug_info

        # Convert points to numpy arrays
        fixed_np = np.float32(fixed_points)
        moving_np = np.float32(moving_points)

        # Find homography
        H, mask = cv2.findHomography(moving_np, fixed_np, cv2.RANSAC, 5.0)

        if H is None:
            return fixed_points[:num_points], moving_points[:num_points], debug_info

        # Count inliers
        inliers = mask.ravel().tolist()
        debug_info["stats"]["ransac_inliers"] = sum(inliers)

        # Filter points using mask
        filtered_fixed = []
        filtered_moving = []
        for i, (fp, mp) in enumerate(zip(fixed_points, moving_points)):
            if inliers[i]:
                filtered_fixed.append(fp)
                filtered_moving.append(mp)

        # Return requested number of points
        return (filtered_fixed[:num_points], filtered_moving[:num_points], debug_info)

    except Exception as e:
        print(f"Error in geometric filtering: {str(e)}")
        return fixed_points[:num_points], moving_points[:num_points], debug_info


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

            if valid:
                valid_pairs.append(i)

        filtered_fixed = [fixed_points[i] for i in valid_pairs]
        filtered_moving = [moving_points[i] for i in valid_pairs]

        return filtered_fixed, filtered_moving

    except Exception as e:
        print(f"Error in constraint filtering: {str(e)}")
        return fixed_points, moving_points


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


def scale_points_to_full_size(points, thumbnail_size, full_size):
    """Scale points from thumbnail coordinates to full image coordinates"""
    if not points:
        return points

    scale_x = full_size["width"] / thumbnail_size[1]  # width is dim 1 in thumbnail
    scale_y = full_size["height"] / thumbnail_size[0]  # height is dim 0 in thumbnail

    return [(x * scale_x, y * scale_y) for x, y in points]


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
    """Main registration point generation function that handles different methods"""
    try:
        # Get initial matches with double the requested points for better filtering
        fixed_points, moving_points, debug_info = find_matching_points(
            fixed_img, moving_img, method=method, max_points=num_points * 2
        )

        print(f"Initial points found: {len(fixed_points)}")

        if not fixed_points or not moving_points:
            print("No initial points found")
            return [], [], debug_info

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
