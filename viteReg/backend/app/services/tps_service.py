"""
Thin Plate Spline (TPS) registration service
Uses LightGlue for feature matching, then applies TPS transformation for non-rigid registration
"""
import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional, Tuple
from app.services.dsa_client import get_dsa_client
from app.services.registration_service import get_thumbnail_image

logger = logging.getLogger(__name__)

try:
    from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
    from lightglue.utils import load_image, rbd
    import torch
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    logger.warning("LightGlue not available. Install with: pip install lightglue")

try:
    from scipy.interpolate import RBFInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install with: pip install scipy")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Spatial consistency filtering will be disabled. Install with: pip install scikit-learn")


def _filter_spatial_consistency(
    points0: np.ndarray,
    points1: np.ndarray,
    max_neighbor_distance_ratio: float = 2.0,
    k_neighbors: int = 5
) -> np.ndarray:
    """
    Filter matches to enforce spatial consistency (topographic constraint)
    
    For each match, check if nearby points in image0 match to nearby points in image1.
    This enforces that local spatial relationships are preserved.
    
    Args:
        points0: Matched points in image0 (N, 2)
        points1: Matched points in image1 (N, 2)
        max_neighbor_distance_ratio: Maximum ratio of neighbor distances between images
        k_neighbors: Number of nearest neighbors to check
    
    Returns:
        Boolean array indicating which matches pass spatial consistency check
    """
    if not SKLEARN_AVAILABLE:
        # sklearn not available, skip spatial filtering
        logger.debug("sklearn not available, skipping spatial consistency filtering")
        return np.ones(len(points0), dtype=bool)
    
    if len(points0) < k_neighbors + 1:
        # Not enough points to check spatial consistency
        return np.ones(len(points0), dtype=bool)
    
    try:
        # Find k nearest neighbors for each point in image0
        nn0 = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(points0)), metric='euclidean')
        nn0.fit(points0)
        distances0, indices0 = nn0.kneighbors(points0)
        
        # Find k nearest neighbors for each point in image1
        nn1 = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(points1)), metric='euclidean')
        nn1.fit(points1)
        distances1, indices1 = nn1.kneighbors(points1)
        
        # For each match, check if neighbors in image0 correspond to neighbors in image1
        valid = np.ones(len(points0), dtype=bool)
        
        for i in range(len(points0)):
            # Get neighbors of point i in image0 (skip self, index 0)
            neighbors0 = set(indices0[i, 1:])  # Skip first (self)
            
            # Get neighbors of point i in image1 (skip self, index 0)
            neighbors1 = set(indices1[i, 1:])  # Skip first (self)
            
            # Check how many neighbors are shared (should be high for good matches)
            # A match is valid if at least some neighbors are also matched to neighbors
            shared_neighbors = 0
            for j in neighbors0:
                if j < len(points1) and j in neighbors1:
                    # Check if the distance ratios are reasonable
                    dist0 = distances0[i, list(indices0[i]).index(j)]
                    dist1 = distances1[i, list(indices1[i]).index(j)]
                    if dist0 > 0 and dist1 > 0:
                        ratio = max(dist0 / dist1, dist1 / dist0)
                        if ratio < max_neighbor_distance_ratio:
                            shared_neighbors += 1
            
            # Require at least 2 shared neighbors (or 30% of neighbors, whichever is less)
            min_shared = min(2, max(1, int(len(neighbors0) * 0.3)))
            if shared_neighbors < min_shared:
                valid[i] = False
        
        return valid
    except Exception as e:
        logger.warning(f"Spatial consistency filtering failed: {e}, using all matches")
        return np.ones(len(points0), dtype=bool)


def apply_tps_warp(
    src_img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    target_shape: Tuple[int, int],
    smoothing: float = 1.0
) -> np.ndarray:
    """
    Apply Thin Plate Spline warping to the source image
    
    Args:
        src_img: Source image to warp
        src_points: Control points in source image (N, 2)
        dst_points: Corresponding control points in destination (N, 2)
        target_shape: Target image shape (height, width)
        smoothing: Smoothing parameter for TPS (higher = smoother)
        
    Returns:
        Warped image
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for TPS. Install with: pip install scipy")
    
    h, w = target_shape[:2]
    
    # Create coordinate grid for target image
    y, x = np.mgrid[0:h, 0:w]
    coords = np.vstack([x.ravel(), y.ravel()]).T
    
    # Use RBFInterpolator with thin_plate_spline kernel
    # This computes the displacement field
    rbf = RBFInterpolator(
        src_points,
        dst_points - src_points,  # Displacement vectors
        kernel="thin_plate_spline",
        smoothing=smoothing
    )
    
    # Compute displacements for all coordinates
    displacements = rbf(coords)
    
    # Apply displacements to get warped coordinates
    warped_coords = coords + displacements
    
    # Reshape for cv2.remap
    map_x = warped_coords[:, 0].reshape(h, w).astype(np.float32)
    map_y = warped_coords[:, 1].reshape(h, w).astype(np.float32)
    
    # Apply the warping
    warped = cv2.remap(
        src_img,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped


def register_with_tps(
    fixed_id: str,
    moving_id: str,
    extractor_type: str = "superpoint",
    thumbnail_width: int = 1024,
    max_keypoints: int = 2048,
    device: Optional[str] = None,
    smoothing: float = 1.0,
    min_matches: int = 6,
    # LightGlue extractor parameters
    detection_threshold: Optional[float] = None,
    nms_window_size: Optional[int] = None,
    # LightGlue matcher parameters
    n_layers: int = 9,
    depth_confidence: float = 0.9,
    width_confidence: float = 0.99,
    filter_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Perform non-rigid registration using Thin Plate Spline transformation
    
    Uses LightGlue to find feature correspondences, then applies TPS warp.
    Good for handling local deformations like tissue rips.
    
    Args:
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        extractor_type: Feature extractor type ('superpoint', 'disk', 'aliked', 'sift')
        thumbnail_width: Width of thumbnail to use for registration
        max_keypoints: Maximum number of keypoints to extract
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None
        smoothing: TPS smoothing parameter (higher = smoother, less local deformation)
        min_matches: Minimum number of matches required (TPS needs at least 3, but more is better)
        
        # LightGlue extractor parameters (for keypoint detection):
        detection_threshold: Minimum score for keypoint detection (lower = more keypoints, default: extractor default)
        nms_window_size: Window size for non-maximum suppression (default: extractor default)
        
        # LightGlue matcher parameters (for matching):
        n_layers: Number of attention layers/iterations (default: 9, more = better quality but slower)
        depth_confidence: Early stopping confidence (0-1, default: 0.9, higher = faster but may miss matches)
        width_confidence: Point pruning confidence (0-1, default: 0.99, higher = more aggressive pruning)
        filter_threshold: Filter threshold for matches (0-1, default: 0.1, lower = stricter filtering)
        
    Returns:
        Dictionary with registration results
    """
    if not LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue is not installed. Install with: pip install lightglue")
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is not installed. Install with: pip install scipy")
    
    try:
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get thumbnail images
        logger.debug(f"Fetching thumbnails for TPS registration: fixed={fixed_id}, moving={moving_id}")
        fixed_img = get_thumbnail_image(fixed_id, width=thumbnail_width)
        moving_img = get_thumbnail_image(moving_id, width=thumbnail_width)
        
        if fixed_img is None or moving_img is None:
            raise ValueError("Failed to retrieve thumbnail images")
        
        # Convert to RGB if needed
        if len(fixed_img.shape) == 2:
            fixed_img = np.stack([fixed_img] * 3, axis=-1)
        if len(moving_img.shape) == 2:
            moving_img = np.stack([moving_img] * 3, axis=-1)
        
        # Convert to float32 and normalize to [0, 1]
        fixed_img = fixed_img.astype(np.float32) / 255.0
        moving_img = moving_img.astype(np.float32) / 255.0
        
        # Convert to torch tensors: (H, W, C) -> (C, H, W)
        fixed_tensor = torch.from_numpy(fixed_img).permute(2, 0, 1).unsqueeze(0).to(device)
        moving_tensor = torch.from_numpy(moving_img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Initialize extractor and matcher
        extractor_map = {
            "superpoint": SuperPoint,
            "disk": DISK,
            "aliked": ALIKED,
            "sift": SIFT,
        }
        
        if extractor_type not in extractor_map:
            raise ValueError(f"Unknown extractor type: {extractor_type}. Choose from {list(extractor_map.keys())}")
        
        ExtractorClass = extractor_map[extractor_type]
        
        # Build extractor with optional parameters
        extractor_kwargs = {"max_num_keypoints": max_keypoints}
        if detection_threshold is not None:
            extractor_kwargs["detection_threshold"] = detection_threshold
        if nms_window_size is not None:
            extractor_kwargs["nms_window_size"] = nms_window_size
        
        extractor = ExtractorClass(**extractor_kwargs).eval().to(device)
        
        # Build matcher with optional parameters
        matcher_kwargs = {"features": extractor_type}
        if n_layers is not None:
            matcher_kwargs["n_layers"] = n_layers
        if depth_confidence is not None:
            matcher_kwargs["depth_confidence"] = depth_confidence
        if width_confidence is not None:
            matcher_kwargs["width_confidence"] = width_confidence
        if filter_threshold is not None:
            matcher_kwargs["filter_threshold"] = filter_threshold
        
        matcher = LightGlue(**matcher_kwargs).eval().to(device)
        
        # Extract features
        logger.debug("Extracting features...")
        feats0 = extractor.extract(fixed_tensor)
        feats1 = extractor.extract(moving_tensor)
        
        # Match features
        logger.debug("Matching features with LightGlue...")
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        
        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        # Get matched points
        matches = matches01['matches']  # Shape: (K, 2)
        match_confidence = matches01.get('matching_scores', None)
        
        if len(matches) < min_matches:
            raise ValueError(f"Insufficient matches: {len(matches)}. Need at least {min_matches} for TPS.")
        
        # Filter matches by confidence if available
        if match_confidence is not None:
            confidence_threshold = 0.1
            valid_matches = match_confidence > confidence_threshold
            matches = matches[valid_matches]
            if len(matches) < min_matches:
                logger.warning(f"After confidence filtering, only {len(matches)} matches remain. Using all matches.")
                matches = matches01['matches']
        
        # Get keypoint coordinates
        keypoints0 = feats0['keypoints']  # Shape: (N, 2)
        keypoints1 = feats1['keypoints']  # Shape: (M, 2)
        
        # Extract matched points
        matched_kpts0 = keypoints0[matches[:, 0]].cpu().numpy()  # Shape: (K, 2)
        matched_kpts1 = keypoints1[matches[:, 1]].cpu().numpy()  # Shape: (K, 2)
        
        logger.info(f"Found {len(matches)} feature matches for TPS")
        
        # Apply TPS warp
        h, w = fixed_img.shape[:2]
        moving_warped = apply_tps_warp(
            moving_img,
            matched_kpts1,  # Source points (moving image)
            matched_kpts0,  # Destination points (fixed image)
            (h, w),
            smoothing=smoothing
        )
        
        # Convert to grayscale for metrics
        fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY) if len(fixed_img.shape) == 3 else fixed_img
        moving_gray_warped = cv2.cvtColor(moving_warped, cv2.COLOR_RGB2GRAY) if len(moving_warped.shape) == 3 else moving_warped
        
        # Calculate metrics
        from app.services.registration_service import (
            calculate_mutual_information,
            calculate_normalized_cross_correlation,
            calculate_structural_similarity
        )
        
        mi_score = calculate_mutual_information(fixed_gray, moving_gray_warped)
        ncc_score = calculate_normalized_cross_correlation(fixed_gray, moving_gray_warped)
        ssim_score = calculate_structural_similarity(fixed_gray, moving_gray_warped)
        
        # Calculate Dice on masks
        from app.services.registration_service import prepare_mask_exit, pad_to_size, find_relative_rotation
        mask1 = prepare_mask_exit(fixed_gray, method="otsu")
        mask2_warped = prepare_mask_exit(moving_gray_warped, method="otsu")
        h1, w1 = mask1.shape
        h2, w2 = mask2_warped.shape
        max_dim = max(h1, w1, h2, w2)
        padded_mask1 = pad_to_size(mask1, max_dim)
        padded_mask2 = pad_to_size(mask2_warped, max_dim)
        _, dice = find_relative_rotation(padded_mask1, padded_mask2)
        
        # For TPS, we don't have a simple transform matrix, but we can estimate
        # an average rotation/scale from the correspondences for display purposes
        # Use centroid-based estimation
        src_centroid = matched_kpts1.mean(axis=0)
        dst_centroid = matched_kpts0.mean(axis=0)
        src_centered = matched_kpts1 - src_centroid
        dst_centered = matched_kpts0 - dst_centroid
        
        # Estimate average rotation and scale
        if len(matched_kpts1) >= 2:
            # Use SVD for optimal rigid transform estimation
            H = src_centered.T @ dst_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            rotation_degrees = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            scale = float(np.sqrt(np.linalg.det(R)))
            offset_x = float(dst_centroid[0] - (R @ src_centroid)[0])
            offset_y = float(dst_centroid[1] - (R @ src_centroid)[1])
        else:
            rotation_degrees = 0.0
            scale = 1.0
            offset_x = float(dst_centroid[0] - src_centroid[0])
            offset_y = float(dst_centroid[1] - src_centroid[1])
        
        # Create identity transform matrix (TPS doesn't have a simple matrix representation)
        # We store the control points instead
        transform_matrix = np.eye(3).tolist()
        
        # Store match data for visualization
        # Note: For TPS, all matches are used as control points (no RANSAC filtering)
        # So we mark all as inliers for visualization purposes
        # Convert all numpy types to native Python types
        kpts0_list = keypoints0.cpu().numpy().tolist()
        kpts1_list = keypoints1.cpu().numpy().tolist()
        matches_list = matches.cpu().numpy().tolist()
        
        match_data = {
            "keypoints0": [[float(x), float(y)] for x, y in kpts0_list],
            "keypoints1": [[float(x), float(y)] for x, y in kpts1_list],
            "matches": [[int(x), int(y)] for x, y in matches_list],
            "match_scores": [float(x) for x in match_confidence.cpu().numpy().tolist()] if match_confidence is not None else None,
            "inliers": [bool(x) for x in np.ones(len(matches), dtype=bool).tolist()],  # All matches used for TPS
        }
        
        return {
            "transform_matrix": transform_matrix,
            "rotation_degrees": float(rotation_degrees),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "scale": float(scale),
            "mutual_information": float(mi_score) if mi_score is not None else 0.0,
            "normalized_cross_correlation": float(ncc_score) if ncc_score is not None else None,
            "structural_similarity": float(ssim_score) if ssim_score is not None else None,
            "dice_coefficient": float(dice) if dice is not None else None,
            "num_matches": int(len(matches)),
            "num_control_points": int(len(matched_kpts0)),
            "tps_smoothing": float(smoothing),
            "control_points_src": [[float(x), float(y)] for x, y in matched_kpts1.tolist()],
            "control_points_dst": [[float(x), float(y)] for x, y in matched_kpts0.tolist()],
            "match_data": match_data,
            "success": True,
        }
        
    except Exception as e:
        logger.error(f"TPS registration failed: {e}", exc_info=True)
        return {
            "transform_matrix": np.eye(3).tolist(),
            "rotation_degrees": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "scale": 1.0,
            "mutual_information": 0.0,
            "normalized_cross_correlation": None,
            "structural_similarity": None,
            "dice_coefficient": None,
            "num_matches": 0,
            "num_control_points": 0,
            "tps_smoothing": smoothing,
            "success": False,
            "error": str(e),
        }


def register_with_affine_tps(
    fixed_id: str,
    moving_id: str,
    extractor_type: str = "disk",  # Changed default: DISK often works better for different stains than SuperPoint
    thumbnail_width: int = 1024,
    max_keypoints: int = 2048,
    device: Optional[str] = None,
    max_matches_for_tps: int = 2000,
    affine_ransac_thresh_px: float = 3.0,
    affine_max_iters: int = 5000,
    tps_min_inliers: int = 30,
    tps_grid_step: int = 2,
    # LightGlue extractor parameters
    detection_threshold: Optional[float] = None,
    nms_window_size: Optional[int] = None,
    # LightGlue matcher parameters
    n_layers: int = 9,
    depth_confidence: float = 0.9,
    width_confidence: float = 0.99,
    filter_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Hybrid registration: LightGlue -> Affine (RANSAC) -> TPS refinement
    
    This method:
    1. Uses LightGlue for feature matching
    2. Estimates coarse affine transform using RANSAC
    3. Refines with non-rigid Thin-Plate Spline using affine inliers
    
    Good for cases where you need both global alignment (affine) and local
    refinement (TPS) for handling tissue deformations.
    
    Args:
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        extractor_type: Feature extractor type ('superpoint', 'disk', 'aliked', 'sift')
        thumbnail_width: Width of thumbnail to use for registration
        max_keypoints: Maximum number of keypoints to extract
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None
        max_matches_for_tps: Maximum number of inlier matches to use for TPS (for performance)
        affine_ransac_thresh_px: RANSAC reprojection threshold in pixels
        affine_max_iters: Maximum RANSAC iterations for affine estimation
        tps_min_inliers: Minimum inliers required to attempt TPS (otherwise returns affine-only)
        tps_grid_step: Grid step size for TPS warp (smaller = more accurate but slower)
        
        # LightGlue extractor parameters (for keypoint detection):
        detection_threshold: Minimum score for keypoint detection (lower = more keypoints, default: extractor default)
        nms_window_size: Window size for non-maximum suppression (default: extractor default)
        
        # LightGlue matcher parameters (for matching):
        n_layers: Number of attention layers/iterations (default: 9, more = better quality but slower)
        depth_confidence: Early stopping confidence (0-1, default: 0.9, higher = faster but may miss matches)
        width_confidence: Point pruning confidence (0-1, default: 0.99, higher = more aggressive pruning)
        filter_threshold: Filter threshold for matches (0-1, default: 0.1, lower = stricter filtering)
        
    Returns:
        Dictionary with registration results
    """
    if not LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue is not installed. Install with: pip install lightglue")
    
    try:
        # Check for OpenCV contrib (required for cv2.createThinPlateSplineShapeTransformer)
        try:
            tps_test = cv2.createThinPlateSplineShapeTransformer()
            del tps_test
        except AttributeError:
            raise RuntimeError("opencv-contrib-python is required for TPS. Install with: pip install opencv-contrib-python")
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get thumbnail images
        logger.debug(f"Fetching thumbnails for affine+TPS registration: fixed={fixed_id}, moving={moving_id}")
        fixed_img = get_thumbnail_image(fixed_id, width=thumbnail_width)
        moving_img = get_thumbnail_image(moving_id, width=thumbnail_width)
        
        if fixed_img is None or moving_img is None:
            raise ValueError("Failed to retrieve thumbnail images")
        
        # Convert to BGR for OpenCV (our images are RGB, need BGR)
        if len(fixed_img.shape) == 2:
            fixed_img = np.stack([fixed_img] * 3, axis=-1)
        if len(moving_img.shape) == 2:
            moving_img = np.stack([moving_img] * 3, axis=-1)
        
        # Convert RGB to BGR for OpenCV
        fixed_bgr = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2BGR) if fixed_img.shape[2] == 3 else fixed_img
        moving_bgr = cv2.cvtColor(moving_img, cv2.COLOR_RGB2BGR) if moving_img.shape[2] == 3 else moving_img
        
        # Convert to uint8
        fixed_bgr = (fixed_bgr * 255.0).astype(np.uint8) if fixed_bgr.dtype == np.float32 else fixed_bgr.astype(np.uint8)
        moving_bgr = (moving_bgr * 255.0).astype(np.uint8) if moving_bgr.dtype == np.float32 else moving_bgr.astype(np.uint8)
        
        # Convert to torch tensors for LightGlue: (H, W, C) -> (C, H, W), float32 [0,1]
        fixed_rgb = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)
        moving_rgb = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2RGB)
        fixed_tensor = torch.from_numpy(fixed_rgb).permute(2, 0, 1).float() / 255.0
        moving_tensor = torch.from_numpy(moving_rgb).permute(2, 0, 1).float() / 255.0
        
        # Move to device
        fixed_tensor = fixed_tensor.to(device)
        moving_tensor = moving_tensor.to(device)
        
        # Initialize extractor and matcher
        extractor_map = {
            "superpoint": SuperPoint,
            "disk": DISK,
            "aliked": ALIKED,
            "sift": SIFT,
        }
        
        if extractor_type not in extractor_map:
            raise ValueError(f"Unknown extractor type: {extractor_type}. Choose from {list(extractor_map.keys())}")
        
        ExtractorClass = extractor_map[extractor_type]
        
        # Build extractor with optional parameters
        extractor_kwargs = {"max_num_keypoints": max_keypoints}
        if detection_threshold is not None:
            extractor_kwargs["detection_threshold"] = detection_threshold
        if nms_window_size is not None:
            extractor_kwargs["nms_window_size"] = nms_window_size
        
        extractor = ExtractorClass(**extractor_kwargs).eval().to(device)
        
        # Build matcher with optional parameters
        matcher_kwargs = {"features": extractor_type}
        if n_layers is not None:
            matcher_kwargs["n_layers"] = n_layers
        if depth_confidence is not None:
            matcher_kwargs["depth_confidence"] = depth_confidence
        if width_confidence is not None:
            matcher_kwargs["width_confidence"] = width_confidence
        if filter_threshold is not None:
            matcher_kwargs["filter_threshold"] = filter_threshold
        
        matcher = LightGlue(**matcher_kwargs).eval().to(device)
        
        # Extract features
        logger.debug("Extracting features...")
        feats0 = extractor.extract(fixed_tensor.unsqueeze(0))
        feats1 = extractor.extract(moving_tensor.unsqueeze(0))
        
        # Match features
        logger.debug("Matching features with LightGlue...")
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        
        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        # Get matched points
        if "matches" not in matches01:
            raise ValueError("LightGlue output missing 'matches'")
        
        m = matches01["matches"]  # Shape: (K, 2)
        match_confidence = matches01.get('matching_scores', None)
        
        logger.info(f"LightGlue found {len(m)} initial matches")
        if match_confidence is not None:
            logger.info(f"Match confidence range: [{match_confidence.min().item():.3f}, {match_confidence.max().item():.3f}], mean: {match_confidence.mean().item():.3f}")
        
        if m.numel() == 0:
            raise ValueError("No matches returned by LightGlue")
        
        # Filter matches by confidence if available
        # LightGlue confidence scores are typically in [0, 1], with good matches often > 0.1
        # We use a lower threshold to keep more matches, letting RANSAC filter outliers
        if match_confidence is not None:
            # Start with a lenient threshold - RANSAC will handle outlier rejection
            confidence_threshold = 0.05  # Lower threshold to keep more matches
            valid_matches = match_confidence > confidence_threshold
            m_filtered = m[valid_matches]
            match_confidence_filtered = match_confidence[valid_matches]
            
            # Only use filtered matches if we still have enough
            if len(m_filtered) >= 10:  # Need enough for RANSAC to work well
                m = m_filtered
                match_confidence = match_confidence_filtered
                logger.info(f"Filtered matches by confidence: {len(m)} matches above threshold {confidence_threshold} (from {len(matches01['matches'])})")
            else:
                logger.info(f"Confidence filtering left only {len(m_filtered)} matches, using all {len(m)} matches (RANSAC will filter outliers)")
                # Keep all matches, let RANSAC do the filtering
        
        if len(m) < 3:
            raise ValueError(f"Not enough matches after filtering: {len(m)}. Need at least 3 for affine transform.")
        
        # Gather matched keypoints
        k0 = feats0["keypoints"]  # Shape: (N, 2)
        k1 = feats1["keypoints"]  # Shape: (M, 2)
        
        points0 = k0[m[:, 0]]
        points1 = k1[m[:, 1]]
        
        p0 = points0.detach().cpu().numpy().astype(np.float32)
        p1 = points1.detach().cpu().numpy().astype(np.float32)
        
        logger.info(f"Using {len(p0)} feature matches for affine estimation (confidence filtered)")
        
        # Optional: Filter matches by spatial consistency (topographic constraint)
        # Remove matches where nearby points in one image match to far-away points in the other
        # This helps enforce that local spatial relationships are preserved
        # Only apply if we have many matches - it can be too aggressive for challenging cases
        if len(p0) > 50:  # Only do this if we have plenty of matches
            spatial_consistent = _filter_spatial_consistency(p0, p1, max_neighbor_distance_ratio=3.0)  # More lenient ratio
            if np.sum(spatial_consistent) >= 20:  # Only use if we keep enough matches
                p0 = p0[spatial_consistent]
                p1 = p1[spatial_consistent]
                m = m[spatial_consistent]
                logger.info(f"After spatial consistency filtering: {len(p0)} matches remain")
            else:
                logger.info(f"Spatial consistency filtering too aggressive ({np.sum(spatial_consistent)} matches), skipping")
        
        # Step 2: Coarse affine with RANSAC
        # cv2.estimateAffine2D(src, dst) estimates transform: src -> dst
        # We want: moving (p1) -> fixed (p0)
        # So we call: estimateAffine2D(p1, p0) to get moving->fixed transform
        
        # Scale RANSAC threshold with image size (threshold is in pixels)
        # For smaller images, use proportionally smaller threshold
        # Base threshold of 3px works well for 1024px images
        image_size_factor = thumbnail_width / 1024.0
        scaled_ransac_thresh = affine_ransac_thresh_px * image_size_factor
        # But don't go below 1px or above 10px
        scaled_ransac_thresh = max(1.0, min(10.0, scaled_ransac_thresh))
        
        logger.info(f"RANSAC threshold: {scaled_ransac_thresh:.2f}px (scaled from {affine_ransac_thresh_px}px for {thumbnail_width}px images)")
        
        A, inliers = cv2.estimateAffine2D(
            p1,  # Source: moving image keypoints
            p0,  # Destination: fixed image keypoints
            method=cv2.RANSAC,
            ransacReprojThreshold=float(scaled_ransac_thresh),
            maxIters=int(affine_max_iters),
            confidence=0.999,
            refineIters=10,
        )
        
        if A is None or inliers is None:
            raise ValueError("cv2.estimateAffine2D failed")
        
        inliers_mask = inliers.ravel().astype(bool)
        p0_in = p0[inliers_mask]
        p1_in = p1[inliers_mask]
        
        num_inliers = len(p0_in)
        logger.info(f"Affine RANSAC found {num_inliers} inliers out of {len(p0)} matches")
        
        # cv2.estimateAffine2D returns transform: moving -> fixed
        # But cv2.warpAffine needs: fixed -> moving (inverse)
        # So we need to invert A before using it with warpAffine
        # Convert A to 3x3 homogeneous matrix for inversion
        A_3x3 = np.vstack([A, [0, 0, 1]])
        A_inv_3x3 = np.linalg.inv(A_3x3)
        A_inv = A_inv_3x3[:2, :].astype(np.float32)
        
        # Warp moving image by inverse affine transform
        out_hw = fixed_bgr.shape[:2]
        moving_affine = cv2.warpAffine(moving_bgr, A_inv, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR)
        
        # Extract affine transform parameters for return
        # A is 2x3: [a00 a01 tx; a10 a11 ty] - this maps moving -> fixed
        # For display, we want the forward transform (moving -> fixed)
        rotation_degrees = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        scale_x = np.sqrt(A[0, 0]**2 + A[0, 1]**2)
        scale_y = np.sqrt(A[1, 0]**2 + A[1, 1]**2)
        scale = float((scale_x + scale_y) / 2.0)
        offset_x = float(A[0, 2])
        offset_y = float(A[1, 2])
        
        # Convert to 3x3 homogeneous matrix (for storage - this is moving -> fixed)
        A_3x3 = np.vstack([A, [0, 0, 1]])
        
        # Step 3: TPS refinement (if enough inliers)
        moving_tps = None
        notes = []
        
        if num_inliers < tps_min_inliers:
            notes.append(f"Only {num_inliers} affine inliers; returning affine-only (TPS skipped, need >= {tps_min_inliers})")
            moving_final = moving_affine
        else:
            # Optionally cap number of control points for TPS
            if num_inliers > max_matches_for_tps:
                rng = np.random.default_rng(0)  # deterministic subsample
                idx = rng.choice(num_inliers, size=max_matches_for_tps, replace=False)
                p0_in = p0_in[idx]
                p1_in = p1_in[idx]
                notes.append(f"Subsampled TPS control points to {max_matches_for_tps}")
            
            # TPS maps from affine-warped coordinates to fixed image coordinates
            # p1_in are points in ORIGINAL moving image (inliers from RANSAC)
            # p0_in are points in fixed image (corresponding inliers)
            # A maps p1_in -> p0_in (approximately, since inliers)
            # After warping moving by A_inv, moving_affine is in fixed space
            # For TPS refinement: we want to map from moving_affine to fixed
            # The source points should be where p1_in ended up in moving_affine
            # Since A maps p1_in->p0_in, after warping by A_inv, p1_in are at approximately p0_in
            # But for exact TPS, we use the actual transformed locations
            ones = np.ones((p1_in.shape[0], 1), dtype=np.float32)
            p1_h = np.concatenate([p1_in, ones], axis=1)  # (N,3)
            # Transform p1_in by A to get where they are in fixed space (should be ~p0_in for inliers)
            p1_in_fixed = (A @ p1_h.T).T.astype(np.float32)  # (N,2) - where p1_in are after affine warp
            
            # For TPS: source = p1_in_fixed (locations in moving_affine/fixed), dest = p0_in (target in fixed)
            # Since they're inliers, p1_in_fixed ≈ p0_in, so TPS will do fine local refinement
            src = p1_in_fixed.reshape(-1, 1, 2)  # source points in moving_affine (fixed space)
            dst = p0_in.reshape(-1, 1, 2)   # destination points in fixed image space
            
            src = p1_in_fixed.reshape(-1, 1, 2)  # source points in moving_affine space
            dst = p0_in.reshape(-1, 1, 2)   # destination points in fixed image space
            
            # DMatch pairs
            dmatches = [cv2.DMatch(i, i, 0) for i in range(len(p1_in_fixed))]
            
            # Estimate TPS transform
            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(dst, src, dmatches)
            
            # Use scipy-based TPS warp instead of OpenCV's applyTransformation
            # OpenCV's applyTransformation has issues with return value format
            # We'll use the existing apply_tps_warp function which uses scipy
            H, W = out_hw
            
            # Convert control points to the format expected by apply_tps_warp
            # p1_in_fixed are points in moving_affine space (source, after affine warp)
            # p0_in are points in fixed image space (destination)
            src_points = p1_in_fixed.astype(np.float64)  # (N, 2) - points in moving_affine space
            dst_points = p0_in.astype(np.float64)    # (N, 2) - points in fixed image space
            
            # Apply TPS warp using scipy-based implementation
            moving_tps = apply_tps_warp(
                src_img=moving_affine,
                src_points=src_points,
                dst_points=dst_points,
                target_shape=(H, W),
                smoothing=1.0
            )
            moving_final = moving_tps
            notes.append(f"Applied TPS refinement with {len(p0_in)} control points")
        
        # Convert back to RGB for metrics
        moving_final_rgb = cv2.cvtColor(moving_final, cv2.COLOR_BGR2RGB)
        fixed_rgb = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for metrics
        fixed_gray = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY) if len(fixed_rgb.shape) == 3 else fixed_rgb
        moving_gray = cv2.cvtColor(moving_final_rgb, cv2.COLOR_RGB2GRAY) if len(moving_final_rgb.shape) == 3 else moving_final_rgb
        
        # Ensure images are in uint8 format [0, 255] for mask generation
        # fixed_rgb and moving_final_rgb are in [0, 1] range from torch, so convert
        if fixed_gray.dtype != np.uint8:
            if fixed_gray.max() <= 1.0:
                fixed_gray = (np.clip(fixed_gray, 0, 1) * 255).astype(np.uint8)
            else:
                fixed_gray = np.clip(fixed_gray, 0, 255).astype(np.uint8)
        if moving_gray.dtype != np.uint8:
            if moving_gray.max() <= 1.0:
                moving_gray = (np.clip(moving_gray, 0, 1) * 255).astype(np.uint8)
            else:
                moving_gray = np.clip(moving_gray, 0, 255).astype(np.uint8)
        
        logger.debug(f"Image stats for Dice: fixed shape={fixed_gray.shape}, dtype={fixed_gray.dtype}, range=[{fixed_gray.min()}, {fixed_gray.max()}], moving shape={moving_gray.shape}, dtype={moving_gray.dtype}, range=[{moving_gray.min()}, {moving_gray.max()}]")
        
        # Calculate metrics
        from app.services.registration_service import (
            calculate_mutual_information,
            calculate_normalized_cross_correlation,
            calculate_structural_similarity,
            prepare_mask_exit,
            pad_to_size,
            find_relative_rotation
        )
        
        # Ensure images are same size for metrics
        h_fixed, w_fixed = fixed_gray.shape[:2]
        h_moving, w_moving = moving_gray.shape[:2]
        if h_fixed != h_moving or w_fixed != w_moving:
            logger.warning(f"Image size mismatch for metrics: fixed={h_fixed}x{w_fixed}, moving={h_moving}x{w_moving}. Resizing moving to match fixed.")
            moving_gray = cv2.resize(moving_gray, (w_fixed, h_fixed), interpolation=cv2.INTER_LINEAR)
        
        mi_score = None
        ncc_score = None
        ssim_score = None
        dice = None
        
        try:
            mi_score = calculate_mutual_information(fixed_gray, moving_gray)
            logger.debug(f"MI score: {mi_score}")
        except Exception as e:
            logger.warning(f"Failed to calculate MI: {e}", exc_info=True)
        
        try:
            ncc_score = calculate_normalized_cross_correlation(fixed_gray, moving_gray)
            logger.debug(f"NCC score: {ncc_score}")
        except Exception as e:
            logger.warning(f"Failed to calculate NCC: {e}", exc_info=True)
        
        try:
            ssim_score = calculate_structural_similarity(fixed_gray, moving_gray)
            logger.debug(f"SSIM score: {ssim_score}")
        except Exception as e:
            logger.warning(f"Failed to calculate SSIM: {e}", exc_info=True)
        
        # Calculate Dice on masks
        # Note: For registered images, we don't need to test rotations - they should already be aligned
        # Different stains may produce different mask shapes even with good registration,
        # so Dice might be lower than expected for different stain types
        try:
            # Images should already be uint8 from above, but double-check
            mask1 = prepare_mask_exit(fixed_gray, method="otsu")
            mask2 = prepare_mask_exit(moving_gray, method="otsu")
            
            # Ensure masks are same size (they should be after registration, but check anyway)
            h1, w1 = mask1.shape
            h2, w2 = mask2.shape
            if h1 != h2 or w1 != w2:
                logger.warning(f"Mask size mismatch: fixed={h1}x{w1}, moving={h2}x{w2}. Resizing moving mask.")
                mask2 = cv2.resize(mask2, (w1, h1), interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary for Dice calculation
            # prepare_mask_exit returns masks where tissue=255, background=0
            mask1_binary = (mask1 > 127).astype(np.uint8)  # Use 127 as threshold (middle of 0-255)
            mask2_binary = (mask2 > 127).astype(np.uint8)
            
            # Log mask statistics for debugging
            mask1_pct = (mask1_binary.sum() / mask1_binary.size) * 100 if mask1_binary.size > 0 else 0
            mask2_pct = (mask2_binary.sum() / mask2_binary.size) * 100 if mask2_binary.size > 0 else 0
            logger.info(f"Mask stats: fixed={mask1_pct:.1f}% tissue ({mask1_binary.sum()} pixels), moving={mask2_pct:.1f}% tissue ({mask2_binary.sum()} pixels), total={mask1_binary.size} pixels")
            
            # Calculate Dice directly (no rotation testing needed - images are already registered)
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            size1 = mask1_binary.sum()
            size2 = mask2_binary.sum()
            
            logger.info(f"Dice calculation: intersection={intersection}, size1={size1}, size2={size2}, total_pixels={mask1_binary.size}")
            
            if (size1 + size2) == 0:
                logger.warning("Both masks are empty - cannot calculate Dice")
                dice = 0.0
            else:
                dice = 2.0 * intersection / (size1 + size2)
                dice = min(1.0, max(0.0, dice))  # Clamp to [0, 1]
            
            logger.info(f"Dice score: {dice:.4f} (intersection={intersection}, union={size1 + size2}, formula=2*{intersection}/({size1}+{size2}))")
            
            # Also calculate overlap percentage for debugging
            overlap_pct = (intersection / mask1_binary.size) * 100 if mask1_binary.size > 0 else 0.0
            logger.info(f"Overlap: {overlap_pct:.1f}% of image area ({intersection}/{mask1_binary.size} pixels)")
            
            # If Dice is suspiciously low, try alternative: use a more lenient threshold
            # This helps when Otsu finds very different thresholds for different stains
            if dice < 0.1 and size1 > 0 and size2 > 0:
                logger.warning(f"Dice is very low ({dice:.4f}). Trying alternative mask generation with fixed threshold...")
                # Try using a fixed threshold based on image intensity percentiles
                # This is more robust to different stain intensities
                fixed_thresh1 = np.percentile(fixed_gray, 50)  # Median intensity
                fixed_thresh2 = np.percentile(moving_gray, 50)
                # Use the lower threshold to be more inclusive
                combined_thresh = min(fixed_thresh1, fixed_thresh2)
                
                alt_mask1 = (fixed_gray < combined_thresh).astype(np.uint8)  # Darker = tissue
                alt_mask2 = (moving_gray < combined_thresh).astype(np.uint8)
                
                alt_intersection = np.logical_and(alt_mask1, alt_mask2).sum()
                alt_size1 = alt_mask1.sum()
                alt_size2 = alt_mask2.sum()
                
                if (alt_size1 + alt_size2) > 0:
                    alt_dice = 2.0 * alt_intersection / (alt_size1 + alt_size2)
                    logger.info(f"Alternative Dice (fixed threshold {combined_thresh:.1f}): {alt_dice:.4f} (intersection={alt_intersection}, sizes={alt_size1}, {alt_size2})")
                    # Use the higher Dice score
                    if alt_dice > dice:
                        dice = alt_dice
                        logger.info(f"Using alternative Dice score: {dice:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to calculate Dice: {e}", exc_info=True)
            dice = None
        
        # Store match data for visualization
        # Note: Store ALL original matches (before confidence filtering) for visualization
        # But mark which ones passed confidence filtering
        m_original = matches01["matches"]
        match_confidence_original = matches01.get('matching_scores', None)
        
        kpts0_list = k0.cpu().numpy().tolist()
        kpts1_list = k1.cpu().numpy().tolist()
        matches_list_original = m_original.cpu().numpy().tolist()
        
        # Create a mask for matches that passed confidence filtering
        confidence_passed = None
        if match_confidence_original is not None:
            confidence_threshold = 0.2
            confidence_passed = (match_confidence_original > confidence_threshold).cpu().numpy().tolist()
            # If too few passed, use lower threshold
            if sum(confidence_passed) < 3:
                confidence_threshold = 0.1
                confidence_passed = (match_confidence_original > confidence_threshold).cpu().numpy().tolist()
        
        match_data = {
            "keypoints0": [[float(x), float(y)] for x, y in kpts0_list],
            "keypoints1": [[float(x), float(y)] for x, y in kpts1_list],
            "matches": [[int(x), int(y)] for x, y in matches_list_original],
            "match_scores": [float(x) for x in match_confidence_original.cpu().numpy().tolist()] if match_confidence_original is not None else None,
            "inliers": [bool(x) for x in inliers_mask.tolist()],
            "confidence_passed": confidence_passed,  # Additional info: which matches passed confidence filtering
        }
        
        # Generate warped overlay thumbnail for visualization (base64 encoded)
        warped_overlay_base64 = None
        try:
            import base64
            # Use the already-aligned images for overlay
            # fixed_rgb and moving_final_rgb should be the same size after registration
            h_fo, w_fo = fixed_rgb.shape[:2]
            h_mo, w_mo = moving_final_rgb.shape[:2]
            
            # Ensure same size (they should be, but check)
            if h_fo != h_mo or w_fo != w_mo:
                moving_overlay = cv2.resize(moving_final_rgb, (w_fo, h_fo), interpolation=cv2.INTER_LINEAR)
            else:
                moving_overlay = moving_final_rgb.copy()
            
            fixed_overlay = fixed_rgb.copy()
            
            # Convert to uint8 if needed (images should already be in [0, 1] range from torch)
            if fixed_overlay.dtype != np.uint8:
                if fixed_overlay.max() <= 1.0:
                    fixed_overlay = (np.clip(fixed_overlay, 0, 1) * 255).astype(np.uint8)
                else:
                    fixed_overlay = np.clip(fixed_overlay, 0, 255).astype(np.uint8)
            if moving_overlay.dtype != np.uint8:
                if moving_overlay.max() <= 1.0:
                    moving_overlay = (np.clip(moving_overlay, 0, 1) * 255).astype(np.uint8)
                else:
                    moving_overlay = np.clip(moving_overlay, 0, 255).astype(np.uint8)
            
            # Resize overlay to match thumbnail_width (but cap at reasonable size for JSON response)
            # Use thumbnail_width as the target, but don't exceed 1024px to keep response size manageable
            target_size = min(thumbnail_width, 1024)
            max_dim = max(w_fo, h_fo)
            if max_dim > target_size:
                scale = target_size / max_dim
                new_w = int(w_fo * scale)
                new_h = int(h_fo * scale)
                fixed_overlay = cv2.resize(fixed_overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                moving_overlay = cv2.resize(moving_overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                logger.debug(f"Resized overlay from {w_fo}x{h_fo} to {new_w}x{new_h} (target: {target_size}px, thumbnail_width: {thumbnail_width}px)")
            
            # Create overlay with 50% opacity
            overlay = cv2.addWeighted(fixed_overlay, 0.5, moving_overlay, 0.5, 0)
            
            # Encode as PNG base64
            success, buffer = cv2.imencode('.png', overlay)
            if success:
                warped_overlay_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                logger.debug(f"Generated warped overlay thumbnail ({overlay.shape[1]}x{overlay.shape[0]})")
        except Exception as e:
            logger.warning(f"Failed to generate warped overlay thumbnail: {e}", exc_info=True)
        
        return {
            "transform_matrix": [[float(x) for x in row] for row in A_3x3.tolist()],
            "rotation_degrees": rotation_degrees,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "scale": scale,
            "thumbnail_width": thumbnail_width,  # Store the image size used for registration
            "mutual_information": float(mi_score) if mi_score is not None else 0.0,
            "normalized_cross_correlation": float(ncc_score) if ncc_score is not None else None,
            "structural_similarity": float(ssim_score) if ssim_score is not None else None,
            "dice_coefficient": float(dice) if dice is not None else None,
            "num_matches": int(len(p0)),
            "num_inliers": int(num_inliers),
            "match_data": match_data,
            "affine_matrix": [[float(x) for x in row] for row in A.tolist()],
            "tps_applied": moving_tps is not None,
            "notes": notes,
            "warped_overlay_base64": warped_overlay_base64,  # Base64-encoded PNG thumbnail
            "success": True,
        }
        
    except Exception as e:
        logger.error(f"Affine+TPS registration failed: {e}", exc_info=True)
        return {
            "transform_matrix": np.eye(3).tolist(),
            "rotation_degrees": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "scale": 1.0,
            "mutual_information": 0.0,
            "normalized_cross_correlation": None,
            "structural_similarity": None,
            "dice_coefficient": None,
            "num_matches": 0,
            "num_inliers": 0,
            "success": False,
            "error": str(e),
        }

