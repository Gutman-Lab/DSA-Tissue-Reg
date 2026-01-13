"""
LightGlue-based registration service
Uses LightGlue for feature matching, then estimates rigid transform
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


def register_with_lightglue(
    fixed_id: str,
    moving_id: str,
    extractor_type: str = "superpoint",
    thumbnail_width: int = 1024,
    max_keypoints: int = 2048,
    device: Optional[str] = None,
    # LightGlue extractor parameters
    detection_threshold: Optional[float] = None,
    nms_window_size: Optional[int] = None,
    # LightGlue matcher parameters
    n_layers: int = 9,
    depth_confidence: float = 0.9,
    width_confidence: float = 0.99,
    filter_threshold: float = 0.1,
    # Note: LightGlue is for rigid transforms only (rotation + translation)
    # For affine transforms, use SimpleITK instead
) -> Dict[str, Any]:
    """
    Perform registration using LightGlue feature matching
    
    Args:
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        extractor_type: Feature extractor type ('superpoint', 'disk', 'aliked', 'sift')
        thumbnail_width: Width of thumbnail to use for registration
        max_keypoints: Maximum number of keypoints to extract
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None
        
        # LightGlue extractor parameters (for keypoint detection):
        detection_threshold: Minimum score for keypoint detection (lower = more keypoints, default: extractor default)
        nms_window_size: Window size for non-maximum suppression (default: extractor default)
        
        # LightGlue matcher parameters (for matching):
        n_layers: Number of attention layers/iterations (default: 9, more = better quality but slower)
        depth_confidence: Early stopping confidence (0-1, default: 0.9, higher = faster but may miss matches)
        width_confidence: Point pruning confidence (0-1, default: 0.99, higher = more aggressive pruning)
        filter_threshold: Filter threshold for matches (0-1, default: 0.1, lower = stricter filtering)
        
    Returns:
        Dictionary with registration results:
        - transform_matrix: 3x3 affine transformation matrix
        - rotation_degrees: Rotation in degrees
        - offset_x: X translation
        - offset_y: Y translation
        - scale: Scale factor (always 1.0 for rigid)
        - num_matches: Number of matched features
        - success: Whether registration succeeded
    """
    if not LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue is not installed. Install with: pip install lightglue")
    
    try:
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get thumbnail images
        logger.debug(f"Fetching thumbnails for LightGlue registration: fixed={fixed_id}, moving={moving_id}")
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
        
        # Minimum matches for rigid transform (needs at least 2 points)
        min_matches = 2
        if len(matches) < min_matches:
            raise ValueError(f"Insufficient matches: {len(matches)}. Need at least {min_matches} for rigid transform estimation.")
        
        # Filter matches by confidence if available
        if match_confidence is not None:
            confidence_threshold = 0.1
            valid_matches = match_confidence > confidence_threshold
            matches = matches[valid_matches]
            if len(matches) < 4:
                logger.warning(f"After confidence filtering, only {len(matches)} matches remain. Using all matches.")
                matches = matches01['matches']
        
        # Get keypoint coordinates
        keypoints0 = feats0['keypoints']  # Shape: (N, 2)
        keypoints1 = feats1['keypoints']  # Shape: (M, 2)
        
        # Extract matched points
        matched_kpts0 = keypoints0[matches[:, 0]].cpu().numpy()  # Shape: (K, 2)
        matched_kpts1 = keypoints1[matches[:, 1]].cpu().numpy()  # Shape: (K, 2)
        
        logger.info(f"Found {len(matches)} feature matches")
        
        # LightGlue is for rigid transforms only (rotation + translation, 3 DOF)
        # The many matches are used by RANSAC for robust outlier rejection
        # For affine transforms, use SimpleITK instead (intensity-based optimization)
        # Transform direction: moving (kpts1) -> fixed (kpts0)
        transform_matrix, inliers = estimate_rigid_transform_ransac(
            matched_kpts1, matched_kpts0, max_iters=2000, threshold=3.0
        )
        
        if transform_matrix is None:
            raise ValueError("Failed to estimate rigid transform from matches")
        
        logger.info(f"RANSAC found {int(np.sum(inliers))} inliers out of {len(matches)} matches using rigid transform")
        
        # Extract transform parameters (convert to native Python floats)
        # Rigid transform: 3x3 homogeneous matrix (rotation + translation, no scale)
        rotation_degrees = float(np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])))
        offset_x = float(transform_matrix[0, 2])
        offset_y = float(transform_matrix[1, 2])
        scale = 1.0  # Rigid transform has no scaling
        transform_matrix_3x3 = transform_matrix
        
        # Calculate metrics on registered image
        h, w = fixed_img.shape[:2]
        # Extract 2x3 matrix for cv2.warpAffine (rigid transform)
        affine_matrix = transform_matrix[:2, :]
        
        moving_registered = cv2.warpAffine(
            moving_img,
            affine_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Convert to grayscale for metrics
        fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY) if len(fixed_img.shape) == 3 else fixed_img
        moving_gray_reg = cv2.cvtColor(moving_registered, cv2.COLOR_RGB2GRAY) if len(moving_registered.shape) == 3 else moving_registered
        
        # Calculate metrics
        from app.services.registration_service import (
            calculate_mutual_information,
            calculate_normalized_cross_correlation,
            calculate_structural_similarity
        )
        
        mi_score = calculate_mutual_information(fixed_gray, moving_gray_reg)
        ncc_score = calculate_normalized_cross_correlation(fixed_gray, moving_gray_reg)
        ssim_score = calculate_structural_similarity(fixed_gray, moving_gray_reg)
        
        # Calculate Dice on masks
        from app.services.registration_service import prepare_mask_exit, pad_to_size, find_relative_rotation
        mask1 = prepare_mask_exit(fixed_gray, method="otsu")
        mask2_reg = cv2.warpAffine(
            prepare_mask_exit(moving_gray_reg, method="otsu"),
            transform_matrix[:2, :],
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        h1, w1 = mask1.shape
        h2, w2 = mask2_reg.shape
        max_dim = max(h1, w1, h2, w2)
        padded_mask1 = pad_to_size(mask1, max_dim)
        padded_mask2 = pad_to_size(mask2_reg, max_dim)
        _, dice = find_relative_rotation(padded_mask1, padded_mask2)
        
        # Store match data for visualization
        # Convert all numpy types to native Python types for JSON serialization
        kpts0_list = keypoints0.cpu().numpy().tolist()
        kpts1_list = keypoints1.cpu().numpy().tolist()
        matches_list = matches.cpu().numpy().tolist()
        
        match_data = {
            "keypoints0": [[float(x), float(y)] for x, y in kpts0_list],
            "keypoints1": [[float(x), float(y)] for x, y in kpts1_list],
            "matches": [[int(x), int(y)] for x, y in matches_list],
            "match_scores": [float(x) for x in match_confidence.cpu().numpy().tolist()] if match_confidence is not None else None,
            "inliers": [bool(x) for x in inliers.tolist()] if inliers is not None else None,
        }
        
        return {
            "transform_matrix": [[float(x) for x in row] for row in transform_matrix_3x3.tolist()],
            "rotation_degrees": float(rotation_degrees),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "scale": float(scale),
            "mutual_information": float(mi_score) if mi_score is not None else 0.0,
            "normalized_cross_correlation": float(ncc_score) if ncc_score is not None else None,
            "structural_similarity": float(ssim_score) if ssim_score is not None else None,
            "dice_coefficient": float(dice) if dice is not None else None,
            "num_matches": int(len(matches)),
            "num_inliers": int(np.sum(inliers)),
            "match_data": match_data,
            "success": True,
        }
        
    except Exception as e:
        logger.error(f"LightGlue registration failed: {e}", exc_info=True)
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


def estimate_rigid_transform_ransac(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    max_iters: int = 2000,
    threshold: float = 3.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate rigid transform (rotation + translation) using RANSAC
    
    Args:
        src_points: Source points (N, 2)
        dst_points: Destination points (N, 2)
        max_iters: Maximum RANSAC iterations
        threshold: Inlier threshold in pixels
        
    Returns:
        transform_matrix: 3x3 transformation matrix (or None if failed)
        inliers: Boolean array indicating inliers (or None if failed)
    """
    if len(src_points) < 2:
        return None, None
    
    best_transform = None
    best_inliers = None
    best_inlier_count = 0
    
    for _ in range(max_iters):
        # Randomly sample 2 points
        indices = np.random.choice(len(src_points), size=2, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]
        
        # Estimate transform from 2 points
        # For rigid transform: rotation + translation
        # We need at least 2 points to estimate rotation and translation
        
        # Calculate translation from centroid
        src_centroid = src_sample.mean(axis=0)
        dst_centroid = dst_sample.mean(axis=0)
        
        # Center points
        src_centered = src_sample - src_centroid
        dst_centered = dst_sample - dst_centroid
        
        # Calculate rotation angle
        # Use atan2 of the angle between vectors
        if np.linalg.norm(src_centered[0]) > 1e-6 and np.linalg.norm(dst_centered[0]) > 1e-6:
            angle = np.arctan2(
                np.cross(src_centered[0], dst_centered[0]),
                np.dot(src_centered[0], dst_centered[0])
            )
        else:
            continue
        
        # Build rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Calculate translation
        t = dst_centroid - R @ src_centroid
        
        # Build 2x3 transform matrix
        transform_2x3 = np.hstack([R, t.reshape(2, 1)])
        
        # Test on all points
        src_homogeneous = np.hstack([src_points, np.ones((len(src_points), 1))])
        dst_predicted = (transform_2x3 @ src_homogeneous.T).T
        
        # Calculate distances
        distances = np.linalg.norm(dst_points - dst_predicted, axis=1)
        inliers = distances < threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_transform = transform_2x3
            best_inliers = inliers
    
    if best_transform is None:
        return None, None
    
    # Refine transform using all inliers
    if best_inlier_count >= 2:
        src_inliers = src_points[best_inliers]
        dst_inliers = dst_points[best_inliers]
        
        # Use SVD to get optimal rigid transform from inliers
        src_centroid = src_inliers.mean(axis=0)
        dst_centroid = dst_inliers.mean(axis=0)
        
        src_centered = src_inliers - src_centroid
        dst_centered = dst_inliers - dst_centroid
        
        # SVD for optimal rotation
        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = dst_centroid - R @ src_centroid
        
        # Build final transform matrix
        transform_2x3 = np.hstack([R, t.reshape(2, 1)])
        
        # Convert to 3x3 homogeneous
        transform_3x3 = np.vstack([transform_2x3, [0, 0, 1]])
        
        return transform_3x3, best_inliers
    
    return None, None

