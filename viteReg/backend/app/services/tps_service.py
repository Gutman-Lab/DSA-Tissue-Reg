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
    min_matches: int = 6
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
        extractor = ExtractorClass(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features=extractor_type).eval().to(device)
        
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
    extractor_type: str = "superpoint",
    thumbnail_width: int = 1024,
    max_keypoints: int = 2048,
    device: Optional[str] = None,
    max_matches_for_tps: int = 2000,
    affine_ransac_thresh_px: float = 3.0,
    affine_max_iters: int = 5000,
    tps_min_inliers: int = 30,
    tps_grid_step: int = 2,
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
        extractor = ExtractorClass(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features=extractor_type).eval().to(device)
        
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
        if m.numel() == 0:
            raise ValueError("No matches returned by LightGlue")
        
        # Gather matched keypoints
        k0 = feats0["keypoints"]  # Shape: (N, 2)
        k1 = feats1["keypoints"]  # Shape: (M, 2)
        
        points0 = k0[m[:, 0]]
        points1 = k1[m[:, 1]]
        
        p0 = points0.detach().cpu().numpy().astype(np.float32)
        p1 = points1.detach().cpu().numpy().astype(np.float32)
        
        if len(p0) < 3:
            raise ValueError(f"Not enough matches for affine (need >= 3, got {len(p0)})")
        
        logger.info(f"Found {len(p0)} feature matches")
        
        # Step 2: Coarse affine with RANSAC
        # cv2.estimateAffine2D(src, dst) estimates transform: src -> dst
        # We want: moving (p1) -> fixed (p0)
        # So we call: estimateAffine2D(p1, p0) to get moving->fixed transform
        A, inliers = cv2.estimateAffine2D(
            p1,  # Source: moving image keypoints
            p0,  # Destination: fixed image keypoints
            method=cv2.RANSAC,
            ransacReprojThreshold=float(affine_ransac_thresh_px),
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
        
        # Warp moving image by affine
        out_hw = fixed_bgr.shape[:2]
        moving_affine = cv2.warpAffine(moving_bgr, A, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR)
        
        # Extract affine transform parameters for return
        # A is 2x3: [a00 a01 tx; a10 a11 ty]
        rotation_degrees = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        scale_x = np.sqrt(A[0, 0]**2 + A[0, 1]**2)
        scale_y = np.sqrt(A[1, 0]**2 + A[1, 1]**2)
        scale = float((scale_x + scale_y) / 2.0)
        offset_x = float(A[0, 2])
        offset_y = float(A[1, 2])
        
        # Convert to 3x3 homogeneous matrix
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
            # p0_in are points in ORIGINAL moving image, p1_in are in fixed image
            # We need to transform p0_in by A to get points in moving_affine space
            ones = np.ones((p0_in.shape[0], 1), dtype=np.float32)
            p0_h = np.concatenate([p0_in, ones], axis=1)  # (N,3)
            p0_aff = (A @ p0_h.T).T.astype(np.float32)  # (N,2) - points in moving_affine space
            
            src = p0_aff.reshape(-1, 1, 2)  # source points in moving_affine space
            dst = p1_in.reshape(-1, 1, 2)   # destination points in fixed image space
            
            # DMatch pairs
            dmatches = [cv2.DMatch(i, i, 0) for i in range(len(p0_aff))]
            
            # Estimate TPS transform
            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(dst, src, dmatches)
            
            # Build dense mapping
            H, W = out_hw
            xs = np.arange(0, W, tps_grid_step, dtype=np.float32)
            ys = np.arange(0, H, tps_grid_step, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
            
            # Apply TPS transformation
            _, warped = tps.applyTransformation(grid)
            warped = warped.reshape(grid_y.shape[0], grid_x.shape[1], 2)
            
            # Upsample maps to full resolution
            map_x_small = warped[..., 0]
            map_y_small = warped[..., 1]
            map_x = cv2.resize(map_x_small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            map_y = cv2.resize(map_y_small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            
            # Apply TPS warp
            moving_tps = cv2.remap(moving_affine, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            moving_final = moving_tps
            notes.append(f"Applied TPS refinement with {len(p0_in)} control points")
        
        # Convert back to RGB for metrics
        moving_final_rgb = cv2.cvtColor(moving_final, cv2.COLOR_BGR2RGB)
        fixed_rgb = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for metrics
        fixed_gray = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY) if len(fixed_rgb.shape) == 3 else fixed_rgb
        moving_gray = cv2.cvtColor(moving_final_rgb, cv2.COLOR_RGB2GRAY) if len(moving_final_rgb.shape) == 3 else moving_final_rgb
        
        # Calculate metrics
        from app.services.registration_service import (
            calculate_mutual_information,
            calculate_normalized_cross_correlation,
            calculate_structural_similarity,
            prepare_mask_exit,
            pad_to_size,
            find_relative_rotation
        )
        
        mi_score = calculate_mutual_information(fixed_gray, moving_gray)
        ncc_score = calculate_normalized_cross_correlation(fixed_gray, moving_gray)
        ssim_score = calculate_structural_similarity(fixed_gray, moving_gray)
        
        # Calculate Dice on masks
        mask1 = prepare_mask_exit(fixed_gray, method="otsu")
        mask2 = prepare_mask_exit(moving_gray, method="otsu")
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        max_dim = max(h1, w1, h2, w2)
        padded_mask1 = pad_to_size(mask1, max_dim)
        padded_mask2 = pad_to_size(mask2, max_dim)
        _, dice = find_relative_rotation(padded_mask1, padded_mask2)
        
        # Store match data for visualization
        kpts0_list = k0.cpu().numpy().tolist()
        kpts1_list = k1.cpu().numpy().tolist()
        matches_list = m.cpu().numpy().tolist()
        match_confidence = matches01.get('matching_scores', None)
        
        match_data = {
            "keypoints0": [[float(x), float(y)] for x, y in kpts0_list],
            "keypoints1": [[float(x), float(y)] for x, y in kpts1_list],
            "matches": [[int(x), int(y)] for x, y in matches_list],
            "match_scores": [float(x) for x in match_confidence.cpu().numpy().tolist()] if match_confidence is not None else None,
            "inliers": [bool(x) for x in inliers_mask.tolist()],
        }
        
        return {
            "transform_matrix": [[float(x) for x in row] for row in A_3x3.tolist()],
            "rotation_degrees": rotation_degrees,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "scale": scale,
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

