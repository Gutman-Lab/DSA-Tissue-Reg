"""
Visualization endpoints for registration results
"""
import numpy as np
import cv2
import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional
from app.services.dsa_client import get_dsa_client
from app.services.registration_service import get_thumbnail_image

logger = logging.getLogger(__name__)
router = APIRouter()


def draw_feature_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    matches: np.ndarray,
    inliers: Optional[np.ndarray] = None,
    max_matches: int = 100
) -> np.ndarray:
    """
    Draw feature matches between two images
    
    Args:
        img0: First image (fixed)
        img1: Second image (moving)
        keypoints0: Keypoints in first image (N, 2)
        keypoints1: Keypoints in second image (M, 2)
        matches: Match indices (K, 2) - matches[:, 0] are indices into keypoints0, matches[:, 1] into keypoints1
        inliers: Boolean array indicating inlier matches (optional)
        max_matches: Maximum number of matches to draw
        
    Returns:
        Visualization image with matches drawn
    """
    # Convert to RGB if needed
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    
    # Ensure uint8
    if img0.dtype != np.uint8:
        img0 = (img0 * 255).astype(np.uint8) if img0.max() <= 1.0 else img0.astype(np.uint8)
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1
    
    # Create side-by-side visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    viz[:h0, :w0] = img0
    viz[:h1, w0:w0+w1] = img1
    
    # Limit number of matches to draw
    num_matches = min(len(matches), max_matches)
    indices = np.random.choice(len(matches), num_matches, replace=False) if len(matches) > max_matches else np.arange(len(matches))
    
    # Generate random colors for each match pair (for keypoints)
    # Use a fixed seed based on match indices for reproducibility
    np.random.seed(42)
    match_colors = []
    for idx in indices:
        # Generate a random color (avoid too dark colors for visibility)
        color = tuple(np.random.randint(50, 255, size=3).tolist())
        match_colors.append(color)
    
    # Draw matches
    for i, idx in enumerate(indices):
        match = matches[idx]
        kp0 = keypoints0[match[0]]
        kp1 = keypoints1[match[1]]
        
        x0, y0 = int(kp0[0]), int(kp0[1])
        x1, y1 = int(kp1[0]) + w0, int(kp1[1])  # Shift x1 to second image
        
        # Get random color for this match pair (use same color for line and points)
        point_color = match_colors[i]
        line_color = point_color
        
        # Determine if this is an inlier or outlier
        is_inlier = False
        if inliers is not None and idx < len(inliers):
            is_inlier = inliers[idx]
        
        # Draw line with same color as points
        # Use dashed line for outliers, solid for inliers
        if is_inlier:
            # Solid line for inliers
            cv2.line(viz, (x0, y0), (x1, y1), line_color, 2, cv2.LINE_AA)
        else:
            # Dashed line for outliers
            # Draw dashed line by drawing multiple short segments
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                num_segments = max(8, int(length / 10))  # Adjust dash pattern based on line length
                for j in range(num_segments):
                    t0 = j / num_segments
                    t1 = (j + 0.5) / num_segments  # 50% dash, 50% gap
                    if t1 > 1.0:
                        t1 = 1.0
                    x_start = int(x0 + dx * t0)
                    y_start = int(y0 + dy * t0)
                    x_end = int(x0 + dx * t1)
                    y_end = int(y0 + dy * t1)
                    cv2.line(viz, (x_start, y_start), (x_end, y_end), line_color, 1, cv2.LINE_AA)
            else:
                # Very short line, just draw it solid
                cv2.line(viz, (x0, y0), (x1, y1), line_color, 1, cv2.LINE_AA)
        
        # Draw keypoints with random color for each pair
        cv2.circle(viz, (x0, y0), 4, point_color, -1)
        cv2.circle(viz, (x1, y1), 4, point_color, -1)
        # Add a small border to keypoints for better visibility
        cv2.circle(viz, (x0, y0), 4, (255, 255, 255), 1)
        cv2.circle(viz, (x1, y1), 4, (255, 255, 255), 1)
    
    return viz


@router.get("/warped-overlay/{fixed_id}/{moving_id}")
async def get_warped_overlay(
    fixed_id: str,
    moving_id: str,
    method: str = Query("simpleitk", description="Registration method to get transform from"),
    opacity: float = Query(0.5, ge=0.0, le=1.0, description="Overlay opacity (0.0-1.0)"),
    width: int = Query(1024, description="Image width for overlay")
):
    """
    Generate a warped overlay visualization
    
    Returns an image with the fixed image as base and the warped moving image overlaid.
    The moving image is warped using the registration transform matrix.
    """
    try:
        dsa = get_dsa_client()
        
        # Get registration result from DSA metadata
        item_data = dsa.get_item(moving_id)
        item_meta = item_data.get("meta", {})
        
        # Get method-specific registration data
        reg_key = f"npReg_{method}" if method != "simpleitk" else "npReg"
        reg_meta = item_meta.get(reg_key) or (item_meta.get("npReg") if method == "simpleitk" else None)
        
        if not reg_meta:
            raise HTTPException(
                status_code=404,
                detail=f"No registration data found for method: {method}"
            )
        
        # Get transform matrix
        xfm_key = f"XFM_{method}" if method != "simpleitk" else "XFM"
        xfm_str = item_meta.get(xfm_key) or (item_meta.get("XFM") if method == "simpleitk" else None)
        
        if not xfm_str:
            raise HTTPException(
                status_code=404,
                detail=f"No transform matrix found for method: {method}"
            )
        
        # Parse transform matrix
        import json
        xfm_dict = json.loads(xfm_str) if isinstance(xfm_str, str) else xfm_str
        transform_matrix = np.array([
            xfm_dict.get("0", [1, 0, 0]),
            xfm_dict.get("1", [0, 1, 0]),
            xfm_dict.get("2", [0, 0, 1])
        ])
        
        # Get the thumbnail_width used for registration (if stored)
        reg_thumbnail_width = reg_meta.get("thumbnail_width", 1024)  # Default to 1024 if not stored
        
        # Load images at the registration size to ensure transform alignment
        # If user requested different size, we'll resize after applying transform
        fixed_img = get_thumbnail_image(fixed_id, width=reg_thumbnail_width)
        moving_img = get_thumbnail_image(moving_id, width=reg_thumbnail_width)
        
        if fixed_img is None or moving_img is None:
            raise HTTPException(
                status_code=404,
                detail="Failed to retrieve images"
            )
        
        # Convert to RGB if needed
        if len(fixed_img.shape) == 2:
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2RGB)
        if len(moving_img.shape) == 2:
            moving_img = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2RGB)
        
        # Ensure uint8
        if fixed_img.dtype != np.uint8:
            fixed_img = (fixed_img * 255).astype(np.uint8) if fixed_img.max() <= 1.0 else fixed_img.astype(np.uint8)
        if moving_img.dtype != np.uint8:
            moving_img = (moving_img * 255).astype(np.uint8) if moving_img.max() <= 1.0 else moving_img.astype(np.uint8)
        
        # CRITICAL: Ensure both images have the same dimensions
        # Even with same width, different aspect ratios can cause different heights
        # Use fixed image dimensions as reference (transform was calculated for these)
        h, w = fixed_img.shape[:2]
        h_moving, w_moving = moving_img.shape[:2]
        
        if h_moving != h or w_moving != w:
            logger.info(f"Resizing moving image from {w_moving}x{h_moving} to match fixed image {w}x{h}")
            moving_img = cv2.resize(moving_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Check if transform is valid (not identity)
        if np.allclose(transform_matrix, np.eye(3), atol=1e-6):
            logger.warning("Transform matrix is identity - registration may not have been applied")
        
        # No need to scale transform - we loaded images at registration size
        # If user requested different size, we'll resize the final result
        
        # Extract 2x3 affine matrix from 3x3 transform matrix
        # The stored transform matrix should map: moving coordinates -> fixed coordinates
        # cv2.warpAffine applies: dst(x,y) = src(M11*x + M12*y + M13, M21*x + M22*y + M23)
        # This means M maps destination (fixed) coords -> source (moving) coords
        # So we need the INVERSE of the stored transform
        
        # Invert the transform for cv2.warpAffine
        try:
            transform_3x3 = transform_matrix.astype(np.float64)
            # Check if it's invertible
            det = np.linalg.det(transform_3x3)
            if abs(det) < 1e-10:
                logger.warning(f"Transform matrix is singular (det={det}), using identity")
                affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            else:
                transform_inv = np.linalg.inv(transform_3x3)
                affine_matrix = transform_inv[:2, :].astype(np.float32)
                logger.debug(f"Using inverse transform for warping (det={det:.6f})")
        except Exception as e:
            logger.warning(f"Failed to invert transform: {e}, using identity")
            affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        
        # Warp the moving image into fixed image space
        # cv2.warpAffine needs the inverse: fixed_coords -> moving_coords
        warped_moving = cv2.warpAffine(
            moving_img,
            affine_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Create overlay with opacity
        overlay = cv2.addWeighted(fixed_img, 1.0 - opacity, warped_moving, opacity, 0)
        
        # Resize to requested width if different from registration size
        if width != reg_thumbnail_width:
            # Calculate new height maintaining aspect ratio
            h_orig, w_orig = overlay.shape[:2]
            aspect_ratio = h_orig / w_orig
            new_height = int(width * aspect_ratio)
            overlay = cv2.resize(overlay, (width, new_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized overlay from {w_orig}x{h_orig} to {width}x{new_height}")
        
        # Encode as PNG
        _, buffer = cv2.imencode('.png', overlay)
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/png"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating warped overlay: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create overlay: {str(e)}"
        )


@router.get("/masks/{fixed_id}/{moving_id}")
async def visualize_masks(
    fixed_id: str,
    moving_id: str,
    method: str = Query("affine_tps", description="Registration method to get transform from"),
    width: int = Query(1024, description="Image width for mask visualization")
):
    """
    Visualize the masks used for Dice calculation
    
    Returns a side-by-side visualization showing:
    - Fixed image with its mask overlay
    - Moving (warped) image with its mask overlay
    - Both masks side by side
    - Intersection of the masks
    """
    try:
        from app.services.registration_service import prepare_mask_exit, get_thumbnail_image
        
        # Get registration metadata to determine registration size
        dsa = get_dsa_client()
        item_data = dsa.get_item(moving_id)
        item_meta = item_data.get("meta", {})
        reg_key = f"npReg_{method}" if method != "simpleitk" else "npReg"
        reg_meta = item_meta.get(reg_key) or (item_meta.get("npReg") if method == "simpleitk" else None)
        reg_thumbnail_width = reg_meta.get("thumbnail_width", 1024) if reg_meta else 1024
        
        # Load images at registration size to ensure transform alignment
        fixed_img = get_thumbnail_image(fixed_id, width=reg_thumbnail_width)
        moving_img_original = get_thumbnail_image(moving_id, width=reg_thumbnail_width)
        
        if fixed_img is None or moving_img_original is None:
            raise HTTPException(
                status_code=404,
                detail="Failed to retrieve images"
            )
        
        # Convert to RGB if needed
        if len(fixed_img.shape) == 2:
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2RGB)
        if len(moving_img_original.shape) == 2:
            moving_img_original = cv2.cvtColor(moving_img_original, cv2.COLOR_GRAY2RGB)
        
        # Ensure uint8
        if fixed_img.dtype != np.uint8:
            fixed_img = (fixed_img * 255).astype(np.uint8) if fixed_img.max() <= 1.0 else fixed_img.astype(np.uint8)
        if moving_img_original.dtype != np.uint8:
            moving_img_original = (moving_img_original * 255).astype(np.uint8) if moving_img_original.max() <= 1.0 else moving_img_original.astype(np.uint8)
        
        # CRITICAL: Ensure both images have the same dimensions
        # Even with same width, different aspect ratios can cause different heights
        h_fixed, w_fixed = fixed_img.shape[:2]
        h_moving, w_moving = moving_img_original.shape[:2]
        
        if h_moving != h_fixed or w_moving != w_fixed:
            logger.info(f"Resizing moving image from {w_moving}x{h_moving} to match fixed image {w_fixed}x{h_fixed}")
            moving_img_original = cv2.resize(moving_img_original, (w_fixed, h_fixed), interpolation=cv2.INTER_LINEAR)
        
        # Start with original moving image - will be warped below
        moving_img = moving_img_original.copy()
        
        # reg_meta already retrieved above
        
        if reg_meta:
            # For affine_tps, we need to regenerate the full warp (affine + TPS) using stored affine matrix and match data
            if method == "affine_tps":
                match_data = reg_meta.get("match_data")
                tps_applied = reg_meta.get("tps_applied", False)  # Check if TPS was actually applied during registration
                num_inliers = reg_meta.get("num_inliers", 0)  # Number of inliers from registration
                stored_affine_matrix = reg_meta.get("affine_matrix")  # 2x3 affine matrix from registration
                
                # Get stored transform matrix (3x3) as fallback
                xfm_key = f"XFM_{method}" if method != "simpleitk" else "XFM"
                xfm_str = item_meta.get(xfm_key) or (item_meta.get("XFM") if method == "simpleitk" else None)
                
                if match_data and (stored_affine_matrix is not None or xfm_str is not None):
                    try:
                        from app.services.tps_service import apply_tps_warp
                        import json
                        
                        # No need to scale - images already loaded at registration size
                        scale_factor = 1.0
                        
                        # Get the stored affine matrix (prefer 2x3, fallback to 3x3)
                        if stored_affine_matrix is not None:
                            # stored_affine_matrix is 2x3: [a00 a01 tx; a10 a11 ty] - maps moving -> fixed
                            A = np.array(stored_affine_matrix, dtype=np.float64)
                        elif xfm_str:
                            # Extract 2x3 from 3x3 transform matrix
                            xfm_dict = json.loads(xfm_str) if isinstance(xfm_str, str) else xfm_str
                            transform_matrix = np.array([
                                xfm_dict.get("0", [1, 0, 0]),
                                xfm_dict.get("1", [0, 1, 0]),
                                xfm_dict.get("2", [0, 0, 1])
                            ], dtype=np.float64)
                            A = transform_matrix[:2, :]  # Extract 2x3 from 3x3
                        else:
                            raise ValueError("No affine matrix found in stored metadata")
                        
                        # Scale translation components if image size differs
                        if scale_factor != 1.0:
                            A[0, 2] *= scale_factor  # Scale tx
                            A[1, 2] *= scale_factor  # Scale ty
                        
                        # Step 1: Apply affine transform
                        # A maps moving -> fixed, but cv2.warpAffine needs the inverse
                        A_3x3 = np.vstack([A, [0, 0, 1]])
                        A_inv_3x3 = np.linalg.inv(A_3x3)
                        A_inv = A_inv_3x3[:2, :].astype(np.float32)
                        
                        h, w = fixed_img.shape[:2]
                        moving_affine = cv2.warpAffine(
                            moving_img,
                            A_inv,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        
                        # Step 2: Apply TPS refinement ONLY if it was applied during registration
                        if tps_applied and num_inliers >= 30 and match_data:
                            # Get match data for TPS control points
                            keypoints0 = np.array(match_data["keypoints0"])
                            keypoints1 = np.array(match_data["keypoints1"])
                            matches = np.array(match_data["matches"])
                            inliers = np.array(match_data["inliers"]) if match_data.get("inliers") is not None else None
                            
                            if inliers is not None and len(inliers) > 0:
                                # Scale keypoints if image size differs
                                if scale_factor != 1.0:
                                    keypoints0 = keypoints0 * scale_factor
                                    keypoints1 = keypoints1 * scale_factor
                                
                                # Extract inlier matches
                                inlier_mask = inliers.ravel().astype(bool)
                                matched_kpts0 = keypoints0[matches[inlier_mask, 0]]  # Fixed image keypoints
                                matched_kpts1 = keypoints1[matches[inlier_mask, 1]]  # Moving image keypoints
                                
                                if len(matched_kpts0) >= 30:
                                    # Subsample control points if needed (same as registration code)
                                    max_matches_for_tps = 2000
                                    p0_in = matched_kpts0.copy()
                                    p1_in = matched_kpts1.copy()
                                    
                                    if len(p0_in) > max_matches_for_tps:
                                        # Use same deterministic subsampling as registration (seed=0)
                                        rng = np.random.default_rng(0)
                                        idx = rng.choice(len(p0_in), size=max_matches_for_tps, replace=False)
                                        p0_in = p0_in[idx]
                                        p1_in = p1_in[idx]
                                        logger.debug(f"Subsampled TPS control points from {len(matched_kpts0)} to {max_matches_for_tps}")
                                    
                                    # Transform moving keypoints by affine to get their positions in moving_affine space
                                    # This matches the registration code exactly
                                    ones = np.ones((p1_in.shape[0], 1), dtype=np.float32)
                                    p1_h = np.concatenate([p1_in, ones], axis=1)  # (N,3)
                                    p1_in_fixed = (A @ p1_h.T).T.astype(np.float64)  # Where p1_in are after affine warp
                                    
                                    # TPS maps from moving_affine space to fixed space
                                    # Source: p1_in_fixed (locations in moving_affine), Dest: p0_in (fixed)
                                    moving_img = apply_tps_warp(
                                        src_img=moving_affine,
                                        src_points=p1_in_fixed,
                                        dst_points=p0_in.astype(np.float64),
                                        target_shape=(h, w),
                                        smoothing=1.0
                                    )
                                    logger.debug(f"Applied TPS refinement with {len(p0_in)} control points")
                                else:
                                    # Not enough control points, use affine-only
                                    moving_img = moving_affine
                                    logger.debug(f"Not enough control points for TPS ({len(matched_kpts0)} < 30), using affine-only")
                            else:
                                # No inliers, use affine-only
                                moving_img = moving_affine
                                logger.debug("No inliers found, using affine-only")
                        else:
                            # TPS was not applied during registration, use affine-only
                            moving_img = moving_affine
                            logger.debug(f"Using affine-only (TPS was not applied during registration: tps_applied={tps_applied}, num_inliers={num_inliers})")
                    except Exception as e:
                        logger.warning(f"Failed to regenerate affine_tps warp for mask visualization: {e}. Falling back to stored transform matrix.", exc_info=True)
                        # Fall through to use stored transform matrix below
                        match_data = None
                else:
                    # No match data or affine matrix, fall through to use stored transform matrix
                    match_data = None
            
            # For other methods or if affine_tps regeneration failed, use stored transform matrix
            if method != "affine_tps" or not match_data:
                xfm_key = f"XFM_{method}" if method != "simpleitk" else "XFM"
                xfm_str = item_meta.get(xfm_key) or (item_meta.get("XFM") if method == "simpleitk" else None)
                
                if xfm_str:
                    import json
                    xfm_dict = json.loads(xfm_str) if isinstance(xfm_str, str) else xfm_str
                    transform_matrix = np.array([
                        xfm_dict.get("0", [1, 0, 0]),
                        xfm_dict.get("1", [0, 1, 0]),
                        xfm_dict.get("2", [0, 0, 1])
                    ])
                    
                    # No need to scale - images already loaded at registration size
                    
                    # Invert transform for warping
                    transform_3x3 = transform_matrix.astype(np.float64)
                    det = np.linalg.det(transform_3x3)
                    if abs(det) > 1e-10:
                        transform_inv = np.linalg.inv(transform_3x3)
                        affine_matrix = transform_inv[:2, :].astype(np.float32)
                        
                        h, w = fixed_img.shape[:2]
                        moving_img = cv2.warpAffine(
                            moving_img,
                            affine_matrix,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        # Verify warped image size matches expected
                        h_warped, w_warped = moving_img.shape[:2]
                        if h_warped != h or w_warped != w:
                            logger.warning(f"Warped image size mismatch: expected {h}x{w}, got {h_warped}x{w_warped}. Resizing.")
                            moving_img = cv2.resize(moving_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # CRITICAL: Ensure images are exactly the same size before mask generation
        # The warped moving_img should already be the same size as fixed_img, but verify and fix if needed
        h_fixed, w_fixed = fixed_img.shape[:2]
        h_moving, w_moving = moving_img.shape[:2]
        
        logger.info(f"Image dimensions before final check: fixed={h_fixed}x{w_fixed}, moving={h_moving}x{w_moving}")
        
        if h_fixed != h_moving or w_fixed != w_moving:
            logger.warning(f"Image size mismatch after warping: fixed={h_fixed}x{w_fixed}, moving={h_moving}x{w_moving}. Resizing moving to match fixed.")
            moving_img = cv2.resize(moving_img, (w_fixed, h_fixed), interpolation=cv2.INTER_LINEAR)
            logger.info(f"After resize: moving={moving_img.shape[0]}x{moving_img.shape[1]}")
        
        # Double-check after resize
        h_moving, w_moving = moving_img.shape[:2]
        if h_fixed != h_moving or w_fixed != w_moving:
            raise ValueError(f"Failed to normalize image sizes: fixed={h_fixed}x{w_fixed}, moving={h_moving}x{w_moving}")
        
        # Ensure images are in uint8 format before mask generation
        if fixed_img.dtype != np.uint8:
            fixed_img = (fixed_img * 255).astype(np.uint8) if fixed_img.max() <= 1.0 else fixed_img.astype(np.uint8)
        if moving_img.dtype != np.uint8:
            moving_img = (moving_img * 255).astype(np.uint8) if moving_img.max() <= 1.0 else moving_img.astype(np.uint8)
        
        # Convert to grayscale for mask generation
        # IMPORTANT: Generate masks on the warped images (both in fixed coordinate space)
        # moving_img should be the warped version at this point
        if len(fixed_img.shape) == 3:
            fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
        else:
            fixed_gray = fixed_img.copy()
            
        if len(moving_img.shape) == 3:
            moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
        else:
            moving_gray = moving_img.copy()
        
        # Ensure same size (should already be, but double-check)
        if fixed_gray.shape != moving_gray.shape:
            moving_gray = cv2.resize(moving_gray, (fixed_gray.shape[1], fixed_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            logger.warning(f"Grayscale size mismatch after conversion, resized moving_gray to {fixed_gray.shape}")
        
        # Generate masks on the warped images (both in fixed coordinate space)
        # mask1: fixed image mask (in fixed coordinate space)
        # mask2: moving image mask (on warped moving image, also in fixed coordinate space)
        
        # Check if warped image has content (not all black)
        moving_nonzero = np.count_nonzero(moving_gray)
        moving_total = moving_gray.size
        moving_coverage = (moving_nonzero / moving_total * 100) if moving_total > 0 else 0.0
        logger.info(f"Warped moving image stats: shape={moving_gray.shape}, dtype={moving_gray.dtype}, "
                   f"min={moving_gray.min()}, max={moving_gray.max()}, "
                   f"non-zero pixels={moving_nonzero}/{moving_total} ({moving_coverage:.1f}%)")
        
        if moving_coverage < 1.0:
            logger.warning(f"Warped moving image has very little content ({moving_coverage:.1f}% non-zero). "
                          f"This might indicate a warping issue or the image is mostly outside the fixed image bounds.")
        
        mask1 = prepare_mask_exit(fixed_gray, method="otsu")
        mask2 = prepare_mask_exit(moving_gray, method="otsu")
        
        # Check mask generation results
        mask1_coverage = (mask1.sum() / (mask1.size * 255) * 100) if mask1.size > 0 else 0.0
        mask2_coverage = (mask2.sum() / (mask2.size * 255) * 100) if mask2.size > 0 else 0.0
        
        logger.info(f"Generated masks: fixed_mask shape={mask1.shape}, coverage={mask1_coverage:.1f}%, "
                   f"moving_mask shape={mask2.shape}, coverage={mask2_coverage:.1f}%")
        
        if mask2_coverage < 0.1:
            logger.warning(f"Moving mask has very low coverage ({mask2_coverage:.1f}%). "
                          f"This might indicate the warped image is mostly background or the mask generation failed.")
        
        # Convert images to RGB for display (after mask generation)
        if len(fixed_img.shape) == 2:
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2RGB)
        if len(moving_img.shape) == 2:
            moving_img = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2RGB)
        
        # Ensure uint8 for display
        if fixed_img.dtype != np.uint8:
            fixed_img = (fixed_img * 255).astype(np.uint8) if fixed_img.max() <= 1.0 else fixed_img.astype(np.uint8)
        if moving_img.dtype != np.uint8:
            moving_img = (moving_img * 255).astype(np.uint8) if moving_img.max() <= 1.0 else moving_img.astype(np.uint8)
        
        # Ensure masks are same size
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary
        mask1_binary = (mask1 > 127).astype(np.uint8)
        mask2_binary = (mask2 > 127).astype(np.uint8)
        
        # Calculate intersection
        intersection = np.logical_and(mask1_binary, mask2_binary).astype(np.uint8) * 255
        
        # CRITICAL: Final size check before creating visualizations
        h_fixed, w_fixed = fixed_img.shape[:2]
        h_moving, w_moving = moving_img.shape[:2]
        if h_fixed != h_moving or w_fixed != w_moving:
            logger.error(f"Size mismatch before visualization: fixed={h_fixed}x{w_fixed}, moving={h_moving}x{w_moving}")
            moving_img = cv2.resize(moving_img, (w_fixed, h_fixed), interpolation=cv2.INTER_LINEAR)
            # Also resize mask2 if needed
            if mask2.shape != mask1.shape:
                mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask2_binary = (mask2 > 127).astype(np.uint8)
        
        # Create visualizations
        # 1. Fixed image with mask overlay (green)
        fixed_with_mask = fixed_img.copy()
        mask1_colored = np.zeros_like(fixed_img)
        mask1_colored[mask1_binary > 0] = [0, 255, 0]  # Green overlay
        fixed_with_mask = cv2.addWeighted(fixed_with_mask, 0.7, mask1_colored, 0.3, 0)
        
        # 2. Moving (warped) image with mask overlay (red)
        moving_with_mask = moving_img.copy()
        mask2_colored = np.zeros_like(moving_img)
        mask2_colored[mask2_binary > 0] = [255, 0, 0]  # Red overlay
        moving_with_mask = cv2.addWeighted(moving_with_mask, 0.7, mask2_colored, 0.3, 0)
        
        # Final verification before stacking
        if fixed_with_mask.shape != moving_with_mask.shape:
            logger.error(f"Size mismatch in final images: fixed={fixed_with_mask.shape}, moving={moving_with_mask.shape}")
            moving_with_mask = cv2.resize(moving_with_mask, (fixed_with_mask.shape[1], fixed_with_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 3. Masks side by side
        mask1_rgb = cv2.cvtColor(mask1_binary * 255, cv2.COLOR_GRAY2RGB)
        mask2_rgb = cv2.cvtColor(mask2_binary * 255, cv2.COLOR_GRAY2RGB)
        masks_side_by_side = np.hstack([mask1_rgb, mask2_rgb])
        
        # 4. Intersection
        intersection_rgb = cv2.cvtColor(intersection, cv2.COLOR_GRAY2RGB)
        
        # CRITICAL: Final size check before combining - images MUST be same size for hstack
        if fixed_with_mask.shape != moving_with_mask.shape:
            logger.error(f"CRITICAL: Size mismatch before hstack! fixed={fixed_with_mask.shape}, moving={moving_with_mask.shape}")
            # Force resize to match fixed image
            h_f, w_f = fixed_with_mask.shape[:2]
            moving_with_mask = cv2.resize(moving_with_mask, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized moving_with_mask to {moving_with_mask.shape} to match fixed")
        
        # Combine all visualizations into a grid
        # Top row: images with masks
        top_row = np.hstack([fixed_with_mask, moving_with_mask])
        # Bottom row: masks and intersection
        bottom_row = np.hstack([masks_side_by_side, intersection_rgb])
        
        # Resize to same width if needed
        if top_row.shape[1] != bottom_row.shape[1]:
            target_width = max(top_row.shape[1], bottom_row.shape[1])
            if top_row.shape[1] < target_width:
                top_row = cv2.resize(top_row, (target_width, top_row.shape[0]), interpolation=cv2.INTER_LINEAR)
            if bottom_row.shape[1] < target_width:
                bottom_row = cv2.resize(bottom_row, (target_width, bottom_row.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Combine vertically
        combined = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)
        
        # Add text labels
        cv2.putText(combined, "Fixed + Mask", (10, 30), font, font_scale, color, thickness)
        cv2.putText(combined, "Moving (Warped) + Mask", (top_row.shape[1] // 2 + 10, 30), font, font_scale, color, thickness)
        cv2.putText(combined, "Fixed Mask | Moving Mask", (10, top_row.shape[0] + 30), font, font_scale, color, thickness)
        cv2.putText(combined, "Intersection", (top_row.shape[1] // 2 + 10, top_row.shape[0] + 30), font, font_scale, color, thickness)
        
        # Resize to requested width if different from registration size
        if width != reg_thumbnail_width:
            # Calculate new height maintaining aspect ratio
            h_orig, w_orig = combined.shape[:2]
            aspect_ratio = h_orig / w_orig
            new_height = int(width * aspect_ratio)
            combined = cv2.resize(combined, (width, new_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized masks visualization from {w_orig}x{h_orig} to {width}x{new_height}")
        
        # Encode as PNG
        _, buffer = cv2.imencode('.png', combined)
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/png"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating mask visualization: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create mask visualization: {str(e)}"
        )


@router.get("/feature-matches-data/{fixed_id}/{moving_id}")
async def get_feature_matches_data(
    fixed_id: str,
    moving_id: str,
    method: str = Query("affine_tps", description="Registration method to get matches from")
):
    """
    Get feature match data as JSON for client-side rendering
    
    Returns match data including keypoints, matches, scores, and inliers.
    """
    try:
        dsa = get_dsa_client()
        
        # Get registration result from DSA metadata
        item_data = dsa.get_item(moving_id)
        item_meta = item_data.get("meta", {})
        
        # Get method-specific registration data
        reg_key = f"npReg_{method}" if method != "simpleitk" else "npReg"
        reg_meta = item_meta.get(reg_key) or (item_meta.get("npReg") if method == "simpleitk" else None)
        
        if not reg_meta:
            raise HTTPException(
                status_code=404,
                detail=f"No registration data found for method: {method}"
            )
        
        # Get match data from stored metadata
        match_data = reg_meta.get("match_data")
        
        # Handle case where match_data might be stored as JSON string
        if isinstance(match_data, str):
            import json
            try:
                match_data = json.loads(match_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse match_data as JSON for item {moving_id}")
                match_data = None
        
        if not match_data and method in ["affine_tps", "lightglue"]:
            raise HTTPException(
                status_code=404,
                detail=f"Match data not available. Please re-run the registration with {method} to generate match data."
            )
        
        if not match_data:
            raise HTTPException(
                status_code=404,
                detail="No match data available for visualization"
            )
        
        # Get transform matrix for affine_tps (to transform keypoints)
        transform_matrix = None
        if method == "affine_tps":
            xfm_key = f"XFM_{method}" if method != "simpleitk" else "XFM"
            xfm_str = item_meta.get(xfm_key) or (item_meta.get("XFM") if method == "simpleitk" else None)
            
            if xfm_str:
                import json
                xfm_dict = json.loads(xfm_str) if isinstance(xfm_str, str) else xfm_str
                transform_matrix = [
                    xfm_dict.get("0", [1, 0, 0]),
                    xfm_dict.get("1", [0, 1, 0]),
                    xfm_dict.get("2", [0, 0, 1])
                ]
        
        return {
            "fixed_id": fixed_id,
            "moving_id": moving_id,
            "method": method,
            "match_data": match_data,
            "transform_matrix": transform_matrix,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature match data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get match data: {str(e)}"
        )


@router.get("/feature-matches/{fixed_id}/{moving_id}")
async def visualize_feature_matches(
    fixed_id: str,
    moving_id: str,
    method: str = Query("affine_tps", description="Registration method to get matches from (affine_tps uses LightGlue for matching)"),
    max_matches: int = Query(100, description="Maximum number of matches to display"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum match confidence score to display (0.0-1.0)"),
    inliers_only: bool = Query(False, description="Show only inlier matches (if available)"),
    outliers_only: bool = Query(False, description="Show only outlier matches (if available)")
):
    """
    Visualize feature matches from LightGlue registration
    
    Returns an image showing matched keypoints between fixed and moving images.
    """
    try:
        dsa = get_dsa_client()
        
        # Get registration result from DSA metadata
        item_data = dsa.get_item(moving_id)
        item_meta = item_data.get("meta", {})
        
        # Get method-specific registration data
        reg_key = f"npReg_{method}" if method != "simpleitk" else "npReg"
        reg_meta = item_meta.get(reg_key) or (item_meta.get("npReg") if method == "simpleitk" else None)
        
        if not reg_meta:
            raise HTTPException(
                status_code=404,
                detail=f"No registration data found for method: {method}"
            )
        
        # For TPS (which uses LightGlue for matching), we need match data stored during registration
        # For now, let's check if we can get it from a recent registration result
        # Or we'll need to add an endpoint that returns match data
        
        # Get thumbnail images
        fixed_img = get_thumbnail_image(fixed_id, width=1024)
        moving_img = get_thumbnail_image(moving_id, width=1024)
        
        if fixed_img is None or moving_img is None:
            raise HTTPException(
                status_code=404,
                detail="Failed to retrieve images"
            )
        
        # Convert to grayscale for visualization
        if len(fixed_img.shape) == 3:
            fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
        else:
            fixed_gray = fixed_img
            
        if len(moving_img.shape) == 3:
            moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
        else:
            moving_gray = moving_img
        
        # Get match data from stored metadata
        match_data = reg_meta.get("match_data")
        
        if not match_data and method in ["affine_tps", "lightglue"]:  # Support 'affine_tps' and legacy 'lightglue' method
            # Check if this is an old registration (before match_data storage was added)
            logger.warning(f"No match_data found for {method} registration on item {moving_id}. Registration may have been run before match_data storage was implemented.")
            raise HTTPException(
                status_code=404,
                detail=f"Match visualization not available. This registration was run before match data storage was implemented. Please re-run the registration with {method} to generate match data for visualization."
            )
        
        if match_data:
            # Extract match data
            keypoints0 = np.array(match_data["keypoints0"])
            keypoints1 = np.array(match_data["keypoints1"])
            matches = np.array(match_data["matches"])
            inliers = np.array(match_data["inliers"]) if match_data.get("inliers") else None
            match_scores = np.array(match_data["match_scores"]) if match_data.get("match_scores") else None
            
            # Apply filters
            filter_mask = np.ones(len(matches), dtype=bool)
            
            # Filter by confidence score
            if match_scores is not None and min_confidence > 0.0:
                filter_mask = filter_mask & (match_scores >= min_confidence)
                logger.debug(f"Filtered by confidence >= {min_confidence}: {np.sum(filter_mask)} matches remain")
            
            # Filter by inlier/outlier status
            if inliers is not None:
                if inliers_only:
                    filter_mask = filter_mask & inliers
                    logger.debug(f"Filtered to inliers only: {np.sum(filter_mask)} matches remain")
                elif outliers_only:
                    filter_mask = filter_mask & (~inliers)
                    logger.debug(f"Filtered to outliers only: {np.sum(filter_mask)} matches remain")
            
            # Apply filters
            matches = matches[filter_mask]
            if inliers is not None:
                inliers = inliers[filter_mask]
            if match_scores is not None:
                match_scores = match_scores[filter_mask]
            
            logger.info(f"Displaying {len(matches)} matches after filtering (from {len(match_data['matches'])})")
            
            # For affine_tps, the inliers are based on the affine transform
            # So we should show matches on the affine-warped moving image for proper visualization
            if method == "affine_tps":
                # Get affine transform matrix from metadata
                xfm_key = f"XFM_{method}" if method != "simpleitk" else "XFM"
                xfm_str = item_meta.get(xfm_key) or (item_meta.get("XFM") if method == "simpleitk" else None)
                
                if xfm_str:
                    import json
                    xfm_dict = json.loads(xfm_str) if isinstance(xfm_str, str) else xfm_str
                    transform_matrix = np.array([
                        xfm_dict.get("0", [1, 0, 0]),
                        xfm_dict.get("1", [0, 1, 0]),
                        xfm_dict.get("2", [0, 0, 1])
                    ])
                    
                    # Extract 2x3 affine matrix
                    # The stored transform_matrix maps: moving -> fixed
                    # cv2.warpAffine needs the inverse: fixed -> moving
                    h, w = fixed_gray.shape[:2]
                    transform_3x3 = transform_matrix.astype(np.float64)
                    
                    # Invert the transform for cv2.warpAffine
                    try:
                        det = np.linalg.det(transform_3x3)
                        if abs(det) < 1e-10:
                            logger.warning(f"Transform matrix is singular (det={det}), cannot warp for visualization")
                            # Fall back to original images
                            moving_gray_warped = moving_gray
                            keypoints1_warped = keypoints1
                        else:
                            transform_inv = np.linalg.inv(transform_3x3)
                            affine_matrix = transform_inv[:2, :].astype(np.float32)
                            
                            # Warp the moving image by the inverse affine transform
                            moving_gray_warped = cv2.warpAffine(
                                moving_gray,
                                affine_matrix,
                                (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0
                            )
                            
                            # Transform moving image keypoints by the inverse affine transform
                            # keypoints1 are in original moving image coordinates
                            # We need to transform them to fixed image coordinates (where the warped image is)
                            ones = np.ones((keypoints1.shape[0], 1), dtype=np.float32)
                            kpts1_homogeneous = np.concatenate([keypoints1, ones], axis=1)  # (N, 3)
                            # Use the forward transform (moving->fixed) to transform keypoints
                            keypoints1_warped = (transform_matrix[:2, :] @ kpts1_homogeneous.T).T  # (N, 2)
                    except Exception as e:
                        logger.warning(f"Failed to invert transform for visualization: {e}, using original images")
                        moving_gray_warped = moving_gray
                        keypoints1_warped = keypoints1
                    
                    # Use the warped moving image and transformed keypoints for visualization
                    moving_gray = moving_gray_warped
                    keypoints1 = keypoints1_warped
            
            # Draw matches
            viz_img = draw_feature_matches(
                fixed_gray,
                moving_gray,
                keypoints0,
                keypoints1,
                matches,
                inliers,
                max_matches=max_matches
            )
            
            # Encode as PNG
            _, buffer = cv2.imencode('.png', viz_img)
            
            return Response(
                content=buffer.tobytes(),
                media_type="image/png"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="No match data available for visualization"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating feature match visualization: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create visualization: {str(e)}"
        )

