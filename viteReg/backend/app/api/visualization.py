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
    
    # Draw matches
    for idx in indices:
        match = matches[idx]
        kp0 = keypoints0[match[0]]
        kp1 = keypoints1[match[1]]
        
        x0, y0 = int(kp0[0]), int(kp0[1])
        x1, y1 = int(kp1[0]) + w0, int(kp1[1])  # Shift x1 to second image
        
        # Color: green for inliers, red for outliers
        if inliers is not None and idx < len(inliers):
            color = (0, 255, 0) if inliers[idx] else (0, 0, 255)
            thickness = 2 if inliers[idx] else 1
        else:
            color = (255, 255, 0)  # Yellow for unknown
            thickness = 1
        
        # Draw line connecting matches
        cv2.line(viz, (x0, y0), (x1, y1), color, thickness)
        
        # Draw keypoints
        cv2.circle(viz, (x0, y0), 3, color, -1)
        cv2.circle(viz, (x1, y1), 3, color, -1)
    
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
        
        # Get thumbnail images
        fixed_img = get_thumbnail_image(fixed_id, width=width)
        moving_img = get_thumbnail_image(moving_id, width=width)
        
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
        
        # Get dimensions
        h, w = fixed_img.shape[:2]
        
        # Check if transform is valid (not identity)
        if np.allclose(transform_matrix, np.eye(3), atol=1e-6):
            logger.warning("Transform matrix is identity - registration may not have been applied")
        
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


@router.get("/feature-matches/{fixed_id}/{moving_id}")
async def visualize_feature_matches(
    fixed_id: str,
    moving_id: str,
    method: str = Query("tps", description="Registration method to get matches from (tps uses LightGlue for matching)"),
    max_matches: int = Query(100, description="Maximum number of matches to display")
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
        
        if not match_data and method in ["tps", "affine_tps", "lightglue"]:  # Support 'tps', 'affine_tps', and legacy 'lightglue' method
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

