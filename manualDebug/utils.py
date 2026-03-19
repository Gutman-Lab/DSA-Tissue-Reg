"""
Transform utilities for manual debugging
"""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent / "viteReg" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import cv2
import json
from typing import Dict, Any, Optional
from app.services.registration_service import get_thumbnail_image


def load_transform_from_json(json_data: Dict[str, Any], matrix_type: str = "Homography") -> np.ndarray:
    """
    Load transformation matrix from JSON data.
    
    Args:
        json_data: Dictionary containing geom_info
        matrix_type: Which matrix to use ('Fundamental', 'Homography', 'H1', 'H2')
    
    Returns:
        3x3 transformation matrix as numpy array
    """
    if 'geom_info' not in json_data:
        raise ValueError("JSON must contain 'geom_info' key")
    
    geom_info = json_data['geom_info']
    
    if matrix_type not in geom_info:
        available = list(geom_info.keys())
        raise ValueError(f"Matrix type '{matrix_type}' not found. Available: {available}")
    
    matrix = np.array(geom_info[matrix_type])
    
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got shape {matrix.shape}")
    
    return matrix


def apply_homography_transform(
    fixed_image_id: str,
    moving_image_id: str,
    transform_matrix: np.ndarray,
    thumbnail_width: int = 512
) -> Dict[str, Any]:
    """
    Apply a homography transformation matrix to align moving image to fixed image.
    
    Args:
        fixed_image_id: DSA item ID of fixed (reference) image
        moving_image_id: DSA item ID of moving image to transform
        transform_matrix: 3x3 homography matrix (moving -> fixed)
        thumbnail_width: Width of thumbnails to use
    
    Returns:
        Dictionary with transformed image and metadata
    """
    # Load images
    fixed_img = get_thumbnail_image(fixed_image_id, width=thumbnail_width)
    moving_img = get_thumbnail_image(moving_image_id, width=thumbnail_width)
    
    if fixed_img is None:
        raise ValueError(f"Failed to load fixed image: {fixed_image_id}")
    if moving_img is None:
        raise ValueError(f"Failed to load moving image: {moving_image_id}")
    
    # Convert to RGB if grayscale
    if len(fixed_img.shape) == 2:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2RGB)
    elif fixed_img.shape[2] == 4:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_RGBA2RGB)
    
    if len(moving_img.shape) == 2:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2RGB)
    elif moving_img.shape[2] == 4:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_RGBA2RGB)
    
    # Ensure uint8
    if fixed_img.dtype != np.uint8:
        fixed_img = (fixed_img * 255).astype(np.uint8) if fixed_img.max() <= 1.0 else fixed_img.astype(np.uint8)
    if moving_img.dtype != np.uint8:
        moving_img = (moving_img * 255).astype(np.uint8) if moving_img.max() <= 1.0 else moving_img.astype(np.uint8)
    
    # Get output dimensions (use fixed image size)
    h, w = fixed_img.shape[:2]
    
    # Apply homography transform
    moving_registered = cv2.warpPerspective(
        moving_img,
        transform_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Create overlay
    overlay = cv2.addWeighted(fixed_img, 0.5, moving_registered, 0.5, 0)
    
    # Extract transform parameters for display
    if abs(transform_matrix[2, 0]) < 1e-6 and abs(transform_matrix[2, 1]) < 1e-6:
        # Nearly affine
        rotation_rad = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
        scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
        scale = (scale_x + scale_y) / 2.0
        offset_x = transform_matrix[0, 2]
        offset_y = transform_matrix[1, 2]
    else:
        # Full homography with perspective
        rotation_deg = None
        scale = None
        offset_x = transform_matrix[0, 2] / transform_matrix[2, 2]
        offset_y = transform_matrix[1, 2] / transform_matrix[2, 2]
    
    return {
        'fixed_image': fixed_img,
        'moving_image': moving_img,
        'transformed_image': moving_registered,
        'overlay': overlay,
        'transform_params': {
            'rotation_degrees': float(rotation_deg) if rotation_deg is not None else None,
            'scale': float(scale) if scale is not None else None,
            'offset_x': float(offset_x),
            'offset_y': float(offset_y),
        }
    }


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image array to base64 data URL"""
    import base64
    from io import BytesIO
    from PIL import Image
    
    # Convert to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    pil_img = Image.fromarray(image)
    
    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"
