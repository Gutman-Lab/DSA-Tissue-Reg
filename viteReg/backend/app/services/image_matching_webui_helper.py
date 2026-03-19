"""
Helper service to integrate with image-matching-webui for registration exploration

This service provides utilities to:
1. Export images from DSA to a format usable by image-matching-webui
2. Call image-matching-webui API programmatically
3. Import results back from image-matching-webui
4. Convert between formats
"""
import os
import logging
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import base64
from app.services.dsa_client import get_dsa_client

logger = logging.getLogger(__name__)

# Default directory for image exchange (should match docker-compose volume)
IMAGE_MATCHING_DATA_DIR = os.getenv("IMAGE_MATCHING_DATA_DIR", "/app/image-matching-data")

# Image matching webui API base URL
IMAGE_MATCHING_WEBUI_URL = os.getenv(
    "IMAGE_MATCHING_WEBUI_URL", 
    "http://image-matching-webui:7860"  # Docker service name, fallback to localhost
)


def export_images_for_matching(
    fixed_id: str,
    moving_id: str,
    output_dir: Optional[str] = None,
    thumbnail_width: int = 1024
) -> Dict[str, str]:
    """
    Export images from DSA to files that can be used by image-matching-webui
    
    Args:
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        output_dir: Directory to save images (defaults to IMAGE_MATCHING_DATA_DIR)
        thumbnail_width: Width of thumbnail to export
        
    Returns:
        Dictionary with paths to exported images:
        {
            'fixed_image': '/path/to/fixed.jpg',
            'moving_image': '/path/to/moving.jpg',
            'fixed_id': '...',
            'moving_id': '...'
        }
    """
    try:
        if output_dir is None:
            output_dir = IMAGE_MATCHING_DATA_DIR
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dsa = get_dsa_client()
        
        # Fetch thumbnails
        fixed_url = dsa.get_thumbnail_url(fixed_id, width=thumbnail_width)
        moving_url = dsa.get_thumbnail_url(moving_id, width=thumbnail_width)
        
        # Download images
        fixed_response = requests.get(fixed_url)
        fixed_response.raise_for_status()
        moving_response = requests.get(moving_url)
        moving_response.raise_for_status()
        
        # Save images
        fixed_path = os.path.join(output_dir, f"fixed_{fixed_id}.jpg")
        moving_path = os.path.join(output_dir, f"moving_{moving_id}.jpg")
        
        with open(fixed_path, 'wb') as f:
            f.write(fixed_response.content)
        
        with open(moving_path, 'wb') as f:
            f.write(moving_response.content)
        
        logger.info(f"Exported images for matching: fixed={fixed_path}, moving={moving_path}")
        
        return {
            'fixed_image': fixed_path,
            'moving_image': moving_path,
            'fixed_id': fixed_id,
            'moving_id': moving_id,
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"Failed to export images for matching: {e}", exc_info=True)
        raise


def get_image_matching_webui_url(port: int = 7860, host: str = "localhost") -> str:
    """
    Get the URL for the image-matching-webui service
    
    Args:
        port: Port number (default: 7860)
        host: Hostname (default: localhost, use 'image-matching-webui' in Docker)
        
    Returns:
        URL string
    """
    return f"http://{host}:{port}"


def create_matching_session_info(
    fixed_id: str,
    moving_id: str,
    exported_paths: Dict[str, str]
) -> Dict[str, Any]:
    """
    Create a session info file that can be used to track matching sessions
    
    Args:
        fixed_id: DSA item ID of fixed image
        moving_id: DSA item ID of moving image
        exported_paths: Output from export_images_for_matching
        
    Returns:
        Session info dictionary
    """
    session_info = {
        'fixed_id': fixed_id,
        'moving_id': moving_id,
        'fixed_image_path': exported_paths['fixed_image'],
        'moving_image_path': exported_paths['moving_image'],
        'output_dir': exported_paths['output_dir'],
        'webui_url': get_image_matching_webui_url()
    }
    
    # Save session info to file
    import json
    session_file = os.path.join(exported_paths['output_dir'], f"session_{fixed_id}_{moving_id}.json")
    with open(session_file, 'w') as f:
        json.dump(session_info, f, indent=2)
    
    session_info['session_file'] = session_file
    return session_info


def call_image_matching_api(
    fixed_image_path: str,
    moving_image_path: str,
    extractor: str = "superpoint",
    matcher: str = "lightglue",
    extractor_params: Optional[Dict[str, Any]] = None,
    matcher_params: Optional[Dict[str, Any]] = None,
    webui_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call image-matching-webui API programmatically to perform matching
    
    Args:
        fixed_image_path: Path to fixed image file
        moving_image_path: Path to moving image file
        extractor: Feature extractor name (e.g., 'superpoint', 'disk', 'aliked', 'sift')
        matcher: Matcher name (e.g., 'lightglue', 'superglue', 'loftr')
        extractor_params: Optional parameters for extractor
        matcher_params: Optional parameters for matcher
        webui_url: Base URL for image-matching-webui (defaults to IMAGE_MATCHING_WEBUI_URL)
        
    Returns:
        Match results dictionary with keypoints, matches, and metadata
    """
    if webui_url is None:
        webui_url = IMAGE_MATCHING_WEBUI_URL
    
    try:
        # The image-matching-webui API typically expects images as base64 or file uploads
        # Check if files exist
        if not os.path.exists(fixed_image_path):
            raise FileNotFoundError(f"Fixed image not found: {fixed_image_path}")
        if not os.path.exists(moving_image_path):
            raise FileNotFoundError(f"Moving image not found: {moving_image_path}")
        
        # Read images and convert to base64
        with open(fixed_image_path, 'rb') as f:
            fixed_image_data = base64.b64encode(f.read()).decode('utf-8')
        
        with open(moving_image_path, 'rb') as f:
            moving_image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare API request
        # Note: The actual API structure may vary - check imcui/api/server.py for exact endpoints
        # Common patterns:
        # - /api/match or /api/v1/match for matching
        # - /api/extract for feature extraction only
        # - May use /api/predict or similar Gradio API endpoints
        # 
        # To find the correct endpoint, check:
        # https://github.com/Vincentqyw/image-matching-webui/tree/main/imcui/api
        api_endpoint = f"{webui_url}/api/match"  # TODO: Verify actual endpoint from imcui/api/server.py
        
        payload = {
            "image0": fixed_image_data,
            "image1": moving_image_data,
            "extractor": extractor,
            "matcher": matcher,
        }
        
        if extractor_params:
            payload["extractor_params"] = extractor_params
        if matcher_params:
            payload["matcher_params"] = matcher_params
        
        # Make API call
        response = requests.post(
            api_endpoint,
            json=payload,
            timeout=300  # 5 minute timeout for processing
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Image matching API call successful: {len(result.get('matches', []))} matches found")
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to call image matching API: {e}", exc_info=True)
        raise


def match_images_from_dsa(
    fixed_id: str,
    moving_id: str,
    extractor: str = "superpoint",
    matcher: str = "lightglue",
    thumbnail_width: int = 1024,
    extractor_params: Optional[Dict[str, Any]] = None,
    matcher_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to match images directly from DSA IDs using image-matching-webui
    
    Args:
        fixed_id: DSA item ID of fixed image
        moving_id: DSA item ID of moving image
        extractor: Feature extractor name
        matcher: Matcher name
        thumbnail_width: Width of thumbnail to use
        extractor_params: Optional extractor parameters
        matcher_params: Optional matcher parameters
        
    Returns:
        Match results dictionary
    """
    # Export images
    exported_paths = export_images_for_matching(
        fixed_id=fixed_id,
        moving_id=moving_id,
        thumbnail_width=thumbnail_width
    )
    
    # Call API
    result = call_image_matching_api(
        fixed_image_path=exported_paths['fixed_image'],
        moving_image_path=exported_paths['moving_image'],
        extractor=extractor,
        matcher=matcher,
        extractor_params=extractor_params,
        matcher_params=matcher_params
    )
    
    # Add metadata
    result['fixed_id'] = fixed_id
    result['moving_id'] = moving_id
    result['exported_paths'] = exported_paths
    
    return result


def convert_matches_to_registration_transform(
    matches: Dict[str, Any],
    fixed_image_shape: Tuple[int, int],
    moving_image_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Convert image-matching-webui match results to registration transform
    
    Args:
        matches: Match results from image-matching-webui API
        fixed_image_shape: Shape of fixed image (height, width)
        moving_image_shape: Shape of moving image (height, width)
        
    Returns:
        Registration transform dictionary compatible with your registration service
    """
    try:
        # Extract keypoints and matches from API response
        # The format may vary - adjust based on actual API response structure
        keypoints0 = np.array(matches.get('keypoints0', []))  # Shape: (N, 2)
        keypoints1 = np.array(matches.get('keypoints1', []))  # Shape: (M, 2)
        match_indices = np.array(matches.get('matches', []))  # Shape: (K, 2) or (K,)
        
        if len(keypoints0) == 0 or len(keypoints1) == 0:
            raise ValueError("No keypoints found in match results")
        
        # Handle different match index formats
        if len(match_indices.shape) == 1:
            # If matches is a flat array, assume it's indices into keypoints
            matched_kpts0 = keypoints0[match_indices]
            matched_kpts1 = keypoints1[match_indices]
        else:
            # If matches is (K, 2), use it as index pairs
            matched_kpts0 = keypoints0[match_indices[:, 0]]
            matched_kpts1 = keypoints1[match_indices[:, 1]]
        
        if len(matched_kpts0) < 2:
            raise ValueError(f"Insufficient matches: {len(matched_kpts0)}. Need at least 2 for rigid transform.")
        
        # Use RANSAC to estimate rigid transform (similar to lightglue_service.py)
        from app.services.lightglue_service import estimate_rigid_transform_ransac
        
        transform_matrix, inliers = estimate_rigid_transform_ransac(
            src_points=matched_kpts1,  # moving image points
            dst_points=matched_kpts0,    # fixed image points
            max_iters=2000,
            threshold=3.0
        )
        
        if transform_matrix is None:
            raise ValueError("Failed to estimate rigid transform from matches")
        
        # Extract transform parameters
        rotation_degrees = float(np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])))
        offset_x = float(transform_matrix[0, 2])
        offset_y = float(transform_matrix[1, 2])
        scale = 1.0  # Rigid transform has no scaling
        
        return {
            'transform_matrix': transform_matrix.tolist(),
            'rotation_degrees': rotation_degrees,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'scale': scale,
            'num_matches': int(len(matched_kpts0)),
            'num_inliers': int(np.sum(inliers)) if inliers is not None else 0,
            'success': True,
            'match_data': {
                'keypoints0': keypoints0.tolist(),
                'keypoints1': keypoints1.tolist(),
                'matches': match_indices.tolist(),
                'inliers': inliers.tolist() if inliers is not None else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to convert matches to registration transform: {e}", exc_info=True)
        return {
            'transform_matrix': np.eye(3).tolist(),
            'rotation_degrees': 0.0,
            'offset_x': 0.0,
            'offset_y': 0.0,
            'scale': 1.0,
            'num_matches': 0,
            'num_inliers': 0,
            'success': False,
            'error': str(e)
        }
