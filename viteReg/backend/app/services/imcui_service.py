"""
Direct integration with imcui (image-matching-webui) as a Python library

This service uses imcui's internal functions directly, bypassing the web UI.
It provides access to all extractors and matchers available in imcui.
"""
import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import torch
from app.services.dsa_client import get_dsa_client
from app.services.registration_service import get_thumbnail_image

logger = logging.getLogger(__name__)

# Try to import imcui
try:
    import imcui
    from imcui.hloc import extract_features, match_features
    from imcui.hloc.extractors import get_extractor
    from imcui.hloc.matchers import get_matcher
    IMCUI_AVAILABLE = True
except ImportError:
    IMCUI_AVAILABLE = False
    logger.warning("imcui not available. Install with: pip install imcui")


def get_available_extractors() -> List[str]:
    """
    Get list of available feature extractors from imcui
    
    Returns:
        List of extractor names
    """
    if not IMCUI_AVAILABLE:
        return []
    
    try:
        # Common extractors in imcui
        # You may need to check imcui/hloc/extract_features.py for the full list
        extractors = [
            'superpoint', 'disk', 'aliked', 'sift', 'd2net', 'r2d2',
            'hardnet', 'sosnet', 'sold2', 'linetr', 'loftr'
        ]
        # Filter to only those that are actually available
        available = []
        for ext in extractors:
            try:
                get_extractor(ext)
                available.append(ext)
            except:
                pass
        return available
    except Exception as e:
        logger.warning(f"Could not get extractor list: {e}")
        return ['superpoint', 'disk', 'aliked', 'sift']  # Fallback


def get_available_matchers() -> List[str]:
    """
    Get list of available matchers from imcui
    
    Returns:
        List of matcher names
    """
    if not IMCUI_AVAILABLE:
        return []
    
    try:
        # Common matchers in imcui
        matchers = [
            'lightglue', 'superglue', 'loftr', 'topicfm', 'aspanformer',
            'sgmnet', 'adalam', 'disk', 'nn'
        ]
        # Filter to only those that are actually available
        available = []
        for mat in matchers:
            try:
                get_matcher(mat)
                available.append(mat)
            except:
                pass
        return available
    except Exception as e:
        logger.warning(f"Could not get matcher list: {e}")
        return ['lightglue', 'superglue', 'loftr']  # Fallback


def get_extractor_config(extractor_name: str) -> Dict[str, Any]:
    """
    Get default configuration for an extractor
    
    Args:
        extractor_name: Name of the extractor
        
    Returns:
        Dictionary with extractor configuration parameters
    """
    if not IMCUI_AVAILABLE:
        return {}
    
    try:
        # Get extractor instance to inspect its parameters
        extractor = get_extractor(extractor_name)
        
        # Common parameters across extractors
        config = {
            'max_num_keypoints': 2048,
            'detection_threshold': None,
            'nms_window_size': None,
        }
        
        # Try to get actual defaults from the extractor
        if hasattr(extractor, 'default_config'):
            config.update(extractor.default_config)
        elif hasattr(extractor, 'config'):
            config.update(extractor.config)
        
        return config
    except Exception as e:
        logger.warning(f"Could not get extractor config for {extractor_name}: {e}")
        return {}


def get_matcher_config(matcher_name: str, extractor_name: str) -> Dict[str, Any]:
    """
    Get default configuration for a matcher
    
    Args:
        matcher_name: Name of the matcher
        extractor_name: Name of the extractor (some matchers need this)
        
    Returns:
        Dictionary with matcher configuration parameters
    """
    if not IMCUI_AVAILABLE:
        return {}
    
    try:
        # Get matcher instance to inspect its parameters
        matcher = get_matcher(matcher_name, extractor_name)
        
        # Common parameters
        config = {
            'n_layers': 9,
            'depth_confidence': 0.9,
            'width_confidence': 0.99,
            'filter_threshold': 0.1,
        }
        
        # Try to get actual defaults
        if hasattr(matcher, 'default_config'):
            config.update(matcher.default_config)
        elif hasattr(matcher, 'config'):
            config.update(matcher.config)
        
        return config
    except Exception as e:
        logger.warning(f"Could not get matcher config for {matcher_name}: {e}")
        return {}


def match_images_with_imcui(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    extractor: str = "superpoint",
    matcher: str = "lightglue",
    extractor_config: Optional[Dict[str, Any]] = None,
    matcher_config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Match two images using imcui's extractors and matchers directly
    
    Args:
        fixed_image: Fixed image as numpy array (H, W, C) or (H, W)
        moving_image: Moving image as numpy array (H, W, C) or (H, W)
        extractor: Feature extractor name
        matcher: Matcher name
        extractor_config: Optional extractor configuration
        matcher_config: Optional matcher configuration
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None
        
    Returns:
        Dictionary with match results:
        - keypoints0: Keypoints from fixed image
        - keypoints1: Keypoints from moving image
        - matches: Match indices
        - match_scores: Match confidence scores
        - descriptors0: Descriptors from fixed image (if available)
        - descriptors1: Descriptors from moving image (if available)
    """
    if not IMCUI_AVAILABLE:
        raise RuntimeError("imcui is not installed. Install with: pip install imcui")
    
    try:
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prepare images
        # Convert to RGB if needed
        if len(fixed_image.shape) == 2:
            fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_GRAY2RGB)
        elif fixed_image.shape[2] == 4:
            fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_RGBA2RGB)
        
        if len(moving_image.shape) == 2:
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_GRAY2RGB)
        elif moving_image.shape[2] == 4:
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_RGBA2RGB)
        
        # Normalize to [0, 1] and convert to torch tensor
        fixed_tensor = torch.from_numpy(fixed_image).float() / 255.0
        moving_tensor = torch.from_numpy(moving_image).float() / 255.0
        
        # Convert to (C, H, W) format
        if len(fixed_tensor.shape) == 3:
            fixed_tensor = fixed_tensor.permute(2, 0, 1)
        if len(moving_tensor.shape) == 3:
            moving_tensor = moving_tensor.permute(2, 0, 1)
        
        # Add batch dimension
        fixed_tensor = fixed_tensor.unsqueeze(0).to(device)
        moving_tensor = moving_tensor.unsqueeze(0).to(device)
        
        # Get extractor and matcher
        extractor_instance = get_extractor(extractor)
        if extractor_config:
            # Update extractor config if provided
            for key, value in extractor_config.items():
                if hasattr(extractor_instance, key):
                    setattr(extractor_instance, key, value)
        
        matcher_instance = get_matcher(matcher, extractor)
        if matcher_config:
            # Update matcher config if provided
            for key, value in matcher_config.items():
                if hasattr(matcher_instance, key):
                    setattr(matcher_instance, key, value)
        
        # Move to device
        extractor_instance = extractor_instance.eval().to(device)
        matcher_instance = matcher_instance.eval().to(device)
        
        # Extract features
        logger.debug(f"Extracting features with {extractor}...")
        feats0 = extractor_instance.extract(fixed_tensor)
        feats1 = extractor_instance.extract(moving_tensor)
        
        # Match features
        logger.debug(f"Matching features with {matcher}...")
        matches = matcher_instance({'image0': feats0, 'image1': feats1})
        
        # Remove batch dimension if present
        from lightglue.utils import rbd
        feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]
        
        # Extract results
        keypoints0 = feats0['keypoints'].cpu().numpy()  # (N, 2)
        keypoints1 = feats1['keypoints'].cpu().numpy()  # (M, 2)
        match_indices = matches['matches'].cpu().numpy()  # (K, 2)
        match_scores = matches.get('matching_scores', None)
        if match_scores is not None:
            match_scores = match_scores.cpu().numpy()
        
        # Get descriptors if available
        descriptors0 = feats0.get('descriptors', None)
        descriptors1 = feats1.get('descriptors', None)
        if descriptors0 is not None:
            descriptors0 = descriptors0.cpu().numpy()
        if descriptors1 is not None:
            descriptors1 = descriptors1.cpu().numpy()
        
        logger.info(f"Found {len(match_indices)} matches using {extractor} + {matcher}")
        
        return {
            'keypoints0': keypoints0.tolist(),
            'keypoints1': keypoints1.tolist(),
            'matches': match_indices.tolist(),
            'match_scores': match_scores.tolist() if match_scores is not None else None,
            'descriptors0': descriptors0.tolist() if descriptors0 is not None else None,
            'descriptors1': descriptors1.tolist() if descriptors1 is not None else None,
            'extractor': extractor,
            'matcher': matcher,
            'num_keypoints0': len(keypoints0),
            'num_keypoints1': len(keypoints1),
            'num_matches': len(match_indices)
        }
        
    except Exception as e:
        logger.error(f"imcui matching failed: {e}", exc_info=True)
        raise


def register_with_imcui(
    fixed_id: str,
    moving_id: str,
    extractor: str = "superpoint",
    matcher: str = "lightglue",
    thumbnail_width: int = 1024,
    extractor_config: Optional[Dict[str, Any]] = None,
    matcher_config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    return_transform: bool = True
) -> Dict[str, Any]:
    """
    Perform registration using imcui's extractors and matchers
    
    Args:
        fixed_id: DSA item ID of fixed image
        moving_id: DSA item ID of moving image
        extractor: Feature extractor name
        matcher: Matcher name
        thumbnail_width: Width of thumbnail to use
        extractor_config: Optional extractor configuration
        matcher_config: Optional matcher configuration
        device: Device to use
        return_transform: Whether to compute and return registration transform
        
    Returns:
        Dictionary with registration results
    """
    if not IMCUI_AVAILABLE:
        raise RuntimeError("imcui is not installed. Install with: pip install imcui")
    
    try:
        # Get images from DSA
        fixed_img = get_thumbnail_image(fixed_id, width=thumbnail_width)
        moving_img = get_thumbnail_image(moving_id, width=thumbnail_width)
        
        # Match images
        match_results = match_images_with_imcui(
            fixed_image=fixed_img,
            moving_image=moving_img,
            extractor=extractor,
            matcher=matcher,
            extractor_config=extractor_config,
            matcher_config=matcher_config,
            device=device
        )
        
        result = {
            'success': True,
            'fixed_id': fixed_id,
            'moving_id': moving_id,
            'match_results': match_results
        }
        
        # Compute transform if requested
        if return_transform:
            from app.services.lightglue_service import estimate_rigid_transform_ransac
            
            keypoints0 = np.array(match_results['keypoints0'])
            keypoints1 = np.array(match_results['keypoints1'])
            matches = np.array(match_results['matches'])
            
            # Extract matched points
            matched_kpts0 = keypoints0[matches[:, 0]]
            matched_kpts1 = keypoints1[matches[:, 1]]
            
            if len(matched_kpts0) >= 2:
                transform_matrix, inliers = estimate_rigid_transform_ransac(
                    src_points=matched_kpts1,  # moving
                    dst_points=matched_kpts0,    # fixed
                    max_iters=2000,
                    threshold=3.0
                )
                
                if transform_matrix is not None:
                    rotation_degrees = float(np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])))
                    offset_x = float(transform_matrix[0, 2])
                    offset_y = float(transform_matrix[1, 2])
                    
                    result['transform'] = {
                        'transform_matrix': transform_matrix.tolist(),
                        'rotation_degrees': rotation_degrees,
                        'offset_x': offset_x,
                        'offset_y': offset_y,
                        'scale': 1.0,
                        'num_matches': len(matched_kpts0),
                        'num_inliers': int(np.sum(inliers)) if inliers is not None else 0,
                        'success': True
                    }
                else:
                    result['transform'] = {'success': False, 'error': 'Failed to estimate transform'}
            else:
                result['transform'] = {'success': False, 'error': 'Insufficient matches'}
        
        return result
        
    except Exception as e:
        logger.error(f"imcui registration failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'fixed_id': fixed_id,
            'moving_id': moving_id
        }
