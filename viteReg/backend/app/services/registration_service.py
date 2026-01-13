"""
Registration service using SimpleITK for rigid image registration
"""
import numpy as np
import cv2
import SimpleITK as sitk
import logging
import requests
import os
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from joblib import Memory
from app.services.dsa_client import get_dsa_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize joblib Memory cache for numpy arrays
# Cache directory is persistent within Docker container
cache_dir = settings.CACHE_DIR
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


@memory.cache
def _fetch_thumbnail_uncached(item_id: str, width: int, token: str) -> Optional[np.ndarray]:
    """Internal cached function to fetch thumbnail (token is part of cache key)"""
    base_url = settings.DSA_BASE_URL.rstrip('/api/v1')
    url = f"{base_url}/api/v1/item/{item_id}/tiles/thumbnail?width={width}"
    if token:
        url += f"&token={token}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        logger.error(f"Failed to get thumbnail for {item_id}: {e}")
        return None


def get_thumbnail_image(item_id: str, width: int = 1024) -> Optional[np.ndarray]:
    """Fetch thumbnail image from DSA and return as numpy array (cached)"""
    dsa = get_dsa_client()
    token = dsa.get_token() or ""
    return _fetch_thumbnail_uncached(item_id, width, token)


def pad_to_size(mask: np.ndarray, target_size: int) -> np.ndarray:
    """Pad a mask to a target square size, centering it"""
    h, w = mask.shape
    pad_height = target_size - h
    pad_width = target_size - w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)


def find_relative_rotation(padded_mask1: np.ndarray, padded_mask2: np.ndarray) -> Tuple[int, float]:
    """
    Find relative rotation between two masks by testing 0, 90, 180, 270 degrees
    
    Uses Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    where |A| is the number of non-zero pixels in mask A
    """
    best_k = 0
    max_dice = 0
    
    # Convert masks to binary (0 or 1) for proper Dice calculation
    mask1_binary = (padded_mask1 > 0).astype(np.uint8)
    mask2_binary = (padded_mask2 > 0).astype(np.uint8)
    
    for k in range(4):
        rotated_mask2 = np.rot90(mask2_binary, k)
        
        # Calculate intersection and individual mask sizes
        intersection = np.logical_and(mask1_binary, rotated_mask2).sum()
        size1 = mask1_binary.sum()
        size2 = rotated_mask2.sum()
        
        # Dice coefficient: 2 * intersection / (size1 + size2)
        # This ensures Dice is always between 0 and 1
        dice = 2.0 * intersection / (size1 + size2) if (size1 + size2) > 0 else 0.0
        
        # Clamp to [0, 1] to handle any edge cases
        dice = min(1.0, max(0.0, dice))
        
        if dice > max_dice:
            max_dice = dice
            best_k = k
    
    return best_k * 90, max_dice


def prepare_mask_exit(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Prepare a binary mask from an image using Gaussian blur and thresholding
    
    Args:
        image: Input image (grayscale or RGB)
        method: Thresholding method - "otsu" (adaptive) or "fixed" (0.95 threshold)
    
    Returns:
        Binary mask (255 for tissue, 0 for background)
    """
    # Normalize to 0-1 if needed
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur((image * 255).astype(np.uint8), (5, 5), 1.0)
    
    if method == "otsu":
        # Use OTSU's method for adaptive thresholding
        # OTSU automatically finds the optimal threshold that minimizes intra-class variance
        threshold_value, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Invert so that tissue is white (255) and background is black (0)
        # THRESH_BINARY_INV already inverts, so tissue (darker) becomes white
        logger.debug(f"OTSU threshold: {threshold_value:.1f}")
    else:
        # Fixed threshold at 0.95 (original method)
        # Assumes background is very bright (>0.95) and tissue is darker
        threshold = int(0.95 * 255)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        logger.debug(f"Fixed threshold: {threshold} (0.95)")
    
    return mask


def calculate_mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 50) -> float:
    """Calculate mutual information between two images"""
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Flatten images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # Calculate 2D histogram
    hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)
    
    # Convert to probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x
    py = np.sum(pxy, axis=0)  # marginal for y
    
    # Calculate mutual information
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # Only include non-zero elements
    
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return float(mi)


def calculate_normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate normalized cross-correlation between two images (better for different stains)"""
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalize images
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-10)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-10)
    
    # Calculate cross-correlation
    ncc = (img1_norm * img2_norm).mean()
    return float(ncc)


def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) - focuses on structural features"""
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalize to 0-1 if needed
    if img1.max() > 1.0:
        img1 = img1.astype(np.float32) / 255.0
    if img2.max() > 1.0:
        img2 = img2.astype(np.float32) / 255.0
    
    # Constants for SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    # Calculate variances and covariance
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / (denominator + 1e-10)
    return float(ssim)


@memory.cache
def _register_rigid_cached(
    fixed_id: str,
    moving_id: str,
    fixed_img_hash: str,
    moving_img_hash: str,
    thumbnail_width: int = 1024
) -> Dict[str, Any]:
    """
    Internal cached registration function.
    Image hashes are used to invalidate cache when images change.
    This caches the expensive SimpleITK registration computation.
    """
    try:
        # Get thumbnail images (these are already cached separately)
        fixed_img = get_thumbnail_image(fixed_id, width=thumbnail_width)
        moving_img = get_thumbnail_image(moving_id, width=thumbnail_width)
        
        if fixed_img is None or moving_img is None:
            raise ValueError("Failed to retrieve thumbnail images")
        
        # Convert grayscale to RGB if needed
        if len(fixed_img.shape) == 2:
            fixed_img = np.stack([fixed_img] * 3, axis=-1)
        if len(moving_img.shape) == 2:
            moving_img = np.stack([moving_img] * 3, axis=-1)
        
        # Create grayscale copies for registration
        fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
        moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
        
        # Convert to float32 for SimpleITK
        fixed_gray = fixed_gray.astype(np.float32) / 255.0
        moving_gray = moving_gray.astype(np.float32) / 255.0
        
        # Prepare masks and find relative rotation (using OTSU for adaptive thresholding)
        mask1 = prepare_mask_exit(fixed_gray, method="otsu")
        mask2 = prepare_mask_exit(moving_gray, method="otsu")
        
        # Pad masks to same size
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        max_dim = max(h1, w1, h2, w2)
        padded_mask1 = pad_to_size(mask1, max_dim)
        padded_mask2 = pad_to_size(mask2, max_dim)
        
        # Find relative rotation
        relative_rotation, dice = find_relative_rotation(padded_mask1, padded_mask2)
        logger.info(f"Relative rotation: {relative_rotation}°, Dice: {dice:.3f}")
        
        # Rotate moving image if needed
        if relative_rotation != 0:
            moving_gray = np.rot90(moving_gray, int(relative_rotation / 90))
            moving_img = np.rot90(moving_img, int(relative_rotation / 90))
        
        # Convert to SimpleITK format
        fixed_sitk = sitk.GetImageFromArray(fixed_gray)
        moving_sitk = sitk.GetImageFromArray(moving_gray)
        
        # Define rigid transform (rotation + translation only, no scaling)
        # For tissue slides, we assume same physical size, so scaling can cause poor registrations
        transform = sitk.Euler2DTransform()
        
        # Initialize transform by aligning centers
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk,
            moving_sitk,
            transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        
        # Set up registration
        registration_method = sitk.ImageRegistrationMethod()
        
        # Use mutual information metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        
        # Configure gradient descent optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        
        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Apply initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        
        # Balance parameter updates
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Use multi-resolution for robustness
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        
        # Execute registration
        logger.info("Executing registration...")
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        # Extract parameters (Euler2DTransform: angle, tx, ty - no scale)
        parameters = final_transform.GetParameters()
        angle = parameters[0]
        tx = parameters[1]
        ty = parameters[2]
        
        # Construct affine matrix (rigid transform: rotation + translation, scale = 1.0)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        affine_matrix = np.array([
            [cos_theta, -sin_theta, tx],
            [sin_theta, cos_theta, ty],
            [0, 0, 1],
        ])
        
        # Calculate metrics
        rotation_degrees = np.degrees(angle)
        offset_x = float(tx)
        offset_y = float(ty)
        scaling = 1.0  # Rigid transform has no scaling
        
        # Apply transform to moving image and calculate MI
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetTransform(final_transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        registered_sitk = resampler.Execute(moving_sitk)
        registered_img = sitk.GetArrayFromImage(registered_sitk)
        
        # Calculate multiple metrics for different stain types
        mi_score = calculate_mutual_information(fixed_gray, registered_img)
        ncc_score = calculate_normalized_cross_correlation(fixed_gray, registered_img)
        ssim_score = calculate_structural_similarity(fixed_gray, registered_img)
        
        logger.info(f"Registration complete: rotation={rotation_degrees:.2f}°, "
                   f"offset=({offset_x:.2f}, {offset_y:.2f}), scale={scaling:.4f}, "
                   f"MI={mi_score:.4f}, NCC={ncc_score:.4f}, SSIM={ssim_score:.4f}, Dice={dice:.4f}")
        
        return {
            "transform_matrix": affine_matrix.tolist(),
            "rotation_degrees": rotation_degrees,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "scale": scaling,
            "mutual_information": mi_score,
            "normalized_cross_correlation": ncc_score,
            "structural_similarity": ssim_score,
            "success": True,
            "relative_rotation": int(relative_rotation),
            "dice_coefficient": float(dice),
            "reg_image_size": thumbnail_width,
            "pre_rotate": "None",  # Default, can be updated later
        }
        
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        return {
            "transform_matrix": np.eye(3).tolist(),
            "rotation_degrees": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "scale": 1.0,
            "mutual_information": 0.0,
            "normalized_cross_correlation": None,
            "structural_similarity": None,
            "success": False,
            "error": str(e),
        }


def _get_image_hash(img: np.ndarray) -> str:
    """Generate a hash for an image array to use as cache key"""
    import hashlib
    return hashlib.md5(img.tobytes()).hexdigest()[:16]


def register_rigid(
    fixed_id: str,
    moving_id: str,
    thumbnail_width: int = 1024
) -> Dict[str, Any]:
    """
    Perform rigid registration using SimpleITK (with caching)
    
    Args:
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        thumbnail_width: Width of thumbnail to use for registration
        
    Returns:
        Dictionary with registration results:
        - transform_matrix: 3x3 affine transformation matrix
        - rotation_degrees: Rotation in degrees
        - offset_x: X translation
        - offset_y: Y translation
        - scale: Scale factor
        - mutual_information: MI score
        - success: Whether registration succeeded
    """
    try:
        # Get thumbnail images first (to compute hashes)
        logger.debug(f"Fetching thumbnails for registration: fixed={fixed_id}, moving={moving_id}")
        fixed_img = get_thumbnail_image(fixed_id, width=thumbnail_width)
        moving_img = get_thumbnail_image(moving_id, width=thumbnail_width)
        
        if fixed_img is None or moving_img is None:
            raise ValueError("Failed to retrieve thumbnail images")
        
        # Compute image hashes for cache invalidation
        fixed_hash = _get_image_hash(fixed_img)
        moving_hash = _get_image_hash(moving_img)
        
        # Call cached registration function
        return _register_rigid_cached(
            fixed_id,
            moving_id,
            fixed_hash,
            moving_hash,
            thumbnail_width
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        return {
            "transform_matrix": np.eye(3).tolist(),
            "rotation_degrees": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "scale": 1.0,
            "mutual_information": 0.0,
            "normalized_cross_correlation": None,
            "structural_similarity": None,
            "success": False,
            "error": str(e),
        }

