"""
Image registration service using SimpleITK
Provides rigid registration functionality and utility functions
"""
import numpy as np
import cv2
import SimpleITK as sitk
import logging
import requests
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import skimage
from skimage import filters
from app.services.dsa_client import get_dsa_client

logger = logging.getLogger(__name__)

# Simple memory cache decorator (can be replaced with joblib.Memory if needed)
class SimpleCache:
    """Simple in-memory cache for function results"""
    def __init__(self):
        self.cache = {}
    
    def cache_decorator(self, func):
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            key = str(args) + str(sorted(kwargs.items()))
            if key not in self.cache:
                self.cache[key] = func(*args, **kwargs)
            return self.cache[key]
        return wrapper

memory = SimpleCache()


def get_thumbnail_image(item_id: str, width: int = 1024) -> np.ndarray:
    """
    Fetch thumbnail image from DSA and return as numpy array
    
    Args:
        item_id: DSA item ID
        width: Thumbnail width in pixels
        
    Returns:
        Image as numpy array (RGB, uint8)
    """
    try:
        dsa = get_dsa_client()
        url = dsa.get_thumbnail_url(item_id, width=width)
        
        response = requests.get(url)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img_np = np.array(img)
        
        # Convert to RGB if image is in RGBA format
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img = img.convert("RGB")
            img_np = np.array(img)
        
        return img_np
    except Exception as e:
        logger.error(f"Error fetching thumbnail for item {item_id}: {e}")
        raise


def prepare_mask_exit(image: np.ndarray, method: str = "threshold") -> np.ndarray:
    """
    Create a mask based on image processing
    
    Args:
        image: Input image (grayscale or RGB)
        method: Method to use ('threshold' or 'otsu')
        
    Returns:
        Binary mask (True for tissue, False for background)
    """
    # If image is already grayscale, use it directly
    if len(image.shape) == 2:
        gray_image = image
    else:
        # Convert RGB to grayscale
        gray_image = skimage.color.rgb2gray(image)
    
    if method == "otsu":
        # Use Otsu's method for thresholding
        threshold = filters.threshold_otsu(gray_image)
        mask = gray_image < threshold
    else:
        # Original method: gaussian filter + threshold
        blurred_image = filters.gaussian(gray_image, sigma=1.0)
        threshold = 0.95
        mask = blurred_image < threshold
    
    return mask


def pad_to_size(mask: np.ndarray, target_size: int) -> np.ndarray:
    """
    Pad a mask to a target square size, centering it
    
    Args:
        mask: Input mask
        target_size: Target size (square)
        
    Returns:
        Padded mask
    """
    h, w = mask.shape
    pad_height = target_size - h
    pad_width = target_size - w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    padded_mask = np.pad(
        mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=False,
    )
    return padded_mask


def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Dice coefficient between two masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dice coefficient (0-1)
    """
    intersection = np.sum(mask1 & mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    if size1 + size2 == 0:
        return 1.0
    return 2 * intersection / (size1 + size2)


def find_relative_rotation(padded_mask1: np.ndarray, padded_mask2: np.ndarray) -> Tuple[int, float]:
    """
    Find the relative rotation between two masks by testing 0°, 90°, 180°, 270°
    
    Args:
        padded_mask1: First mask
        padded_mask2: Second mask
        
    Returns:
        Tuple of (rotation_angle_degrees, dice_coefficient)
    """
    max_dice = 0
    best_k = 0
    first_dice = 0
    
    for k in range(4):  # 0°, 90°, 180°, 270°
        rotated_mask2 = np.rot90(padded_mask2, k)
        dice = dice_coefficient(padded_mask1, rotated_mask2)
        
        if k == 0:
            first_dice = dice
        
        if dice > max_dice:
            if dice > (first_dice + 0.005):
                best_k = k
            max_dice = dice
    
    return best_k * 90, max_dice  # Rotation angle and similarity score


def metrics_registration(affine_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Extract registration metrics from affine transformation matrix
    
    Args:
        affine_matrix: 3x3 affine transformation matrix
        
    Returns:
        Tuple of (rotation_degrees, offset_x, offset_y, scaling)
    """
    a, b, t_x = affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[0, 2]
    c, d, t_y = affine_matrix[1, 0], affine_matrix[1, 1], affine_matrix[1, 2]
    
    rotation_degrees = np.arctan2(c, a) * 180 / np.pi
    offset_x = t_x
    offset_y = t_y
    scaling = np.sqrt(a * d - b * c)
    
    return rotation_degrees, offset_x, offset_y, scaling


def register_rigid(fixed_id: str, moving_id: str, thumbnail_width: int = 1024) -> Dict[str, Any]:
    """
    Perform rigid registration using SimpleITK
    
    Uses Similarity2DTransform (rigid + uniform scaling) with mutual information metric
    
    Args:
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        thumbnail_width: Width of thumbnail to use for registration
        
    Returns:
        Dictionary with registration results:
        - success: Whether registration succeeded
        - transform_matrix: 3x3 affine transformation matrix
        - rotation_degrees: Rotation in degrees
        - offset_x: X translation
        - offset_y: Y translation
        - scale: Scale factor
        - mutual_information: Mutual information metric value
        - reg_image_size: Size of image used for registration
        - thumbnail_width: Width of thumbnail used
    """
    try:
        logger.info(f"Starting rigid registration: fixed={fixed_id}, moving={moving_id}")
        
        # Get thumbnail images
        fixed_img_sk = get_thumbnail_image(fixed_id, width=thumbnail_width)
        moving_img_sk = get_thumbnail_image(moving_id, width=thumbnail_width)
        
        # Convert grayscale to RGB if needed
        if len(fixed_img_sk.shape) == 2:
            fixed_img_sk = np.stack([fixed_img_sk] * 3, axis=-1)
        if len(moving_img_sk.shape) == 2:
            moving_img_sk = np.stack([moving_img_sk] * 3, axis=-1)
        
        # CRITICAL: Normalize image sizes - ensure both have same dimensions
        # Even with same width, different aspect ratios can cause different heights
        h_fixed, w_fixed = fixed_img_sk.shape[:2]
        h_moving, w_moving = moving_img_sk.shape[:2]
        
        if h_moving != h_fixed or w_moving != w_fixed:
            logger.info(f"Normalizing image sizes: fixed={w_fixed}x{h_fixed}, moving={w_moving}x{h_moving}")
            # Resize moving image to match fixed image dimensions
            moving_img_sk = cv2.resize(moving_img_sk, (w_fixed, h_fixed), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized moving image to {w_fixed}x{h_fixed} to match fixed image")
        
        # Create grayscale copies for registration while preserving original color
        fixed_gray = cv2.cvtColor(fixed_img_sk, cv2.COLOR_RGB2GRAY)
        moving_gray = cv2.cvtColor(moving_img_sk, cv2.COLOR_RGB2GRAY)
        
        # Convert to float32 for SimpleITK
        fixed_gray = fixed_gray.astype(np.float32) / 255.0
        moving_gray = moving_gray.astype(np.float32) / 255.0
        
        # Load and process both images for mask creation
        mask = prepare_mask_exit(fixed_gray)
        mask2 = prepare_mask_exit(moving_gray)
        
        # Determine the maximum dimension and pad both masks
        h1, w1 = mask.shape
        h2, w2 = mask2.shape
        max_dim = max(h1, w1, h2, w2)
        padded_mask1 = pad_to_size(mask, max_dim)
        padded_mask2 = pad_to_size(mask2, max_dim)
        
        # Determine the orientation
        relative_rotation, dice = find_relative_rotation(padded_mask1, padded_mask2)
        
        logger.debug(f"Relative rotation: {relative_rotation} degrees, Dice: {dice:.3f}")
        
        # Rotate moving image if needed
        if relative_rotation != 0:
            moving_image_rgb = np.rot90(moving_img_sk, int(relative_rotation / 90))
            moving_gray_rotated = np.rot90(moving_gray, int(relative_rotation / 90))
        else:
            moving_image_rgb = moving_img_sk
            moving_gray_rotated = moving_gray
        
        # Convert to SimpleITK format - use grayscale for registration
        fixed_img = sitk.GetImageFromArray(fixed_gray)
        moving_img = sitk.GetImageFromArray(moving_gray_rotated)
        
        # Define a similarity transform
        transform = sitk.Similarity2DTransform()
        
        # Initialize transform by aligning centers
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img,
            moving_img,
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
        final_transform = registration_method.Execute(fixed_img, moving_img)
        
        # Extract parameters from the final transform
        parameters = final_transform.GetParameters()
        scale = parameters[0]
        angle = parameters[1]
        tx = parameters[2]
        ty = parameters[3]
        
        # Construct the affine matrix
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Similarity transform matrix
        affine_matrix = np.array(
            [
                [scale * cos_theta, -scale * sin_theta, tx],
                [scale * sin_theta, scale * cos_theta, ty],
                [0, 0, 1],
            ]
        )
        
        # Extract metrics
        rotation_degrees, offset_x, offset_y, scaling = metrics_registration(affine_matrix)
        
        # Get mutual information value (if available)
        try:
            mutual_information = registration_method.GetMetricValue()
        except:
            mutual_information = 0.0
        
        # Get image size
        reg_image_size = max(fixed_img_sk.shape[0], fixed_img_sk.shape[1])
        
        result = {
            "success": True,
            "transform_matrix": affine_matrix.tolist(),
            "rotation_degrees": float(rotation_degrees),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "scale": float(scaling),
            "mutual_information": float(mutual_information),
            "reg_image_size": int(reg_image_size),
            "thumbnail_width": int(thumbnail_width),
            "pre_rotate": str(relative_rotation) if relative_rotation != 0 else "None",
            "relative_rotation": int(relative_rotation),
        }
        
        logger.info(f"Registration completed successfully: rotation={rotation_degrees:.2f}°, scale={scaling:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "transform_matrix": np.eye(3).tolist(),
            "rotation_degrees": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "scale": 1.0,
            "mutual_information": 0.0,
            "reg_image_size": thumbnail_width,
            "thumbnail_width": thumbnail_width,
        }


def calculate_mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 32) -> float:
    """
    Calculate mutual information between two images
    
    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        bins: Number of bins for histogram
        
    Returns:
        Mutual information value
    """
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Flatten the images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # Calculate joint histogram
    hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)
    
    # Convert to probability distribution
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x
    py = np.sum(pxy, axis=0)  # marginal for y
    px_py = px[:, None] * py[None, :]
    
    # Avoid log(0)
    nonzero = pxy > 0
    
    # Calculate mutual information
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    return float(mi)


def calculate_normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate normalized cross-correlation between two images
    
    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        
    Returns:
        Normalized cross-correlation coefficient (-1 to 1)
    """
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to float and normalize
    img1_norm = img1.astype(np.float64) - np.mean(img1)
    img2_norm = img2.astype(np.float64) - np.mean(img2)
    
    # Calculate cross-correlation
    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))
    
    if denominator == 0:
        return 0.0
    
    ncc = numerator / denominator
    return float(ncc)


def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate structural similarity index (SSIM) between two images
    
    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        
    Returns:
        SSIM value (0 to 1, higher is better)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        # Fallback implementation if skimage is not available
        logger.warning("skimage.metrics.structural_similarity not available, using simple implementation")
        return calculate_normalized_cross_correlation(img1, img2)
    
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate SSIM
    # data_range is the range of the image data (e.g., 255 for uint8, 1.0 for float)
    if img1.dtype == np.uint8:
        data_range = 255
    else:
        data_range = img1.max() - img1.min()
        if data_range == 0:
            data_range = 1.0
    
    ssim_value = ssim(img1, img2, data_range=data_range)
    return float(ssim_value)
