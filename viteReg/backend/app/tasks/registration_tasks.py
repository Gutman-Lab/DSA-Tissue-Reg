"""
Celery tasks for image registration
"""
import logging
import json
import numpy as np
from datetime import datetime
from app.celery_app import celery_app
from app.services.registration_service import register_rigid
from app.services.lightglue_service import register_with_lightglue, LIGHTGLUE_AVAILABLE
from app.services.tps_service import register_with_tps, register_with_affine_tps, register_with_affine_tps
from app.services.dsa_client import get_dsa_client

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def _save_registration_to_dsa(moving_id: str, fixed_id: str, result: dict, method: str):
    """
    Save registration results to DSA metadata
    
    Args:
        moving_id: DSA item ID of moving image
        fixed_id: DSA item ID of fixed image
        result: Registration result dictionary
        method: Registration method name (e.g., 'simpleitk', 'lightglue')
    """
    try:
        dsa = get_dsa_client()
        
        # Get current item metadata
        item_data = dsa.get_item(moving_id)
        item_meta = item_data.get("meta", {})
        
        # Store registration parameters with method-specific key
        # Format: npReg_{method} for multiple methods
        reg_key = f"npReg_{method}"
        item_meta[reg_key] = {
            "srcImage": fixed_id,
            "method": method,
            "xOffset": result.get("offset_x", 0.0),
            "yOffset": result.get("offset_y", 0.0),
            "scale": result.get("scale", 1.0),
            "rotation": result.get("rotation_degrees", 0.0),
            "regImageSize": result.get("reg_image_size", 1024),
            "preRotate": result.get("pre_rotate", "None"),
            "mutual_information": result.get("mutual_information", 0.0),
            "dice_coefficient": result.get("dice_coefficient"),
            "normalized_cross_correlation": result.get("normalized_cross_correlation"),
            "structural_similarity": result.get("structural_similarity"),
            "relative_rotation": result.get("relative_rotation", 0),
        }
        
        # Store method-specific additional fields
        if method in ["lightglue", "tps", "affine_tps"]:
            item_meta[reg_key]["num_matches"] = result.get("num_matches", 0)
            item_meta[reg_key]["num_inliers"] = result.get("num_inliers", 0)
            # Store match data for visualization (if available)
            if result.get("match_data"):
                # Store match data as JSON (it's already a dict with lists)
                import json
                item_meta[reg_key]["match_data"] = result.get("match_data")
        
        # Store transform matrix as JSON string (XFM_{method})
        transform_matrix = result.get("transform_matrix")
        if transform_matrix:
            # Convert to dict format: {0: [row0], 1: [row1], 2: [row2]}
            xfm_dict = {str(i): list(row) for i, row in enumerate(transform_matrix)}
            item_meta[f"XFM_{method}"] = json.dumps(xfm_dict)
        
        # Also maintain backward compatibility: store in npReg if it's the default method
        if method == "simpleitk":
            item_meta["npReg"] = item_meta[reg_key]
            item_meta["XFM"] = item_meta.get(f"XFM_{method}")
        
        # Update metadata in DSA
        dsa.update_item_metadata(moving_id, item_meta)
        logger.info(f"Saved {method} registration metadata to DSA for item {moving_id}")
        
    except Exception as e:
        logger.warning(f"Failed to save registration metadata to DSA: {e}")
        # Don't fail the task if metadata save fails


@celery_app.task(bind=True, name="registration.rigid")
def task_register_rigid(self, *args, **kwargs):
    """
    Celery task to perform rigid registration
    
    Supports both old signature (job_id, fixed_id, moving_id) and new (with method parameter)
    
    Args (positional or keyword):
        job_id: Unique job identifier
        fixed_id: DSA item ID of fixed (reference) image
        moving_id: DSA item ID of moving image
        method: Registration method ('simpleitk', 'lightglue', or 'tps'). Defaults to 'simpleitk'.
        
    Returns:
        Dictionary with registration results
    """
    try:
        # Extract parameters - handle both old and new call signatures
        if len(args) >= 3:
            # Old signature: (job_id, fixed_id, moving_id) or new: (job_id, fixed_id, moving_id, method)
            job_id = args[0]
            fixed_id = args[1]
            moving_id = args[2]
            method = args[3] if len(args) > 3 else kwargs.get('method', 'simpleitk')
        elif 'job_id' in kwargs:
            # All keyword arguments
            job_id = kwargs['job_id']
            fixed_id = kwargs['fixed_id']
            moving_id = kwargs['moving_id']
            method = kwargs.get('method', 'simpleitk')
        else:
            raise ValueError("Invalid arguments: must provide job_id, fixed_id, moving_id")
        
        # Validate method
        if method not in ["simpleitk", "tps", "affine_tps"]:
            method = "simpleitk"
        
        logger.info(f"Starting {method} registration task {job_id}: fixed={fixed_id}, moving={moving_id}")
        
        # Update task state
        self.update_state(
            state="RUNNING",
            meta={
                "job_id": job_id,
                "status": "running",
                "message": f"{method} registration in progress",
            },
        )
        
        # Perform registration based on method
        # Note: LightGlue is only used internally by TPS methods (non-rigid), not as a standalone method
        # For rigid/affine, use SimpleITK (intensity-based, more appropriate for low DOF)
        if method == "simpleitk":
            result = register_rigid(fixed_id, moving_id)
        elif method == "tps":
            if not LIGHTGLUE_AVAILABLE:
                raise RuntimeError("LightGlue is not installed. Install with: pip install lightglue")
            result = register_with_tps(fixed_id, moving_id)
        elif method == "affine_tps":
            if not LIGHTGLUE_AVAILABLE:
                raise RuntimeError("LightGlue is not installed. Install with: pip install lightglue")
            result = register_with_affine_tps(fixed_id, moving_id)
        else:
            raise ValueError(f"Unknown registration method: {method}. Choose 'simpleitk', 'tps', or 'affine_tps'")
        
        # Add job metadata
        result["job_id"] = job_id
        result["method"] = method
        result["status"] = "completed" if result.get("success") else "failed"
        result["completed_at"] = datetime.utcnow().isoformat()
        
        # Convert all numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        # Save registration results to DSA metadata if successful
        if result.get("success"):
            _save_registration_to_dsa(moving_id, fixed_id, result, method)
        
        logger.info(f"Registration task {job_id} ({method}) completed: success={result.get('success')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Registration task {job_id} failed: {e}", exc_info=True)
        return {
            "job_id": job_id,
            "method": method,
            "status": "failed",
            "success": False,
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        }

