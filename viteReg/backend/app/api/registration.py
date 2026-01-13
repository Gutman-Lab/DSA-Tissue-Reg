"""Image registration endpoints"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import uuid
import logging
import os
from datetime import datetime
from app.tasks.registration_tasks import task_register_rigid
from app.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter()

# Job status cache (backed by Celery results in Redis)
_registration_jobs: Dict[str, Dict[str, Any]] = {}


class RegistrationRequest(BaseModel):
    """Registration request model"""
    fixed_image_id: str
    moving_image_id: str
    method: Literal["sift", "orb", "akaze", "lightglue", "simpleitk", "tps"] = "sift"
    roi_size: int = 256


class Point(BaseModel):
    """Point coordinate model"""
    x: float
    y: float


class TransformParams(BaseModel):
    """Transform parameters model"""
    rotation: float
    scale: float
    offset_x: float
    offset_y: float


class RegistrationResponse(BaseModel):
    """Registration response model"""
    job_id: str
    status: str
    message: str


class RegistrationResult(BaseModel):
    """Registration result model"""
    job_id: str
    status: str
    fixed_image_id: str
    moving_image_id: str
    transform_matrix: List[List[float]]
    rotation_degrees: float
    offset_x: float
    offset_y: float
    scale: float
    mutual_information: float
    normalized_cross_correlation: Optional[float] = None  # NCC - better for different stains
    structural_similarity: Optional[float] = None  # SSIM - structural similarity
    success: bool
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    pre_rotate: Optional[str] = None  # Pre-rotation: "-90", "FlipXY", or "None"
    reg_image_size: Optional[int] = None  # Registration image size used
    relative_rotation: Optional[int] = None  # Relative rotation detected (0, 90, 180, 270)
    dice_coefficient: Optional[float] = None  # Dice coefficient from mask matching (best for different stains)
    method: Optional[str] = None  # Registration method: 'simpleitk' (rigid/affine), 'tps' (non-rigid), or 'affine_tps' (affine+TPS hybrid)
    num_matches: Optional[int] = None  # Number of feature matches (LightGlue/TPS only)
    num_inliers: Optional[int] = None  # Number of inlier matches (LightGlue/TPS only)


def _get_task_result(celery_task_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Get task result from Celery using Celery task ID"""
    try:
        # Get result from Celery using the Celery task ID
        task = celery_app.AsyncResult(celery_task_id)
        
        if task.ready():
            try:
                result = task.get(timeout=1)
                if isinstance(result, dict):
                    return result
                else:
                    logger.warning(f"Unexpected result type from Celery task {celery_task_id}: {type(result)}")
                    return None
            except Exception as e:
                logger.error(f"Error getting result from Celery task {celery_task_id}: {e}")
                # Check if task failed
                if task.failed():
                    return {
                        "status": "failed",
                        "job_id": job_id,
                        "error": str(task.info) if task.info else "Task failed",
                    }
                return None
        elif task.state == "PENDING":
            return {"status": "pending", "job_id": job_id}
        elif task.state == "STARTED" or task.state == "RUNNING":
            return {"status": "running", "job_id": job_id}
        else:
            # Task is in another state
            logger.debug(f"Task {celery_task_id} in state: {task.state}")
            return {
                "status": task.state.lower(),
                "job_id": job_id,
            }
    except Exception as e:
        logger.error(f"Error getting Celery result for task {celery_task_id}: {e}", exc_info=True)
        return None


@router.post("/rigid", response_model=RegistrationResponse)
async def perform_rigid_registration(request: RegistrationRequest):
    """
    Start rigid registration job (async)
    
    Uses SimpleITK with Similarity2DTransform (rigid + uniform scaling)
    Jobs are processed by Celery workers
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job metadata
    _registration_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "fixed_image_id": request.fixed_image_id,
        "moving_image_id": request.moving_image_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    # Start Celery task - use apply_async with kwargs for better compatibility
    method = request.method if hasattr(request, 'method') and request.method in ['simpleitk', 'tps', 'affine_tps'] else 'simpleitk'
    task = task_register_rigid.apply_async(
        args=(job_id, request.fixed_image_id, request.moving_image_id),
        kwargs={"method": method}
    )
    logger.info(f"Started Celery task {task.id} for registration job {job_id}")
    
    # Store Celery task ID and method for result lookup
    _registration_jobs[job_id]["celery_task_id"] = task.id
    _registration_jobs[job_id]["method"] = method
    
    return RegistrationResponse(
        job_id=job_id,
        status="pending",
        message="Registration job started"
    )


@router.get("/job/{job_id}", response_model=RegistrationResult)
async def get_registration_status(job_id: str):
    """Get registration job status"""
    if job_id not in _registration_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _registration_jobs[job_id]
    
    # Try to get result from Celery
    celery_task_id = job.get("celery_task_id")
    if celery_task_id:
        task_result = _get_task_result(celery_task_id, job_id)
        if task_result:
            # Update job with result
            task_status = task_result.get("status", "pending")
            if task_status in ["completed", "failed"]:
                job["status"] = task_status
                job["result"] = task_result
                job["completed_at"] = task_result.get("completed_at")
            elif task_status == "running":
                job["status"] = "running"
            elif task_status == "pending":
                job["status"] = "pending"
    
    # Return result if available
    if job.get("status") == "completed" and "result" in job:
        result = job["result"]
        return RegistrationResult(
            job_id=job_id,
            status=job["status"],
            fixed_image_id=job["fixed_image_id"],
            moving_image_id=job["moving_image_id"],
            transform_matrix=result.get("transform_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            rotation_degrees=result.get("rotation_degrees", 0.0),
            offset_x=result.get("offset_x", 0.0),
            offset_y=result.get("offset_y", 0.0),
            scale=result.get("scale", 1.0),
            mutual_information=result.get("mutual_information", 0.0),
            normalized_cross_correlation=result.get("normalized_cross_correlation"),
            structural_similarity=result.get("structural_similarity"),
            dice_coefficient=result.get("dice_coefficient"),
            success=result.get("success", False),
            error=result.get("error"),
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            pre_rotate=result.get("pre_rotate", "None"),
            reg_image_size=result.get("reg_image_size", 1024),
            relative_rotation=result.get("relative_rotation"),
            method=result.get("method") or job.get("method", "simpleitk"),  # Prefer method from result, fallback to job metadata
            num_matches=result.get("num_matches"),  # Include match counts for LightGlue/TPS
            num_inliers=result.get("num_inliers"),  # Include inlier counts for LightGlue/TPS
        )
    else:
        # Return current status
        return RegistrationResult(
            job_id=job_id,
            status=job.get("status", "pending"),
            fixed_image_id=job["fixed_image_id"],
            moving_image_id=job["moving_image_id"],
            transform_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            rotation_degrees=0.0,
            offset_x=0.0,
            offset_y=0.0,
            scale=1.0,
            mutual_information=0.0,
            normalized_cross_correlation=None,
            structural_similarity=None,
            dice_coefficient=None,
            success=False,
            error=job.get("error"),
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            method=job.get("method", "simpleitk"),  # Include method from job metadata
            num_matches=None,  # No matches for pending/running/failed jobs
            num_inliers=None,  # No inliers for pending/running/failed jobs
        )


@router.post("/transform")
async def calculate_transform(
    fixed_points: List[Point],
    moving_points: List[Point]
):
    """Calculate transform from point pairs"""
    # TODO: Implement transform calculation
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/stored-registrations/{case_id}", response_model=List[RegistrationResult])
async def list_stored_registrations(case_id: str, method: Optional[str] = Query(None, description="Filter by method: 'simpleitk', 'lightglue', or 'tps'. If None, returns all methods.")):
    """
    Load stored registrations for a case from DSA metadata
    
    Returns RegistrationResult objects for all items that have registration metadata stored in DSA.
    Supports multiple methods (simpleitk, tps) - can filter by method or return all.
    """
    from app.services.dsa_client import get_dsa_client
    
    try:
        dsa = get_dsa_client()
        items = dsa.list_items(case_id)
        
        stored_registrations = []
        for item in items:
            item_id = item["_id"]
            item_meta = item.get("meta", {})
            
            # Check for method-specific registrations
            methods_to_check = [method] if method else ["simpleitk", "tps", "affine_tps"]
            
            for reg_method in methods_to_check:
                reg_key = f"npReg_{reg_method}" if reg_method != "simpleitk" else "npReg"
                
                # Try method-specific first, then fall back to npReg for backward compatibility
                reg_meta = item_meta.get(reg_key) or (item_meta.get("npReg") if reg_method == "simpleitk" else None)
                
                if reg_meta:
                    # Get transform matrix
                    xfm_key = f"XFM_{reg_method}" if reg_method != "simpleitk" else "XFM"
                    transform_matrix = None
                    xfm_str = item_meta.get(xfm_key) or (item_meta.get("XFM") if reg_method == "simpleitk" else None)
                    
                    if xfm_str:
                        try:
                            if isinstance(xfm_str, str):
                                xfm_dict = json.loads(xfm_str)
                            else:
                                xfm_dict = xfm_str
                            
                            # Convert dict to matrix
                            if isinstance(xfm_dict, dict):
                                transform_matrix = [
                                    [xfm_dict.get(str(i), [0, 0, 0])[j] if isinstance(xfm_dict.get(str(i), [0, 0, 0]), list) else 0 for j in range(3)]
                                    for i in range(3)
                                ]
                        except Exception as e:
                            logger.debug(f"Could not parse transform matrix for {item_id}: {e}")
                            # Create identity matrix as fallback
                            transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    
                    if not transform_matrix:
                        # Create identity matrix if no transform found
                        transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    
                    # Create RegistrationResult
                    result = RegistrationResult(
                        job_id=f"stored_{item_id}_{reg_method}",
                        status="completed",
                        fixed_image_id=reg_meta.get("srcImage", ""),
                        moving_image_id=item_id,
                        transform_matrix=transform_matrix,
                        rotation_degrees=reg_meta.get("rotation", 0.0),
                        offset_x=reg_meta.get("xOffset", 0.0),
                        offset_y=reg_meta.get("yOffset", 0.0),
                        scale=reg_meta.get("scale", 1.0),
                        mutual_information=reg_meta.get("mutual_information", 0.0),
                        normalized_cross_correlation=reg_meta.get("normalized_cross_correlation"),
                        structural_similarity=reg_meta.get("structural_similarity"),
                        dice_coefficient=reg_meta.get("dice_coefficient"),
                        success=True,
                        created_at=item.get("created", ""),
                        completed_at=item.get("updated", ""),
                        method=reg_method,
                        num_matches=reg_meta.get("num_matches"),  # Include match counts for LightGlue/TPS
                        num_inliers=reg_meta.get("num_inliers"),  # Include inlier counts for LightGlue/TPS
                    )
                    stored_registrations.append(result)
        
        logger.info(f"Loaded {len(stored_registrations)} stored registrations for case {case_id}")
        return stored_registrations
        
    except Exception as e:
        logger.error(f"Error loading stored registrations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load stored registrations: {str(e)}"
        )


@router.post("/auto-register/{case_id}", response_model=List[RegistrationResponse])
async def auto_register_case(
    case_id: str,
    block_id: Optional[str] = Query(None, description="Block ID to filter slides (uses HE image's block ID if not provided)"),
    method: Optional[str] = Query("simpleitk", description="Registration method: 'simpleitk' (rigid/affine), 'tps' (non-rigid), or 'affine_tps' (affine+TPS hybrid)")
):
    """
    Automatically register all non-HE stains to the HE image for a case
    
    Finds the HE (fixed) image and registers all other stains to it.
    If block_id is provided, only registers slides from that block.
    If not provided, uses the block_id from the HE (fixed) image.
    """
    from app.services.dsa_client import get_dsa_client
    
    try:
        logger.info(f"Auto-registering case: {case_id}, block_id: {block_id}")
        dsa = get_dsa_client()
        
        # Get all slides for the case
        logger.info(f"Listing items in folder: {case_id}")
        items = dsa.list_items(case_id)
        logger.info(f"Found {len(items)} items in case folder")
        
        # Find HE (fixed) image
        he_item = None
        he_block_id = None
        
        for item in items:
            stain_id = item.get("meta", {}).get("npSchema", {}).get("stainID", "").upper()
            item_name = item.get("name", "unknown")
            logger.debug(f"Item {item_name}: stainID={stain_id}")
            if stain_id == "HE":
                he_item = item
                he_block_id = item.get("meta", {}).get("npSchema", {}).get("blockID")
                logger.info(f"Found HE (fixed) image: {item_name} ({item['_id']}), blockID: {he_block_id}")
                break
        
        if not he_item:
            error_msg = f"No HE (fixed) image found in case {case_id}. Found {len(items)} items total."
            logger.error(error_msg)
            raise HTTPException(
                status_code=404,
                detail=error_msg
            )
        
        # Use provided block_id or the HE image's block_id
        target_block_id = block_id or he_block_id
        logger.info(f"Using block_id: {target_block_id}")
        
        # Filter moving items: must be non-HE and same block ID (if block_id is specified)
        moving_items = []
        for item in items:
            stain_id = item.get("meta", {}).get("npSchema", {}).get("stainID", "").upper()
            item_block_id = item.get("meta", {}).get("npSchema", {}).get("blockID")
            item_name = item.get("name", "unknown")
            
            # Skip HE images
            if stain_id == "HE":
                continue
            
            # If we have a target block_id, only include items from that block
            if target_block_id:
                if item_block_id != target_block_id:
                    logger.debug(f"Skipping {item_name}: blockID mismatch ({item_block_id} != {target_block_id})")
                    continue
            
            moving_items.append(item)
            logger.debug(f"Added moving item: {item_name} (stainID={stain_id}, blockID={item_block_id})")
        
        if not moving_items:
            if target_block_id:
                error_msg = f"No moving images found to register in case {case_id} for block {target_block_id}."
            else:
                error_msg = f"No moving images found to register in case {case_id}. Only HE image found."
            logger.warning(error_msg)
            raise HTTPException(
                status_code=404,
                detail=error_msg
            )
        
        logger.info(f"Starting registration for {len(moving_items)} moving images (block_id: {target_block_id})")
        
        # Start registration jobs for all moving images
        job_responses = []
        for moving_item in moving_items:
            job_id = str(uuid.uuid4())
            _registration_jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "fixed_image_id": he_item["_id"],
                "moving_image_id": moving_item["_id"],
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Validate method
            if method not in ["simpleitk", "tps", "affine_tps"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid method: {method}. Must be 'simpleitk', 'tps', or 'affine_tps'"
                )
            
            # Start Celery task - use apply_async with kwargs for better compatibility
            task = task_register_rigid.apply_async(
                args=(job_id, he_item["_id"], moving_item["_id"]),
                kwargs={"method": method}
            )
            _registration_jobs[job_id]["celery_task_id"] = task.id
            _registration_jobs[job_id]["method"] = method
            logger.debug(f"Started Celery task {task.id} for job {job_id}")
            
            job_responses.append(RegistrationResponse(
                job_id=job_id,
                status="pending",
                message=f"Registration started for {moving_item.get('name', 'unknown')}"
            ))
        
        logger.info(f"Started {len(job_responses)} registration jobs")
        return job_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in auto-register endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start auto-registration: {str(e)}"
        )


@router.post("/clear-cache")
async def clear_registration_cache():
    """
    Clear the registration cache (thumbnails and registration results)
    
    Uses joblib Memory.clear() to invalidate cached results.
    Useful for debugging and when images are updated.
    """
    from app.core.config import settings
    from app.services.registration_service import memory
    
    cache_dir = settings.CACHE_DIR
    
    try:
        # Clear joblib Memory cache - this invalidates all cached function results
        # Joblib handles the actual file cleanup internally
        memory.clear(warn=False)
        
        logger.info(f"Cache cleared: {cache_dir}")
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "cache_dir": cache_dir
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

