"""
API endpoints for integrating with image-matching-webui service

Supports both:
1. Direct library integration (imcui as Python package) - recommended
2. HTTP API integration (webui service)
"""
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.services.image_matching_webui_helper import (
    export_images_for_matching,
    create_matching_session_info,
    get_image_matching_webui_url,
    call_image_matching_api,
    match_images_from_dsa,
    convert_matches_to_registration_transform
)

# Try to import direct imcui integration
try:
    from app.services.imcui_service import (
        get_available_extractors,
        get_available_matchers,
        get_extractor_config,
        get_matcher_config,
        register_with_imcui,
        match_images_with_imcui
    )
    IMCUI_DIRECT_AVAILABLE = True
except ImportError:
    IMCUI_DIRECT_AVAILABLE = False

router = APIRouter()


class MatchingRequest(BaseModel):
    """Request model for image matching"""
    fixed_id: str
    moving_id: str
    extractor: str = "superpoint"
    matcher: str = "lightglue"
    thumbnail_width: int = 1024
    extractor_params: Optional[Dict[str, Any]] = None
    matcher_params: Optional[Dict[str, Any]] = None
    return_transform: bool = True  # Whether to convert matches to registration transform


@router.post("/export-for-matching")
async def export_for_matching(
    fixed_id: str = Query(..., description="DSA item ID of fixed (reference) image"),
    moving_id: str = Query(..., description="DSA item ID of moving image"),
    thumbnail_width: int = Query(1024, description="Thumbnail width in pixels")
):
    """
    Export images from DSA to files that can be used by image-matching-webui
    
    This endpoint:
    1. Downloads thumbnails from DSA
    2. Saves them to a shared directory accessible by image-matching-webui
    3. Returns paths and session info
    """
    try:
        exported_paths = export_images_for_matching(
            fixed_id=fixed_id,
            moving_id=moving_id,
            thumbnail_width=thumbnail_width
        )
        
        session_info = create_matching_session_info(
            fixed_id=fixed_id,
            moving_id=moving_id,
            exported_paths=exported_paths
        )
        
        return {
            "success": True,
            "session_info": session_info,
            "webui_url": get_image_matching_webui_url(host="image-matching-webui"),  # Docker service name
            "instructions": {
                "step1": "Open the image-matching-webui at the URL above",
                "step2": f"Upload fixed image: {exported_paths['fixed_image']}",
                "step3": f"Upload moving image: {exported_paths['moving_image']}",
                "step4": "Experiment with different feature extractors and matchers",
                "step5": "Export results or use the API to retrieve them"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export images: {str(e)}")


@router.get("/webui-url")
async def get_webui_url():
    """
    Get the URL for the image-matching-webui service
    """
    return {
        "url": get_image_matching_webui_url(host="image-matching-webui"),
        "local_url": get_image_matching_webui_url(host="localhost"),
        "description": "Use 'local_url' if accessing from host machine, or 'url' from within Docker network"
    }


@router.post("/match")
async def match_images(request: MatchingRequest):
    """
    Perform image matching using image-matching-webui
    
    This endpoint tries direct library integration first, then falls back to HTTP API.
    For best performance, use /match-direct endpoint with imcui installed.
    """
    # Try direct integration first
    if IMCUI_DIRECT_AVAILABLE:
        try:
            return await match_images_direct(request)
        except Exception as e:
            logger.warning(f"Direct integration failed, falling back to HTTP API: {e}")
            # Fall through to HTTP API
    
    # Fallback to HTTP API
    try:
        # Match images using the webui API
        match_results = match_images_from_dsa(
            fixed_id=request.fixed_id,
            moving_id=request.moving_id,
            extractor=request.extractor,
            matcher=request.matcher,
            thumbnail_width=request.thumbnail_width,
            extractor_params=request.extractor_params,
            matcher_params=request.matcher_params
        )
        
        result = {
            "success": True,
            "match_results": match_results,
            "source": "http_api"
        }
        
        # Optionally convert to registration transform
        if request.return_transform:
            # Get image shapes from exported paths
            import os
            from PIL import Image
            
            fixed_path = match_results['exported_paths']['fixed_image']
            moving_path = match_results['exported_paths']['moving_image']
            
            fixed_img = Image.open(fixed_path)
            moving_img = Image.open(moving_path)
            
            transform = convert_matches_to_registration_transform(
                matches=match_results,
                fixed_image_shape=(fixed_img.height, fixed_img.width),
                moving_image_shape=(moving_img.height, moving_img.width)
            )
            
            result["registration_transform"] = transform
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image matching failed: {str(e)}")


@router.post("/match-simple")
async def match_images_simple(
    fixed_id: str = Query(..., description="DSA item ID of fixed image"),
    moving_id: str = Query(..., description="DSA item ID of moving image"),
    extractor: str = Query("superpoint", description="Feature extractor"),
    matcher: str = Query("lightglue", description="Matcher algorithm"),
    thumbnail_width: int = Query(1024, description="Thumbnail width")
):
    """
    Simplified endpoint for image matching with query parameters
    
    Uses direct imcui library integration if available, otherwise falls back to HTTP API
    """
    request = MatchingRequest(
        fixed_id=fixed_id,
        moving_id=moving_id,
        extractor=extractor,
        matcher=matcher,
        thumbnail_width=thumbnail_width
    )
    return await match_images(request)


@router.get("/extractors")
async def list_extractors():
    """
    Get list of available feature extractors from imcui
    """
    if IMCUI_DIRECT_AVAILABLE:
        extractors = get_available_extractors()
        return {
            "extractors": extractors,
            "source": "imcui_direct"
        }
    else:
        # Fallback list
        return {
            "extractors": ["superpoint", "disk", "aliked", "sift"],
            "source": "fallback",
            "note": "Install imcui for full list: pip install imcui"
        }


@router.get("/matchers")
async def list_matchers():
    """
    Get list of available matchers from imcui
    """
    if IMCUI_DIRECT_AVAILABLE:
        matchers = get_available_matchers()
        return {
            "matchers": matchers,
            "source": "imcui_direct"
        }
    else:
        # Fallback list
        return {
            "matchers": ["lightglue", "superglue", "loftr"],
            "source": "fallback",
            "note": "Install imcui for full list: pip install imcui"
        }


@router.get("/extractor/{extractor_name}/config")
async def get_extractor_config_endpoint(extractor_name: str):
    """
    Get configuration parameters for a specific extractor
    """
    if IMCUI_DIRECT_AVAILABLE:
        config = get_extractor_config(extractor_name)
        return {
            "extractor": extractor_name,
            "config": config
        }
    else:
        raise HTTPException(
            status_code=501,
            detail="imcui not available. Install with: pip install imcui"
        )


@router.get("/matcher/{matcher_name}/config")
async def get_matcher_config_endpoint(
    matcher_name: str,
    extractor_name: str = Query(..., description="Extractor name (some matchers need this)")
):
    """
    Get configuration parameters for a specific matcher
    """
    if IMCUI_DIRECT_AVAILABLE:
        config = get_matcher_config(matcher_name, extractor_name)
        return {
            "matcher": matcher_name,
            "extractor": extractor_name,
            "config": config
        }
    else:
        raise HTTPException(
            status_code=501,
            detail="imcui not available. Install with: pip install imcui"
        )


@router.post("/match-direct")
async def match_images_direct(request: MatchingRequest):
    """
    Direct matching using imcui library (bypasses web UI)
    
    This is the recommended endpoint for programmatic use.
    It uses imcui as a Python library directly, providing:
    - Faster execution (no HTTP overhead)
    - Access to all extractors and matchers
    - Full parameter control
    - Better error handling
    """
    if not IMCUI_DIRECT_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Direct imcui integration not available. Install with: pip install imcui"
        )
    
    try:
        result = register_with_imcui(
            fixed_id=request.fixed_id,
            moving_id=request.moving_id,
            extractor=request.extractor,
            matcher=request.matcher,
            thumbnail_width=request.thumbnail_width,
            extractor_config=request.extractor_params,
            matcher_config=request.matcher_params,
            return_transform=request.return_transform
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Direct matching failed: {str(e)}")
