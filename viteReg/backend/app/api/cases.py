"""Case and slide management endpoints"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
from app.services.dsa_client import get_dsa_client
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Collection ID for fetching cases dynamically
# This is the collection containing all case folders
# Default: http://bdsa.pathology.emory.edu:8080/#collection/695d6a148c871f3a02969b00


class CaseInfo(BaseModel):
    """Case information model"""
    label: str
    value: str


class SlideInfo(BaseModel):
    """Slide information model"""
    id: str
    name: str
    case_id: str
    block_id: Optional[str] = None
    stain_type: Optional[str] = None
    annotation_count: int = 0
    # Include other fields from DSA item
    meta: Optional[Dict[str, Any]] = None


@router.get("/", response_model=List[CaseInfo])
async def list_cases(
    collection_id: Optional[str] = None
):
    """
    List available cases from the configured collection
    
    Args:
        collection_id: Optional collection ID to override default. 
                       Defaults to CASE_COLLECTION_ID from settings.
    """
    dsa = get_dsa_client()
    
    # Use provided collection_id or default from settings
    target_collection_id = collection_id or settings.CASE_COLLECTION_ID
    
    try:
        # List folders from the collection (parentType='collection')
        folders = dsa.list_folders(target_collection_id, parent_type='collection')
        
        # Format folders as cases
        cases = [
            CaseInfo(label=folder["name"], value=folder["_id"])
            for folder in folders
        ]
        
        logger.info(f"Found {len(cases)} cases in collection {target_collection_id}")
        return cases
    
    except Exception as e:
        logger.error(f"Error fetching cases from collection {target_collection_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cases from collection: {str(e)}"
        )


@router.get("/{case_id}/slides", response_model=List[SlideInfo])
async def get_case_slides(
    case_id: str,
    block_id: Optional[str] = None,
    only_annotated: bool = False
):
    """Get slides for a case"""
    try:
        dsa = get_dsa_client()
        
        # Verify folder exists first
        if not dsa.folder_exists(case_id):
            raise HTTPException(
                status_code=404,
                detail=f"Case folder {case_id} not found in DSA. It may have been deleted."
            )
        
        # List items in the case folder
        items = dsa.list_items(case_id)
        
        if not items:
            return []
        
        # Get annotation counts for all items
        item_ids = [item["_id"] for item in items]
        annotation_counts = dsa.get_annotation_counts(item_ids)
        
        # Build slide info list
        slides = []
        for item in items:
            # Extract metadata
            meta = item.get("meta", {})
            np_schema = meta.get("npSchema", {})
            
            slide_info = SlideInfo(
                id=item["_id"],
                name=item.get("name", ""),
                case_id=case_id,
                block_id=np_schema.get("blockID"),
                stain_type=np_schema.get("stainID"),
                annotation_count=annotation_counts.get(item["_id"], 0),
                meta=meta
            )
            
            # Filter by block ID if specified
            if block_id and block_id != "all":
                if slide_info.block_id != block_id:
                    continue
            
            # Filter by annotation status if specified
            if only_annotated and slide_info.annotation_count == 0:
                continue
            
            slides.append(slide_info)
        
        return slides
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching slides: {str(e)}")


@router.get("/slides/{slide_id}", response_model=SlideInfo)
async def get_slide_info(slide_id: str):
    """Get slide metadata"""
    try:
        dsa = get_dsa_client()
        item = dsa.get_item(slide_id)
        
        # Get annotation count
        annotation_counts = dsa.get_annotation_counts([slide_id])
        
        meta = item.get("meta", {})
        np_schema = meta.get("npSchema", {})
        
        # Try to determine case_id from parent folder
        case_id = item.get("folderId", "")
        
        return SlideInfo(
            id=item["_id"],
            name=item.get("name", ""),
            case_id=case_id,
            block_id=np_schema.get("blockID"),
            stain_type=np_schema.get("stainID"),
            annotation_count=annotation_counts.get(slide_id, 0),
            meta=meta
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching slide info: {str(e)}")

