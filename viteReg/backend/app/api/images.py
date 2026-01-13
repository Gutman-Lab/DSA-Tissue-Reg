"""Image viewing endpoints"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from app.services.dsa_client import get_dsa_client

router = APIRouter()


class TileMetadata(BaseModel):
    """Tile metadata model"""
    sizeX: int
    sizeY: int
    tileWidth: int
    tileHeight: int
    levels: int
    magnification: Optional[float] = None


@router.get("/{image_id}/tiles")
async def get_tile_metadata(image_id: str):
    """Get tile metadata for an image"""
    try:
        dsa = get_dsa_client()
        tiles_info = dsa.get_tiles(image_id)
        return tiles_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tile metadata: {str(e)}")


@router.get("/{image_id}/dzi")
async def get_dzi_url(image_id: str):
    """Get DZI tile source URL and token for authentication"""
    try:
        dsa = get_dsa_client()
        dzi_url = dsa.get_dzi_url(image_id)
        token = dsa.get_token()
        # Return the URL and token separately so frontend can use apiHeaders
        return {
            "dzi_url": dzi_url,
            "token": token
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating DZI URL: {str(e)}")


@router.get("/{image_id}/thumbnail")
async def get_thumbnail(image_id: str, width: int = 1024):
    """Get thumbnail URL (redirects to DSA thumbnail endpoint)"""
    try:
        dsa = get_dsa_client()
        thumbnail_url = dsa.get_thumbnail_url(image_id, width)
        # Return redirect to DSA thumbnail
        return RedirectResponse(url=thumbnail_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting thumbnail: {str(e)}")

