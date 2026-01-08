"""
DSA (Digital Slide Archive) client service
Wraps girder_client for use in FastAPI
"""
import os
import logging
from typing import Optional, Dict, Any, List
import girder_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class DSAClient:
    """DSA client wrapper for Girder API"""
    
    def __init__(self):
        """Initialize DSA client with authentication"""
        self.base_url = settings.DSA_BASE_URL
        self.api_key = settings.DSAKEY
        self.client: Optional[girder_client.GirderClient] = None
        self.token: Optional[str] = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with DSA using API key"""
        try:
            self.client = girder_client.GirderClient(apiUrl=self.base_url)
            
            if self.api_key:
                response = self.client.authenticate(apiKey=self.api_key)
                # Get the actual token ID for use in URLs (same as token_info['_id'] in Python version)
                try:
                    token_info = self.client.get("token/current")
                    self.token = token_info.get("_id")
                    if not self.token:
                        logger.warning("Token info missing _id, falling back to API key")
                        self.token = self.api_key
                    else:
                        logger.info(f"Authenticated with DSA, token ID: {self.token[:10]}...")
                except Exception as e:
                    logger.warning(f"Could not get token info: {e}, falling back to API key")
                    # Fallback to API key if token endpoint fails
                    self.token = self.api_key
            else:
                logger.warning("No DSAKEY provided, DSA client will have limited functionality")
                self.token = None
        except Exception as e:
            logger.error(f"Failed to authenticate with DSA: {e}")
            raise
    
    def list_items(self, folder_id: str) -> List[Dict[str, Any]]:
        """List items in a folder"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            items = list(self.client.listItem(folder_id))
            return items
        except Exception as e:
            logger.error(f"Error listing items from folder {folder_id}: {e}")
            raise
    
    def get_item(self, item_id: str) -> Dict[str, Any]:
        """Get item metadata"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            return self.client.getItem(item_id)
        except Exception as e:
            logger.error(f"Error getting item {item_id}: {e}")
            raise
    
    def get_tiles(self, item_id: str) -> Dict[str, Any]:
        """Get tile metadata for an item"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            return self.client.get(f"item/{item_id}/tiles")
        except Exception as e:
            logger.error(f"Error getting tiles for item {item_id}: {e}")
            raise
    
    def get_thumbnail_url(self, item_id: str, width: int = 1024) -> str:
        """Get thumbnail URL for an item"""
        base_url = self.base_url.rstrip('/api/v1')
        token_param = f"&token={self.token}" if self.token else ""
        return f"{base_url}/api/v1/item/{item_id}/tiles/thumbnail?width={width}{token_param}"
    
    def get_dzi_url(self, item_id: str) -> str:
        """Get DZI tile source URL"""
        base_url = self.base_url.rstrip('/api/v1')
        token_param = f"?token={self.token}" if self.token else ""
        return f"{base_url}/api/v1/item/{item_id}/tiles/dzi.dzi{token_param}"
    
    def get_token(self) -> Optional[str]:
        """Get the current authentication token"""
        return self.token
    
    def get_annotation_counts(self, item_ids: List[str]) -> Dict[str, int]:
        """Get annotation counts for multiple items"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            if not item_ids:
                return {}
            
            items_str = ','.join(item_ids)
            counts = self.client.get(f"annotation/counts?items={items_str}")
            return counts
        except Exception as e:
            logger.error(f"Error getting annotation counts: {e}")
            return {item_id: 0 for item_id in item_ids}
    
    def get_annotations(self, item_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get annotations for an item"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            annotations = self.client.get(
                f"annotation?itemId={item_id}&limit={limit}&offset={offset}"
            )
            return annotations
        except Exception as e:
            logger.error(f"Error getting annotations for item {item_id}: {e}")
            return []
    
    def list_folders(
        self, 
        parent_id: str, 
        parent_type: str = 'folder'
    ) -> List[Dict[str, Any]]:
        """
        List folders in a parent folder or collection
        
        Args:
            parent_id: ID of the parent folder or collection
            parent_type: Type of parent - 'folder', 'collection', or 'user' (default: 'folder')
        """
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            folders = list(self.client.listFolder(parent_id, parentFolderType=parent_type))
            return folders
        except Exception as e:
            logger.error(f"Error listing folders from {parent_id} (type: {parent_type}): {e}")
            raise
    
    def get_folder(self, folder_id: str) -> Dict[str, Any]:
        """Get folder metadata"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            return self.client.getFolder(folder_id)
        except Exception as e:
            logger.error(f"Error getting folder {folder_id}: {e}")
            raise
    
    def folder_exists(self, folder_id: str) -> bool:
        """Check if a folder exists"""
        try:
            self.get_folder(folder_id)
            return True
        except Exception:
            return False
    
    def update_item_metadata(self, item_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update item metadata in DSA"""
        if not self.client:
            raise RuntimeError("DSA client not authenticated")
        
        try:
            return self.client.addMetadataToItem(item_id, metadata)
        except Exception as e:
            logger.error(f"Error updating metadata for item {item_id}: {e}")
            raise
    
    def get_registration_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get registration metadata from item if it exists"""
        try:
            item = self.get_item(item_id)
            meta = item.get("meta", {})
            np_reg = meta.get("npReg")
            if np_reg:
                # Also get transform matrix if available
                xfm = meta.get("XFM")
                if xfm and isinstance(xfm, str):
                    import json
                    try:
                        xfm_dict = json.loads(xfm)
                        # Convert back to list of lists
                        if isinstance(xfm_dict, dict):
                            xfm_matrix = [[xfm_dict.get(str(i), [0, 0, 0])[j] if isinstance(xfm_dict.get(str(i), [0, 0, 0]), list) else 0 for j in range(3)] for i in range(3)]
                            np_reg["transform_matrix"] = xfm_matrix
                    except Exception:
                        pass
                return np_reg
            return None
        except Exception as e:
            logger.debug(f"No registration metadata found for item {item_id}: {e}")
            return None


# Global DSA client instance
_dsa_client: Optional[DSAClient] = None


def get_dsa_client() -> DSAClient:
    """Get or create DSA client instance"""
    global _dsa_client
    if _dsa_client is None:
        _dsa_client = DSAClient()
    return _dsa_client

