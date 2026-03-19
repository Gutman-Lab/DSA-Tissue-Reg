"""
Minimal web service for manual registration debugging.
Allows pasting transformation matrices and applying them to images.
"""
import sys
import os
from pathlib import Path

# Set local cache directory before importing anything that uses settings
# This prevents trying to write to /app (Docker path) when running locally
cache_dir = Path(__file__).parent / ".cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["CACHE_DIR"] = str(cache_dir)

# Load .env file if it exists (for DSAKEY and DSA_BASE_URL)
# Load it BEFORE importing anything that uses settings
# Filter out frontend-specific variables (VITE_*) to avoid pydantic validation errors
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    try:
        from dotenv import dotenv_values
        env_vars = dotenv_values(env_file)
        # Filter out VITE_* variables (frontend-only)
        filtered_vars = {k: v for k, v in env_vars.items() if not k.startswith('VITE_')}
        # Set only the filtered variables
        for k, v in filtered_vars.items():
            if v is not None:
                os.environ[k] = v
        print(f"Loaded .env file from {env_file} (filtered {len(env_vars) - len(filtered_vars)} frontend variables)")
    except ImportError:
        print(f"Warning: python-dotenv not installed. Install with: uv pip install python-dotenv")
        print(f"Will try to use environment variables or pydantic-settings .env support.")
else:
    print(f"Info: .env file not found at {env_file}. Using environment variables or defaults.")

# Add parent directory to path to import from app
backend_dir = Path(__file__).parent.parent / "viteReg" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import json

from utils import apply_homography_transform, load_transform_from_json, image_to_base64

app = FastAPI(title="Manual Registration Debug Tool")

# Mount static files (for HTML/JS/CSS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class TransformRequest(BaseModel):
    fixed_image_id: str
    moving_image_id: str
    matrix_type: str = "Homography"  # Fundamental, Homography, H1, H2
    geom_info: Dict[str, Any]  # The geom_info object from JSON


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return html_file.read_text()
    else:
        return """
        <html>
            <body>
                <h1>Manual Registration Debug Tool</h1>
                <p>Please create static/index.html</p>
            </body>
        </html>
        """


@app.post("/api/apply-transform")
async def apply_transform(request: TransformRequest):
    """
    Apply a transformation matrix to images.
    
    Request body:
    {
        "fixed_image_id": "6903df8dd26a6d93de19a9b2",
        "moving_image_id": "6903df8dd26a6d93de19a9b3",
        "matrix_type": "Homography",
        "geom_info": {
            "Homography": [[...], [...], [...]],
            ...
        }
    }
    """
    try:
        # Load transform matrix
        json_data = {"geom_info": request.geom_info}
        transform_matrix = load_transform_from_json(json_data, request.matrix_type)
        
        # Apply transform
        result = apply_homography_transform(
            request.fixed_image_id,
            request.moving_image_id,
            transform_matrix,
            thumbnail_width=512
        )
        
        # Convert images to base64
        return {
            "success": True,
            "fixed_image": image_to_base64(result['fixed_image']),
            "moving_image": image_to_base64(result['moving_image']),
            "transformed_image": image_to_base64(result['transformed_image']),
            "overlay": image_to_base64(result['overlay']),
            "transform_params": result['transform_params']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))  # Default to 8002 to avoid conflict with main backend
    uvicorn.run(app, host="0.0.0.0", port=port)
