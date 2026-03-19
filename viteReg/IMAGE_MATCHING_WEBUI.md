# Image Matching WebUI Integration

This document describes how to use the [image-matching-webui](https://github.com/Vincentqyw/image-matching-webui) (imcui) for registration exploration.

## Integration Modes

This integration supports **two modes**:

1. **Direct Library Integration** (Recommended) - Uses `imcui` as a Python package directly
   - Faster execution (no HTTP overhead)
   - Access to all extractors and matchers
   - Full parameter control
   - Better error handling

2. **HTTP API Integration** - Uses the webui service via HTTP
   - Useful if running imcui as a separate service
   - Good for distributed setups

## Overview

The image-matching-webui provides an interactive Gradio-based interface for exploring different feature matching algorithms. It's useful for:

- **Algorithm Comparison**: Compare SuperPoint, SuperGlue, LightGlue, LoFTR, TopicFM, ASpanFormer, and many more
- **Parameter Tuning**: Interactive UI for adjusting detection thresholds, matching parameters, etc.
- **Visual Feedback**: See keypoints and matches overlaid on images
- **Exploration**: Quickly test which algorithms work best for your tissue registration use case

## Setup

### Option 1: Docker Compose (Recommended)

The service is already configured in `docker-compose.yml`. Just start it:

```bash
cd viteReg
docker-compose up -d image-matching-webui
```

The web UI will be available at:
- **From host machine**: http://localhost:7860
- **From Docker network**: http://image-matching-webui:7860

### Option 2: Standalone Installation

If you prefer to run it separately:

```bash
# Install via pip
pip install imcui

# Run the service
imcui --server-port 7860
```

Or using Docker directly:

```bash
docker pull vincentqin/image-matching-webui:latest
docker run -p 7860:7860 vincentqin/image-matching-webui:latest
```

## Installation

### Direct Library Integration (Recommended)

```bash
pip install imcui
```

This installs imcui as a Python package, allowing direct use of all extractors and matchers.

### HTTP API Integration (Optional)

If you prefer to run imcui as a separate service, use Docker Compose (see Setup section below).

## Usage

### Option 1: Direct Library Integration (Recommended)

Use the direct library integration for best performance:

```bash
# Get available extractors and matchers
curl "http://localhost:8000/api/image-matching/extractors"
curl "http://localhost:8000/api/image-matching/matchers"

# Get configuration for an extractor
curl "http://localhost:8000/api/image-matching/extractor/superpoint/config"

# Get configuration for a matcher
curl "http://localhost:8000/api/image-matching/matcher/lightglue/config?extractor_name=superpoint"

# Direct matching (uses imcui library directly)
curl -X POST "http://localhost:8000/api/image-matching/match-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_id": "IMAGE_ID_1",
    "moving_id": "IMAGE_ID_2",
    "extractor": "disk",
    "matcher": "lightglue",
    "thumbnail_width": 1024,
    "extractor_params": {"max_num_keypoints": 2048},
    "matcher_params": {"n_layers": 9},
    "return_transform": true
  }'
```

### Option 2: HTTP API Integration

Use the API endpoint to match images programmatically:

```bash
# Simple matching
curl -X POST "http://localhost:8000/api/image-matching/match-simple?fixed_id=IMAGE_ID_1&moving_id=IMAGE_ID_2&extractor=superpoint&matcher=lightglue"

# Advanced matching with parameters
curl -X POST "http://localhost:8000/api/image-matching/match" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_id": "IMAGE_ID_1",
    "moving_id": "IMAGE_ID_2",
    "extractor": "disk",
    "matcher": "lightglue",
    "thumbnail_width": 1024,
    "extractor_params": {"max_num_keypoints": 2048},
    "matcher_params": {"n_layers": 9},
    "return_transform": true
  }'
```

This will:
- Export images from DSA automatically
- Call image-matching-webui API
- Return match results and optionally registration transform

### Option 3: Manual Web UI Exploration

#### 1. Export Images from DSA

Use the API endpoint to export images for matching:

```bash
curl "http://localhost:8000/api/image-matching/export-for-matching?fixed_id=IMAGE_ID_1&moving_id=IMAGE_ID_2"
```

This will:
- Download thumbnails from DSA
- Save them to `./image-matching-data/` directory
- Return paths and session info

#### 2. Access the Web UI

Open http://localhost:7860 in your browser.

### 3. Upload Images

In the web UI:
1. Upload the fixed image (from the exported paths)
2. Upload the moving image
3. Select a feature extractor (SuperPoint, DISK, ALIKED, SIFT, etc.)
4. Select a matcher (LightGlue, SuperGlue, LoFTR, etc.)
5. Adjust parameters as needed
6. Click "Match" to see results

### 4. Compare Algorithms

Try different combinations:
- **SuperPoint + LightGlue**: Fast, good for general use
- **DISK + LightGlue**: Good for texture-rich images
- **LoFTR**: Dense matching, no keypoint detection needed
- **TopicFM**: Good for challenging cases
- **ASpanFormer**: Transformer-based, robust

### 5. Export Results

The web UI allows you to:
- Download matched keypoints
- Export visualization images
- Copy transform matrices (if available)

## Integration with Your Registration Pipeline

### Direct Library Integration

The service supports using imcui as a Python library directly:

```python
from app.services.imcui_service import (
    register_with_imcui,
    get_available_extractors,
    get_extractor_config,
    get_matcher_config
)

# Get available algorithms
extractors = get_available_extractors()  # ['superpoint', 'disk', 'aliked', 'sift', ...]
matchers = get_available_matchers()      # ['lightglue', 'superglue', 'loftr', ...]

# Get default configurations
extractor_config = get_extractor_config('superpoint')
matcher_config = get_matcher_config('lightglue', 'superpoint')

# Perform registration
result = register_with_imcui(
    fixed_id="IMAGE_ID_1",
    moving_id="IMAGE_ID_2",
    extractor="disk",
    matcher="lightglue",
    extractor_config={"max_num_keypoints": 2048},
    matcher_config={"n_layers": 9},
    return_transform=True
)
```

### HTTP API Integration

The service also supports calling image-matching-webui via HTTP API:

```python
from app.services.image_matching_webui_helper import match_images_from_dsa

# Match images directly from DSA IDs
result = match_images_from_dsa(
    fixed_id="IMAGE_ID_1",
    moving_id="IMAGE_ID_2",
    extractor="superpoint",
    matcher="lightglue",
    thumbnail_width=1024
)

# Convert to registration transform
transform = convert_matches_to_registration_transform(
    matches=result,
    fixed_image_shape=(1024, 1024),
    moving_image_shape=(1024, 1024)
)
```

### Using Results in Your App

1. **Programmatic Matching**: Use the `/api/image-matching/match` endpoint for automated matching
2. **Manual Exploration**: Use the web UI for interactive parameter tuning
3. **Hybrid Approach**: Use web UI to find best parameters, then use API for batch processing

### API Endpoints

**Direct Library Integration:**
- `POST /api/image-matching/match-direct` - Direct matching using imcui library (recommended)
- `GET /api/image-matching/extractors` - List available extractors
- `GET /api/image-matching/matchers` - List available matchers
- `GET /api/image-matching/extractor/{name}/config` - Get extractor configuration
- `GET /api/image-matching/matcher/{name}/config` - Get matcher configuration

**HTTP API Integration:**
- `POST /api/image-matching/match` - Matching via HTTP API (falls back to direct if available)
- `POST /api/image-matching/match-simple` - Simplified matching with query params
- `POST /api/image-matching/export-for-matching` - Export images for manual use
- `GET /api/image-matching/webui-url` - Get web UI URL

**Note:** The `/match` endpoint automatically uses direct integration if imcui is installed, otherwise falls back to HTTP API.

## Available Algorithms

### Feature Extractors
- SuperPoint
- DISK
- ALIKED
- SIFT
- D2Net
- R2D2
- And many more...

### Matchers
- LightGlue
- SuperGlue
- LoFTR
- TopicFM
- ASpanFormer
- SGMNet
- And many more...

## Tips for Tissue Registration

1. **Start with LightGlue**: You already have this integrated, so compare web UI results with your implementation
2. **Try LoFTR**: Dense matching can be better for tissue images with repetitive patterns
3. **Adjust detection thresholds**: Lower thresholds = more keypoints (may help with sparse tissue)
4. **Use RANSAC**: The web UI shows inliers/outliers - use this to judge match quality
5. **Compare with SimpleITK**: Use web UI for feature-based exploration, SimpleITK for intensity-based refinement

## Troubleshooting

### Images not appearing in web UI

- Check that `./image-matching-data/` directory exists and is writable
- Verify Docker volume mount in `docker-compose.yml`
- Check file permissions

### Web UI not accessible

- Verify container is running: `docker-compose ps image-matching-webui`
- Check logs: `docker-compose logs image-matching-webui`
- Verify port 7860 is not in use

### Performance

- The web UI can be GPU-intensive
- Consider running on a machine with CUDA support
- For CPU-only, algorithms will be slower but still functional

## References

- [image-matching-webui GitHub](https://github.com/Vincentqyw/image-matching-webui)
- [HuggingFace Space](https://huggingface.co/spaces/Realcat/image-matching-webui)
- [LightGlue Documentation](https://github.com/cvg/LightGlue)
