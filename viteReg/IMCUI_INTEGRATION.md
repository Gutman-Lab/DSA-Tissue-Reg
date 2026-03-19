# Deep Integration with imcui (image-matching-webui)

This document describes the **direct library integration** with imcui, bypassing the web UI entirely.

## Overview

Instead of using the web UI or HTTP API, we import imcui as a Python library and use its functions directly. This provides:

- ✅ **Faster execution** - No HTTP overhead
- ✅ **Full algorithm access** - All extractors and matchers available
- ✅ **Parameter control** - Access to all configuration options
- ✅ **Better error handling** - Direct Python exceptions
- ✅ **Type safety** - Full IDE support

## Installation

```bash
pip install imcui
```

## Available Algorithms

### Feature Extractors

Get the full list programmatically:

```python
from app.services.imcui_service import get_available_extractors

extractors = get_available_extractors()
# Returns: ['superpoint', 'disk', 'aliked', 'sift', 'd2net', 'r2d2', ...]
```

Common extractors:
- **SuperPoint** - Fast, good general-purpose extractor
- **DISK** - Good for texture-rich images
- **ALIKED** - Accurate keypoint detection
- **SIFT** - Classic, reliable
- **D2Net** - Dense feature extraction
- **R2D2** - Repeatable and reliable

### Matchers

```python
from app.services.imcui_service import get_available_matchers

matchers = get_available_matchers()
# Returns: ['lightglue', 'superglue', 'loftr', 'topicfm', 'aspanformer', ...]
```

Common matchers:
- **LightGlue** - Fast, efficient (you already have this)
- **SuperGlue** - Robust, good for challenging cases
- **LoFTR** - Dense matching, no keypoints needed
- **TopicFM** - Good for challenging cases
- **ASpanFormer** - Transformer-based, robust

## Configuration

### Get Default Configurations

```python
from app.services.imcui_service import get_extractor_config, get_matcher_config

# Get extractor configuration
extractor_config = get_extractor_config('superpoint')
# Returns: {
#     'max_num_keypoints': 2048,
#     'detection_threshold': None,
#     'nms_window_size': None,
#     ...
# }

# Get matcher configuration
matcher_config = get_matcher_config('lightglue', 'superpoint')
# Returns: {
#     'n_layers': 9,
#     'depth_confidence': 0.9,
#     'width_confidence': 0.99,
#     'filter_threshold': 0.1,
#     ...
# }
```

### Custom Configuration

You can override any parameter:

```python
result = register_with_imcui(
    fixed_id="IMAGE_ID_1",
    moving_id="IMAGE_ID_2",
    extractor="disk",
    matcher="lightglue",
    extractor_config={
        "max_num_keypoints": 4096,  # More keypoints
        "detection_threshold": 0.01  # Lower threshold = more keypoints
    },
    matcher_config={
        "n_layers": 12,  # More layers = better quality
        "filter_threshold": 0.05  # Stricter filtering
    }
)
```

## Usage Examples

### Basic Matching

```python
from app.services.imcui_service import register_with_imcui

result = register_with_imcui(
    fixed_id="6903df8dd26a6d93de19a9b2",
    moving_id="6903df8dd26a6d93de19a9b3",
    extractor="superpoint",
    matcher="lightglue"
)

if result['success']:
    transform = result['transform']
    print(f"Rotation: {transform['rotation_degrees']:.2f}°")
    print(f"Translation: ({transform['offset_x']:.2f}, {transform['offset_y']:.2f})")
    print(f"Matches: {transform['num_matches']}")
    print(f"Inliers: {transform['num_inliers']}")
```

### Advanced: Match Images Directly

```python
from app.services.imcui_service import match_images_with_imcui
import numpy as np
from PIL import Image

# Load images
fixed_img = np.array(Image.open("fixed.jpg"))
moving_img = np.array(Image.open("moving.jpg"))

# Match
match_results = match_images_with_imcui(
    fixed_image=fixed_img,
    moving_image=moving_img,
    extractor="disk",
    matcher="lightglue",
    extractor_config={"max_num_keypoints": 2048},
    matcher_config={"n_layers": 9}
)

print(f"Keypoints in fixed image: {match_results['num_keypoints0']}")
print(f"Keypoints in moving image: {match_results['num_keypoints1']}")
print(f"Matches found: {match_results['num_matches']}")
```

### Compare Multiple Algorithms

```python
from app.services.imcui_service import register_with_imcui

algorithms = [
    ("superpoint", "lightglue"),
    ("disk", "lightglue"),
    ("superpoint", "superglue"),
    ("disk", "superglue"),
]

results = {}
for extractor, matcher in algorithms:
    try:
        result = register_with_imcui(
            fixed_id="IMAGE_ID_1",
            moving_id="IMAGE_ID_2",
            extractor=extractor,
            matcher=matcher
        )
        if result['success']:
            results[f"{extractor}+{matcher}"] = {
                'matches': result['match_results']['num_matches'],
                'inliers': result['transform']['num_inliers'],
                'rotation': result['transform']['rotation_degrees']
            }
    except Exception as e:
        print(f"Failed {extractor}+{matcher}: {e}")

# Find best algorithm
best = max(results.items(), key=lambda x: x[1]['inliers'])
print(f"Best algorithm: {best[0]} with {best[1]['inliers']} inliers")
```

## API Endpoints

### List Available Algorithms

```bash
# Get extractors
curl "http://localhost:8000/api/image-matching/extractors"

# Get matchers
curl "http://localhost:8000/api/image-matching/matchers"
```

### Get Configurations

```bash
# Get extractor config
curl "http://localhost:8000/api/image-matching/extractor/superpoint/config"

# Get matcher config
curl "http://localhost:8000/api/image-matching/matcher/lightglue/config?extractor_name=superpoint"
```

### Perform Matching

```bash
curl -X POST "http://localhost:8000/api/image-matching/match-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_id": "IMAGE_ID_1",
    "moving_id": "IMAGE_ID_2",
    "extractor": "disk",
    "matcher": "lightglue",
    "extractor_params": {
      "max_num_keypoints": 2048,
      "detection_threshold": 0.01
    },
    "matcher_params": {
      "n_layers": 9,
      "filter_threshold": 0.1
    },
    "return_transform": true
  }'
```

## Integration with Existing Services

The imcui service integrates seamlessly with your existing registration services:

```python
# Use imcui for feature matching
from app.services.imcui_service import register_with_imcui

imcui_result = register_with_imcui(
    fixed_id=fixed_id,
    moving_id=moving_id,
    extractor="disk",
    matcher="lightglue"
)

# Then use SimpleITK for refinement
from app.services.registration_service import register_rigid

simpleitk_result = register_rigid(
    fixed_id=fixed_id,
    moving_id=moving_id
)

# Compare results
print(f"imcui rotation: {imcui_result['transform']['rotation_degrees']:.2f}°")
print(f"SimpleITK rotation: {simpleitk_result['rotation_degrees']:.2f}°")
```

## Tips for Tissue Registration

1. **Start with DISK + LightGlue**: Often works better for different stains than SuperPoint
2. **Adjust keypoint count**: More keypoints (4096) can help with sparse tissue
3. **Try LoFTR**: Dense matching can be better for repetitive patterns
4. **Compare algorithms**: Use the comparison example above to find the best for your data
5. **Combine with SimpleITK**: Use imcui for initial alignment, SimpleITK for refinement

## Troubleshooting

### Import Errors

If you get import errors, ensure imcui is installed:

```bash
pip install imcui
```

### Algorithm Not Available

Some algorithms may require additional dependencies. Check the imcui documentation for specific requirements.

### Performance

- Use GPU (`device="cuda"`) for faster processing
- Reduce `max_num_keypoints` for faster extraction
- Reduce `n_layers` for faster matching (at cost of quality)

## References

- [imcui GitHub](https://github.com/Vincentqyw/image-matching-webui)
- [imcui PyPI](https://pypi.org/project/imcui/)
- [LightGlue Documentation](https://github.com/cvg/LightGlue)
