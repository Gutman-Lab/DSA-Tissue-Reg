# Apply Transform Utility

Quick debugging tool to apply transformation matrices from another UI to your selected images.

## Quick Start

### Method 1: Command Line

```bash
# From the backend directory
python -m app.utils.apply_transform <fixed_id> <moving_id> Homography --matrix-file transform.json
```

### Method 2: Python Script

Copy and modify `quick_apply_example.py`:

```python
from app.utils.apply_transform import quick_apply_transform, load_transform_from_json

# Load matrix from JSON
matrix = load_transform_from_json("transform.json", "Homography")

# Apply to images
transformed = quick_apply_transform(
    "6903df8dd26a6d93de19a9b2",  # fixed image ID
    "6903df8dd26a6d93de19a9b3",  # moving image ID
    matrix,
    output_path="result.png"
)
```

### Method 3: Direct Matrix

```python
import numpy as np
from app.utils.apply_transform import quick_apply_transform

# Your matrix from the other UI
matrix = np.array([
    [0.8036541512898507, -0.026251719324804954, 7.991064800476521],
    [0.010849798858044617, 0.8128097314434242, 1.9440023459021754],
    [-0.00007160226364392854, -0.000026090723081625177, 1]
])

transformed = quick_apply_transform(
    "fixed_id",
    "moving_id", 
    matrix
)
```

## JSON Format

The utility expects JSON files with this structure:

```json
{
  "geom_info": {
    "Fundamental": [[...], [...], [...]],
    "Homography": [[...], [...], [...]],
    "H1": [[...], [...], [...]],
    "H2": [[...], [...], [...]]
  }
}
```

## Command Line Options

```
python -m app.utils.apply_transform <fixed_id> <moving_id> [matrix_type] [options]

Arguments:
  fixed_id          DSA item ID of fixed image
  moving_id         DSA item ID of moving image
  matrix_type       Matrix type: Fundamental, Homography, H1, H2 (default: Homography)

Options:
  --matrix-file     Path to JSON file containing geom_info
  --matrix          Matrix as comma-separated values (9 values for 3x3)
  --output, -o      Save transformed image to this path
  --no-show         Don't display result with matplotlib
  --width           Thumbnail width (default: 1024)
```

## Examples

```bash
# Apply Homography from JSON file
python -m app.utils.apply_transform 6903df8dd26a6d93de19a9b2 6903df8dd26a6d93de19a9b3 Homography --matrix-file transform.json

# Apply H1 matrix and save result
python -m app.utils.apply_transform <fixed_id> <moving_id> H1 --matrix-file transform.json --output result.png

# Use matrix directly (no JSON file)
python -m app.utils.apply_transform <fixed_id> <moving_id> --matrix "0.8,0,8,0,0.8,2,0,0,1" --output result.png
```

## Output

The utility will:
1. Load both images from DSA
2. Apply the transformation matrix
3. Display a 2x2 grid showing: Fixed, Moving (original), Moving (transformed), Overlay
4. Optionally save the transformed image and overlay

## Notes

- The matrix should map **moving image coordinates → fixed image coordinates**
- Uses `cv2.warpPerspective` for homography transforms
- Images are loaded as thumbnails (default 1024px width) for speed
- The overlay shows 50% fixed + 50% transformed for visual comparison
