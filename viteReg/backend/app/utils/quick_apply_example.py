"""
Quick example script for applying transforms from another UI.
Copy this and modify the image IDs and matrix.
"""
import numpy as np
import sys
from pathlib import Path

# Add backend directory to path if needed
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.utils.apply_transform import quick_apply_transform, load_transform_from_json

# ===== CONFIGURATION =====
# Replace these with your actual image IDs
FIXED_IMAGE_ID = "6903df8dd26a6d93de19a9b2"  # Your fixed image ID
MOVING_IMAGE_ID = "6903df8dd26a6d93de19a9b3"  # Your moving image ID

# Option 1: Load from JSON file
MATRIX_FILE = "transform.json"  # Path to your JSON file
MATRIX_TYPE = "Homography"  # or "H1", "H2", "Fundamental"

# Option 2: Use matrix directly (uncomment and modify)
# MATRIX = np.array([
#     [0.8036541512898507, -0.026251719324804954, 7.991064800476521],
#     [0.010849798858044617, 0.8128097314434242, 1.9440023459021754],
#     [-0.00007160226364392854, -0.000026090723081625177, 1]
# ])

# Output path (optional)
OUTPUT_PATH = "transformed_result.png"

# ===== APPLY TRANSFORM =====
if __name__ == "__main__":
    # Load matrix
    if 'MATRIX' in globals():
        transform_matrix = MATRIX
        print("Using direct matrix")
    else:
        transform_matrix = load_transform_from_json(MATRIX_FILE, MATRIX_TYPE)
        print(f"Loaded {MATRIX_TYPE} from {MATRIX_FILE}")
    
    # Apply transform
    print(f"\nApplying transform to:")
    print(f"  Fixed: {FIXED_IMAGE_ID}")
    print(f"  Moving: {MOVING_IMAGE_ID}")
    
    transformed = quick_apply_transform(
        FIXED_IMAGE_ID,
        MOVING_IMAGE_ID,
        transform_matrix,
        output_path=OUTPUT_PATH
    )
    
    print(f"\n✓ Done! Shape: {transformed.shape}")
    if OUTPUT_PATH:
        print(f"  Saved to: {OUTPUT_PATH}")
