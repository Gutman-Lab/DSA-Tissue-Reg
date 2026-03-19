"""
Standalone utility to apply a transformation matrix to images for debugging.
Usage:
    python -m app.utils.apply_transform <fixed_image_id> <moving_image_id> <matrix_type> [--matrix-file <json_file>]
    
Example:
    python -m app.utils.apply_transform 6903df8dd26a6d93de19a9b2 6903df8dd26a6d93de19a9b3 Homography --matrix-file transform.json
"""
import numpy as np
import cv2
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

# Import from app services
try:
    from app.services.registration_service import get_thumbnail_image
    from app.services.dsa_client import get_dsa_client
except ImportError:
    # If running as standalone, add parent to path
    backend_dir = Path(__file__).parent.parent.parent
    if backend_dir not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from app.services.registration_service import get_thumbnail_image
    from app.services.dsa_client import get_dsa_client


def quick_apply_transform(
    fixed_image_id: str,
    moving_image_id: str,
    transform_matrix: np.ndarray,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Quick function to apply transform - returns transformed image array.
    Simple wrapper for debugging.
    
    Args:
        fixed_image_id: DSA item ID of fixed image
        moving_image_id: DSA item ID of moving image
        transform_matrix: 3x3 numpy array (homography matrix)
        output_path: Optional path to save result
    
    Returns:
        Transformed moving image as numpy array
    """
    result = apply_homography_transform(
        fixed_image_id,
        moving_image_id,
        transform_matrix,
        show_result=False,
        output_path=output_path
    )
    return result['transformed_image']


def load_transform_from_json(json_file: str, matrix_type: str = "Homography") -> Optional[np.ndarray]:
    """
    Load transformation matrix from JSON file.
    
    Args:
        json_file: Path to JSON file containing geom_info
        matrix_type: Which matrix to use ('Fundamental', 'Homography', 'H1', 'H2')
    
    Returns:
        3x3 transformation matrix as numpy array
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'geom_info' not in data:
        raise ValueError("JSON file must contain 'geom_info' key")
    
    geom_info = data['geom_info']
    
    if matrix_type not in geom_info:
        available = list(geom_info.keys())
        raise ValueError(f"Matrix type '{matrix_type}' not found. Available: {available}")
    
    matrix = np.array(geom_info[matrix_type])
    
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got shape {matrix.shape}")
    
    return matrix


def apply_homography_transform(
    fixed_image_id: str,
    moving_image_id: str,
    transform_matrix: np.ndarray,
    thumbnail_width: int = 1024,
    output_path: Optional[str] = None,
    show_result: bool = True
) -> Dict[str, Any]:
    """
    Apply a homography transformation matrix to align moving image to fixed image.
    
    Args:
        fixed_image_id: DSA item ID of fixed (reference) image
        moving_image_id: DSA item ID of moving image to transform
        transform_matrix: 3x3 homography matrix (moving -> fixed)
        thumbnail_width: Width of thumbnails to use
        output_path: Optional path to save result image
        show_result: Whether to display result using matplotlib
    
    Returns:
        Dictionary with result info and transformed image
    """
    print(f"Loading images...")
    print(f"  Fixed: {fixed_image_id}")
    print(f"  Moving: {moving_image_id}")
    
    # Load images
    fixed_img = get_thumbnail_image(fixed_image_id, width=thumbnail_width)
    moving_img = get_thumbnail_image(moving_image_id, width=thumbnail_width)
    
    if fixed_img is None:
        raise ValueError(f"Failed to load fixed image: {fixed_image_id}")
    if moving_img is None:
        raise ValueError(f"Failed to load moving image: {moving_image_id}")
    
    # Convert to RGB if grayscale
    if len(fixed_img.shape) == 2:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2RGB)
    elif fixed_img.shape[2] == 4:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_RGBA2RGB)
    
    if len(moving_img.shape) == 2:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2RGB)
    elif moving_img.shape[2] == 4:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_RGBA2RGB)
    
    print(f"  Fixed image shape: {fixed_img.shape}")
    print(f"  Moving image shape: {moving_img.shape}")
    
    # Get output dimensions (use fixed image size)
    h, w = fixed_img.shape[:2]
    
    print(f"\nApplying transformation matrix:")
    print(f"  {transform_matrix}")
    
    # Apply homography transform
    # cv2.warpPerspective expects transform: moving -> fixed
    # The matrix should map moving image coordinates to fixed image coordinates
    moving_registered = cv2.warpPerspective(
        moving_img,
        transform_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"  Transformed image shape: {moving_registered.shape}")
    
    # Extract transform parameters for display
    # For homography, we can extract approximate rotation/translation
    # by treating it as an affine transform (ignoring perspective effects)
    if abs(transform_matrix[2, 0]) < 1e-6 and abs(transform_matrix[2, 1]) < 1e-6:
        # Nearly affine (bottom row is [0, 0, 1])
        rotation_rad = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
        scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
        scale = (scale_x + scale_y) / 2.0
        offset_x = transform_matrix[0, 2]
        offset_y = transform_matrix[1, 2]
    else:
        # Full homography with perspective
        rotation_deg = None
        scale = None
        offset_x = transform_matrix[0, 2] / transform_matrix[2, 2]
        offset_y = transform_matrix[1, 2] / transform_matrix[2, 2]
    
    print(f"\nTransform parameters:")
    if rotation_deg is not None:
        print(f"  Rotation: {rotation_deg:.2f}°")
    if scale is not None:
        print(f"  Scale: {scale:.4f}")
    print(f"  Offset: ({offset_x:.2f}, {offset_y:.2f})")
    
    # Create overlay visualization
    overlay = cv2.addWeighted(fixed_img, 0.5, moving_registered, 0.5, 0)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(moving_registered, cv2.COLOR_RGB2BGR))
        print(f"\nSaved transformed image to: {output_path}")
        
        overlay_path = output_path.replace('.png', '_overlay.png').replace('.jpg', '_overlay.jpg')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay to: {overlay_path}")
    
    # Display if requested
    if show_result:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        axes[0, 0].imshow(fixed_img)
        axes[0, 0].set_title('Fixed Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(moving_img)
        axes[0, 1].set_title('Moving Image (Original)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(moving_registered)
        axes[1, 0].set_title('Moving Image (Transformed)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (50% Fixed + 50% Transformed)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'fixed_image': fixed_img,
        'moving_image': moving_img,
        'transformed_image': moving_registered,
        'overlay': overlay,
        'transform_params': {
            'rotation_degrees': rotation_deg,
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Apply transformation matrix to images for debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply Homography matrix from JSON file
  python -m app.utils.apply_transform 6903df8dd26a6d93de19a9b2 6903df8dd26a6d93de19a9b3 Homography --matrix-file transform.json
  
  # Apply H1 matrix and save result
  python -m app.utils.apply_transform <fixed_id> <moving_id> H1 --matrix-file transform.json --output result.png
  
  # Apply matrix directly (as comma-separated values)
  python -m app.utils.apply_transform <fixed_id> <moving_id> --matrix "0.8,0,8,0,0.8,2,0,0,1"
        """
    )
    
    parser.add_argument('fixed_id', help='DSA item ID of fixed image')
    parser.add_argument('moving_id', help='DSA item ID of moving image')
    parser.add_argument('matrix_type', nargs='?', default='Homography',
                       choices=['Fundamental', 'Homography', 'H1', 'H2'],
                       help='Type of matrix to use from JSON (default: Homography)')
    parser.add_argument('--matrix-file', type=str, help='Path to JSON file containing geom_info')
    parser.add_argument('--matrix', type=str, help='Matrix as comma-separated values (9 values for 3x3)')
    parser.add_argument('--output', '-o', type=str, help='Output path to save transformed image')
    parser.add_argument('--no-show', action='store_true', help='Do not display result with matplotlib')
    parser.add_argument('--width', type=int, default=1024, help='Thumbnail width (default: 1024)')
    
    args = parser.parse_args()
    
    # Load transformation matrix
    if args.matrix_file:
        if not Path(args.matrix_file).exists():
            print(f"Error: Matrix file not found: {args.matrix_file}")
            sys.exit(1)
        transform_matrix = load_transform_from_json(args.matrix_file, args.matrix_type)
        print(f"Loaded {args.matrix_type} matrix from {args.matrix_file}")
    elif args.matrix:
        # Parse comma-separated values
        values = [float(x.strip()) for x in args.matrix.split(',')]
        if len(values) != 9:
            print(f"Error: Matrix must have 9 values (3x3), got {len(values)}")
            sys.exit(1)
        transform_matrix = np.array(values).reshape(3, 3)
        print(f"Using provided matrix directly")
    else:
        print("Error: Must provide either --matrix-file or --matrix")
        parser.print_help()
        sys.exit(1)
    
    # Apply transform
    try:
        result = apply_homography_transform(
            args.fixed_id,
            args.moving_id,
            transform_matrix,
            thumbnail_width=args.width,
            output_path=args.output,
            show_result=not args.no_show
        )
        print("\n✓ Transformation applied successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
