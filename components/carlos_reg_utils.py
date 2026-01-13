import subprocess
import skimage
from girder_client import GirderClient
from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2

import io
import os
import yaml
import numpy as np
from PIL import Image
import json
import utils.registration_utils as reg_utils

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

bdsa_url = config["bdsa_url"]
url2 = config["url2"]
ftype = config["ftype"]
folder_id = config["folder_id"]

region = config["region"]
size = config["size"]


from settings import gc, memory, DSA_BASE_URL, token_info


def get_thumbnail_img() -> np.array:
    """Collect image thumbnails list with GC listclient
    according to the region and their case number"""

    itms_list = [i for i in gc.listItem(folder_id)]

    arr_slide = np.array([["_id", "slide", "region", "name", "case", "stainID"]])

    for small_img in itms_list:
        schema = small_img["meta"]["npSchema"]
        if len(schema) == 0:
            continue
        if schema["regionName"] != region:
            continue

        slide_name = small_img["name"].replace(" ", "_")
        slide = "_".join(slide_name.split("_")[0:2])
        case_id = schema["caseID"]
        stain_id = schema["stainID"]

        arr_slide = np.vstack(
            (
                arr_slide,
                [
                    small_img["_id"],
                    slide,
                    schema["regionName"],
                    small_img["name"],
                    case_id,
                    stain_id,
                ],
            )
        )

        outpth = f"thumbnails/{small_img['_id']}"
    return arr_slide


def return_byte(img_id) -> bytes:
    """Rest API call to the BDSA to  get the images as bytes."""
    cmd = [
        "curl",
        "-X",
        "GET",
        f"{bdsa_url}/{img_id}/{url2}",
        "-H",
        ftype,
        "-H",
        f"Girder-Token: {token_info['_id']}",
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0:
        image_data = result.stdout  # Image bytes are in
    else:
        print(f"Error in curl command: {result.stderr.decode()}")
        return None

    return image_data


def pad_to_size(mask, target_size) -> list:
    """
    Function to pad a mask to a target square size, centering it
    """
    h, w = mask.shape
    pad_height = target_size - h
    pad_width = target_size - w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_mask = np.pad(
        mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=False,
    )
    return padded_mask


def apply_affine_transform(reg_image, transf_matrix):
    print("\nApply Affine Transform Debug:")
    print("Input image type:", type(reg_image))
    print("Input image shape:", reg_image.shape)
    print("Input image dtype:", reg_image.dtype)
    print("Input image min/max:", np.min(reg_image), np.max(reg_image))
    print("Transform matrix:\n", transf_matrix)

    try:
        # Ensure the image is in uint8 format
        if reg_image.dtype != np.uint8:
            reg_image = (reg_image * 255).astype(np.uint8)

        # Get image dimensions
        height, width = reg_image.shape[:2]

        # Extract the 2x3 affine matrix from the 3x3 transformation matrix
        affine_matrix = transf_matrix[:2, :]

        # Split the channels
        b, g, r = cv2.split(reg_image)

        # Apply transformation to each channel separately
        warped_r = cv2.warpAffine(
            r,
            affine_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_g = cv2.warpAffine(
            g,
            affine_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_b = cv2.warpAffine(
            b,
            affine_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Merge the channels back together
        warped_image = cv2.merge([warped_b, warped_g, warped_r])

        print("Final output shape:", warped_image.shape)
        print("Final output dtype:", warped_image.dtype)
        print("Final output min/max:", np.min(warped_image), np.max(warped_image))

        return warped_image

    except Exception as e:
        print(f"Error in affine transform: {str(e)}")
        return reg_image  # Return original image if transform fails


def dice_coefficient(mask1, mask2) -> np.array:
    """Define the Dice coefficient function"""
    intersection = np.sum(mask1 & mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    if size1 + size2 == 0:
        return 1.0
    return 2 * intersection / (size1 + size2)


def find_relative_rotation(padded_mask1, padded_mask2) -> tuple[int, float]:
    """Function to find the relative rotation. Rotate counterclockwise"""
    max_dice = 0
    best_k = 0
    for k in range(4):  # 0°, 90°, 180°, 270°
        rotated_mask2 = np.rot90(padded_mask2, k)
        dice = dice_coefficient(padded_mask1, rotated_mask2)
        if k == 0:
            first_dice = dice
        if dice > max_dice:

            if dice > (first_dice + 0.005):

                best_k = k
            # else:
            #     max_dice = dice
            max_dice = dice

    return best_k * 90, max_dice  # Rotation angle and similarity score


def prepare_mask_exit(image) -> np.array:
    """Create a mask based on gaussian filter, blur, threshold"""
    # If image is already grayscale, use it directly
    if len(image.shape) == 2:
        gray_image = image
    else:
        # Convert RGB to grayscale
        gray_image = skimage.color.rgb2gray(image)

    blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
    threshold = 0.95
    mask = blurred_image < threshold

    return mask


def metrics_registration(affine_matrix) -> tuple[int, float, float, float]:
    """ """
    a, b, t_x = affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[0, 2]
    c, d, t_y = affine_matrix[1, 0], affine_matrix[1, 1], affine_matrix[1, 2]

    rotation_degrees = np.arctan2(c, a) * 180 / np.pi
    offset_x = t_x
    offset_y = t_y
    scaling = np.sqrt(a * d - b * c)

    return rotation_degrees, offset_x, offset_y, scaling


def register_fixed_moving(fixed_id, moving_id) -> np.array:
    import numpy as np

    try:
        print("\nDebug: Getting thumbnail images")
        fixed_img_sk = reg_utils.get_thumbnail_image(fixed_id)
        moving_img_sk = reg_utils.get_thumbnail_image(moving_id)

        print("Fixed image type:", type(fixed_img_sk))
        print(
            "Fixed image shape:",
            fixed_img_sk.shape if hasattr(fixed_img_sk, "shape") else "No shape",
        )
        print(
            "Fixed image dtype:",
            fixed_img_sk.dtype if hasattr(fixed_img_sk, "dtype") else "No dtype",
        )
        print("Moving image type:", type(moving_img_sk))
        print(
            "Moving image shape:",
            moving_img_sk.shape if hasattr(moving_img_sk, "shape") else "No shape",
        )
        print(
            "Moving image dtype:",
            moving_img_sk.dtype if hasattr(moving_img_sk, "dtype") else "No dtype",
        )

    except Exception as e:
        print(f"Error getting thumbnail images: {str(e)}")
        return np.eye(3), np.zeros((100, 100, 3), dtype=np.uint8)

    # Convert grayscale to RGB if needed (keeping original color space)
    if len(fixed_img_sk.shape) == 2:
        print("Converting fixed image from grayscale to RGB")
        fixed_img_sk = np.stack([fixed_img_sk] * 3, axis=-1)
    if len(moving_img_sk.shape) == 2:
        print("Converting moving image from grayscale to RGB")
        moving_img_sk = np.stack([moving_img_sk] * 3, axis=-1)

    print("After RGB conversion:")
    print("Fixed image shape:", fixed_img_sk.shape)
    print("Moving image shape:", moving_img_sk.shape)
    print("Fixed image dtype:", fixed_img_sk.dtype)
    print("Moving image dtype:", moving_img_sk.dtype)
    print("Fixed image min/max:", np.min(fixed_img_sk), np.max(fixed_img_sk))
    print("Moving image min/max:", np.min(moving_img_sk), np.max(moving_img_sk))

    # Create grayscale copies for registration while preserving original color
    fixed_gray = cv2.cvtColor(fixed_img_sk, cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_img_sk, cv2.COLOR_RGB2GRAY)

    # Convert to float32 for SimpleITK
    fixed_gray = fixed_gray.astype(np.float32) / 255.0
    moving_gray = moving_gray.astype(np.float32) / 255.0

    # Load and process both images for mask creation
    mask = prepare_mask_exit(fixed_gray)
    mask2 = prepare_mask_exit(moving_gray)

    # Determine the maximum dimension and pad both masks
    h1, w1 = mask.shape
    h2, w2 = mask2.shape
    max_dim = max(h1, w1, h2, w2)
    padded_mask1 = pad_to_size(mask, max_dim)
    padded_mask2 = pad_to_size(mask2, max_dim)

    # Determine the orientation
    relative_rotation, dice = find_relative_rotation(padded_mask1, padded_mask2)

    # Output the result
    print(f"Relative rotation: {relative_rotation} degrees")
    print(f"Dice coefficient: {dice:.3f}")

    if relative_rotation == 0:
        print("The images are in the same orientation.")
    else:
        print("The images are rotated.")

    padded_mask2 = np.rot90(padded_mask2, int(relative_rotation / 90))
    moving_image_rgb = np.rot90(moving_img_sk, int(relative_rotation / 90))

    # ----------------------------------------------
    #                REGISTRATION
    # ----------------------------------------------

    try:
        # Convert to SimpleITK format - use grayscale for registration
        fixed_img = sitk.GetImageFromArray(fixed_gray)
        moving_img = sitk.GetImageFromArray(
            cv2.cvtColor(moving_image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            / 255.0
        )

        # Define a similarity transform
        transform = sitk.Similarity2DTransform()

        # Initialize transform by aligning centers
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img,
            moving_img,
            transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        # Set up registration
        registration_method = sitk.ImageRegistrationMethod()

        # Use mutual information metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # Configure gradient descent optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )

        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Apply initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=True)

        # Balance parameter updates
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Use multi-resolution for robustness
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])

        # Execute registration
        final_transform = registration_method.Execute(fixed_img, moving_img)

        # Extract parameters from the final transform
        parameters = final_transform.GetParameters()
        scale = parameters[0]
        angle = parameters[1]
        tx = parameters[2]
        ty = parameters[3]

        # Construct the affine matrix
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        # Similarity transform matrix
        affine_matrix = np.array(
            [
                [scale * cos_theta, -scale * sin_theta, tx],
                [scale * sin_theta, scale * cos_theta, ty],
                [0, 0, 1],
            ]
        )
        rotation_degrees, offset_x, offset_y, scaling = metrics_registration(
            affine_matrix
        )

        # ----------------------------------------------
        #         GC POST TO BDSA
        # ----------------------------------------------

        try:
            all_item_data = gc.getItem(moving_id)
            item_meta = all_item_data["meta"]
            item_meta["XFM"] = affine_matrix

            all_item_data["meta"]["npReg"] = {
                "srcImage": fixed_id,
                "xOffset": offset_x,
                "yOffset": offset_y,
                "scale": scaling,
                "rotation": rotation_degrees,
                "regImageSize": size,
            }
            item_meta["npReg"] = all_item_data["meta"]["npReg"]

            all_item_data["meta"] = item_meta
            xfm_dict = item_meta["XFM"]
            xfm_dict = {i: list(xfm_dict[i]) for i in range(xfm_dict.shape[0])}
            xfm_json = json.dumps(xfm_dict)
            item_meta["XFM"] = xfm_json
            gc.addMetadataToItem(moving_id, item_meta)
        except Exception as e:
            print(f"Error updating metadata: {str(e)}")

        return affine_matrix, moving_image_rgb

    except Exception as e:
        print(f"Error in registration: {str(e)}")
        # Return identity transform and original image if registration fails
        return np.eye(3), moving_image_rgb
