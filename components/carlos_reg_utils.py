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
    print(type(reg_image), "was received for reg image...")
    print("NumPy array shape:", reg_image.shape)

    # Extract each channel as a 2D scalar image
    channel1 = sitk.GetImageFromArray(reg_image[:, :, 0])
    channel2 = sitk.GetImageFromArray(reg_image[:, :, 1])
    channel3 = sitk.GetImageFromArray(reg_image[:, :, 2])

    # Compose the channels into a 2D vector image
    reg_image_sitk = sitk.Compose([channel1, channel2, channel3])
    print("Image Size:", reg_image_sitk.GetSize())  # Should be (1024, 927)
    print(
        "Number of Components:", reg_image_sitk.GetNumberOfComponentsPerPixel()
    )  # Should be 3

    # Define the 2D affine transform
    transform = sitk.AffineTransform(2)
    transform.SetMatrix(transf_matrix[:2, :2].flatten().tolist())
    transform.SetTranslation(transf_matrix[:2, 2].tolist())
    inverse_transform = transform.GetInverse()

    # Resample the image with explicit parameters
    resampled_image = sitk.Resample(
        reg_image_sitk,
        size=reg_image_sitk.GetSize(),
        transform=inverse_transform,
        interpolator=sitk.sitkLinear,
        outputOrigin=reg_image_sitk.GetOrigin(),
        outputSpacing=reg_image_sitk.GetSpacing(),
        outputDirection=reg_image_sitk.GetDirection(),
        defaultPixelValue=0.0,
        outputPixelType=reg_image_sitk.GetPixelID(),
    )
    # Save the transformed image

    resampled_image = np.array(resampled_image).reshape(
        (reg_image_sitk.GetSize()[0], reg_image_sitk.GetSize()[1], 3)
    )
    return resampled_image


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
    image = skimage.color.rgb2gray(image)
    blurred_image = skimage.filters.gaussian(image, sigma=1.0)
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

    fixed_img_sk = reg_utils.get_thumbnail_image(fixed_id)
    moving_img_sk = reg_utils.get_thumbnail_image(moving_id)

    # Ensure images are in RGB format
    if len(fixed_img_sk.shape) == 2:  # If grayscale
        fixed_img_sk = cv2.cvtColor(fixed_img_sk, cv2.COLOR_GRAY2RGB)
    if len(moving_img_sk.shape) == 2:  # If grayscale
        moving_img_sk = cv2.cvtColor(moving_img_sk, cv2.COLOR_GRAY2RGB)

    # Load and process both images
    mask = prepare_mask_exit(fixed_img_sk)
    mask2 = prepare_mask_exit(moving_img_sk)

    # Determine the maximum dimension and pad both masks
    h1, w1 = mask.shape
    h2, w2 = mask2.shape
    max_dim = max(h1, w1, h2, w2)  # max(1108, 1024, 748, 1024) = 1108
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
    to_save_rotate = Image.fromarray(moving_image_rgb)  # .convert("RGB")

    # ----------------------------------------------
    #                REGITRATION
    # ----------------------------------------------

    # Load images as RGB
    fixed_img = sitk.GetImageFromArray(fixed_img_sk, sitk.sitkVectorFloat32)
    moving_img = sitk.GetImageFromArray(moving_image_rgb, sitk.sitkVectorFloat32)

    # Convert to grayscale for initialization
    fixed_image_gray = sitk.Cast(
        sitk.VectorIndexSelectionCast(fixed_img, 0), sitk.sitkFloat32
    )
    moving_image_gray = sitk.Cast(
        sitk.VectorIndexSelectionCast(moving_img, 0), sitk.sitkFloat32
    )
    # Define a similarity transform
    transform = sitk.Similarity2DTransform()
    # transform = sitk.AffineTransform(2)

    # Initialize transform by aligning centers
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image_gray,
        moving_image_gray,
        transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Set up registration
    registration_method = sitk.ImageRegistrationMethod()

    # Use mutual information metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Configure gradient descent optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,  # Reduced for stability
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

    # ----------------------------------------------
    #         TRANSFORMATION MATRIX
    # ----------------------------------------------

    # Execute registration
    final_transform = registration_method.Execute(fixed_image_gray, moving_image_gray)

    # Extract parameters from the final transform
    parameters = final_transform.GetParameters()
    scale = parameters[0]
    angle = parameters[1]
    tx = parameters[2]
    ty = parameters[3]

    # Construct the affine matrix
    import numpy as np

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
    rotation_degrees, offset_x, offset_y, scaling = metrics_registration(affine_matrix)

    # ----------------------------------------------
    #         GC POST TO BDSA
    # ----------------------------------------------

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
    # xfm_dict = dict(enumerate(xfm_dict.flatten(), 1))
    xfm_json = json.dumps(xfm_dict)
    item_meta["XFM"] = xfm_json
    gc.addMetadataToItem(moving_id, item_meta)

    # ----------------------------------------------
    #         TRANSFORMATION MATRIX
    # ----------------------------------------------

    # Resample each color channel separately and cast to uint8
    registered_channels = []
    for channel in range(moving_img.GetNumberOfComponentsPerPixel()):
        moving_channel = sitk.VectorIndexSelectionCast(
            moving_img, channel, sitk.sitkFloat32
        )
        registered_channel = sitk.Resample(
            moving_channel, fixed_image_gray, final_transform, sitk.sitkLinear, 0.0
        )
        registered_channel_uint8 = sitk.Cast(registered_channel, sitk.sitkUInt8)
        registered_channels.append(registered_channel_uint8)

    # Combine the registered channels into a color image
    registered_image = sitk.Compose(registered_channels)

    # Save the result
    # os.makedirs( f'registered/{case}/{subcase}', exist_ok=True)
    # sitk.WriteImage(registered_image, f'registered/{case}/{subcase}/{reg_name}.jpg')

    reg_image = sitk.GetArrayFromImage(registered_image)
    np_image = sitk.GetArrayFromImage(fixed_img).astype("uint8")

    mask_reg = prepare_mask_exit(reg_image)
    mask_orig = prepare_mask_exit(np_image)

    # Determine the orientation
    prev_dice = dice
    dice = dice_coefficient(mask_orig, mask_reg)
    print(dice)

    print("Affine Matrix:\n", affine_matrix)
    return affine_matrix, reg_image
