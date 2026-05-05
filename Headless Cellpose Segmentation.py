#!/usr/bin/env python
# coding: utf-8

# In[3]:


# =============================================================================
# RECURSIVE CELLPOSE SEGMENTATION + MASK POST-PROCESSING PIPELINE
# =============================================================================
#
# Purpose:
#   Recursively find median projection TIFFs, segment them with Cellpose,
#   optionally erode or dilate the resulting masks, save processed masks, and
#   write a segmentation log in each processed folder.
#
# New feature:
#   Easily switch between:
#       1. A custom trained Cellpose model
#       2. A built-in Cellpose model
#
# Outputs per processed image:
#   - <input_name>_mask.tif
#
# Outputs per processed folder:
#   - segmentation_run_log.csv
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from skimage import morphology, exposure, io as skimage_io
from scipy.ndimage import distance_transform_edt


# =============================================================================
# USER SETTINGS: FOLDERS AND FILE MATCHING
# =============================================================================

# Parent folder to recursively search
MASTER_FOLDER = r"YourDirectoryHere"

# Target folder structure
STABILIZED_FOLDER_KEYWORD = "stabilized"
PROJECTION_FOLDER_NAME = "median_projections"

# Target file filter
TARGET_FILENAME_ENDING = "stabilized_median_projection.tif"
TARGET_FILENAME_KEYWORD = "channel"


# =============================================================================
# USER SETTINGS: CELLPOSE MODEL SELECTION
# =============================================================================

# Choose model source:
#   "custom"  = use a trained model from MODEL_DIR / CUSTOM_MODEL_NAME
#   "builtin" = use a built-in Cellpose model
MODEL_SOURCE = "custom"

# Custom model settings
MODEL_DIR = r"PathToModels"
CUSTOM_MODEL_NAME = "ModelName"

# Built-in model settings
# Common built-in options may include:
#   "cyto"
#   "cyto2"
#   "cyto3"
#   "nuclei"
BUILTIN_MODEL_TYPE = "cyto2"

# Cellpose runtime/settings
USE_GPU = True
AVG_CELL_DIAMETER = 40
FLOW_THRESHOLD = 0.8
CHANNELS = [0, 0]


# =============================================================================
# USER SETTINGS: IMAGE NORMALIZATION
# =============================================================================

# If True, normalize each image before Cellpose segmentation
NORMALIZE_BEFORE_SEGMENTATION = True

# Percentiles used for intensity normalization
LOWER_PERCENTILE = 50
UPPER_PERCENTILE = 99

# Output intensity range after percentile normalization
NORMALIZED_OUTPUT_RANGE = (0, 65535)


# =============================================================================
# USER SETTINGS: MASK POST-PROCESSING
# =============================================================================

# Mask processing mode:
#   "erode"  = shrink masks inward
#   "dilate" = expand masks outward
#   "none"   = save raw Cellpose masks
MASK_PROCESSING_MODE = "erode"

# Pixel radius used for erosion or dilation
PIXEL_VALUE = 5


# =============================================================================
# USER SETTINGS: OUTPUTS
# =============================================================================

# Suffix added to output mask files
MASK_SUFFIX = "_maskTEST.tif"

# Folder-level log filename
LOG_FILENAME = "segmentation_run_log.csv"

# If True, overwrite existing masks
OVERWRITE_EXISTING_MASKS = True

# If True, print progress messages
VERBOSE = True


# =============================================================================
# CELLPOSE MODEL LOADING
# =============================================================================

def load_cellpose_model():
    """
    Load either a custom Cellpose model or a built-in Cellpose model.

    Controlled by MODEL_SOURCE.

    Returns
    -------
    cellpose.models.CellposeModel
        Loaded Cellpose model object.
    """

    model_source = MODEL_SOURCE.lower()

    if model_source == "custom":
        custom_model_path = Path(MODEL_DIR) / CUSTOM_MODEL_NAME

        print(f"Loading custom Cellpose model:")
        print(f"  {custom_model_path}")

        model = models.CellposeModel(
            gpu=USE_GPU,
            pretrained_model=str(custom_model_path),
        )

        return model

    if model_source == "builtin":
        print(f"Loading built-in Cellpose model:")
        print(f"  {BUILTIN_MODEL_TYPE}")

        model = models.CellposeModel(
            gpu=USE_GPU,
            model_type=BUILTIN_MODEL_TYPE,
        )

        return model

    raise ValueError("MODEL_SOURCE must be either 'custom' or 'builtin'.")


# =============================================================================
# IMAGE NORMALIZATION
# =============================================================================

def normalize_percentiles(image, lower_percentile, upper_percentile):
    """
    Normalize image intensities between selected percentile values.

    This improves segmentation consistency when brightness varies between files.
    """

    p_low, p_high = np.percentile(
        image,
        (lower_percentile, upper_percentile),
    )

    normalized = exposure.rescale_intensity(
        image,
        in_range=(p_low, p_high),
        out_range=NORMALIZED_OUTPUT_RANGE,
    )

    return normalized


# =============================================================================
# MASK POST-PROCESSING
# =============================================================================

def centered_erosion(mask, erosion_level=10):
    """
    Erode a binary mask using a distance transform.

    Pixels are retained only if they are farther than erosion_level pixels from
    the ROI boundary.
    """

    distance = distance_transform_edt(mask)
    eroded_mask = distance > erosion_level

    return eroded_mask


def postprocess_masks(masks, mode, pixel_value):
    """
    Apply optional erosion or dilation to labeled Cellpose masks.

    Parameters
    ----------
    masks : numpy.ndarray
        Labeled Cellpose mask image.

    mode : str
        "erode", "dilate", or "none".

    pixel_value : int
        Erosion or dilation radius in pixels.

    Returns
    -------
    numpy.ndarray
        Post-processed labeled mask.
    """

    mode = mode.lower()

    if mode == "none":
        return masks.astype(np.uint16)

    processed_masks = np.zeros_like(masks, dtype=np.uint16)

    roi_labels = np.unique(masks)
    roi_labels = roi_labels[roi_labels != 0]

    if mode == "dilate":
        for label in roi_labels:
            single_mask = masks == label

            dilated_mask = morphology.dilation(
                single_mask,
                morphology.disk(pixel_value),
            )

            # Assign dilated pixels only where no label has already been written.
            # This prevents later ROIs from overwriting earlier ROIs in overlap zones.
            processed_masks[dilated_mask & (processed_masks == 0)] = label

    elif mode == "erode":
        for label in roi_labels:
            single_mask = masks == label

            eroded_mask = centered_erosion(
                single_mask,
                erosion_level=pixel_value,
            )

            processed_masks[eroded_mask] = label

    else:
        raise ValueError("MASK_PROCESSING_MODE must be 'erode', 'dilate', or 'none'.")

    return processed_masks


# =============================================================================
# LOGGING
# =============================================================================

def build_log_row(
    image_path,
    mask_path,
    raw_image_shape,
    raw_image_dtype,
    mask_shape,
    mask_dtype,
    num_masks,
):
    """
    Build one row for the segmentation log file.

    This captures the key variables needed to reproduce the mask.
    """

    custom_model_path = Path(MODEL_DIR) / CUSTOM_MODEL_NAME

    return {
        # Date/time of mask creation
        "mask_creation_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # Input/output files
        "input_image": str(image_path),
        "output_mask": str(mask_path),

        # Image/mask metadata
        "raw_image_shape": str(raw_image_shape),
        "raw_image_dtype": str(raw_image_dtype),
        "mask_shape": str(mask_shape),
        "mask_dtype": str(mask_dtype),
        "num_masks_detected": int(num_masks),

        # Model-selection settings
        "model_source": MODEL_SOURCE,
        "custom_model_dir": str(MODEL_DIR),
        "custom_model_name": CUSTOM_MODEL_NAME,
        "custom_model_path": str(custom_model_path),
        "builtin_model_type": BUILTIN_MODEL_TYPE,

        # Cellpose segmentation variables
        "use_gpu": USE_GPU,
        "avg_cell_diameter": AVG_CELL_DIAMETER,
        "flow_threshold": FLOW_THRESHOLD,
        "channels": str(CHANNELS),

        # Normalization variables
        "normalize_before_segmentation": NORMALIZE_BEFORE_SEGMENTATION,
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "normalized_output_range": str(NORMALIZED_OUTPUT_RANGE),

        # Mask post-processing variables
        "mask_processing_mode": MASK_PROCESSING_MODE,
        "pixel_value": PIXEL_VALUE,

        # File matching/output variables
        "target_filename_ending": TARGET_FILENAME_ENDING,
        "target_filename_keyword": TARGET_FILENAME_KEYWORD,
        "mask_suffix": MASK_SUFFIX,
        "overwrite_existing_masks": OVERWRITE_EXISTING_MASKS,
    }


def append_segmentation_log(folder_path, log_row):
    """
    Append one segmentation record to the folder-level CSV log.
    """

    folder_path = Path(folder_path)
    log_path = folder_path / LOG_FILENAME

    new_row_df = pd.DataFrame([log_row])

    if log_path.exists():
        existing_df = pd.read_csv(log_path)
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df

    combined_df.to_csv(log_path, index=False)

    if VERBOSE:
        print(f"Updated segmentation log:")
        print(f"  {log_path}")


# =============================================================================
# SINGLE-IMAGE PROCESSING
# =============================================================================

def process_image(image_path, model):
    """
    Segment one image with Cellpose, post-process the mask, save the mask,
    and return one log row.
    """

    image_path = Path(image_path)

    if VERBOSE:
        print(f"\nProcessing image:")
        print(f"  {image_path}")

    # Load image
    image = skimage_io.imread(image_path)

    raw_image_shape = image.shape
    raw_image_dtype = image.dtype

    # Optional percentile normalization before segmentation
    if NORMALIZE_BEFORE_SEGMENTATION:
        image_for_segmentation = normalize_percentiles(
            image,
            LOWER_PERCENTILE,
            UPPER_PERCENTILE,
        )
    else:
        image_for_segmentation = image

    # Construct mask path beside the source image
    mask_path = image_path.with_name(f"{image_path.stem}{MASK_SUFFIX}")

    # Skip existing output if requested
    if mask_path.exists() and not OVERWRITE_EXISTING_MASKS:
        print(f"Skipping existing mask:")
        print(f"  {mask_path}")
        return None

    # Run Cellpose segmentation
    results = model.eval(
        image_for_segmentation,
        diameter=AVG_CELL_DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        channels=CHANNELS,
    )

    masks = results[0]

    if masks is None or not np.any(masks):
        print(f"No cells detected in:")
        print(f"  {image_path}")
        return None

    # Erode, dilate, or keep raw masks
    processed_masks = postprocess_masks(
        masks=masks,
        mode=MASK_PROCESSING_MODE,
        pixel_value=PIXEL_VALUE,
    )

    # Save processed mask
    tifffile.imwrite(
        mask_path,
        processed_masks.astype(np.uint16),
    )

    # Count non-background labels
    roi_labels = np.unique(processed_masks)
    roi_labels = roi_labels[roi_labels != 0]
    num_masks = len(roi_labels)

    print(f"Saved mask:")
    print(f"  {mask_path}")
    print(f"Masks detected: {num_masks}")

    # Build log record
    log_row = build_log_row(
        image_path=image_path,
        mask_path=mask_path,
        raw_image_shape=raw_image_shape,
        raw_image_dtype=raw_image_dtype,
        mask_shape=processed_masks.shape,
        mask_dtype=processed_masks.dtype,
        num_masks=num_masks,
    )

    return log_row


# =============================================================================
# FOLDER PROCESSING
# =============================================================================

def process_specific_folder(folder_path, model):
    """
    Process all matching TIFF files in one projection folder.

    A file is processed only if:
      - it ends with TARGET_FILENAME_ENDING
      - it contains TARGET_FILENAME_KEYWORD
    """

    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Folder does not exist, skipping:")
        print(f"  {folder_path}")
        return 0

    processed_count = 0

    for image_path in sorted(folder_path.iterdir()):
        if not image_path.is_file():
            continue

        filename = image_path.name

        if not filename.endswith(TARGET_FILENAME_ENDING):
            continue

        if TARGET_FILENAME_KEYWORD not in filename:
            continue

        log_row = process_image(
            image_path=image_path,
            model=model,
        )

        if log_row is not None:
            append_segmentation_log(
                folder_path=folder_path,
                log_row=log_row,
            )

            processed_count += 1

    if VERBOSE:
        print(f"\nFinished folder:")
        print(f"  {folder_path}")
        print(f"Images processed in folder: {processed_count}")

    return processed_count


# =============================================================================
# RECURSIVE BATCH PROCESSING
# =============================================================================

def recursive_folder_processing(master_folder):
    """
    Recursively find stabilized/median_projections folders and process
    matching projection TIFF images.
    """

    master_folder = Path(master_folder)

    if not master_folder.exists():
        raise FileNotFoundError(f"Master folder does not exist: {master_folder}")

    # Load the selected Cellpose model once, then reuse it for all images.
    model = load_cellpose_model()

    total_processed = 0

    for root, dirs, _ in os.walk(master_folder):
        for dir_name in dirs:

            # Look for stabilized folders
            if STABILIZED_FOLDER_KEYWORD not in dir_name:
                continue

            stabilized_folder = Path(root) / dir_name
            projection_folder = stabilized_folder / PROJECTION_FOLDER_NAME

            if projection_folder.exists():
                print(f"\nFound projection folder:")
                print(f"  {projection_folder}")

                processed_here = process_specific_folder(
                    folder_path=projection_folder,
                    model=model,
                )

                total_processed += processed_here

            else:
                if VERBOSE:
                    print(f"No projection folder found in:")
                    print(f"  {stabilized_folder}")

    print(f"\nDone.")
    print(f"Total images processed: {total_processed}")


# =============================================================================
# RUN
# =============================================================================

recursive_folder_processing(MASTER_FOLDER)


# In[ ]:




