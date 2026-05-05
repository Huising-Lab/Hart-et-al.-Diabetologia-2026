#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# RECURSIVE MASK-BASED INTENSITY QUANTIFICATION PIPELINE
# =============================================================================
#
# Purpose:
#   Recursively find multi-frame microscopy TIFFs, match them to labeled mask
#   TIFFs by FOV number, quantify mean intensity inside each ROI for every
#   frame, and save one CSV per matched movie/mask pair.
#
# Main use case:
#   - Quantify different imaging channels using the same masks
#   - Easily switch target channel by editing CHANNEL_KEYWORD
#
# Input expectations:
#   - Microscopy TIFF:
#       shape = (T, Y, X)
#
#   - Mask TIFF:
#       shape = (Y, X)
#       0 = background
#       positive integers = ROI labels
#
# Output:
#   - CSV with:
#       Frame, Cell_1, Cell_2, Cell_3, ...
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread


# =============================================================================
# USER SETTINGS
# =============================================================================

# Parent folder to recursively search
PARENT_DIRECTORY = r"YourDirectory"

# Subfolder containing the multi-frame TIFF files
MICROSCOPY_SUBFOLDER = "stabilized"

# Subfolder containing the mask files
# This is matched by replacing MICROSCOPY_SUBFOLDER with MASK_SUBFOLDER
MASK_SUBFOLDER = r"stabilized\median_projections"

# -------------------------------------------------------------------------
# Channel selection
# -------------------------------------------------------------------------
# Edit this to quantify a different channel.
#
# Examples:
#   "channel_0"
#   "channel_1"
#   "GFP"
#   "Cy3"
#   "YFP"
#   "CFP"
#   "GCaMP"
#
CHANNEL_KEYWORD = "channel"

# Optional extra keyword required in the microscopy filename.
# Leave as "" to ignore.
EXTRA_MOVIE_KEYWORD = ""

# Microscopy file extension
MICROSCOPY_EXTENSIONS = (".tif", ".tiff")

# Mask file settings
MASK_KEYWORD = "_mask"
MASK_EXTENSIONS = (".tif", ".tiff")

# FOV parsing pattern
# Matches filenames containing:
#   fov0, FOV0, fov_0, FOV-0, fov 0
FOV_REGEX = re.compile(r"fov[_\- ]*(\d+)", re.IGNORECASE)

# Output naming
# If True, output file number is FOV + 1, matching your original behavior.
ADD_ONE_TO_FOV_FOR_OUTPUT = True

# Prefix added to output CSVs
OUTPUT_PREFIX = "TEST"

# Optional channel label added to output filename
# Example output:
#   GCaMP6s1_channel_0.csv
ADD_CHANNEL_TO_OUTPUT_NAME = True

# Output folder behavior
# "movie_folder" = save beside the microscopy TIFF
# "mask_folder"  = save beside the mask TIFF
OUTPUT_LOCATION = "movie_folder"

# If True, overwrite existing output CSVs
OVERWRITE_EXISTING = True

# If True, print detailed processing messages
VERBOSE = True


# =============================================================================
# PATH HELPERS
# =============================================================================

def as_path(path):
    """
    Convert string or Path input into a Path object.
    """

    return Path(path)


def normalize_path_string(path):
    """
    Normalize a path string for robust substring replacement.
    """

    return str(path).replace("/", os.sep).replace("\\", os.sep)


def get_corresponding_mask_folder(microscopy_folder):
    """
    Convert a microscopy folder path into the corresponding mask folder path.

    This uses the original workflow:
        stabilized  ->  stabilized/median_projections

    Example:
        .../analysis/stabilized
        becomes
        .../analysis/stabilized/median_projections
    """

    microscopy_folder = Path(microscopy_folder)

    microscopy_str = normalize_path_string(microscopy_folder)
    microscopy_subfolder_str = normalize_path_string(MICROSCOPY_SUBFOLDER)
    mask_subfolder_str = normalize_path_string(MASK_SUBFOLDER)

    mask_folder_str = microscopy_str.replace(
        microscopy_subfolder_str,
        mask_subfolder_str,
    )

    return Path(mask_folder_str)


# =============================================================================
# FILE MATCHING HELPERS
# =============================================================================

def parse_fov_number(path):
    """
    Extract FOV number from a filename.

    Returns
    -------
    int or None
        Parsed FOV number, or None if no FOV is found.
    """

    path = Path(path)

    match = FOV_REGEX.search(path.name)

    if match is None:
        return None

    return int(match.group(1))


def file_has_extension(path, valid_extensions):
    """
    Check whether a file has one of the allowed extensions.
    """

    return Path(path).suffix.lower() in valid_extensions


def find_microscopy_files(folder):
    """
    Find microscopy TIFF files in a folder matching the selected channel.

    A file must:
      - have a valid microscopy extension
      - contain CHANNEL_KEYWORD
      - contain EXTRA_MOVIE_KEYWORD, if provided
    """

    folder = Path(folder)

    microscopy_files = []

    for path in sorted(folder.iterdir()):

        if not path.is_file():
            continue

        if not file_has_extension(path, MICROSCOPY_EXTENSIONS):
            continue

        name_lower = path.name.lower()

        if CHANNEL_KEYWORD.lower() not in name_lower:
            continue

        if EXTRA_MOVIE_KEYWORD and EXTRA_MOVIE_KEYWORD.lower() not in name_lower:
            continue

        microscopy_files.append(path)

    return microscopy_files


def find_mask_files(folder):
    """
    Find labeled mask TIFF files in a mask folder.

    A file must:
      - have a valid mask extension
      - contain MASK_KEYWORD
    """

    folder = Path(folder)

    if not folder.exists():
        return []

    mask_files = []

    for path in sorted(folder.iterdir()):

        if not path.is_file():
            continue

        if not file_has_extension(path, MASK_EXTENSIONS):
            continue

        if MASK_KEYWORD.lower() not in path.name.lower():
            continue

        mask_files.append(path)

    return mask_files


def index_files_by_fov(files):
    """
    Build a dictionary mapping FOV number to file path.

    If multiple files share the same FOV, the first sorted file is retained.
    """

    indexed = {}

    for path in files:
        fov = parse_fov_number(path)

        if fov is None:
            if VERBOSE:
                print(f"No FOV found in filename, skipping index: {path.name}")
            continue

        if fov not in indexed:
            indexed[fov] = path

    return indexed


# =============================================================================
# INTENSITY QUANTIFICATION
# =============================================================================

def calculate_mean_intensities(microscopy_path, mask_path):
    """
    Calculate mean intensity inside each labeled mask ROI for every frame.

    Parameters
    ----------
    microscopy_path : str or Path
        Multi-frame TIFF path with shape (T, Y, X).

    mask_path : str or Path
        2D labeled mask TIFF path with shape (Y, X).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per frame and one column per ROI.
    """

    microscopy_path = Path(microscopy_path)
    mask_path = Path(mask_path)

    # Load microscopy movie and mask
    microscopy_data = imread(microscopy_path)
    mask_data = imread(mask_path)

    if VERBOSE:
        print(f"Microscopy shape: {microscopy_data.shape}")
        print(f"Mask shape:        {mask_data.shape}")

    # Require movie to be 3D: time, y, x
    if microscopy_data.ndim != 3:
        raise ValueError(
            f"Microscopy TIFF must be 3D with shape (T, Y, X). "
            f"Got {microscopy_data.shape}"
        )

    # Require mask to be 2D: y, x
    if mask_data.ndim != 2:
        raise ValueError(
            f"Mask TIFF must be 2D with shape (Y, X). "
            f"Got {mask_data.shape}"
        )

    # Check XY dimensions
    if microscopy_data.shape[1:] != mask_data.shape:
        raise ValueError(
            f"Dimension mismatch: movie XY {microscopy_data.shape[1:]} "
            f"does not match mask XY {mask_data.shape}"
        )

    # Identify ROI labels, excluding background label 0
    roi_labels = np.unique(mask_data)
    roi_labels = roi_labels[roi_labels != 0]

    if len(roi_labels) == 0:
        raise ValueError(f"No non-zero ROI labels found in mask: {mask_path}")

    # Precompute pixel coordinates for each ROI for speed
    roi_pixels = {
        int(label): mask_data == label
        for label in roi_labels
    }

    # Initialize output dataframe
    results = pd.DataFrame({"Frame": np.arange(microscopy_data.shape[0])})

    # Calculate one trace per ROI
    for label, pixels in roi_pixels.items():
        results[f"Cell_{label}"] = microscopy_data[:, pixels].mean(axis=1)

    return results


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def build_output_filename(fov_number):
    """
    Build output CSV filename from FOV and channel settings.
    """

    output_number = fov_number + 1 if ADD_ONE_TO_FOV_FOR_OUTPUT else fov_number

    if ADD_CHANNEL_TO_OUTPUT_NAME:
        safe_channel = CHANNEL_KEYWORD.replace(" ", "_").replace(os.sep, "_")
        return f"{OUTPUT_PREFIX}{output_number}_{safe_channel}.csv"

    return f"{OUTPUT_PREFIX}{output_number}.csv"


def get_output_folder(microscopy_file, mask_file):
    """
    Select output folder based on OUTPUT_LOCATION.
    """

    microscopy_file = Path(microscopy_file)
    mask_file = Path(mask_file)

    if OUTPUT_LOCATION == "movie_folder":
        return microscopy_file.parent

    if OUTPUT_LOCATION == "mask_folder":
        return mask_file.parent

    raise ValueError("OUTPUT_LOCATION must be 'movie_folder' or 'mask_folder'")


def save_results_csv(results, microscopy_file, mask_file, fov_number):
    """
    Save quantified intensity results to CSV.
    """

    output_folder = get_output_folder(microscopy_file, mask_file)
    output_filename = build_output_filename(fov_number)
    output_path = output_folder / output_filename

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping existing output: {output_path}")
        return

    results.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# =============================================================================
# FOLDER PROCESSING
# =============================================================================

def process_microscopy_folder(microscopy_folder):
    """
    Process one microscopy folder.

    Steps:
      1. Find target channel TIFFs
      2. Locate corresponding mask folder
      3. Match movie and mask files by FOV
      4. Quantify mean ROI intensities
      5. Save CSV output
    """

    microscopy_folder = Path(microscopy_folder)

    if VERBOSE:
        print(f"\nProcessing microscopy folder: {microscopy_folder}")

    microscopy_files = find_microscopy_files(microscopy_folder)

    if VERBOSE:
        print(f"Microscopy files found: {len(microscopy_files)}")

    if not microscopy_files:
        return 0

    mask_folder = get_corresponding_mask_folder(microscopy_folder)
    mask_files = find_mask_files(mask_folder)

    if VERBOSE:
        print(f"Mask folder: {mask_folder}")
        print(f"Mask files found: {len(mask_files)}")

    if not mask_files:
        print(f"No mask files found for folder: {microscopy_folder}")
        return 0

    microscopy_by_fov = index_files_by_fov(microscopy_files)
    masks_by_fov = index_files_by_fov(mask_files)

    processed_count = 0

    for fov_number, microscopy_file in microscopy_by_fov.items():

        mask_file = masks_by_fov.get(fov_number)

        if mask_file is None:
            print(f"No matching mask found for FOV {fov_number}: {microscopy_file.name}")
            continue

        print(f"\nMatched FOV {fov_number}")
        print(f"Movie: {microscopy_file.name}")
        print(f"Mask:  {mask_file.name}")

        try:
            results = calculate_mean_intensities(
                microscopy_path=microscopy_file,
                mask_path=mask_file,
            )

            save_results_csv(
                results=results,
                microscopy_file=microscopy_file,
                mask_file=mask_file,
                fov_number=fov_number,
            )

            processed_count += 1

        except Exception as error:
            print(f"Failed FOV {fov_number}: {error}")

    return processed_count


def process_parent_directory(parent_directory):
    """
    Recursively search parent directory for microscopy folders and process them.
    """

    parent_directory = Path(parent_directory)

    if not parent_directory.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_directory}")

    total_processed = 0

    for root, _, _ in os.walk(parent_directory):
        root_path = Path(root)

        # Process only folders that contain the microscopy subfolder token
        if MICROSCOPY_SUBFOLDER not in normalize_path_string(root_path):
            continue

        processed_here = process_microscopy_folder(root_path)
        total_processed += processed_here

    print(f"\nDone. Total movie/mask pairs processed: {total_processed}")


# =============================================================================
# RUN
# =============================================================================

process_parent_directory(PARENT_DIRECTORY)


# In[ ]:




