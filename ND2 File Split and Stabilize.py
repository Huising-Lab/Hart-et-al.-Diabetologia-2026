#!/usr/bin/env python
# coding: utf-8

# In[3]:


# =============================================================================
# ND2 CHANNEL SPLITTING + AFFINE STABILIZATION + PROJECTION PIPELINE
# =============================================================================
#
# Purpose:
#   Process ND2 microscopy files by:
#     1. Splitting fields of view and channels
#     2. Saving raw TIFF stacks
#     3. Stabilizing stacks using affine ECC registration
#     4. Applying the same stabilization transforms to all channels in each FOV
#     5. Saving median, maximum, and minimum projections
#
# Input:
#   - Folder containing .nd2 files
#
# Outputs:
#   For each ND2 file:
#     <filename>_analysisYYYYMMDD/
#       median_projections/
#       maximum_projections/
#       minimum_projections/
#       stabilized/
#         median_projections/
#         maximum_projections/
#         minimum_projections/
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tifffile as tiff
from nd2reader import ND2Reader
from tqdm import tqdm


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder containing ND2 files
INPUT_FOLDER = r"YourDirectory"

# ND2 file extension
ND2_EXTENSION = ".nd2"

# Analysis folder date suffix
# Example output:
#   MyFile_analysis20260330
ANALYSIS_DATE_TAG = "DateofAnalysis"

# If True, process all ND2 files in INPUT_FOLDER
PROCESS_ALL_ND2_FILES = True

# Optional single ND2 file path.
# Used only if PROCESS_ALL_ND2_FILES = False
SINGLE_ND2_FILE = r""

# Stabilization settings
RUN_STABILIZATION = True
ECC_ITERATIONS = 1000
ECC_ERROR_TOLERANCE = 1e-5

# Number of previous stabilized frames to average for the moving reference
REFERENCE_WINDOW_SIZE = 10

# Reference channel selection
# If True, use the first channel that does NOT contain EXCLUDED_REFERENCE_CHANNEL_TOKEN.
# If False, use REFERENCE_CHANNEL_INDEX.
AUTO_SELECT_REFERENCE_CHANNEL = True
EXCLUDED_REFERENCE_CHANNEL_TOKEN = "CY5"
REFERENCE_CHANNEL_INDEX = 0

# TIFF output settings
TIFF_PHOTOMETRIC = "minisblack"

# Output projection toggles
SAVE_MEDIAN_PROJECTION = True
SAVE_MAXIMUM_PROJECTION = True
SAVE_MINIMUM_PROJECTION = True

# Save raw and stabilized full stacks
SAVE_RAW_STACKS = True
SAVE_STABILIZED_STACKS = True

# If True, print progress information
VERBOSE = True


# =============================================================================
# OUTPUT FOLDER CREATION
# =============================================================================

def create_output_folders(base_folder, base_filename):
    """
    Create the analysis output folder tree for one ND2 file.

    Parameters
    ----------
    base_folder : str or Path
        Folder containing the ND2 file.

    base_filename : str
        ND2 filename without extension.

    Returns
    -------
    dict
        Dictionary of output folder paths.
    """

    base_folder = Path(base_folder)

    analysis_folder = base_folder / f"{base_filename}_analysis{ANALYSIS_DATE_TAG}"

    folders = {
        "analysis": analysis_folder,
        "raw_median": analysis_folder / "median_projections",
        "raw_maximum": analysis_folder / "maximum_projections",
        "raw_minimum": analysis_folder / "minimum_projections",
        "stabilized": analysis_folder / "stabilized",
        "stabilized_median": analysis_folder / "stabilized" / "median_projections",
        "stabilized_maximum": analysis_folder / "stabilized" / "maximum_projections",
        "stabilized_minimum": analysis_folder / "stabilized" / "minimum_projections",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


# =============================================================================
# AFFINE STABILIZATION HELPERS
# =============================================================================

def compute_affine_transform(reference_frame, moving_frame, iterations=1000, error_tolerance=1e-5):
    """
    Compute affine transform aligning moving_frame to reference_frame.

    Uses OpenCV ECC registration.

    Parameters
    ----------
    reference_frame : numpy.ndarray
        Reference image.

    moving_frame : numpy.ndarray
        Frame to align to the reference.

    iterations : int
        Maximum ECC iterations.

    error_tolerance : float
        ECC convergence tolerance.

    Returns
    -------
    numpy.ndarray
        2 x 3 affine warp matrix.
    """

    reference_gray = (
        cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        if reference_frame.ndim == 3
        else reference_frame
    )

    moving_gray = (
        cv2.cvtColor(moving_frame, cv2.COLOR_BGR2GRAY)
        if moving_frame.ndim == 3
        else moving_frame
    )

    reference_gray = reference_gray.astype(np.float32)
    moving_gray = moving_gray.astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        iterations,
        error_tolerance,
    )

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    _, warp_matrix = cv2.findTransformECC(
        reference_gray,
        moving_gray,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria,
    )

    return warp_matrix


def average_previous_frames(stabilized_stack, current_index, window_size=3):
    """
    Average previous stabilized frames to create a rolling reference image.

    This helps reduce frame-to-frame noise during stabilization.
    """

    start_index = max(0, current_index - window_size)
    frames_to_average = stabilized_stack[start_index:current_index]

    if len(frames_to_average) == 0:
        return stabilized_stack[0]

    return np.mean(frames_to_average, axis=0)


def stabilize_stack_with_affine_averaging(
    stack,
    iterations=1000,
    error_tolerance=1e-5,
    window_size=10,
):
    """
    Stabilize a 3D stack using affine ECC registration.

    The first frame is retained as-is. Each subsequent frame is aligned to an
    average of recent stabilized frames.

    Parameters
    ----------
    stack : numpy.ndarray
        Input stack with shape (T, Y, X).

    Returns
    -------
    tuple
        stabilized_stack, warp_matrices
    """

    stabilized_stack = [stack[0]]
    warp_matrices = []

    for frame_index in tqdm(
        range(1, len(stack)),
        desc="Stabilizing frames",
        unit="frame",
    ):
        reference_frame = average_previous_frames(
            stabilized_stack,
            frame_index,
            window_size=window_size,
        )

        warp_matrix = compute_affine_transform(
            reference_frame=reference_frame,
            moving_frame=stack[frame_index],
            iterations=iterations,
            error_tolerance=error_tolerance,
        )

        warp_matrices.append(warp_matrix)

        stabilized_frame = cv2.warpAffine(
            stack[frame_index],
            warp_matrix,
            (reference_frame.shape[1], reference_frame.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

        stabilized_stack.append(stabilized_frame)

    return np.asarray(stabilized_stack), warp_matrices


def apply_warp_matrices(stack, warp_matrices):
    """
    Apply precomputed warp matrices to another channel from the same FOV.

    This keeps all channels spatially aligned using transforms computed from
    the reference channel.
    """

    stabilized_stack = [stack[0]]

    for frame_index in range(1, len(stack)):
        stabilized_frame = cv2.warpAffine(
            stack[frame_index],
            warp_matrices[frame_index - 1],
            (stack[0].shape[1], stack[0].shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

        stabilized_stack.append(stabilized_frame)

    return np.asarray(stabilized_stack)


# =============================================================================
# TIFF SAVING HELPERS
# =============================================================================

def save_tiff_stack(path, stack):
    """
    Save a TIFF stack with consistent photometric settings.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tiff.imwrite(
        path,
        stack,
        photometric=TIFF_PHOTOMETRIC,
    )

    if VERBOSE:
        print(f"Saved: {path}")


def save_projection(stack, output_folder, output_name, projection_type):
    """
    Save one projection image from a stack.

    projection_type options:
      - "median"
      - "maximum"
      - "minimum"
    """

    output_folder = Path(output_folder)

    if projection_type == "median":
        projection = np.median(stack, axis=0).astype(np.uint16)

    elif projection_type == "maximum":
        projection = np.max(stack, axis=0).astype(np.uint16)

    elif projection_type == "minimum":
        projection = np.min(stack, axis=0).astype(np.uint16)

    else:
        raise ValueError("projection_type must be 'median', 'maximum', or 'minimum'.")

    save_tiff_stack(
        output_folder / output_name,
        projection,
    )


def save_all_projections(stack, folders, channel_filename_base, stabilized=False):
    """
    Save selected projection types for a raw or stabilized stack.
    """

    if stabilized:
        median_folder = folders["stabilized_median"]
        maximum_folder = folders["stabilized_maximum"]
        minimum_folder = folders["stabilized_minimum"]
        prefix = f"{channel_filename_base}_stabilized"

    else:
        median_folder = folders["raw_median"]
        maximum_folder = folders["raw_maximum"]
        minimum_folder = folders["raw_minimum"]
        prefix = channel_filename_base

    if SAVE_MEDIAN_PROJECTION:
        save_projection(
            stack,
            median_folder,
            f"{prefix}_median_projection.tif",
            projection_type="median",
        )

    if SAVE_MAXIMUM_PROJECTION:
        save_projection(
            stack,
            maximum_folder,
            f"{prefix}_maximum_projection.tif",
            projection_type="maximum",
        )

    if SAVE_MINIMUM_PROJECTION:
        save_projection(
            stack,
            minimum_folder,
            f"{prefix}_minimum_projection.tif",
            projection_type="minimum",
        )


# =============================================================================
# ND2 METADATA HELPERS
# =============================================================================

def get_fields_of_view(images):
    """
    Return available fields of view.

    If no FOV axis exists, returns [0].
    """

    available_axes = images.axes

    if "v" in available_axes:
        return images.metadata.get("fields_of_view", [0])

    return [0]


def get_channel_names(images):
    """
    Return channel names from ND2 metadata.

    If channel metadata are unavailable, returns ['channel_0'].
    """

    channel_names = images.metadata.get("channels", None)

    if channel_names is None:
        return ["channel_0"]

    if len(channel_names) == 0:
        return ["channel_0"]

    return list(channel_names)


def choose_reference_channel(channel_names):
    """
    Choose the channel used to compute stabilization transforms.

    If AUTO_SELECT_REFERENCE_CHANNEL is True, the first channel not containing
    EXCLUDED_REFERENCE_CHANNEL_TOKEN is used.
    """

    if not AUTO_SELECT_REFERENCE_CHANNEL:
        return REFERENCE_CHANNEL_INDEX

    excluded = EXCLUDED_REFERENCE_CHANNEL_TOKEN.upper()

    for index, channel_name in enumerate(channel_names):
        if excluded not in str(channel_name).upper():
            return index

    return 0


def sanitize_channel_name(channel_name, channel_index):
    """
    Convert ND2 channel name into a filename-safe string.
    """

    if channel_name is None:
        channel_name = f"channel_{channel_index}"

    channel_name = str(channel_name)

    # Replace characters that commonly break filenames
    channel_name = (
        channel_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    if channel_name == "":
        channel_name = f"channel_{channel_index}"

    return channel_name


# =============================================================================
# ND2 PROCESSING
# =============================================================================

def process_nd2_file(file_path):
    """
    Process one ND2 file.

    Steps:
      1. Open ND2 file
      2. Create output folders
      3. Loop through FOVs
      4. Compute stabilization transforms from reference channel
      5. Apply transforms to all channels
      6. Save raw/stabilized stacks and projections
    """

    file_path = Path(file_path)

    print(f"\nProcessing ND2 file:")
    print(f"  {file_path}")

    base_filename = file_path.stem
    base_folder = file_path.parent

    folders = create_output_folders(
        base_folder=base_folder,
        base_filename=base_filename,
    )

    with ND2Reader(str(file_path)) as images:
        available_axes = images.axes
        fields_of_view = get_fields_of_view(images)
        channel_names = get_channel_names(images)
        num_channels = len(channel_names)

        if VERBOSE:
            print(f"Available axes: {available_axes}")
            print(f"Fields of view: {fields_of_view}")
            print(f"Channels: {channel_names}")

        # Configure ND2 reader to iterate through time
        if "t" in available_axes:
            images.iter_axes = "t"
        else:
            images.iter_axes = ""

        images.bundle_axes = "yx"

        for fov in fields_of_view:
            print(f"\nProcessing FOV {fov}")

            if "v" in available_axes:
                images.default_coords["v"] = fov

            reference_channel_index = choose_reference_channel(channel_names)

            if VERBOSE:
                print(f"Reference channel index: {reference_channel_index}")
                print(f"Reference channel name: {channel_names[reference_channel_index]}")

            warp_matrices = None

            # -----------------------------------------------------------------
            # First pass:
            #   Process the reference channel and compute stabilization matrices
            # -----------------------------------------------------------------

            if "c" in available_axes:
                images.default_coords["c"] = reference_channel_index

            reference_stack = np.asarray(images)

            if RUN_STABILIZATION:
                stabilized_reference_stack, warp_matrices = stabilize_stack_with_affine_averaging(
                    reference_stack,
                    iterations=ECC_ITERATIONS,
                    error_tolerance=ECC_ERROR_TOLERANCE,
                    window_size=REFERENCE_WINDOW_SIZE,
                )
            else:
                stabilized_reference_stack = reference_stack
                warp_matrices = []

            # -----------------------------------------------------------------
            # Second pass:
            #   Save each channel using the reference-channel transforms
            # -----------------------------------------------------------------

            for channel_index in range(num_channels):
                if "c" in available_axes:
                    images.default_coords["c"] = channel_index
                    channel_name = channel_names[channel_index]
                else:
                    channel_name = "channel_0"

                safe_channel_name = sanitize_channel_name(channel_name, channel_index)

                stack = np.asarray(images)

                if RUN_STABILIZATION:
                    if channel_index == reference_channel_index:
                        stabilized_stack = stabilized_reference_stack
                    else:
                        stabilized_stack = apply_warp_matrices(
                            stack,
                            warp_matrices,
                        )
                else:
                    stabilized_stack = stack

                channel_filename_base = f"FOV{fov}_{safe_channel_name}_{base_filename}"

                # Save raw stack
                if SAVE_RAW_STACKS:
                    raw_stack_path = folders["analysis"] / f"{channel_filename_base}.tif"
                    save_tiff_stack(raw_stack_path, stack)

                # Save stabilized stack
                if SAVE_STABILIZED_STACKS:
                    stabilized_stack_path = (
                        folders["stabilized"] /
                        f"{channel_filename_base}_stabilized.tif"
                    )
                    save_tiff_stack(stabilized_stack_path, stabilized_stack)

                # Save raw projections
                save_all_projections(
                    stack=stack,
                    folders=folders,
                    channel_filename_base=channel_filename_base,
                    stabilized=False,
                )

                # Save stabilized projections
                save_all_projections(
                    stack=stabilized_stack,
                    folders=folders,
                    channel_filename_base=channel_filename_base,
                    stabilized=True,
                )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def get_nd2_files(input_folder):
    """
    Return ND2 files to process.
    """

    input_folder = Path(input_folder)

    if PROCESS_ALL_ND2_FILES:
        return sorted(input_folder.glob(f"*{ND2_EXTENSION}"))

    single_file = Path(SINGLE_ND2_FILE)

    if not single_file.exists():
        raise FileNotFoundError(f"Single ND2 file does not exist: {single_file}")

    return [single_file]


def process_nd2_files_in_folder(input_folder):
    """
    Process all selected ND2 files.
    """

    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    nd2_files = get_nd2_files(input_folder)

    if len(nd2_files) == 0:
        print("No ND2 files found.")
        return

    print(f"ND2 files found: {len(nd2_files)}")

    for file_path in nd2_files:
        process_nd2_file(file_path)

    print("\nDone.")


# =============================================================================
# RUN
# =============================================================================

process_nd2_files_in_folder(INPUT_FOLDER)


# In[2]:


pip install nd2reader


# In[ ]:




