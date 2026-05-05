#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# MASK + CSV INTENSITY VISUALIZATION PIPELINE
# =============================================================================
#
# Purpose:
#   This notebook cell creates a multi-frame RGB TIFF where each labeled ROI
#   in a mask image is colored according to its corresponding CSV intensity
#   value over time.
#
# Expected input files in TARGET_FOLDER:
#   1. A labeled mask TIFF
#      - Preferred filename pattern: *mask_5pix.tif
#      - Falls back to the first .tif found in the folder
#
#   2. Optional alpha CSV
#      - Filename must contain "alpha" or "Alpha"
#
#   3. Optional beta CSV
#      - Filename must contain "beta" or "Beta"
#
# CSV expectations:
#   - First column = frame/time/index column
#   - Remaining columns = cell/ROI traces
#   - Cell columns should contain numeric IDs, e.g. Cell_1, Cell 1, Cell12
#   - Numeric cell IDs must match integer labels in the mask TIFF
#
# Output:
#   - One RGB multi-frame TIFF saved in TARGET_FOLDER
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.colors as mcolors


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder containing:
#   - labeled mask TIFF
#   - alpha CSV
#   - beta CSV
TARGET_FOLDER = r"YourParentDirectory"

# Preferred mask filename pattern
# The script looks for this first, then falls back to any .tif file.
MASK_PREFERRED_PATTERN = "*mask"

# CSV name matching
ALPHA_CSV_KEYWORD = "alpha"
BETA_CSV_KEYWORD = "beta"

# Output filename suffix
OUTPUT_SUFFIX = "ROIVIZ.tif"

# Percentile scaling for intensity-to-color conversion
# Values at or below PCT_LOW map to the low end of the colormap.
# Values at or above PCT_HIGH map to the high end of the colormap.
PCT_LOW = 0
PCT_HIGH = 100

# Background color for pixels outside labeled ROIs
# Current setting = black
BACKGROUND_VALUE = 0

# TIFF compression
TIFF_COMPRESSION = "zlib"

# Alpha ROI color gradient
# Low intensity -> black
# High intensity -> pink
ALPHA_LOW_RGB = (0, 0, 0)
ALPHA_HIGH_RGB = (236, 0, 140)

# Beta ROI color gradient
# Low intensity -> black
# High intensity -> blue
BETA_LOW_RGB = (0, 0, 0)
BETA_HIGH_RGB = (57, 83, 164)

# If True, print extra status information
VERBOSE = True


# =============================================================================
# COLOR HELPERS
# =============================================================================

def rgb_255_to_01(r, g, b):
    """
    Convert RGB values from 0-255 scale to 0-1 scale.

    Matplotlib colormaps expect RGB values between 0 and 1.
    """

    return (r / 255.0, g / 255.0, b / 255.0)


def build_linear_colormap(name, low_rgb, high_rgb):
    """
    Build a linear RGB colormap from a low color to a high color.

    Parameters
    ----------
    name : str
        Name assigned to the colormap.

    low_rgb : tuple
        RGB tuple on 0-255 scale for low intensity values.

    high_rgb : tuple
        RGB tuple on 0-255 scale for high intensity values.
    """

    return mcolors.LinearSegmentedColormap.from_list(
        name,
        [
            rgb_255_to_01(*low_rgb),
            rgb_255_to_01(*high_rgb),
        ],
    )


# Build alpha and beta colormaps using user-defined RGB values
alpha_gradient = build_linear_colormap(
    name="AlphaGradient",
    low_rgb=ALPHA_LOW_RGB,
    high_rgb=ALPHA_HIGH_RGB,
)

beta_gradient = build_linear_colormap(
    name="BetaGradient",
    low_rgb=BETA_LOW_RGB,
    high_rgb=BETA_HIGH_RGB,
)


def generate_gradient_color(value_norm01, gradient):
    """
    Convert a normalized value into an RGB color.

    Parameters
    ----------
    value_norm01 : float
        Normalized value between 0 and 1.

    gradient : matplotlib.colors.Colormap
        Colormap used to convert the normalized value into RGB.

    Returns
    -------
    numpy.ndarray
        RGB color as uint8 array with values from 0-255.
    """

    value = float(value_norm01)

    # Protect against NaN or infinite values
    if not np.isfinite(value):
        value = 0.0

    # Force value into 0-1 range
    value = np.clip(value, 0.0, 1.0)

    # Convert colormap output from 0-1 RGB to 0-255 RGB
    color_rgb_01 = gradient(value)[:3]
    color_rgb_255 = (np.array(color_rgb_01) * 255).astype(np.uint8)

    return color_rgb_255


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_files_in_folder(folder):
    """
    Find the mask TIFF, alpha CSV, and beta CSV in a target folder.

    Mask selection order:
      1. First file matching MASK_PREFERRED_PATTERN
      2. First .tif file in the folder

    CSV selection:
      - First CSV containing ALPHA_CSV_KEYWORD
      - First CSV containing BETA_CSV_KEYWORD

    Parameters
    ----------
    folder : str or Path
        Folder containing input files.

    Returns
    -------
    tuple
        mask_path, alpha_csv_path, beta_csv_path
    """

    folder = Path(folder)

    # Prefer explicitly named mask files
    preferred_masks = sorted(folder.glob(MASK_PREFERRED_PATTERN))

    # Fall back to any TIFF if no preferred mask is found
    all_tifs = sorted(folder.glob("*.tif"))

    mask_candidates = preferred_masks + [
        path for path in all_tifs if path not in preferred_masks
    ]

    mask_path = mask_candidates[0] if mask_candidates else None

    # Find alpha and beta CSV files
    alpha_csv_path = None
    beta_csv_path = None

    csv_files = sorted(folder.glob("*.csv"))

    for csv_path in csv_files:
        name_lower = csv_path.name.lower()

        if ALPHA_CSV_KEYWORD.lower() in name_lower and alpha_csv_path is None:
            alpha_csv_path = csv_path

        if BETA_CSV_KEYWORD.lower() in name_lower and beta_csv_path is None:
            beta_csv_path = csv_path

    return mask_path, alpha_csv_path, beta_csv_path


# =============================================================================
# CSV HELPERS
# =============================================================================

def safe_read_csv(path):
    """
    Safely read a CSV file.

    Returns None if:
      - path is None
      - file does not exist
      - file cannot be read
      - file has no usable rows or columns
    """

    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        if df.shape[0] == 0 or df.shape[1] == 0:
            return None

        return df

    except Exception as error:
        print(f"Could not read CSV: {path}")
        print(f"Reason: {error}")
        return None


def extract_cell_label_columns(df):
    """
    Rename cell columns so they match mask label IDs.

    Example:
      "Cell_12" -> "12"
      "Cell 12" -> "12"
      "Cell12"  -> "12"

    The first column is assumed to be frame/time/index and is preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Input CSV dataframe.

    Returns
    -------
    tuple
        df_renamed, cell_columns

    df_renamed : pandas.DataFrame
        Copy of the dataframe with cell columns renamed to numeric strings.

    cell_columns : list
        List of cell columns that correspond to numeric ROI labels.
    """

    renamed_columns = []

    for col in df.columns:

        # Rename columns containing "Cell" and a number
        if isinstance(col, str) and "Cell" in col:
            match = re.search(r"(\d+)", col)
            renamed_columns.append(match.group(1) if match else col)

        else:
            renamed_columns.append(col)

    df_renamed = df.copy()
    df_renamed.columns = renamed_columns

    # Cell columns are numeric-label columns after the first column
    cell_columns = [
        col
        for col in df_renamed.columns[1:]
        if isinstance(col, str) and col.isdigit()
    ]

    return df_renamed, cell_columns


def compute_percentile_bounds(df, cell_columns, pct_low, pct_high):
    """
    Compute lower and upper intensity bounds using percentiles.

    The bounds are calculated across all selected cell columns.

    These bounds are used to normalize values to 0-1 before applying
    the RGB colormap.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing cell intensity values.

    cell_columns : list
        Columns to use for percentile scaling.

    pct_low : float
        Lower percentile.

    pct_high : float
        Upper percentile.

    Returns
    -------
    tuple
        vmin, vmax
    """

    if not cell_columns:
        return 0.0, 1.0

    # Flatten all cell values into one array
    values = pd.to_numeric(
        df[cell_columns].values.flatten(),
        errors="coerce",
    )

    # Keep only finite values
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    vmin = np.percentile(values, pct_low)
    vmax = np.percentile(values, pct_high)

    # Safety checks
    if not np.isfinite(vmin):
        vmin = 0.0

    if not np.isfinite(vmax):
        vmax = 1.0

    # Prevent divide-by-zero during normalization
    if vmax <= vmin:
        vmax = vmin + 1e-9

    return float(vmin), float(vmax)


# =============================================================================
# MASK HELPERS
# =============================================================================

def cache_label_pixels(mask_image):
    """
    Precompute pixel coordinates for each labeled ROI in the mask.

    This avoids repeatedly scanning the mask image for each frame.

    Parameters
    ----------
    mask_image : numpy.ndarray
        2D labeled mask image where:
          - 0 = background
          - positive integers = ROI labels

    Returns
    -------
    dict
        Dictionary mapping label string -> pixel coordinates.

        Example:
          {
              "1": (row_indices, col_indices),
              "2": (row_indices, col_indices)
          }
    """

    label_pixels = {}

    labels = np.unique(mask_image)

    for label in labels:

        # Skip background
        if label == 0:
            continue

        rows, cols = np.nonzero(mask_image == label)

        if rows.size > 0:
            label_pixels[str(int(label))] = (rows, cols)

    return label_pixels


# =============================================================================
# FRAME PAINTING
# =============================================================================

def paint_frames_from_csv(
    csv_df,
    cell_columns,
    label_pixels,
    processed_frames,
    gradient,
    vmin,
    vmax,
):
    """
    Paint each ROI in each frame based on its CSV intensity value.

    Parameters
    ----------
    csv_df : pandas.DataFrame
        Dataframe containing frame-by-frame cell intensity values.

    cell_columns : list
        Numeric cell label columns to paint.

    label_pixels : dict
        Mapping of ROI labels to pixel locations.

    processed_frames : list
        List of RGB image frames that will be modified in place.

    gradient : matplotlib.colors.Colormap
        Colormap used for this cell class.

    vmin : float
        Lower intensity bound.

    vmax : float
        Upper intensity bound.
    """

    n_frames = min(csv_df.shape[0], len(processed_frames))

    for frame_idx in range(n_frames):

        frame = processed_frames[frame_idx]

        for cell_label in cell_columns:

            # Skip if this CSV cell label does not exist in the mask
            pixels = label_pixels.get(cell_label)

            if pixels is None:
                continue

            try:
                raw_value = csv_df.at[frame_idx, cell_label]

                # Normalize value to 0-1 using percentile bounds
                value_norm = (raw_value - vmin) / (vmax - vmin)

                # Convert normalized value to RGB
                color = generate_gradient_color(value_norm, gradient)

                rows, cols = pixels

                # Paint all pixels belonging to this ROI
                frame[rows, cols, 0] = color[0]
                frame[rows, cols, 1] = color[1]
                frame[rows, cols, 2] = color[2]

            except Exception as error:
                print(
                    f"Skipping cell {cell_label} at frame {frame_idx}. "
                    f"Reason: {error}"
                )


# =============================================================================
# TIFF OUTPUT
# =============================================================================

def save_rgb_tiff(processed_frames, output_path):
    """
    Save RGB frames as a compressed multi-frame TIFF.

    Parameters
    ----------
    processed_frames : list
        List of RGB uint8 image frames.

    output_path : str or Path
        Output TIFF path.
    """

    output_path = Path(output_path)

    output_array = np.stack(processed_frames, axis=0).astype(np.uint8)

    tiff.imwrite(
        output_path,
        output_array,
        photometric="rgb",
        compression=TIFF_COMPRESSION,
    )

    print(f"Saved: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_folder(target_folder):
    """
    Run the full visualization pipeline for one folder.

    Steps:
      1. Find mask TIFF, alpha CSV, and beta CSV
      2. Load the mask
      3. Cache ROI pixel coordinates
      4. Load available CSV files
      5. Create empty RGB frames
      6. Paint beta ROIs
      7. Paint alpha ROIs
      8. Save output TIFF
    """

    target_folder = Path(target_folder)

    if VERBOSE:
        print(f"Target folder: {target_folder}")

    if not target_folder.exists():
        raise FileNotFoundError(f"Target folder does not exist: {target_folder}")

    # -------------------------------------------------------------------------
    # Find input files
    # -------------------------------------------------------------------------

    mask_path, alpha_csv_path, beta_csv_path = find_files_in_folder(target_folder)

    if mask_path is None:
        print("No mask TIFF found.")
        return

    if VERBOSE:
        print(f"Mask:  {mask_path.name}")
        print(f"Alpha: {alpha_csv_path.name if alpha_csv_path else '(none found)'}")
        print(f"Beta:  {beta_csv_path.name if beta_csv_path else '(none found)'}")

    # -------------------------------------------------------------------------
    # Load mask
    # -------------------------------------------------------------------------

    mask_image = tiff.imread(mask_path)

    # This script expects one 2D labeled mask, not a time series
    if mask_image.ndim != 2:
        raise ValueError(
            "Mask image must be a single 2D labeled image with shape (H, W)."
        )

    # Ensure mask labels are integer values
    if not np.issubdtype(mask_image.dtype, np.integer):
        mask_image = mask_image.astype(np.int32)

    height, width = mask_image.shape

    # -------------------------------------------------------------------------
    # Cache label pixel locations
    # -------------------------------------------------------------------------

    label_pixels = cache_label_pixels(mask_image)

    if not label_pixels:
        print("No non-zero labels found in mask. Nothing to paint.")

    # -------------------------------------------------------------------------
    # Load CSV files
    # -------------------------------------------------------------------------

    alpha_df = safe_read_csv(alpha_csv_path)
    beta_df = safe_read_csv(beta_csv_path)

    # If no usable CSVs are found, save a one-frame black image and stop
    if alpha_df is None and beta_df is None:
        print("No usable alpha or beta CSV found. Saving empty black frame.")

        processed_frames = [
            np.zeros((height, width, 3), dtype=np.uint8)
        ]

        output_path = target_folder / f"{mask_path.stem}{OUTPUT_SUFFIX}"
        save_rgb_tiff(processed_frames, output_path)
        return

    # -------------------------------------------------------------------------
    # Determine number of frames
    # -------------------------------------------------------------------------

    n_alpha = alpha_df.shape[0] if alpha_df is not None else 0
    n_beta = beta_df.shape[0] if beta_df is not None else 0

    num_frames = max(n_alpha, n_beta)

    if num_frames == 0:
        num_frames = 1

    if VERBOSE:
        print(f"Output frames: {num_frames}")

    # -------------------------------------------------------------------------
    # Create empty RGB output frames
    # -------------------------------------------------------------------------

    processed_frames = [
        np.full(
            (height, width, 3),
            fill_value=BACKGROUND_VALUE,
            dtype=np.uint8,
        )
        for _ in range(num_frames)
    ]

    # -------------------------------------------------------------------------
    # Paint beta ROIs first
    # -------------------------------------------------------------------------
    # If alpha and beta masks overlap, alpha will overwrite beta because
    # alpha is painted after beta.
    # -------------------------------------------------------------------------

    if beta_df is not None:
        beta_df, beta_cells = extract_cell_label_columns(beta_df)

        beta_vmin, beta_vmax = compute_percentile_bounds(
            beta_df,
            beta_cells,
            PCT_LOW,
            PCT_HIGH,
        )

        if VERBOSE:
            print(f"Beta cells matched in CSV: {len(beta_cells)}")
            print(f"Beta scaling range: {beta_vmin:.4f} to {beta_vmax:.4f}")

        paint_frames_from_csv(
            csv_df=beta_df,
            cell_columns=beta_cells,
            label_pixels=label_pixels,
            processed_frames=processed_frames,
            gradient=beta_gradient,
            vmin=beta_vmin,
            vmax=beta_vmax,
        )

    # -------------------------------------------------------------------------
    # Paint alpha ROIs second
    # -------------------------------------------------------------------------

    if alpha_df is not None:
        alpha_df, alpha_cells = extract_cell_label_columns(alpha_df)

        alpha_vmin, alpha_vmax = compute_percentile_bounds(
            alpha_df,
            alpha_cells,
            PCT_LOW,
            PCT_HIGH,
        )

        if VERBOSE:
            print(f"Alpha cells matched in CSV: {len(alpha_cells)}")
            print(f"Alpha scaling range: {alpha_vmin:.4f} to {alpha_vmax:.4f}")

        paint_frames_from_csv(
            csv_df=alpha_df,
            cell_columns=alpha_cells,
            label_pixels=label_pixels,
            processed_frames=processed_frames,
            gradient=alpha_gradient,
            vmin=alpha_vmin,
            vmax=alpha_vmax,
        )

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------

    output_path = target_folder / f"{mask_path.stem}{OUTPUT_SUFFIX}"

    save_rgb_tiff(processed_frames, output_path)


# =============================================================================
# RUN
# =============================================================================

process_folder(TARGET_FOLDER)


# In[ ]:




