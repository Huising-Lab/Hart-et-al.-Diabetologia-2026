#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# RECURSIVE CSV NORMALIZATION PIPELINE
# =============================================================================
#
# Purpose:
#   Recursively search through a master folder, find matching CSV files,
#   normalize the first column to a user-defined range, normalize all remaining
#   columns to 0-1, and save a new normalized CSV beside each input file.
#
# Input CSV structure:
#   - First column = independent variable, e.g. time/frame
#   - Remaining columns = traces/signals/ROI measurements
#
# Output:
#   - normalized_<original_filename>.csv
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# USER SETTINGS
# =============================================================================

# Parent folder to recursively search
MASTER_FOLDER = r"YourDirectoryHere"

# Process only CSV files whose names start with this text
TARGET_FILE_PREFIX = "FileStartswithThis"

# Prefix added to output files
OUTPUT_PREFIX = "Normalized"

# Normalization range for the first column
# Typically used for time/frame normalization
# (StartTime, EndTime)
FIRST_COLUMN_RANGE = (0, 10)

# Normalization range for all remaining columns
DATA_COLUMN_RANGE = (0, 1)

# If True, columns containing NaN or infinite values are removed before normalization
DROP_COLUMNS_WITH_ANY_NAN_OR_INF = True

# If True, skip files that already start with OUTPUT_PREFIX
SKIP_ALREADY_NORMALIZED = True

# If True, print progress messages
VERBOSE = True


# =============================================================================
# NORMALIZATION FUNCTION
# =============================================================================

def normalize_dataframe(data):
    """
    Normalize a dataframe.

    The first column is normalized separately from the remaining columns.

    First column:
        normalized to FIRST_COLUMN_RANGE

    Remaining columns:
        normalized to DATA_COLUMN_RANGE

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe.
    """

    data = data.copy()

    # Do not process empty dataframes
    if data.empty:
        return data

    # -------------------------------------------------------------------------
    # Normalize first column
    # -------------------------------------------------------------------------
    # The first column is treated as the independent variable, commonly frame
    # number or time. It is scaled to FIRST_COLUMN_RANGE.
    # -------------------------------------------------------------------------

    first_col = data.columns[0]

    first_col_values = pd.to_numeric(data[first_col], errors="coerce")

    if first_col_values.notna().sum() > 0:
        scaler = MinMaxScaler(feature_range=FIRST_COLUMN_RANGE)

        data[first_col] = scaler.fit_transform(
            first_col_values.values.reshape(-1, 1)
        )

    # -------------------------------------------------------------------------
    # Normalize remaining columns
    # -------------------------------------------------------------------------
    # All other columns are treated as measured signal traces and are scaled
    # independently to DATA_COLUMN_RANGE.
    # -------------------------------------------------------------------------

    if data.shape[1] > 1:
        signal_cols = data.columns[1:]

        # Force signal columns to numeric before normalization
        data[signal_cols] = data[signal_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        scaler = MinMaxScaler(feature_range=DATA_COLUMN_RANGE)

        data[signal_cols] = scaler.fit_transform(data[signal_cols])

    return data


# =============================================================================
# FILE CLEANING FUNCTION
# =============================================================================

def clean_dataframe_before_normalization(data):
    """
    Prepare a dataframe for normalization.

    Infinite values are converted to NaN. Depending on user settings,
    columns containing NaN or infinite values are removed.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw input dataframe.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.
    """

    data = data.copy()

    # Convert positive/negative infinity to NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    if DROP_COLUMNS_WITH_ANY_NAN_OR_INF:
        # Remove any column containing at least one NaN
        data = data.dropna(axis=1, how="any")

    return data


# =============================================================================
# SINGLE-FILE PROCESSING FUNCTION
# =============================================================================

def process_single_csv(file_path):
    """
    Load, clean, normalize, and save one CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    """

    file_path = Path(file_path)

    if VERBOSE:
        print(f"\nProcessing: {file_path}")

    # Read input CSV
    data = pd.read_csv(file_path)

    # Remove columns containing invalid values, if enabled
    data = clean_dataframe_before_normalization(data)

    # Skip if cleaning removed all usable data
    if data.empty or data.shape[1] == 0:
        print(f"Skipping empty file after cleaning: {file_path}")
        return

    # Normalize data
    normalized_data = normalize_dataframe(data)

    # Save beside original file
    output_path = file_path.parent / f"{OUTPUT_PREFIX}{file_path.name}"
    normalized_data.to_csv(output_path, index=False)

    if VERBOSE:
        print(f"Saved normalized CSV: {output_path}")


# =============================================================================
# FOLDER PROCESSING FUNCTION
# =============================================================================

def process_folder(folder_path):
    """
    Process all matching CSV files in one folder.

    A file is processed only if:
      - it ends with .csv
      - it starts with TARGET_FILE_PREFIX
      - it does not already start with OUTPUT_PREFIX, if skipping is enabled
    """

    folder_path = Path(folder_path)

    for file_path in sorted(folder_path.glob("*.csv")):

        filename = file_path.name

        if not filename.startswith(TARGET_FILE_PREFIX):
            continue

        if SKIP_ALREADY_NORMALIZED and filename.startswith(OUTPUT_PREFIX):
            continue

        process_single_csv(file_path)


# =============================================================================
# RECURSIVE BATCH PROCESSING FUNCTION
# =============================================================================

def recursive_process_csv_files(master_folder):
    """
    Recursively walk through a master folder and normalize matching CSV files.
    """

    master_folder = Path(master_folder)

    if not master_folder.exists():
        raise FileNotFoundError(f"Master folder does not exist: {master_folder}")

    files_processed = 0

    for root, _, _ in os.walk(master_folder):
        root = Path(root)

        matching_files = [
            file_path
            for file_path in sorted(root.glob("*.csv"))
            if file_path.name.startswith(TARGET_FILE_PREFIX)
        ]

        if SKIP_ALREADY_NORMALIZED:
            matching_files = [
                file_path
                for file_path in matching_files
                if not file_path.name.startswith(OUTPUT_PREFIX)
            ]

        for file_path in matching_files:
            process_single_csv(file_path)
            files_processed += 1

    print(f"\nDone. Files processed: {files_processed}")


# =============================================================================
# RUN PIPELINE
# =============================================================================

recursive_process_csv_files(MASTER_FOLDER)


# In[ ]:




