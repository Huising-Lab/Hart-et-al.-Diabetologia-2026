#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# RECURSIVE AUC + HORIZONTAL REORGANIZATION PIPELINE
# =============================================================================
#
# Purpose:
#   This notebook cell recursively searches for target CSV files, calculates
#   statistics across user-defined baseline/test windows, saves the results,
#   and then automatically reorganizes the output into a horizontal format.
#
# Input CSV structure:
#   - First column = independent variable, e.g. time or frame
#   - Remaining columns = cell traces / ROI traces / replicate signals
#
# Outputs:
#   1. Vertical statistics CSV
#   2. Horizontal reorganized statistics CSV
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simpson


# =============================================================================
# USER SETTINGS
# =============================================================================

# Root folder to recursively search
ROOT_FOLDER = r"YourDirectoryHere"

# Target file-matching rule
# Files must start with this text and end with .csv
TARGET_FILE_PREFIX = "FileStartswithThis"

# Output folder created inside each folder containing a target CSV
OUTPUT_FOLDER_NAME = "Analysis"

# Output filename suffixes
VERTICAL_OUTPUT_SUFFIX = "Analysis.csv"
HORIZONTAL_OUTPUT_SUFFIX = "_Horizontal.csv"

# Baseline/test range pairs
# Format:
#   "Label": ((baseline_start, baseline_end), (test_start, test_end))
RANGE_DATA = {
    "10mM Glucose": ((0, 10), (10, 35)),
    "1mM Glutamine": ((20, 35), (35, 50)),
    "1mM Alanine": ((50, 60), (60, 75)),
    "1mM Arginine": ((75, 85), (85, 100)),
}

# If True, missing/invalid columns are reported
VERBOSE = True


# =============================================================================
# STATISTICS FUNCTION
# =============================================================================

def calculate_statistics(df, baseline_range, test_range, label):
    """
    Calculate mean, percent change, normalized AUC, and normalized AUC percent
    change for each data column in a CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe. First column is treated as the independent variable.

    baseline_range : tuple
        Inclusive range in the first column used as the baseline window.

    test_range : tuple
        Inclusive range in the first column used as the test window.

    label : str
        Name assigned to this baseline/test comparison.

    Returns
    -------
    dict
        Nested dictionary of statistics for each column.
    """

    df = df.copy()

    # Convert the first column to numeric.
    # Any non-numeric values are forced to NaN and removed.
    x_col = df.columns[0]
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col])

    # Select rows within the baseline and test ranges.
    baseline_data = df[
        (df[x_col] >= baseline_range[0]) &
        (df[x_col] <= baseline_range[1])
    ]

    test_data = df[
        (df[x_col] >= test_range[0]) &
        (df[x_col] <= test_range[1])
    ]

    stats = {}

    # Loop through every trace column.
    for column in df.columns[1:]:

        baseline_values = pd.to_numeric(
            baseline_data[column],
            errors="coerce"
        ).dropna()

        test_values = pd.to_numeric(
            test_data[column],
            errors="coerce"
        ).dropna()

        # Skip columns that do not contain usable data in both ranges.
        if baseline_values.empty or test_values.empty:
            if VERBOSE:
                print(
                    f"Warning: Skipping column '{column}' for '{label}' "
                    "because baseline or test data are missing."
                )
            continue

        # Calculate mean response in each range.
        baseline_mean = baseline_values.mean()
        test_mean = test_values.mean()

        # Percent change based on means.
        percent_change = (
            ((test_mean - baseline_mean) / baseline_mean) * 100
            if baseline_mean != 0
            else np.nan
        )

        # Get corresponding x-values for AUC calculation.
        baseline_x = baseline_data.loc[baseline_values.index, x_col]
        test_x = test_data.loc[test_values.index, x_col]

        # Simpson AUC using actual x-axis values.
        baseline_auc = simpson(y=baseline_values, x=baseline_x)
        test_auc = simpson(y=test_values, x=test_x)

        # Normalize AUC by number of valid data points.
        normalized_baseline_auc = baseline_auc / len(baseline_values)
        normalized_test_auc = test_auc / len(test_values)

        # Percent change based on normalized AUC.
        normalized_auc_percent_change = (
            (
                (normalized_test_auc - normalized_baseline_auc) /
                normalized_baseline_auc
            ) * 100
            if normalized_baseline_auc != 0
            else np.nan
        )

        # Store results.
        # Tuple key allows later conversion into Cell + Range columns.
        stats[(column, label)] = {
            "Baseline Mean": baseline_mean,
            "Test Mean": test_mean,
            "Percent Change": percent_change,
            "Normalized Baseline AUC": normalized_baseline_auc,
            "Normalized Test AUC": normalized_test_auc,
            "Normalized AUC Percent Change": normalized_auc_percent_change,
        }

    return stats


# =============================================================================
# OUTPUT REORGANIZATION FUNCTION
# =============================================================================

def reorganize_vertical_to_horizontal(vertical_df):
    """
    Convert vertical statistics output into a horizontal format.

    Input vertical structure:
        Cell | Range | Baseline Mean | Test Mean | ...

    Output horizontal structure:
        Cell | 10mM Glucose_Baseline Mean | 10mM Glucose_Test Mean | ...

    Parameters
    ----------
    vertical_df : pandas.DataFrame
        Statistics dataframe produced by calculate_statistics.

    Returns
    -------
    pandas.DataFrame
        Horizontally reorganized dataframe.
    """

    df = vertical_df.copy()

    # The vertical statistics dataframe has a MultiIndex after pd.DataFrame(stats).T.
    # Resetting index converts it into regular columns.
    df = df.reset_index()

    # Rename index columns to meaningful labels.
    df = df.rename(
        columns={
            "level_0": "Cell",
            "level_1": "Range",
        }
    )

    # Pivot so each Cell gets one row and each Range/statistic combination
    # gets its own column.
    horizontal_df = df.pivot(index="Cell", columns="Range")

    # Flatten MultiIndex columns.
    # Example:
    #   ("Baseline Mean", "10mM Glucose") -> "10mM Glucose_Baseline Mean"
    horizontal_df.columns = [
        f"{range_label}_{stat_name}"
        for stat_name, range_label in horizontal_df.columns
    ]

    # Restore Cell as a normal column.
    horizontal_df = horizontal_df.reset_index()

    return horizontal_df


# =============================================================================
# SINGLE-FILE PROCESSING FUNCTION
# =============================================================================

def process_single_csv(file_path):
    """
    Process one CSV file:
      1. Calculate statistics for all defined ranges
      2. Save vertical output
      3. Reorganize to horizontal format
      4. Save horizontal output
    """

    file_path = Path(file_path)

    if VERBOSE:
        print(f"\nProcessing: {file_path}")

    # Load input CSV.
    df = pd.read_csv(file_path)

    # Create output folder beside the input CSV.
    output_folder = file_path.parent / OUTPUT_FOLDER_NAME
    output_folder.mkdir(exist_ok=True)

    # Collect all statistics across all user-defined ranges.
    combined_results = pd.DataFrame()

    for range_label, (baseline_range, test_range) in RANGE_DATA.items():

        stats = calculate_statistics(
            df=df,
            baseline_range=baseline_range,
            test_range=test_range,
            label=range_label,
        )

        range_results = pd.DataFrame(stats).T

        combined_results = pd.concat(
            [combined_results, range_results],
            axis=0,
        )

    # Build output paths.
    base_name = file_path.stem

    vertical_output_path = output_folder / f"{base_name}{VERTICAL_OUTPUT_SUFFIX}"
    horizontal_output_path = output_folder / f"{base_name}{HORIZONTAL_OUTPUT_SUFFIX}"

    # Save vertical results.
    combined_results.to_csv(vertical_output_path, index=True)

    # Convert vertical results to horizontal format.
    horizontal_results = reorganize_vertical_to_horizontal(combined_results)

    # Save horizontal results.
    horizontal_results.to_csv(horizontal_output_path, index=False)

    if VERBOSE:
        print(f"Saved vertical output:   {vertical_output_path}")
        print(f"Saved horizontal output: {horizontal_output_path}")


# =============================================================================
# RECURSIVE BATCH PROCESSING FUNCTION
# =============================================================================

def process_csv_files_recursively(root_folder):
    """
    Recursively search a root folder and process all matching CSV files.

    A file is processed if:
      - it ends with .csv
      - it starts with TARGET_FILE_PREFIX
    """

    root_folder = Path(root_folder)

    if not root_folder.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")

    files_processed = 0

    for dirpath, _, files in os.walk(root_folder):
        dirpath = Path(dirpath)

        for filename in files:

            if not filename.lower().endswith(".csv"):
                continue

            if not filename.startswith(TARGET_FILE_PREFIX):
                continue

            file_path = dirpath / filename
            process_single_csv(file_path)

            files_processed += 1

    print(f"\nDone. Files processed: {files_processed}")


# =============================================================================
# RUN PIPELINE
# =============================================================================

process_csv_files_recursively(ROOT_FOLDER)


# In[ ]:




