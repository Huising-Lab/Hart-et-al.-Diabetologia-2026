#!/usr/bin/env python
# coding: utf-8

# In[3]:


# =============================================================================
# CONFIGURATION (EDIT THESE VARIABLES ONLY)
# =============================================================================

import os
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

# Root folder containing all subdirectories to process
ROOT_FOLDER = r"Yourdirectoryhere"

# File filtering rules
# - File must contain KEYWORD
# - File must start with REQUIRED_FILENAME_START
KEYWORD = ""
REQUIRED_FILENAME_START = ""

# Define ranges (based on FIRST COLUMN values)
# These define the two groups being compared
TEST_RANGE = (StartTime, EndTime)        # "experimental" window
STANDARD_RANGE = (StartTime, EndTime)    # "baseline/control" window

# Statistical threshold for significance
SIGNIFICANCE_LEVEL = 0.005

# T-test assumption
# False = Welch's t-test (recommended; does NOT assume equal variance)
# True  = Student's t-test (assumes equal variance)
ASSUME_EQUAL_VARIANCE = False

# Output labeling
SIGNIFICANT_DECREASE_LABEL = "Alpha"
SIGNIFICANT_INCREASE_LABEL = "Check"
INSIGNIFICANT_LABEL = "Beta"


# =============================================================================
# HELPER FUNCTION: SAVE SELECTED COLUMNS
# =============================================================================

def save_sorted_columns(df, x_col, subdir, filename, keyword, columns, label):
    """
    Save a subset of columns to a new CSV file.

    Always preserves the first column (independent variable),
    then appends only the selected columns.
    """

    # If no columns matched this category, skip writing file
    if not columns:
        return

    # Build output dataframe: first column + selected columns
    output_df = df[[x_col] + columns].copy()

    # Construct output filename
    output_file = subdir / f"{keyword}_{label}_{filename}"

    # Save CSV
    output_df.to_csv(output_file, index=False)

    print(f"Saved {label}: {output_file}")


# =============================================================================
# MAIN FUNCTION: RECURSIVE T-TEST PROCESSING
# =============================================================================

def process_csvs_ttest(root_folder_path):
    """
    Recursively scans directories for CSV files and performs
    column-wise t-tests between two defined ranges.

    Each column is categorized into:
    - Significant decrease
    - Significant increase
    - Not significant
    """

    root_folder_path = Path(root_folder_path)

    # Validate input directory
    if not root_folder_path.exists():
        raise FileNotFoundError(f"Invalid root folder: {root_folder_path}")

    # Walk through all subdirectories
    for subdir, _, files in os.walk(root_folder_path):
        subdir = Path(subdir)

        for filename in files:

            # -----------------------------------------------------------------
            # FILE FILTERING
            # -----------------------------------------------------------------
            # Only process:
            # - CSV files
            # - Containing keyword
            # - Matching naming convention
            # -----------------------------------------------------------------

            if not filename.lower().endswith(".csv"):
                continue

            if KEYWORD not in filename:
                continue

            if not filename.startswith(REQUIRED_FILENAME_START):
                continue

            file_path = subdir / filename
            print(f"\nProcessing: {file_path}")

            # -----------------------------------------------------------------
            # LOAD DATA
            # -----------------------------------------------------------------

            df = pd.read_csv(file_path)

            # Require at least 2 columns:
            # 1st column = independent variable
            # Remaining = traces/replicates
            if df.shape[1] < 2:
                print("Skipping (not enough columns)")
                continue

            # First column = independent variable (e.g., time/frame)
            x_col = df.columns[0]
            x_vals = df.iloc[:, 0]

            # Remaining columns = independent signals/traces
            analysis_df = df.iloc[:, 1:]

            # -----------------------------------------------------------------
            # BUILD RANGE MASKS
            # -----------------------------------------------------------------
            # These define which rows belong to each comparison group
            # -----------------------------------------------------------------

            test_mask = (x_vals >= TEST_RANGE[0]) & (x_vals <= TEST_RANGE[1])
            standard_mask = (x_vals >= STANDARD_RANGE[0]) & (x_vals <= STANDARD_RANGE[1])

            test_data = analysis_df.loc[test_mask]
            standard_data = analysis_df.loc[standard_mask]

            # Skip if either group has no data
            if test_data.empty or standard_data.empty:
                print("Skipping (empty test or standard range)")
                continue

            # -----------------------------------------------------------------
            # PREPARE OUTPUT CONTAINERS
            # -----------------------------------------------------------------

            dec_cols = []     # test < standard (significant decrease)
            inc_cols = []     # test > standard (significant increase)
            insig_cols = []   # no significant difference

            # -----------------------------------------------------------------
            # COLUMN-WISE T-TEST LOOP
            # -----------------------------------------------------------------
            # Each column is treated independently
            # -----------------------------------------------------------------

            for col in analysis_df.columns:

                # Extract values and remove NaNs
                t_vals = test_data[col].dropna()
                s_vals = standard_data[col].dropna()

                # Require minimum data points for t-test
                if len(t_vals) < 2 or len(s_vals) < 2:
                    insig_cols.append(col)
                    continue

                # Perform t-test
                stat, p = ttest_ind(
                    t_vals,
                    s_vals,
                    equal_var=ASSUME_EQUAL_VARIANCE,
                    nan_policy="omit"
                )

                # Compute means for directionality
                t_mean = t_vals.mean()
                s_mean = s_vals.mean()

                # -----------------------------------------------------------------
                # CLASSIFICATION LOGIC
                # -----------------------------------------------------------------
                # Uses BOTH:
                # - statistical significance (p-value)
                # - direction of effect (mean comparison)
                # -----------------------------------------------------------------

                if p <= SIGNIFICANCE_LEVEL and t_mean < s_mean:
                    dec_cols.append(col)

                elif p <= SIGNIFICANCE_LEVEL and t_mean > s_mean:
                    inc_cols.append(col)

                else:
                    insig_cols.append(col)

            # -----------------------------------------------------------------
            # SAVE OUTPUT FILES
            # -----------------------------------------------------------------

            save_sorted_columns(df, x_col, subdir, filename, KEYWORD, dec_cols, SIGNIFICANT_DECREASE_LABEL)
            save_sorted_columns(df, x_col, subdir, filename, KEYWORD, inc_cols, SIGNIFICANT_INCREASE_LABEL)
            save_sorted_columns(df, x_col, subdir, filename, KEYWORD, insig_cols, INSIGNIFICANT_LABEL)


# =============================================================================
# EXECUTION
# =============================================================================

process_csvs_ttest(ROOT_FOLDER)


# In[ ]:




