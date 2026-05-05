#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import re
import numpy as np

# ============================================================
# CONFIG
# ============================================================
#Use this code to calculate FRET ratios of CFP and YFP Channels
#Run this after splitting, stabilizing, segmenting and quantifying cell ROIS
#insert the path to directory with CSVs below
parent_directory = r'YourDirectoryHere'

# Choose numerator and denominator for the regular ratio
# Options: "CFP" or "YFP"
REGULAR_RATIO_NUMERATOR = "CFP"
REGULAR_RATIO_DENOMINATOR = "YFP"

# Choose numerator for the normalized ratio:
# CFP/(CFP+YFP) or YFP/(CFP+YFP)
# Options: "CFP" or "YFP"
NORMALIZED_RATIO_NUMERATOR = "CFP"

# Avoid division-by-zero problems
REPLACE_ZERO_DENOMINATOR_WITH_NAN = True

# ============================================================
# FUNCTIONS
# ============================================================

def safe_divide(numerator_df, denominator_df):
    numerator = numerator_df.copy()
    denominator = denominator_df.copy()

    if REPLACE_ZERO_DENOMINATOR_WITH_NAN:
        denominator = denominator.replace(0, np.nan)

    return numerator / denominator


def process_files_recursively(target_folder):
    for root, _, files in os.walk(target_folder):

        cfp_files = {}
        yfp_files = {}

        for file in files:
            if file.endswith(".csv"):
                match = re.search(r"(CFP|YFP)(\d+)", file)
                if match:
                    prefix, number = match.groups()

                    if prefix == "CFP":
                        cfp_files[number] = os.path.join(root, file)
                    elif prefix == "YFP":
                        yfp_files[number] = os.path.join(root, file)

        for number in cfp_files:
            if number not in yfp_files:
                print(f"No matching YFP file for CFP{number}.csv in {root}, skipping.")
                continue

            cfp_file = cfp_files[number]
            yfp_file = yfp_files[number]

            cfp_df = pd.read_csv(cfp_file)
            yfp_df = pd.read_csv(yfp_file)

            if cfp_df.shape != yfp_df.shape:
                print(f"Shape mismatch between {os.path.basename(cfp_file)} and {os.path.basename(yfp_file)}, skipping.")
                continue

            # Keep first column unchanged, assumed x-axis/frame/time
            regular_result_df = cfp_df.copy()
            normalized_result_df = cfp_df.copy()

            cfp_values = cfp_df.iloc[:, 1:]
            yfp_values = yfp_df.iloc[:, 1:]

            channel_map = {
                "CFP": cfp_values,
                "YFP": yfp_values
            }

            # -----------------------------
            # Regular ratio: numerator / denominator
            # -----------------------------
            regular_numerator = channel_map[REGULAR_RATIO_NUMERATOR]
            regular_denominator = channel_map[REGULAR_RATIO_DENOMINATOR]

            regular_result_df.iloc[:, 1:] = safe_divide(
                regular_numerator,
                regular_denominator
            )

            regular_output_filename = (
                f"FRET_{REGULAR_RATIO_NUMERATOR}_over_"
                f"{REGULAR_RATIO_DENOMINATOR}_{number}.csv"
            )

            regular_output_path = os.path.join(root, regular_output_filename)
            regular_result_df.to_csv(regular_output_path, index=False)
            print(f"Saved regular ratio: {regular_output_path}")

            # -----------------------------
            # Normalized ratio: numerator / (CFP + YFP)
            # -----------------------------
            normalized_numerator = channel_map[NORMALIZED_RATIO_NUMERATOR]
            normalized_denominator = cfp_values + yfp_values

            normalized_result_df.iloc[:, 1:] = safe_divide(
                normalized_numerator,
                normalized_denominator
            )

            normalized_output_filename = (
                f"FRET_normalized_{NORMALIZED_RATIO_NUMERATOR}_over_"
                f"CFP_plus_YFP_{number}.csv"
            )

            normalized_output_path = os.path.join(root, normalized_output_filename)
            normalized_result_df.to_csv(normalized_output_path, index=False)
            print(f"Saved normalized ratio: {normalized_output_path}")


# ============================================================
# RUN
# ============================================================

process_files_recursively(parent_directory)


# In[ ]:




