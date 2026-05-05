#!/usr/bin/env python
# coding: utf-8

# In[2]:


# =============================================================================
# BATCH ROI CONTAMINATION-CORRECTION PIPELINE
# =============================================================================
#
# Purpose:
#   Recursively match calcium-imaging movie TIFFs, labeled mask TIFFs, alpha CSVs,
#   and beta CSVs, then estimate and subtract cross-contamination between
#   nearby alpha and beta ROI traces.
#
# Main use case:
#   - Remove beta-like signal from alpha ROIs
#   - Optionally remove alpha-like signal from beta ROIs
#   - Export raw, corrected, removed-contamination, fit-summary, and
#     contamination-percentage CSV files
#
# Expected inputs:
#   1. Movie TIFF:
#        - 3D stack with shape (T, Y, X)
#
#   2. Mask TIFF:
#        - 2D labeled mask with shape (Y, X)
#        - 0 = background
#        - positive integers = ROI labels
#
#   3. Alpha CSV:
#        - headers contain numeric ROI IDs matching the mask
#
#   4. Beta CSV:
#        - headers contain numeric ROI IDs matching the mask
#
# Output:
#   - A short output folder is created beside each matched movie:
#       CleanedCSVBatch/FOV<#>__<hash>/
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tifffile


# =============================================================================
# USER SETTINGS: FILE DISCOVERY
# =============================================================================

# Parent folder to search
BASE_DIR = Path(r"YourDirectoryHere")

# If True, search through all subfolders under BASE_DIR
# If False, search only BASE_DIR itself
RECURSIVE = True

# Movie file-matching keyword
# The movie filename must contain this text
MOVIE_KEYWORD = "_stabilized.tif"

# Mask file-matching keyword
# The mask filename must contain this text
MASK_KEYWORD = "_mask"

# Alpha and beta CSV file-matching tokens
ALPHA_TOKEN = "Alpha"
BETA_TOKEN = "Beta"

# File extensions
MOVIE_EXTS = (".tif", ".tiff")
MASK_EXTS = (".tif", ".tiff")
CSV_EXT = ".csv"

# Output folder created next to each movie
OUTPUT_ROOT_NAME = "CleanedCSVBatch"


# =============================================================================
# USER SETTINGS: MATCHING RULES
# =============================================================================

# Extract FOV from filenames like:
#   FOV0
#   FOV_0
#   FOV-12
#   FOV 12
FOV_RE = re.compile(r"FOV[_\- ]*(\d+)", re.IGNORECASE)

# Extract all integers from a filename.
# The final integer is used for CSV matching.
ALL_INTS_RE = re.compile(r"(\d+)")

# CSV matching rule:
#   CSV number = FOV + CSV_NUMBER_OFFSET
#
# Example:
#   Movie FOV0 matches CSV ending in 1
#   Movie FOV1 matches CSV ending in 2
CSV_NUMBER_OFFSET = 1


# =============================================================================
# USER SETTINGS: CORRECTION DIRECTION
# =============================================================================

# Remove beta contamination from alpha ROIs
DO_BETA_TO_ALPHA_CLEAN = True

# Remove alpha contamination from beta ROIs
DO_ALPHA_TO_BETA_CLEAN = False

# Save contamination-only traces, i.e. the signal that was subtracted
SAVE_REMOVED_TRACES = True


# =============================================================================
# USER SETTINGS: CONTAMINATION PERCENT OUTPUTS
# =============================================================================

# Write percent contamination per-frame CSV
WRITE_CONTAM_PCT_CSV = True

# Percent contamination type:
#   "abs"    = absolute removed signal / denominator
#   "signed" = signed removed signal / denominator
CONTAM_PCT_KIND = "abs"

# Denominator for contamination percentage:
#   "raw"     = removed / raw
#   "cleaned" = removed / cleaned
CONTAM_PCT_DENOM = "raw"

# Small value used to avoid divide-by-zero
CONTAM_PCT_EPS = 1e-9

# Write one CSV containing both absolute and signed contamination percentages
WRITE_CONTAM_PCT_BOTH_CSV = True


# =============================================================================
# USER SETTINGS: SIGNAL ATTRIBUTION OUTPUTS
# =============================================================================

# Write per-frame attribution fraction CSV
WRITE_SIGNAL_ATTRIB_FRACTION_PER_FRAME_CSV = True

# Write per-cell summary attribution CSV
WRITE_SIGNAL_ATTRIB_SUMMARY_PER_CELL_CSV = True

# If True, only positive removed signal is treated as contaminating signal
ATTRIB_POSITIVE_ONLY = True

# If True, attribution percentages are clipped between 0 and 100%
ATTRIB_CLIP_0_1 = True

# Small value used to avoid divide-by-zero
ATTRIB_EPS = 1e-9


# =============================================================================
# USER SETTINGS: ANALYSIS WINDOWS
# =============================================================================

# Baseline window used for baseline subtraction and optional fitting
# Python-style interval: [BASELINE_START, BASELINE_END)
BASELINE_START = 0
BASELINE_END = 90

# Window where beta-driven signal is expected
BETA_WIN_START = 97
BETA_WIN_END = 134

# Window where alpha-driven signal is expected
ALPHA_WIN_START = 680
ALPHA_WIN_END = 699


# =============================================================================
# USER SETTINGS: MODEL / FITTING
# =============================================================================

# Number of nearest source cells used to build contamination reference
TOP_K_SOURCES = 5

# Distance weighting exponent
# Higher values weight nearby cells more strongly
DIST_POWER = 2.0

# Minimum distance used during weighting to avoid singular weights
MIN_DIST = 2.0

# Ridge penalty for linear fitting
RIDGE_LAMBDA = 1e-3

# Correction mode:
#   "fluct" = subtract k * (source_reference - source_reference_baseline)
#   "full"  = subtract intercept + k * source_reference
CORRECTION_MODE = "fluct"

# Used only in "full" mode
FIT_INTERCEPT = True

# Fitting window for beta-to-alpha correction:
#   "baseline" = fit during baseline window
#   "beta"     = fit during beta response window
K_FIT_WINDOW_B2A = "beta"

# Fitting window for alpha-to-beta correction:
#   "baseline" = fit during baseline window
#   "alpha"    = fit during alpha response window
K_FIT_WINDOW_A2B = "alpha"


# =============================================================================
# USER SETTINGS: OPTIONAL COMPOSITE TIFF OUTPUTS
# =============================================================================

# Master switch for composite TIFF creation
MAKE_CONSOLIDATED_COMPOSITES = False

# Which ROI class to export as composites
MAKE_ALPHA_COMPOSITES = False
MAKE_BETA_COMPOSITES = False

# Composite frame range
# COMPOSITE_FRAME_END = None means use the end of the movie
COMPOSITE_FRAME_START = 0
COMPOSITE_FRAME_END = None

# Individual composite outputs
WRITE_COMPOSITE_RAW_ROI_ONLY = False
WRITE_COMPOSITE_PRED_CONTAM_ONLY = False
WRITE_COMPOSITE_RAW_MINUS_CONTAM = False
WRITE_COMPOSITE_CLEANED_ROI_ONLY = False

# Visualization outputs
WRITE_2CH_HYPERSTACK_ROI = False
WRITE_2CH_HYPERSTACK_FULLFIELD = False
WRITE_RGB_OVERLAY_ROI = False
WRITE_RGB_OVERLAY_FULLFIELD = False

# Export full-field predicted contamination stack
WRITE_FULLFIELD_CONTAM_ONLY = True

# RGB overlay scaling for contamination channel
RGB_CONTAM_GAIN = 2.0

# Composite output format
OUTPUT_UINT16 = True

# If True, negative cleaned composite pixel values are clipped to zero
CLEANED_CLIP_NONNEG = True

# Global multiplier for predicted contamination composite
COMPOSITE_SCALE = 1.0

# Spatial field settings for predicted contamination visualization
SIGMA_PIX = 12.0
FIELD_FLOOR = 1e-6


# =============================================================================
# USER SETTINGS: GENERAL
# =============================================================================

# If True, ask before running matched datasets
ASK_BEFORE_RUNNING = True

# If True, print detailed progress messages
VERBOSE = True

# Header parser used to extract cell IDs from CSV column names
ANY_INT_RE = re.compile(r"(\d+)")


# =============================================================================
# WINDOWS LONG-PATH AND SAFE I/O HELPERS
# =============================================================================

def as_long_path(path):
    """
    Convert a path to a Windows extended-length path.

    This helps avoid Windows MAX_PATH failures when saving deeply nested files.

    Examples
    --------
    C:\\folder\\file.tif -> \\\\?\\C:\\folder\\file.tif
    """

    path = Path(path)
    path_str = str(path)

    if os.name != "nt":
        return path_str

    if path_str.startswith("\\\\?\\"):
        return path_str

    if not path.is_absolute():
        path = path.resolve()
        path_str = str(path)

    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")

    return "\\\\?\\" + path_str


def ensure_dir(path):
    """
    Create a directory if it does not already exist.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def df_to_csv_safe(df, path, **kwargs):
    """
    Save a dataframe using long-path-safe file handling.
    """

    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(as_long_path(path), index=False, **kwargs)


def tiff_write_safe(path, array, **kwargs):
    """
    Save a TIFF using long-path-safe file handling.
    """

    path = Path(path)
    ensure_dir(path.parent)
    tifffile.imwrite(as_long_path(path), array, **kwargs)


def short_dataset_tag(movie_path):
    """
    Create a short, stable output-folder name.

    This keeps output paths short and reproducible.

    Example
    -------
    FOV0__9a31b2c1
    """

    movie_path = Path(movie_path)

    fov = parse_fov(movie_path)
    fov_part = f"FOV{fov}" if fov is not None else "FOVx"

    digest = hashlib.md5(str(movie_path).encode("utf-8")).hexdigest()[:8]

    return f"{fov_part}__{digest}"


# =============================================================================
# FILE DISCOVERY AND MATCHING
# =============================================================================

def rglob_exts(base, extensions):
    """
    Yield files matching a set of extensions.

    Searches recursively if RECURSIVE is True.
    """

    base = Path(base)

    iterator = base.rglob("*") if RECURSIVE else base.glob("*")

    for path in iterator:
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def parse_fov(path):
    """
    Parse FOV number from a filename.

    Returns None if no FOV token is found.
    """

    match = FOV_RE.search(Path(path).stem)
    return int(match.group(1)) if match else None


def last_int_anywhere(path):
    """
    Return the last integer found anywhere in a filename stem.

    Used for CSV number matching.
    """

    nums = ALL_INTS_RE.findall(Path(path).stem)
    return int(nums[-1]) if nums else None


def pick_best(candidates, movie_parent):
    """
    Pick the best candidate file from a list.

    Preference order:
      1. Candidate in the same folder as the movie
      2. Candidate with the shallowest path
    """

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda path: (path.parent != movie_parent, len(path.parts)),
    )

    return candidates[0]


def find_pairs(base_dir):
    """
    Find matched movie, mask, alpha CSV, and beta CSV sets.

    Matching logic:
      - Movie filename contains MOVIE_KEYWORD
      - Mask filename contains MASK_KEYWORD
      - Alpha CSV filename contains ALPHA_TOKEN
      - Beta CSV filename contains BETA_TOKEN
      - FOV is parsed from movie and mask filenames
      - CSV number is matched as FOV + CSV_NUMBER_OFFSET
    """

    base_dir = Path(base_dir)

    movies = [
        path
        for path in rglob_exts(base_dir, MOVIE_EXTS)
        if MOVIE_KEYWORD.lower() in path.name.lower()
    ]

    masks = [
        path
        for path in rglob_exts(base_dir, MASK_EXTS)
        if MASK_KEYWORD.lower() in path.name.lower()
    ]

    csvs = (
        list(base_dir.rglob(f"*{CSV_EXT}"))
        if RECURSIVE
        else list(base_dir.glob(f"*{CSV_EXT}"))
    )

    alpha_token = ALPHA_TOKEN.lower()
    beta_token = BETA_TOKEN.lower()

    # Index masks by FOV
    masks_by_fov = {}

    for mask_path in masks:
        fov = parse_fov(mask_path)

        if fov is not None:
            masks_by_fov.setdefault(fov, []).append(mask_path)

    # Index CSVs by their last integer and class token
    alpha_by_number = {}
    beta_by_number = {}

    for csv_path in csvs:
        name_lower = csv_path.name.lower()

        is_alpha = alpha_token in name_lower
        is_beta = beta_token in name_lower

        if not (is_alpha or is_beta):
            continue

        csv_number = last_int_anywhere(csv_path)

        if csv_number is None:
            continue

        if is_alpha:
            alpha_by_number.setdefault(csv_number, []).append(csv_path)

        if is_beta:
            beta_by_number.setdefault(csv_number, []).append(csv_path)

    matched = []
    skipped = []

    for movie_path in sorted(movies):
        fov = parse_fov(movie_path)

        if fov is None:
            skipped.append((movie_path, "no FOV parsed from filename"))
            continue

        csv_number = fov + CSV_NUMBER_OFFSET

        mask_path = pick_best(
            masks_by_fov.get(fov, []),
            movie_path.parent,
        )

        alpha_csv = pick_best(
            alpha_by_number.get(csv_number, []),
            movie_path.parent,
        )

        beta_csv = pick_best(
            beta_by_number.get(csv_number, []),
            movie_path.parent,
        )

        if mask_path is None or alpha_csv is None or beta_csv is None:
            skipped.append(
                (
                    movie_path,
                    (
                        "missing match "
                        f"(mask={mask_path is not None}, "
                        f"alpha={alpha_csv is not None}, "
                        f"beta={beta_csv is not None})"
                    ),
                )
            )
            continue

        matched.append((movie_path, mask_path, alpha_csv, beta_csv))

    return matched, skipped


def print_matches(matched, skipped, max_skipped=80):
    """
    Print matched datasets and skipped movie files.
    """

    print("\n================= MATCHED DATASETS =================")

    if not matched:
        print("None found.")

    else:
        for i, (movie_path, mask_path, alpha_csv, beta_csv) in enumerate(
            matched,
            start=1,
        ):
            fov = parse_fov(movie_path)
            expected_csv_number = (
                fov + CSV_NUMBER_OFFSET
                if fov is not None
                else None
            )

            print(f"\n[{i}] FOV={fov} | expected CSV number={expected_csv_number}")
            print(f"  movie: {movie_path}")
            print(f"  mask : {mask_path}")
            print(f"  alpha: {alpha_csv}")
            print(f"  beta : {beta_csv}")

    print("\n================== SKIPPED MOVIES ==================")

    if not skipped:
        print("None skipped.")

    else:
        for movie_path, reason in skipped[:max_skipped]:
            print(f"- {movie_path} -> {reason}")

        if len(skipped) > max_skipped:
            print(f"... plus {len(skipped) - max_skipped} more skipped")


def prompt_yes_no(message="Proceed with these matches? [Y/N]: "):
    """
    Prompt user before running the batch.
    """

    while True:
        answer = input(message).strip().lower()

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter Y or N.")


def build_out_dir(movie_path):
    """
    Build output directory for one movie dataset.

    Output folder is created next to the movie.
    """

    movie_path = Path(movie_path)

    output_dir = movie_path.parent / OUTPUT_ROOT_NAME / short_dataset_tag(movie_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# =============================================================================
# TRACE AND FITTING HELPERS
# =============================================================================

def parse_cell_ids_from_headers(headers, label="CSV"):
    """
    Extract numeric cell IDs from CSV headers.

    Example
    -------
    Cell_12 -> 12
    Track 5 -> 5

    Duplicate numeric IDs are reported but not fatal.
    """

    ids = []
    header_map = {}

    for header in headers:
        header_str = str(header).strip()
        match = ANY_INT_RE.search(header_str)

        if match:
            cell_id = int(match.group(1))
            ids.append(cell_id)
            header_map.setdefault(cell_id, []).append(header_str)

    counts = Counter(ids)

    duplicates = {
        cell_id: count
        for cell_id, count in counts.items()
        if count > 1
    }

    if duplicates:
        print(f"\nWARNING: duplicate numeric IDs detected in {label}:")

        for cell_id, count in duplicates.items():
            print(f"  ID {cell_id} appears {count} times:")

            for col in header_map[cell_id]:
                print(f"    - {col}")

        print("Proceeding anyway.\n")

    return sorted(set(ids))


def compute_centroid(mask, cell_id):
    """
    Compute the centroid of one labeled ROI.

    Returns
    -------
    tuple or None
        (x, y) centroid coordinates.
    """

    ys, xs = np.where(mask == cell_id)

    if xs.size == 0:
        return None

    return float(xs.mean()), float(ys.mean())


def extract_trace(movie, mask, cell_id):
    """
    Extract the mean intensity trace for one labeled ROI.

    Parameters
    ----------
    movie : numpy.ndarray
        Movie stack with shape (T, Y, X).

    mask : numpy.ndarray
        2D labeled mask with shape (Y, X).

    cell_id : int
        ROI label to extract.

    Returns
    -------
    numpy.ndarray or None
        Mean intensity trace over time.
    """

    roi_pixels = mask == cell_id

    if not np.any(roi_pixels):
        return None

    return movie[:, roi_pixels].mean(axis=1)


def corrcoef_safe(a, b):
    """
    Safely compute Pearson correlation coefficient.

    Returns NaN if vectors are too short or have zero variance.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    if a.size < 2 or b.size < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def fit_k_ridge(y, x, lam=1e-3, fit_intercept=True):
    """
    Fit y = intercept + k*x using ridge regression.

    The ridge penalty is applied only to k, not the intercept.
    """

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if fit_intercept:
        X = np.column_stack([np.ones_like(x), x])

        XtX = X.T @ X

        ridge_matrix = np.array(
            [
                [0.0, 0.0],
                [0.0, lam],
            ]
        )

        beta = np.linalg.solve(XtX + ridge_matrix, X.T @ y)

        intercept = beta[0]
        k = beta[1]

    else:
        denominator = (x @ x) + lam
        k = float((x @ y) / denominator) if denominator != 0 else 0.0
        intercept = 0.0

    return float(intercept), float(k)


def fit_k_from_window_demeaned(y, x, window_start, window_end, lam=1e-3):
    """
    Fit contamination scale factor k using demeaned traces within a window.

    This estimates shared fluctuations rather than absolute offsets.
    """

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    window_start = int(max(0, window_start))
    window_end = int(min(len(y), window_end))

    if window_end <= window_start + 2:
        return 0.0

    y_window = y[window_start:window_end]
    x_window = x[window_start:window_end]

    y_demeaned = y_window - y_window.mean()
    x_demeaned = x_window - x_window.mean()

    denominator = (x_demeaned @ x_demeaned) + lam

    k = (
        float((x_demeaned @ y_demeaned) / denominator)
        if denominator != 0
        else 0.0
    )

    return k


# =============================================================================
# SPATIAL FIELD HELPERS FOR OPTIONAL COMPOSITES
# =============================================================================

def gaussian_falloff(distance, sigma):
    """
    Gaussian spatial falloff based on distance.
    """

    return np.exp(-(distance ** 2) / (2.0 * sigma ** 2))


def make_distance_map(shape, x0, y0):
    """
    Make a distance map from a point coordinate.
    """

    yy, xx = np.indices(shape)

    return np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)


def build_source_field(mask, source_ids_near, centroids, sigma_pix=12.0):
    """
    Build a spatial field representing influence from nearby source ROIs.
    """

    field = np.zeros(mask.shape, dtype=np.float32)

    for source_id in source_ids_near:
        if source_id not in centroids:
            continue

        sx, sy = centroids[source_id]

        distance_map = make_distance_map(mask.shape, sx, sy)

        field += gaussian_falloff(distance_map, sigma_pix).astype(np.float32)

    max_value = float(field.max())

    if max_value < FIELD_FLOOR:
        max_value = FIELD_FLOOR

    field /= max_value

    return field


def to_u16(stack):
    """
    Clip a stack to uint16 range.
    """

    return np.clip(stack, 0, 65535).astype(np.uint16)


# =============================================================================
# CONTAMINATION PERCENTAGE HELPERS
# =============================================================================

def percent_from_removed(raw, removed, kind="abs", denom_mode="raw", eps=1e-9):
    """
    Convert removed contamination signal into percent contamination.

    Parameters
    ----------
    raw : array-like
        Raw trace.

    removed : array-like
        Removed contamination trace.

    kind : str
        "abs" or "signed".

    denom_mode : str
        "raw" or "cleaned".
    """

    raw = raw.astype(np.float64)
    removed = removed.astype(np.float64)

    if kind == "abs":
        numerator = np.abs(removed)
        raw_base = np.abs(raw)

    elif kind == "signed":
        numerator = removed
        raw_base = raw

    else:
        raise ValueError("kind must be 'abs' or 'signed'")

    if denom_mode == "raw":
        denominator = raw_base

    elif denom_mode == "cleaned":
        denominator = (
            raw_base - numerator
            if kind == "abs"
            else raw - removed
        )

    else:
        raise ValueError("denom_mode must be 'raw' or 'cleaned'")

    denominator = np.where(np.abs(denominator) < eps, np.nan, denominator)

    return 100.0 * (numerator / denominator)


def compute_contam_percent_df(
    raw_df,
    removed_df,
    kind="abs",
    denom_mode="raw",
    label_prefix="Cell_",
):
    """
    Compute one contamination-percent dataframe.
    """

    if "Frame" not in raw_df.columns or "Frame" not in removed_df.columns:
        raise ValueError("raw_df and removed_df must include 'Frame'")

    out = pd.DataFrame({"Frame": raw_df["Frame"].values})

    cell_cols = [
        col
        for col in raw_df.columns
        if col != "Frame" and str(col).startswith(label_prefix)
    ]

    if len(cell_cols) == 0:
        return out

    percent_matrix = []

    for col in cell_cols:
        raw = raw_df[col].to_numpy(dtype=np.float64)
        removed = removed_df[col].to_numpy(dtype=np.float64)

        pct = percent_from_removed(
            raw,
            removed,
            kind=kind,
            denom_mode=denom_mode,
            eps=CONTAM_PCT_EPS,
        )

        out[col] = pct
        percent_matrix.append(pct)

    percent_matrix = np.vstack(percent_matrix)

    out["Mean_%"] = np.nanmean(percent_matrix, axis=0)

    return out


def compute_contam_percent_both_df(raw_df, removed_df, label_prefix="Cell_"):
    """
    Compute both absolute and signed contamination percentages.
    """

    if "Frame" not in raw_df.columns or "Frame" not in removed_df.columns:
        raise ValueError("raw_df and removed_df must include 'Frame'")

    out = pd.DataFrame({"Frame": raw_df["Frame"].values})

    cell_cols = [
        col
        for col in raw_df.columns
        if col != "Frame" and str(col).startswith(label_prefix)
    ]

    if len(cell_cols) == 0:
        return out

    abs_matrix = []
    signed_matrix = []

    for col in cell_cols:
        raw = raw_df[col].to_numpy(dtype=np.float64)
        removed = removed_df[col].to_numpy(dtype=np.float64)

        pct_abs = percent_from_removed(
            raw,
            removed,
            kind="abs",
            denom_mode=CONTAM_PCT_DENOM,
            eps=CONTAM_PCT_EPS,
        )

        pct_signed = percent_from_removed(
            raw,
            removed,
            kind="signed",
            denom_mode=CONTAM_PCT_DENOM,
            eps=CONTAM_PCT_EPS,
        )

        out[f"{col}__pct_abs"] = pct_abs
        out[f"{col}__pct_signed"] = pct_signed

        abs_matrix.append(pct_abs)
        signed_matrix.append(pct_signed)

    abs_matrix = np.vstack(abs_matrix)
    signed_matrix = np.vstack(signed_matrix)

    out["Mean_abs_%"] = np.nanmean(abs_matrix, axis=0)
    out["Mean_signed_%"] = np.nanmean(signed_matrix, axis=0)

    return out


# =============================================================================
# SIGNAL ATTRIBUTION HELPERS
# =============================================================================

def compute_signal_attribution_fraction_per_frame_df(
    raw_df,
    removed_df,
    baseline_start,
    baseline_end,
    positive_only=True,
    clip_0_1=True,
    eps=1e-9,
    label_prefix="Cell_",
):
    """
    Estimate what fraction of baseline-subtracted signal is attributable to
    removed contamination on a per-frame basis.
    """

    if "Frame" not in raw_df.columns or "Frame" not in removed_df.columns:
        raise ValueError("raw_df and removed_df must include 'Frame'")

    out = pd.DataFrame({"Frame": raw_df["Frame"].values})

    cell_cols = [
        col
        for col in raw_df.columns
        if col != "Frame" and str(col).startswith(label_prefix)
    ]

    if len(cell_cols) == 0:
        return out

    frac_matrix = []

    b0 = int(max(0, baseline_start))
    b1 = int(min(len(out), baseline_end))

    if b1 <= b0 + 1:
        raise ValueError("Baseline window too small for attribution fraction.")

    for col in cell_cols:
        raw = raw_df[col].to_numpy(dtype=np.float64)
        removed = removed_df[col].to_numpy(dtype=np.float64)

        baseline = float(np.mean(raw[b0:b1]))
        signal = raw - baseline

        contaminating_signal = removed.copy()

        if positive_only:
            contaminating_signal = np.clip(contaminating_signal, 0, None)

        denominator = np.maximum(signal, eps)
        fraction = contaminating_signal / denominator

        if clip_0_1:
            fraction = np.clip(fraction, 0, 1)

        percent = 100.0 * fraction

        out[col] = percent
        frac_matrix.append(percent)

    frac_matrix = np.vstack(frac_matrix)

    out["Mean_%"] = np.nanmean(frac_matrix, axis=0)

    return out


def compute_signal_attribution_summary_per_cell_df(
    raw_df,
    removed_df,
    baseline_start,
    baseline_end,
    windows_dict,
    positive_only=True,
    clip_0_1=True,
    eps=1e-9,
    label_prefix="Cell_",
):
    """
    Summarize attribution fraction by cell across defined windows.
    """

    cell_cols = [
        col
        for col in raw_df.columns
        if col != "Frame" and str(col).startswith(label_prefix)
    ]

    if len(cell_cols) == 0:
        return pd.DataFrame()

    n_frames = len(raw_df)

    b0 = int(max(0, baseline_start))
    b1 = int(min(n_frames, baseline_end))

    if b1 <= b0 + 1:
        raise ValueError("Baseline window too small for summary attribution.")

    rows = []

    for col in cell_cols:
        raw = raw_df[col].to_numpy(dtype=np.float64)
        removed = removed_df[col].to_numpy(dtype=np.float64)

        baseline = float(np.mean(raw[b0:b1]))
        signal = raw - baseline

        contaminating_signal = removed.copy()

        if positive_only:
            contaminating_signal = np.clip(contaminating_signal, 0, None)

        cell_row = {
            "Cell": col,
            "baseline_mean": baseline,
        }

        for window_name, (window_start, window_end) in windows_dict.items():
            w0 = int(max(0, window_start))
            w1 = int(min(n_frames, window_end))

            if w1 <= w0 + 1:
                continue

            signal_window = signal[w0:w1]
            contam_window = contaminating_signal[w0:w1]

            denominator = np.maximum(signal_window, eps)

            fraction = contam_window / denominator

            if clip_0_1:
                fraction = np.clip(fraction, 0, 1)

            cell_row[f"mean_signal_{window_name}"] = float(np.mean(signal_window))
            cell_row[f"mean_contam_like_{window_name}"] = float(np.mean(contam_window))
            cell_row[f"mean_attrib_pct_{window_name}"] = float(100.0 * np.mean(fraction))
            cell_row[f"frac_frames_signal_pos_{window_name}"] = float(np.mean(signal_window > eps))

        rows.append(cell_row)

    return pd.DataFrame(rows)


# =============================================================================
# OPTIONAL VISUALIZATION HELPERS
# =============================================================================

def normalize_to_u8(array, eps=1e-9):
    """
    Normalize array to uint8 for RGB visualization.
    """

    array = array.astype(np.float32)

    max_value = float(np.nanmax(array))

    if max_value < eps:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = np.clip(array / max_value, 0, 1)

    return (255.0 * normalized).astype(np.uint8)


def write_2ch_hyperstack_tif(path, channel_1, channel_2):
    """
    Save two stacks as a two-channel ImageJ hyperstack.

    Output axes:
        T, C, Y, X
    """

    ch1 = to_u16(channel_1) if OUTPUT_UINT16 else channel_1.astype(np.float32)
    ch2 = to_u16(channel_2) if OUTPUT_UINT16 else channel_2.astype(np.float32)

    stack = np.stack([ch1, ch2], axis=1)

    tiff_write_safe(
        path,
        stack,
        imagej=True,
        metadata={"axes": "TCYX"},
    )


def write_rgb_overlay_tif(path, raw_stack, contam_stack, contam_gain=2.0):
    """
    Save RGB overlay of raw signal plus contamination prediction.

    Red channel:
        raw + contamination

    Green/blue channels:
        raw only
    """

    raw8 = normalize_to_u8(raw_stack)
    contam8 = normalize_to_u8(np.clip(contam_gain * contam_stack, 0, None))

    rgb = np.zeros(
        (
            raw8.shape[0],
            raw8.shape[1],
            raw8.shape[2],
            3,
        ),
        dtype=np.uint8,
    )

    rgb[..., 0] = np.clip(raw8 + contam8, 0, 255)
    rgb[..., 1] = raw8
    rgb[..., 2] = raw8

    tiff_write_safe(
        path,
        rgb,
        imagej=True,
        metadata={"axes": "TYXS"},
    )


# =============================================================================
# CORE DATASET PROCESSOR
# =============================================================================

def process_one_dataset(movie_path, mask_path, alpha_csv, beta_csv, out_dir):
    """
    Process one matched movie/mask/alpha/beta dataset.
    """

    movie_path = Path(movie_path)
    mask_path = Path(mask_path)
    alpha_csv = Path(alpha_csv)
    beta_csv = Path(beta_csv)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load inputs
    # -------------------------------------------------------------------------

    movie = tifffile.imread(as_long_path(movie_path))
    mask = tifffile.imread(as_long_path(mask_path))

    if movie.ndim != 3:
        raise ValueError(f"Movie must be 3D (T, Y, X). Got shape {movie.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D (Y, X). Got shape {mask.shape}")

    if movie.shape[1:] != mask.shape:
        raise ValueError(f"Movie XY {movie.shape[1:]} does not match mask XY {mask.shape}")

    alpha_df = pd.read_csv(as_long_path(alpha_csv))
    beta_df = pd.read_csv(as_long_path(beta_csv))

    # -------------------------------------------------------------------------
    # Parse IDs and keep only IDs present in the mask
    # -------------------------------------------------------------------------

    alpha_ids = parse_cell_ids_from_headers(
        alpha_df.columns,
        label=f"ALPHA CSV: {alpha_csv.name}",
    )

    beta_ids = parse_cell_ids_from_headers(
        beta_df.columns,
        label=f"BETA CSV: {beta_csv.name}",
    )

    mask_ids = set(np.unique(mask).astype(int).tolist())
    mask_ids.discard(0)

    alpha_ids = [cell_id for cell_id in alpha_ids if cell_id in mask_ids]
    beta_ids = [cell_id for cell_id in beta_ids if cell_id in mask_ids]

    print("\n=== DATASET ===")
    print(f"Movie: {movie_path}")
    print(f"Mask : {mask_path}")
    print(f"Alpha: {alpha_csv}")
    print(f"Beta : {beta_csv}")
    print(f"OUT  : {out_dir}")
    print(f"Parsed alpha IDs in mask: {len(alpha_ids)}")
    print(f"Parsed beta IDs in mask:  {len(beta_ids)}")
    print(f"Correction mode: {CORRECTION_MODE}")
    print(f"Baseline window: [{BASELINE_START}, {BASELINE_END})")
    print(f"Beta window:     [{BETA_WIN_START}, {BETA_WIN_END})")
    print(f"Alpha window:    [{ALPHA_WIN_START}, {ALPHA_WIN_END})")

    if len(alpha_ids) == 0:
        raise ValueError("No alpha IDs found that exist in the mask.")

    if len(beta_ids) == 0:
        raise ValueError("No beta IDs found that exist in the mask.")

    # -------------------------------------------------------------------------
    # Precompute centroids
    # -------------------------------------------------------------------------

    centroids = {}

    for cell_id in set(alpha_ids + beta_ids):
        centroid = compute_centroid(mask, cell_id)

        if centroid is not None:
            centroids[cell_id] = centroid

    def distance_between_cells(cell_a, cell_b):
        """
        Euclidean distance between two ROI centroids.
        """

        ax, ay = centroids[cell_a]
        bx, by = centroids[cell_b]

        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    # -------------------------------------------------------------------------
    # Extract traces
    # -------------------------------------------------------------------------

    alpha_traces = {}

    for alpha_id in alpha_ids:
        trace = extract_trace(movie, mask, alpha_id)

        if trace is not None:
            alpha_traces[alpha_id] = trace

    beta_traces = {}

    for beta_id in beta_ids:
        trace = extract_trace(movie, mask, beta_id)

        if trace is not None:
            beta_traces[beta_id] = trace

    # -------------------------------------------------------------------------
    # Nested helper: run contamination correction in one direction
    # -------------------------------------------------------------------------

    def run_contam_correction(
        target_ids,
        source_ids,
        target_traces,
        source_traces,
        target_label="alpha",
        source_label="beta",
        k_fit_window="baseline",
        custom_fit_window=None,
        baseline_window=(0, 100),
        top_k=5,
        dist_power=2.0,
        min_dist=2.0,
        ridge_lambda=1e-3,
        correction_mode="fluct",
        fit_intercept=True,
    ):
        """
        Estimate and subtract source-like contamination from target ROI traces.

        For each target ROI:
          1. Find nearest source ROIs
          2. Build a distance-weighted source reference trace
          3. Fit scale factor k
          4. Subtract predicted source contamination from target trace
        """

        n_frames = movie.shape[0]
        frames = np.arange(n_frames)

        raw_df = pd.DataFrame({"Frame": frames})
        corrected_df = pd.DataFrame({"Frame": frames})
        removed_df = pd.DataFrame({"Frame": frames})

        fit_rows = []
        fit_info = {}

        baseline_start, baseline_end = baseline_window

        for target_id in target_ids:
            if target_id not in centroids:
                continue

            if target_id not in target_traces:
                continue

            target_trace = target_traces[target_id].astype(np.float64)

            # Find nearest source cells with valid centroids and traces
            distances = [
                (source_id, distance_between_cells(target_id, source_id))
                for source_id in source_ids
                if source_id in centroids and source_id in source_traces
            ]

            distances.sort(key=lambda item: item[1])

            nearest_sources = distances[:min(top_k, len(distances))]

            # If no source cells exist, leave trace unchanged
            if len(nearest_sources) == 0:
                raw_df[f"Cell_{target_id}"] = target_trace
                corrected_df[f"Cell_{target_id}"] = target_trace
                removed_df[f"Cell_{target_id}"] = np.zeros_like(target_trace)
                continue

            source_ids_near = [source_id for source_id, _ in nearest_sources]

            source_distances = np.array(
                [distance for _, distance in nearest_sources],
                dtype=float,
            )

            # Distance-weighted source reference
            weights = 1.0 / np.maximum(source_distances, min_dist) ** dist_power
            weights = weights / weights.sum()

            source_reference = np.zeros_like(target_trace, dtype=np.float64)

            for weight, source_id in zip(weights, source_ids_near):
                source_reference += weight * source_traces[source_id].astype(np.float64)

            source_reference_baseline = float(
                np.mean(source_reference[baseline_start:baseline_end])
            )

            # Select fitting window
            if k_fit_window.lower() == "baseline":
                fit_start = baseline_start
                fit_end = baseline_end
                fit_window_name = "baseline"

            else:
                if custom_fit_window is None:
                    raise ValueError(
                        "custom_fit_window must be provided when "
                        "k_fit_window is not 'baseline'"
                    )

                fit_start, fit_end = custom_fit_window
                fit_window_name = k_fit_window.lower()

            # Fit k using demeaned traces inside selected window
            k = fit_k_from_window_demeaned(
                target_trace,
                source_reference,
                fit_start,
                fit_end,
                lam=ridge_lambda,
            )

            # Apply correction
            if correction_mode.lower() == "full":
                intercept, k_full = fit_k_ridge(
                    target_trace,
                    source_reference,
                    lam=ridge_lambda,
                    fit_intercept=fit_intercept,
                )

                k = k_full

                removed = (
                    intercept + k * source_reference
                    if fit_intercept
                    else k * source_reference
                )

                corrected = target_trace - removed

            else:
                intercept = 0.0

                removed = k * (
                    source_reference - source_reference_baseline
                )

                corrected = target_trace - removed

            raw_df[f"Cell_{target_id}"] = target_trace
            corrected_df[f"Cell_{target_id}"] = corrected
            removed_df[f"Cell_{target_id}"] = removed

            corr_before = corrcoef_safe(target_trace, source_reference)
            corr_after = corrcoef_safe(corrected, source_reference)

            fit_rows.append(
                {
                    "Cell": f"Cell_{target_id}",
                    "target_id": target_id,
                    "target_label": target_label,
                    "source_label": source_label,
                    "k": float(k),
                    "intercept": float(intercept),
                    "k_fit_window": fit_window_name,
                    "fit_start": int(fit_start),
                    "fit_end": int(fit_end),
                    "baseline_start": int(baseline_start),
                    "baseline_end": int(baseline_end),
                    "nearest_source_ids": ",".join(
                        [f"Cell_{source_id}" for source_id in source_ids_near]
                    ),
                    "nearest_source_dist_mean": float(source_distances.mean()),
                    "corr_with_ref_before": corr_before,
                    "corr_with_ref_after": corr_after,
                }
            )

            fit_info[target_id] = {
                "intercept": float(intercept),
                "k": float(k),
                "source_ids_near": source_ids_near,
                "ref_trace": source_reference.astype(np.float64),
                "ref0": float(source_reference_baseline),
            }

        fit_df = pd.DataFrame(fit_rows)

        return raw_df, corrected_df, removed_df, fit_df, fit_info

    # -------------------------------------------------------------------------
    # Run beta-to-alpha correction
    # -------------------------------------------------------------------------

    alpha_raw_df = None
    alpha_corr_df = None
    alpha_removed_df = None
    alpha_fit_df = None
    alpha_fit_info = {}

    if DO_BETA_TO_ALPHA_CLEAN:
        print("\n=== Running BETA -> ALPHA contamination correction ===")

        if K_FIT_WINDOW_B2A.lower() == "baseline":
            k_fit_window = "baseline"
            custom_window = None

        elif K_FIT_WINDOW_B2A.lower() == "beta":
            k_fit_window = "beta"
            custom_window = (BETA_WIN_START, BETA_WIN_END)

        else:
            raise ValueError("K_FIT_WINDOW_B2A must be 'baseline' or 'beta'")

        (
            alpha_raw_df,
            alpha_corr_df,
            alpha_removed_df,
            alpha_fit_df,
            alpha_fit_info,
        ) = run_contam_correction(
            target_ids=alpha_ids,
            source_ids=beta_ids,
            target_traces=alpha_traces,
            source_traces=beta_traces,
            target_label="alpha",
            source_label="beta",
            k_fit_window=k_fit_window,
            custom_fit_window=custom_window,
            baseline_window=(BASELINE_START, BASELINE_END),
            top_k=TOP_K_SOURCES,
            dist_power=DIST_POWER,
            min_dist=MIN_DIST,
            ridge_lambda=RIDGE_LAMBDA,
            correction_mode=CORRECTION_MODE,
            fit_intercept=FIT_INTERCEPT,
        )

        alpha_out = out_dir / "alpha_outputs"
        alpha_out.mkdir(exist_ok=True)

        df_to_csv_safe(alpha_raw_df, alpha_out / "alpha_traces_raw.csv")
        df_to_csv_safe(alpha_corr_df, alpha_out / "alpha_traces_corrected.csv")
        df_to_csv_safe(alpha_fit_df, alpha_out / "alpha_contamination_fits.csv")

        if SAVE_REMOVED_TRACES and alpha_removed_df is not None:
            df_to_csv_safe(
                alpha_removed_df,
                alpha_out / "alpha_traces_removed_contamination_only.csv",
            )

        if WRITE_CONTAM_PCT_CSV and alpha_removed_df is not None:
            pct_df = compute_contam_percent_df(
                alpha_raw_df,
                alpha_removed_df,
                kind=CONTAM_PCT_KIND,
                denom_mode=CONTAM_PCT_DENOM,
            )

            pct_path = (
                alpha_out /
                f"alpha_contamination_percent_of_{CONTAM_PCT_DENOM}_{CONTAM_PCT_KIND}_per_frame.csv"
            )

            df_to_csv_safe(pct_df, pct_path)
            print(f"Saved: {pct_path}")

        if WRITE_CONTAM_PCT_BOTH_CSV and alpha_removed_df is not None:
            pct_both_df = compute_contam_percent_both_df(
                alpha_raw_df,
                alpha_removed_df,
            )

            pct_both_path = (
                alpha_out /
                f"alpha_contamination_percent_of_{CONTAM_PCT_DENOM}_BOTH_per_frame.csv"
            )

            df_to_csv_safe(pct_both_df, pct_both_path)
            print(f"Saved: {pct_both_path}")

        if WRITE_SIGNAL_ATTRIB_FRACTION_PER_FRAME_CSV and alpha_removed_df is not None:
            attrib_df = compute_signal_attribution_fraction_per_frame_df(
                alpha_raw_df,
                alpha_removed_df,
                baseline_start=BASELINE_START,
                baseline_end=BASELINE_END,
                positive_only=ATTRIB_POSITIVE_ONLY,
                clip_0_1=ATTRIB_CLIP_0_1,
                eps=ATTRIB_EPS,
            )

            attrib_path = (
                alpha_out /
                "alpha_signal_attribution_fraction_from_beta_per_frame.csv"
            )

            df_to_csv_safe(attrib_df, attrib_path)
            print(f"Saved: {attrib_path}")

        if WRITE_SIGNAL_ATTRIB_SUMMARY_PER_CELL_CSV and alpha_removed_df is not None:
            windows = {
                "beta_win": (BETA_WIN_START, BETA_WIN_END),
                "alpha_win": (ALPHA_WIN_START, ALPHA_WIN_END),
            }

            summary_df = compute_signal_attribution_summary_per_cell_df(
                alpha_raw_df,
                alpha_removed_df,
                baseline_start=BASELINE_START,
                baseline_end=BASELINE_END,
                windows_dict=windows,
                positive_only=ATTRIB_POSITIVE_ONLY,
                clip_0_1=ATTRIB_CLIP_0_1,
                eps=ATTRIB_EPS,
            )

            summary_path = (
                alpha_out /
                "alpha_signal_attribution_fraction_from_beta_summary_per_cell.csv"
            )

            df_to_csv_safe(summary_df, summary_path)
            print(f"Saved: {summary_path}")

        print(f"Saved alpha outputs in: {alpha_out}")

    # -------------------------------------------------------------------------
    # Run alpha-to-beta correction
    # -------------------------------------------------------------------------

    beta_raw_df = None
    beta_corr_df = None
    beta_removed_df = None
    beta_fit_df = None
    beta_fit_info = {}

    if DO_ALPHA_TO_BETA_CLEAN:
        print("\n=== Running ALPHA -> BETA contamination correction ===")

        if K_FIT_WINDOW_A2B.lower() == "baseline":
            k_fit_window = "baseline"
            custom_window = None

        elif K_FIT_WINDOW_A2B.lower() == "alpha":
            k_fit_window = "alpha"
            custom_window = (ALPHA_WIN_START, ALPHA_WIN_END)

        else:
            raise ValueError("K_FIT_WINDOW_A2B must be 'baseline' or 'alpha'")

        (
            beta_raw_df,
            beta_corr_df,
            beta_removed_df,
            beta_fit_df,
            beta_fit_info,
        ) = run_contam_correction(
            target_ids=beta_ids,
            source_ids=alpha_ids,
            target_traces=beta_traces,
            source_traces=alpha_traces,
            target_label="beta",
            source_label="alpha",
            k_fit_window=k_fit_window,
            custom_fit_window=custom_window,
            baseline_window=(BASELINE_START, BASELINE_END),
            top_k=TOP_K_SOURCES,
            dist_power=DIST_POWER,
            min_dist=MIN_DIST,
            ridge_lambda=RIDGE_LAMBDA,
            correction_mode=CORRECTION_MODE,
            fit_intercept=FIT_INTERCEPT,
        )

        beta_out = out_dir / "beta_outputs"
        beta_out.mkdir(exist_ok=True)

        df_to_csv_safe(beta_raw_df, beta_out / "beta_traces_raw.csv")
        df_to_csv_safe(beta_corr_df, beta_out / "beta_traces_corrected.csv")
        df_to_csv_safe(beta_fit_df, beta_out / "beta_contamination_fits.csv")

        if SAVE_REMOVED_TRACES and beta_removed_df is not None:
            df_to_csv_safe(
                beta_removed_df,
                beta_out / "beta_traces_removed_contamination_only.csv",
            )

        if WRITE_CONTAM_PCT_CSV and beta_removed_df is not None:
            pct_df = compute_contam_percent_df(
                beta_raw_df,
                beta_removed_df,
                kind=CONTAM_PCT_KIND,
                denom_mode=CONTAM_PCT_DENOM,
            )

            pct_path = (
                beta_out /
                f"beta_contamination_percent_of_{CONTAM_PCT_DENOM}_{CONTAM_PCT_KIND}_per_frame.csv"
            )

            df_to_csv_safe(pct_df, pct_path)
            print(f"Saved: {pct_path}")

        if WRITE_CONTAM_PCT_BOTH_CSV and beta_removed_df is not None:
            pct_both_df = compute_contam_percent_both_df(
                beta_raw_df,
                beta_removed_df,
            )

            pct_both_path = (
                beta_out /
                f"beta_contamination_percent_of_{CONTAM_PCT_DENOM}_BOTH_per_frame.csv"
            )

            df_to_csv_safe(pct_both_df, pct_both_path)
            print(f"Saved: {pct_both_path}")

        if WRITE_SIGNAL_ATTRIB_FRACTION_PER_FRAME_CSV and beta_removed_df is not None:
            attrib_df = compute_signal_attribution_fraction_per_frame_df(
                beta_raw_df,
                beta_removed_df,
                baseline_start=BASELINE_START,
                baseline_end=BASELINE_END,
                positive_only=ATTRIB_POSITIVE_ONLY,
                clip_0_1=ATTRIB_CLIP_0_1,
                eps=ATTRIB_EPS,
            )

            attrib_path = (
                beta_out /
                "beta_signal_attribution_fraction_from_alpha_per_frame.csv"
            )

            df_to_csv_safe(attrib_df, attrib_path)
            print(f"Saved: {attrib_path}")

        if WRITE_SIGNAL_ATTRIB_SUMMARY_PER_CELL_CSV and beta_removed_df is not None:
            windows = {
                "beta_win": (BETA_WIN_START, BETA_WIN_END),
                "alpha_win": (ALPHA_WIN_START, ALPHA_WIN_END),
            }

            summary_df = compute_signal_attribution_summary_per_cell_df(
                beta_raw_df,
                beta_removed_df,
                baseline_start=BASELINE_START,
                baseline_end=BASELINE_END,
                windows_dict=windows,
                positive_only=ATTRIB_POSITIVE_ONLY,
                clip_0_1=ATTRIB_CLIP_0_1,
                eps=ATTRIB_EPS,
            )

            summary_path = (
                beta_out /
                "beta_signal_attribution_fraction_from_alpha_summary_per_cell.csv"
            )

            df_to_csv_safe(summary_df, summary_path)
            print(f"Saved: {summary_path}")

        print(f"Saved beta outputs in: {beta_out}")

    print("\nDataset done.\n")


# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_batch_with_preflight():
    """
    Run the full batch after printing matched datasets.

    In Jupyter:
      - Set ASK_BEFORE_RUNNING = True for safety
      - Set ASK_BEFORE_RUNNING = False for unattended batch runs
    """

    matched, skipped = find_pairs(BASE_DIR)

    print_matches(matched, skipped)

    if not matched:
        print("\nNo matched datasets found. Exiting.")
        return

    if ASK_BEFORE_RUNNING:
        proceed = prompt_yes_no("\nProceed with these matches? [Y/N]: ")

        if not proceed:
            print("Aborted by user.")
            return

    print("\nRunning batch...\n")

    for movie_path, mask_path, alpha_csv, beta_csv in matched:
        out_dir = build_out_dir(movie_path)

        process_one_dataset(
            movie_path=movie_path,
            mask_path=mask_path,
            alpha_csv=alpha_csv,
            beta_csv=beta_csv,
            out_dir=out_dir,
        )

    print("\nAll done.")


# =============================================================================
# RUN
# =============================================================================

run_batch_with_preflight()


# In[ ]:




