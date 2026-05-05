# Somatostatin receptors shape insulin and glucagon output within the pancreatic islet in mice through direct and paracrine effects

&#x20;

Ryan G. Hart1, Jordan J. Lee1, Karen Zhai1, Sharlene Lee1, Rashita Chauhan1, Aidean Hosseini1, Austin D. Nguyen1, Mark O. Huising1,2



1 Department of Neurobiology, Physiology and Behavior, University of California Davis, Davis, CA, USA

2 Department of Physiology and Membrane Biology, University of California Davis, Davis, CA, USA





# Islet Imaging Analysis Pipeline Scripts



This repository contains Python scripts used for processing and analyzing microscopy imaging data from islet experiments. The workflow presented is designed to begin with multi image TIFs acquired in NIS Elements, Nikon's proprietary software.  These files are saved with the file extension .ND2.  We anticipate these approaches can be applied to multi image TIF stacks acquired with other software.  The scripts support ND2 file splitting, affine image stabilization, Cellpose-based ROI segmentation, ROI intensity extraction, FRET ratio calculation, CSV normalization, statistical range extraction, ROI contamination correction, and ROI-based visualization.



## Repository contents



* `ND2 File Split and Stabilize.py` - Splits ND2 microscopy files into fields of view and channels, saves raw TIFF stacks, performs affine ECC-based stabilization, applies the same stabilization transforms across channels, and exports median, maximum, and minimum projections.



* `Headless Cellpose Segmentation.py` - Recursively finds projection TIFFs, segments ROIs with either a custom or built-in Cellpose model, optionally erodes or dilates masks, saves labeled mask TIFFs, and writes a segmentation run log.



* `Mean Intensity of Cellpose ROIS .py` - Recursively matches multi-frame microscopy TIFFs to labeled Cellpose mask TIFFs by FOV number, calculates mean intensity inside each ROI for every frame, and saves one CSV per movie/mask pair.



* `Raw and Internal Normalization FRET Calculator.py` - Recursively matches CFP and YFP ROI intensity CSVs, calculates raw channel ratios such as CFP/YFP or YFP/CFP, and also calculates internally normalized FRET ratios such as CFP/(CFP+YFP) or YFP/(CFP+YFP).



* `Rescale Imaging Data.py` - Recursively finds matching CSV files, rescales the first column to a user-defined range, rescales all remaining signal columns to 0-1, removes invalid columns when enabled, and saves normalized CSV files.



* `CSV splitting T Test .py` - Recursively scans CSV files, compares user-defined test and standard ranges using column-wise t-tests, and separates columns into significant decrease, significant increase, or not significant output CSVs.



* `Extract Baselines and Test Ranges from CSV.py` - Recursively searches target CSV files, calculates baseline/test statistics across user-defined ranges, including mean response, percent change, normalized AUC, and normalized AUC percent change, then saves vertical and horizontal summary CSVs.



* `ROI Contamination Cleanup.py` - Recursively matches movie TIFFs, labeled masks, alpha CSVs, and beta CSVs, estimates distance-weighted signal contamination between nearby ROI classes, subtracts fitted contamination traces, and exports corrected traces, removed-contamination traces, fit summaries, and contamination percentage outputs.



* `ROI Visualization Pipeline.py` - Generates multi-frame RGB TIFF visualizations where labeled ROIs are colored according to alpha and/or beta CSV intensity values over time.



## Suggested workflow



1. Use `ND2 File Split and Stabilize.py` to split ND2 files, save raw/stabilized TIFF stacks, and generate projections.
2. Use `Headless Cellpose Segmentation.py` to segment median projection images and create labeled ROI masks.
3. Use `Mean Intensity of Cellpose ROIS .py` to quantify mean ROI intensity over time from the stabilized image stacks.
4. Use `Raw and Internal Normalization FRET Calculator.py` for CFP/YFP FRET ratio calculations when applicable.
5. Use `Rescale Imaging Data.py` to normalize trace data before downstream statistical analysis.
6. Use `CSV splitting T Test .py` to separate cell columns using cell specific behavior EG. Adrenaline activation of alpha and deactivation of beta cell calcium or cAMP.
7. Use `Extract Baselines and Test Ranges from CSV.py` for window-based response classification and summary statistics during treatment.
8. Use `ROI Contamination Cleanup.py` when alpha/beta ROI cross-contamination correction is required.
9. Use `ROI Visualization Pipeline.py` to create ROI-colored time-series TIFF visualizations.



## Getting started



All scripts were written in Python and are intended to be run locally after editing the user-configurable variables near the top of each file. Most scripts were written in a Jupyter-notebook style but can also be run as `.py` files.

Before running a script, edit the relevant folder paths and settings, such as:



* input or parent directory
* filename keywords
* channel identifiers
* FOV matching rules
* Cellpose model settings
* baseline/test windows
* normalization ranges
* output naming options



## Python version

Recommended:

```text
Python 3.10+
```

The scripts may also work in other Python 3 versions, but the imaging and Cellpose components are most likely to be environment-sensitive.



## Required packages by script

|Script|Required third-party packages|
|-|-|
|`ND2 File Split and Stabilize.py`|`numpy`, `opencv-python`, `tifffile`, `nd2reader`, `tqdm`|
|`Headless Cellpose Segmentation.py`|`numpy`, `pandas`, `tifffile`, `cellpose`, `scikit-image`, `scipy`|
|`Mean Intensity of Cellpose ROIS .py`|`numpy`, `pandas`, `scikit-image`|
|`Raw and Internal Normalization FRET Calculator.py`|`numpy`, `pandas`|
|`Rescale Imaging Data.py`|`numpy`, `pandas`, `scikit-learn`|
|`CSV splitting T Test .py`|`pandas`, `scipy`|
|`Extract Baselines and Test Ranges from CSV.py`|`numpy`, `pandas`, `scipy`|
|`ROI Contamination Cleanup.py`|`numpy`, `pandas`, `tifffile`|
|`ROI Visualization Pipeline.py`|`numpy`, `pandas`, `tifffile`, `matplotlib`|

Standard-library modules used across the scripts include `os`, `re`, `pathlib`, `datetime`, `glob`, `hashlib`, and `collections`.



## Consolidated package list

Install the following packages to support the full repository:

```bash
pip install numpy pandas scipy matplotlib tifffile opencv-python scikit-image scikit-learn tqdm nd2reader 
```

## Package Versions Used in Original Environment

These versions reflect the environment used while generating the data in the manuscript listed. The package is likely to have been updated since.

|Package|Version identified|
|-|-|
|`numpy`|2.3.1|
|`pandas`|2.2.3|
|`scipy`|1.15.3|
|`matplotlib`|3.10.0|
|`tifffile`|2025.2.18|
|`opencv-python-headless'|4.12.0.88|
|`scikit-image`|0.25.0|
|`scikit-learn`|1.6.1|
|`tqdm`|4.67.3|
|`nd2reader`|3.3.1|
|`cellpose`|Installed in conda environment.  The segmentation environment used cellpose version 3.1.1.2|

## 

## Notes on reproducibility

* Update target path before running script
* Cellpose scripts may require GPU-compatible PyTorch and CUDA installation depending on the local machine and `USE\_GPU` settings.
* The ND2 splitting script requires `nd2reader` and may depend on the metadata structure of the ND2 files.
* Output filenames and matching behavior are controlled by user-editable constants near the top of each script.





## 

