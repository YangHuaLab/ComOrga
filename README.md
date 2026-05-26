# ComOrga Profiling Framework

ComOrga is an automated computational framework designed for high-resolution 3D cellular image processing and spatial phenotyping. The pipeline precisely quantifies subcellular morphological features and organelle interaction networks (such as mitochondria, endoplasmic reticulum, and MERCs/MAMs) across multi-channel fluorescence images. By leveraging high-dimensional dimensionality reduction and a comprehensive scoring system, it enables accurate phenotypic discrimination of cell populations (e.g., macrophage polarization states: M0, M1, and M2).

This repository contains the complete workflow from 3D image reconstruction and multi-worker feature extraction to t-SNE visualization and PCA comprehensive scoring.

## 🛠 Dependencies & Environment

Running this framework requires setting up both Python and R environments:

* **Python Environment (Image Stacking, Feature Extraction & PCA Scoring):**
    * `Python >= 3.8`
    * `tifffile`, `tifftools`
    * `CellProfiler` (configured for headless multi-worker execution)
    * `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`
* **R Environment (t-SNE Dimensionality Reduction & Visualization):**
    * `R >= 4.0`
    * `Rtsne`, `ggplot2`, `ggpubr`, `ggthemes`

## 🚀 Core Workflow

The ComOrga framework consists of four sequential stages corresponding to Figures A–D:

### (A) 3D Image Reconstruction
**Script:** `1. concattif.ipynb`
* **Description:** Reads, stacks, and compiles sequential single-layer Z-stack raw TIF images (e.g., 45 layers) into a unified 3D TIF file.
* **Mechanism:** Utilizes `tifffile` and `tifftools` for high-throughput I/O operations, establishing standardized inputs for 3D volumetric rendering and downstream feature extraction.

### (B) Parameter Extraction
**Script:** `2. multiworker.ipynb`
* **Description:** Executes headless CellProfiler pipelines to segment 3D cellular volumes and extract 754 spatial and morphological parameters into a comprehensive CSV dataset.
* **Mechanism:** Optimizes processing for massive high-content screening datasets using a custom multi-worker `Runner` pipeline. Configuring 32 parallel workers significantly reduces processing times for single-cell level feature profiling (including nucleus, mitochondria, ER, and their colocalization/interaction metrics).

### (C) Dimensionality Reduction
**Scripts:** `3. tsne_welll.R` (Well-level) & `4. tsne_single cell.R` (Single-cell level)
* **Description:** Imports the extracted feature matrices from CellProfiler, applies the t-SNE algorithm to project high-dimensional data onto a 2D plane, and visualizes cellular clustering.
* **Mechanism:** Features a dual-track analysis capability for both well-averaged population profiles and single-cell level datasets (e.g., `nuc+mi+er.csv`). Built-in color palettes and publication-ready plotting logic clearly differentiate distinct clusters and polarization states (Control/M0, M1, M2).

### (D) Comprehensive Scoring
**Script:** `5. PCA_comprehensive score.py`
* **Description:** Performs Principal Component Analysis (PCA) on the cleaned high-dimensional feature datasets and automatically computes a **Comprehensive Phenotypic Score (S)** for each sample group.
* **Algorithmic Detail:** Rather than relying on a simple projection onto a single principal component, the framework calculates the score as the weighted sum of principal component scores multiplied by their corresponding explained variance ratios:
  $$S = \sum_{i=1}^{n} (\text{PC}_i \times \text{Explained Variance Ratio}_i)$$
  This strategy scientifically and comprehensively quantifies the global phenotypic shifts induced by natural products, treatments, or genetic knockdowns.
* **Data Processing & Tolerance:** Automatically detects and mitigates duplicate column naming anomalies (such as `Metadata_Well.1`) typically generated during pandas CSV concatenation, ensuring absolute accuracy in sample group mapping.

## 💻 Quick Start

1. **Image Preprocessing:** Open `1. concattif.ipynb` in Jupyter Notebook, configure your raw image directory paths, and execute the cells to generate merged 3D `.tif` files.
2. **Feature Extraction:** Initialize your CellProfiler-supported python environment and run `2. multiworker.ipynb`. Adjust the `num_workers` parameter based on your workstation's CPU core capacity.
3. **Dimensionality Reduction:** Feed the exported feature CSV files into the R environment and execute the t-SNE scripts to output high-resolution `.tiff` plots.
4. **PCA Scoring & Statistics:** Open `5. PCA_comprehensive score.py`, navigate to `# 1. Experimental Configuration`, update the `INPUT_PATH` (supports a single file or directory traversal) and `GROUP_CONFIG` (set group names and colors), then run the script to generate comprehensive score boxplots with automated statistical validations.

---
**Note:** Before running the PCA script, verify your feature axis direction (`FEATURE_DIRECTION = 'vertical'` or `'horizontal'`) to ensure proper matrix transposition.
